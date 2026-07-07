"""Unit tests for trio retrieval and the embedding cache — no network.

Uses fake embedding functions: three orthogonal unit vectors, so cosine
similarities are exact and predictable.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

import pytest

from tools.trio_retrieval import TrioRetriever

QUESTIONS_TO_VECTORS: dict[str, list[float]] = {
    "Who are our top customers?": [1.0, 0.0, 0.0],
    "What was monthly revenue in 2023?": [0.0, 1.0, 0.0],
    "Which products sell best?": [0.0, 0.0, 1.0],
}


def fake_embed_texts(texts: Sequence[str]) -> list[list[float]]:
    return [QUESTIONS_TO_VECTORS[t] for t in texts]


def _write_trios(directory: Path) -> None:
    for i, question in enumerate(QUESTIONS_TO_VECTORS, start=1):
        payload = {
            "question": question,
            "sql": f"SELECT {i} AS x LIMIT 1",
            "report_style_notes": f"style {i}",
            "tables_used": ["orders"],
        }
        (directory / f"{i:02d}_trio.json").write_text(json.dumps(payload), encoding="utf-8")


@pytest.fixture
def retriever(tmp_path: Path) -> TrioRetriever:
    trios_dir = tmp_path / "trios"
    trios_dir.mkdir()
    _write_trios(trios_dir)
    return TrioRetriever(
        trios_dir=trios_dir,
        cache_path=tmp_path / "embeddings.npy",
        meta_path=tmp_path / "embeddings.meta.json",
        embed_texts_fn=fake_embed_texts,
        embed_query_fn=lambda _q: [0.9, 0.3, 0.0],
    )


class TestCosineRetrieval:
    def test_orders_by_similarity_and_applies_floor(self, retriever: TrioRetriever) -> None:
        results = retriever.retrieve("top customers question")
        # cosine vs [0.9, 0.3, 0] normalized: ~0.95, ~0.32, 0.0 — only the
        # first clears the 0.60 floor.
        assert [t["id"] for t in results] == ["01_trio"]
        assert results[0]["question"] == "Who are our top customers?"
        assert results[0]["score"] > 0.9

    def test_top_k_limits_results(self, tmp_path: Path) -> None:
        trios_dir = tmp_path / "trios"
        trios_dir.mkdir()
        _write_trios(trios_dir)
        retriever = TrioRetriever(
            trios_dir=trios_dir,
            cache_path=tmp_path / "e.npy",
            meta_path=tmp_path / "e.meta.json",
            # cosine ~0.71 and ~0.70 to the first two trios, 0.0 to the third
            embed_texts_fn=fake_embed_texts,
            embed_query_fn=lambda _q: [1.0, 0.98, 0.0],
        )
        assert len(retriever.retrieve("anything", k=1)) == 1
        # k=3 still returns only 2: the third trio is below the 0.60 floor
        assert [t["id"] for t in retriever.retrieve("anything", k=3)] == ["01_trio", "02_trio"]

    def test_maps_trio_fields(self, retriever: TrioRetriever) -> None:
        trio = retriever.retrieve("q")[0]
        assert trio["sql"] == "SELECT 1 AS x LIMIT 1"
        assert trio["report"] == "style 1"

    def test_empty_bucket_returns_no_trios(self, tmp_path: Path) -> None:
        trios_dir = tmp_path / "empty"
        trios_dir.mkdir()
        retriever = TrioRetriever(
            trios_dir=trios_dir,
            cache_path=tmp_path / "e.npy",
            meta_path=tmp_path / "e.meta.json",
            embed_texts_fn=fake_embed_texts,
            embed_query_fn=lambda _q: [1.0, 0.0, 0.0],
        )
        assert retriever.retrieve("anything") == []


class TestEmbeddingCache:
    def _make(self, tmp_path: Path, calls: list[int]) -> TrioRetriever:
        def counting_embed_texts(texts: Sequence[str]) -> list[list[float]]:
            calls.append(len(texts))
            return fake_embed_texts(texts)

        return TrioRetriever(
            trios_dir=tmp_path / "trios",
            cache_path=tmp_path / "embeddings.npy",
            meta_path=tmp_path / "embeddings.meta.json",
            embed_texts_fn=counting_embed_texts,
            embed_query_fn=lambda _q: [1.0, 0.0, 0.0],
        )

    def test_cache_written_then_reused(self, tmp_path: Path) -> None:
        (tmp_path / "trios").mkdir()
        _write_trios(tmp_path / "trios")
        calls: list[int] = []

        first = self._make(tmp_path, calls)
        first.load()
        assert calls == [3]  # embedded once
        assert (tmp_path / "embeddings.npy").exists()
        assert (tmp_path / "embeddings.meta.json").exists()

        second = self._make(tmp_path, calls)
        second.load()
        assert calls == [3]  # cache hit — no new embedding call
        assert second.retrieve("q")[0]["id"] == "01_trio"

    def test_cache_invalidated_when_trio_file_changes(self, tmp_path: Path) -> None:
        (tmp_path / "trios").mkdir()
        _write_trios(tmp_path / "trios")
        calls: list[int] = []
        self._make(tmp_path, calls).load()
        assert calls == [3]

        # Edit one trio (same question key so the fake embedder still works).
        target = tmp_path / "trios" / "01_trio.json"
        record = json.loads(target.read_text(encoding="utf-8"))
        record["sql"] = "SELECT 99 AS x LIMIT 1"
        target.write_text(json.dumps(record), encoding="utf-8")

        self._make(tmp_path, calls).load()
        assert calls == [3, 3]  # recomputed

    def test_cache_invalidated_when_trio_added(self, tmp_path: Path) -> None:
        (tmp_path / "trios").mkdir()
        _write_trios(tmp_path / "trios")
        calls: list[int] = []
        self._make(tmp_path, calls).load()

        QUESTIONS_TO_VECTORS["Extra question?"] = [0.5, 0.5, 0.0]
        try:
            extra = {
                "question": "Extra question?",
                "sql": "SELECT 4 LIMIT 1",
                "report_style_notes": "",
                "tables_used": [],
            }
            (tmp_path / "trios" / "04_trio.json").write_text(json.dumps(extra), encoding="utf-8")
            self._make(tmp_path, calls).load()
            assert calls == [3, 4]
        finally:
            del QUESTIONS_TO_VECTORS["Extra question?"]

    def test_corrupt_meta_triggers_recompute(self, tmp_path: Path) -> None:
        (tmp_path / "trios").mkdir()
        _write_trios(tmp_path / "trios")
        calls: list[int] = []
        self._make(tmp_path, calls).load()
        (tmp_path / "embeddings.meta.json").write_text("not json", encoding="utf-8")
        self._make(tmp_path, calls).load()
        assert calls == [3, 3]
