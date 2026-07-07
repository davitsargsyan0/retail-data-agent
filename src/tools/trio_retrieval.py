"""Golden-bucket trio retrieval — numpy cosine similarity over Gemini embeddings.

Trios live as JSON files in ``golden_bucket/trios/`` (``{question, sql,
report_style_notes, tables_used}``). Question embeddings are computed once via
the single LLM wrapper (`tools.llm`) and cached to
``golden_bucket/embeddings.npy`` with a content-hash sidecar
(``embeddings.meta.json``) — the cache is recomputed automatically whenever any
trio file or the embedding model changes. No vector DB (ADR-002): at
golden-bucket scale an O(n) numpy scan is sub-millisecond.
"""

from __future__ import annotations

import hashlib
import json
import logging
from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np
import numpy.typing as npt

from agent.state import Trio
from tools import llm

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]
TRIOS_DIR = REPO_ROOT / "golden_bucket" / "trios"
CACHE_PATH = REPO_ROOT / "golden_bucket" / "embeddings.npy"
META_PATH = REPO_ROOT / "golden_bucket" / "embeddings.meta.json"

TOP_K = 3
SIMILARITY_FLOOR = 0.60  # drop weak matches rather than inject misleading examples

EmbedTextsFn = Callable[[Sequence[str]], list[list[float]]]
EmbedQueryFn = Callable[[str], list[float]]


def _normalize_rows(matrix: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """L2-normalize each row so dot products become cosine similarities."""
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return matrix / norms


class TrioRetriever:
    """Loads trios, manages the embedding cache, and answers top-k queries."""

    def __init__(
        self,
        trios_dir: Path = TRIOS_DIR,
        cache_path: Path = CACHE_PATH,
        meta_path: Path = META_PATH,
        embed_texts_fn: EmbedTextsFn = llm.embed_texts,
        embed_query_fn: EmbedQueryFn = llm.embed_query,
    ) -> None:
        self._trios_dir = trios_dir
        self._cache_path = cache_path
        self._meta_path = meta_path
        self._embed_texts = embed_texts_fn
        self._embed_query = embed_query_fn
        self._trios: list[Trio] = []
        self._matrix: npt.NDArray[np.float64] | None = None

    def _read_trio_files(self) -> list[tuple[str, str]]:
        """Return sorted ``(file_name, raw_content)`` pairs."""
        return [
            (path.name, path.read_text(encoding="utf-8"))
            for path in sorted(self._trios_dir.glob("*.json"))
        ]

    def _corpus_hash(self, files: list[tuple[str, str]]) -> str:
        """Content hash of every trio file plus the embedding model name."""
        digest = hashlib.sha256()
        digest.update(llm.embedding_model_name().encode("utf-8"))
        for name, content in files:
            digest.update(name.encode("utf-8"))
            digest.update(content.encode("utf-8"))
        return digest.hexdigest()

    def _load_cached_matrix(self, corpus_hash: str, count: int) -> npt.NDArray[np.float64] | None:
        """Load the cached embedding matrix if it matches the current corpus."""
        if not (self._cache_path.exists() and self._meta_path.exists()):
            return None
        try:
            meta = json.loads(self._meta_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        if meta.get("corpus_hash") != corpus_hash:
            return None
        matrix: npt.NDArray[np.float64] = np.asarray(
            np.load(self._cache_path), dtype=np.float64
        )
        if matrix.shape[0] != count:
            return None
        logger.info("trio embeddings loaded from cache (%d trios)", count)
        return matrix

    def _save_cache(self, matrix: npt.NDArray[np.float64], corpus_hash: str) -> None:
        self._cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(self._cache_path, matrix)
        meta = {
            "corpus_hash": corpus_hash,
            "embedding_model": llm.embedding_model_name(),
            "count": matrix.shape[0],
        }
        self._meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")

    def load(self) -> None:
        """Load trios and their embeddings (cache hit or fresh embedding call)."""
        files = self._read_trio_files()
        self._trios = []
        for name, content in files:
            record = json.loads(content)
            self._trios.append(
                Trio(
                    id=Path(name).stem,
                    question=str(record["question"]),
                    sql=str(record["sql"]),
                    report=str(record.get("report_style_notes", "")),
                    score=0.0,
                )
            )
        if not self._trios:
            self._matrix = None
            logger.warning("no trios found in %s", self._trios_dir)
            return

        corpus_hash = self._corpus_hash(files)
        matrix = self._load_cached_matrix(corpus_hash, len(self._trios))
        if matrix is None:
            logger.info("embedding %d trio questions (cache miss)", len(self._trios))
            vectors = self._embed_texts([t["question"] for t in self._trios])
            matrix = np.asarray(vectors, dtype=np.float64)
            self._save_cache(matrix, corpus_hash)
        self._matrix = _normalize_rows(matrix)

    def _ensure_loaded(self) -> None:
        if self._matrix is None and not self._trios:
            self.load()

    def retrieve(self, question: str, k: int = TOP_K) -> list[Trio]:
        """Top-k trios by cosine similarity, dropping matches below the floor."""
        self._ensure_loaded()
        if self._matrix is None or not self._trios:
            return []
        query = np.asarray(self._embed_query(question), dtype=np.float64)
        norm = float(np.linalg.norm(query))
        if norm == 0.0:
            return []
        similarities = self._matrix @ (query / norm)
        order = np.argsort(similarities)[::-1][:k]
        results: list[Trio] = []
        for index in order:
            score = float(similarities[index])
            if score < SIMILARITY_FLOOR:
                continue
            trio = self._trios[int(index)].copy()
            trio["score"] = round(score, 4)
            results.append(trio)
        return results
