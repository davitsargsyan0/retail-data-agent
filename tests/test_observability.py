"""Observability-lite node instrumentation (architecture §5.7, prototype slice).

Verifies the uniform ``instrument`` wrapper: one JSON line per node execution
with the required fields, LLM model/tokens attribution when the node called the
LLM, and error capture that re-raises rather than swallowing.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from agent import observability
from agent.observability import instrument
from tools import llm


class _FakeResponse:
    """Stands in for an AIMessage carrying usage_metadata."""

    usage_metadata = {"input_tokens": 3, "output_tokens": 5, "total_tokens": 8}


@pytest.fixture
def log_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    path = tmp_path / "agent.jsonl"
    monkeypatch.setattr(observability, "LOG_DIR", tmp_path)
    monkeypatch.setattr(observability, "LOG_PATH", path)
    return path


def _records(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_one_line_per_node_with_required_fields(log_path: Path) -> None:
    def node(state: Any) -> Any:
        return {"masked": True}

    instrument("pii_mask", node)({"trace_id": "trace-1"})

    records = _records(log_path)
    assert len(records) == 1
    rec = records[0]
    assert set(rec) >= {"trace_id", "node", "latency_ms", "model", "tokens", "error"}
    assert rec["trace_id"] == "trace-1"
    assert rec["node"] == "pii_mask"
    assert rec["error"] is None
    assert rec["model"] is None and rec["tokens"] is None
    assert isinstance(rec["latency_ms"], (int, float))


def test_attributes_model_and_tokens_when_node_calls_llm(log_path: Path) -> None:
    def node(state: Any) -> Any:
        # Simulate an in-node LLM call recording usage into the active sink.
        llm._record_usage(_FakeResponse(), "gemini-test")  # type: ignore[arg-type]
        return {}

    instrument("sql_generation", node)({"trace_id": "t"})

    rec = _records(log_path)[0]
    assert rec["model"] == "gemini-test"
    assert rec["tokens"] == 8


def test_records_error_then_reraises(log_path: Path) -> None:
    def node(state: Any) -> Any:
        raise ValueError("boom")

    with pytest.raises(ValueError, match="boom"):
        instrument("bigquery_execute", node)({"trace_id": "t"})

    rec = _records(log_path)[0]
    assert rec["error"] == "ValueError: boom"


def test_debug_mirrors_to_stderr(
    log_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(observability, "_debug", True)

    def node(state: Any) -> Any:
        return {}

    instrument("load_context", node)({"trace_id": "t"})
    err = capsys.readouterr().err
    assert '"node": "load_context"' in err
