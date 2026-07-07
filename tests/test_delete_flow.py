"""End-to-end test of the interrupt-guarded delete branch (architecture §5.3).

Runs the real compiled graph with an in-memory checkpointer. The delete path
("delete today's reports") is classified by a deterministic rule and touches no
LLM or BigQuery, so this whole flow runs offline. The store is redirected to a
temp directory."""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any

import pytest
from langchain_core.messages import HumanMessage
from langgraph.types import Command

from agent import graph as graph_module
from reports_store.store import ReportStore


def _run(graph: Any, graph_input: Any, config: dict[str, Any]) -> dict[str, Any]:
    """Drive the stream; return the interrupt payload (if any) and last response."""
    payload: dict[str, Any] | None = None
    final = ""
    for chunk in graph.stream(graph_input, config=config, stream_mode="updates"):
        for node, update in chunk.items():
            if node == "__interrupt__":
                payload = update[0].value
            elif isinstance(update, dict) and update.get("final_response"):
                final = str(update["final_response"])
    return {"interrupt": payload, "final": final}


@pytest.fixture
def store(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> ReportStore:
    store = ReportStore(tmp_path)
    monkeypatch.setattr(graph_module, "_store", store)
    return store


def _start(question: str) -> tuple[Any, dict[str, Any], dict[str, Any]]:
    graph = graph_module.build_graph()
    config: dict[str, Any] = {"configurable": {"thread_id": uuid.uuid4().hex}}
    state: dict[str, Any] = {
        "messages": [HumanMessage(content=question)],
        "user_id": "cli-user",
    }
    return graph, config, state


class TestDeleteFlow:
    def test_preview_is_owner_scoped(self, store: ReportStore) -> None:
        store.save("cli-user", "Report A", "body")
        store.save("cli-user", "Report B", "body")
        store.save("other-user", "Not mine", "body")

        graph, config, state = _start("delete today's reports")
        result = _run(graph, state, config)

        assert result["interrupt"] is not None
        assert result["interrupt"]["count"] == 2  # excludes other-user's report

    def test_typed_confirmation_deletes(self, store: ReportStore) -> None:
        store.save("cli-user", "Report A", "body")
        store.save("cli-user", "Report B", "body")

        graph, config, state = _start("delete today's reports")
        _run(graph, state, config)
        result = _run(graph, Command(resume="delete 2 reports"), config)

        assert "Deleted 2 reports" in result["final"]
        assert store.list_for_owner("cli-user") == []

    def test_wrong_phrase_cancels_cleanly(self, store: ReportStore) -> None:
        store.save("cli-user", "Report A", "body")
        store.save("cli-user", "Report B", "body")

        graph, config, state = _start("delete today's reports")
        _run(graph, state, config)
        result = _run(graph, Command(resume="yes"), config)

        assert "Cancelled" in result["final"]
        assert len(store.list_for_owner("cli-user")) == 2  # untouched

    def test_audit_record_written(self, store: ReportStore, tmp_path: Path) -> None:
        store.save("cli-user", "Report A", "body")
        graph, config, state = _start("delete all my reports")
        _run(graph, state, config)
        _run(graph, Command(resume="delete 1 report"), config)
        audit = (tmp_path / "audit.jsonl").read_text(encoding="utf-8")
        assert '"delete_reports"' in audit
        assert '"confirmation_text": "delete 1 report"' in audit

    def test_no_matches_short_circuits(self, store: ReportStore) -> None:
        graph, config, state = _start("delete all reports mentioning Nonexistent")
        result = _run(graph, state, config)
        assert result["interrupt"] is None
        assert "No saved reports match" in result["final"]
