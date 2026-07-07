"""Full-graph self-heal loop: the retry counter is hard-capped at 2 (§5.5).

Drives the real compiled graph but stubs the two live seams (intent gate and the
LLM/BigQuery calls) so the test is deterministic and offline. The point is to
prove the *cost bound*: a query that never validates triggers exactly
``MAX_HEAL_ATTEMPTS`` repair rounds and then fails gracefully — never a stack
trace, never an unbounded loop.
"""

from __future__ import annotations

import uuid
from typing import Any

import pytest
from langchain_core.messages import HumanMessage

from agent import graph as graph_module
from agent.graph import MAX_HEAL_ATTEMPTS
from safety import intent_gate as intent_gate_module
from safety.intent_gate import IntentResult
from tools.bigquery_client import BigQueryExecutionError


def _drain(graph: Any, state: Any, config: dict[str, Any]) -> dict[str, Any]:
    """Run the stream, returning the last final_response and the max retry_count."""
    final = ""
    max_retry = 0
    for chunk in graph.stream(state, config=config, stream_mode="updates"):
        for update in chunk.values():
            if isinstance(update, dict):
                if update.get("final_response"):
                    final = str(update["final_response"])
                if isinstance(update.get("retry_count"), int):
                    max_retry = max(max_retry, int(update["retry_count"]))
    return {"final": final, "max_retry": max_retry}


@pytest.fixture
def counting_graph(monkeypatch: pytest.MonkeyPatch) -> tuple[Any, dict[str, int]]:
    """A graph whose intent is forced to analysis, trio retrieval is empty, the
    LLM always returns the same (invalid) SQL, and the dry-run always fails."""
    monkeypatch.setattr(
        intent_gate_module,
        "classify",
        lambda _q: IntentResult("analysis", "forced", "rule"),
    )
    monkeypatch.setattr(graph_module._retriever, "retrieve", lambda _q, k=3: [])

    calls = {"generate": 0}

    def fake_generate(_system: str, _user: str) -> str:
        calls["generate"] += 1
        return "SELECT nonexistent_col FROM `bigquery-public-data.thelook_ecommerce.orders`"

    monkeypatch.setattr(graph_module, "generate", fake_generate)

    def boom(_sql: str) -> int:
        raise BigQueryExecutionError("Unrecognized name: nonexistent_col at [1:8]")

    monkeypatch.setattr(graph_module._bq, "dry_run", boom)

    return graph_module.build_graph(), calls


class TestSelfHealCounterCap:
    def test_caps_at_two_repairs_then_fails_gracefully(
        self, counting_graph: tuple[Any, dict[str, int]]
    ) -> None:
        graph, calls = counting_graph
        config = {"configurable": {"thread_id": uuid.uuid4().hex}}
        state = {"messages": [HumanMessage(content="what was revenue last year?")]}

        result = _drain(graph, state, config)

        # 1 initial generation + exactly MAX_HEAL_ATTEMPTS repair calls.
        assert calls["generate"] == 1 + MAX_HEAL_ATTEMPTS
        assert result["max_retry"] == MAX_HEAL_ATTEMPTS
        assert "couldn't complete that analysis reliably" in result["final"]
        assert "Traceback" not in result["final"]

    def test_generation_is_bounded_never_unbounded(
        self, counting_graph: tuple[Any, dict[str, int]]
    ) -> None:
        graph, calls = counting_graph
        config = {"configurable": {"thread_id": uuid.uuid4().hex}}
        state = {"messages": [HumanMessage(content="another failing question?")]}
        _drain(graph, state, config)
        # The cost bound: never more than 1 + cap LLM calls, no matter what.
        assert calls["generate"] <= 1 + MAX_HEAL_ATTEMPTS
