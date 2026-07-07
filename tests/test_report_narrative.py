"""The persona-toned executive summary in ``report_generation`` (§5.8).

The narrative is an LLM call whose SYSTEM prompt is the hot-loaded persona and
whose user prompt carries the masked rows plus retrieved-trio style notes. It
must degrade — never block — when the LLM is unavailable, and the deterministic
PII sweep must still run over whatever the LLM produced. All offline.
"""

from __future__ import annotations

from typing import Any

import pytest
from langchain_core.messages import HumanMessage

from agent import graph as graph_module
from tools.llm import LLMError

_ROWS: list[dict[str, object]] = [
    {"category": "Jeans", "revenue": 12345.6},
    {"category": "Shoes", "revenue": 9876.5},
]


def _state() -> dict[str, Any]:
    return {
        "messages": [HumanMessage(content="Which categories drive revenue?")],
        "persona": "You are a blunt, numbers-first analyst.",
        "masked": True,
        "result_rows": list(_ROWS),
        "retrieved_trios": [
            {
                "id": "05",
                "question": "q",
                "sql": "s",
                "report": "Lead with the top category.",
                "score": 0.9,
            }
        ],
    }


class TestNarrativeSummary:
    def test_summary_leads_the_report(self, monkeypatch: pytest.MonkeyPatch) -> None:
        captured: dict[str, str] = {}

        def fake_generate(system: str, user: str) -> str:
            captured["system"] = system
            captured["user"] = user
            return "Jeans lead revenue at $12,345.60, ahead of Shoes."

        monkeypatch.setattr(graph_module, "generate", fake_generate)
        out = str(graph_module.report_generation(_state())["final_response"])  # type: ignore[arg-type]

        assert out.startswith("Jeans lead revenue")
        assert "Shoes" in out  # data body still present below the summary
        # Persona is the system prompt; rows and style notes reach the user prompt.
        assert captured["system"] == "You are a blunt, numbers-first analyst."
        assert "12345.6" in captured["user"]
        assert "Lead with the top category." in captured["user"]

    def test_llm_failure_degrades_to_data_only(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def boom(system: str, user: str) -> str:
            raise LLMError("provider down")

        monkeypatch.setattr(graph_module, "generate", boom)
        out = str(graph_module.report_generation(_state())["final_response"])  # type: ignore[arg-type]
        assert out.startswith("Here is what I found")
        assert "Jeans" in out

    def test_summary_prose_is_still_pii_masked(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            graph_module,
            "generate",
            lambda system, user: "Top buyer reachable at jane@example.com spent the most.",
        )
        out = str(graph_module.report_generation(_state())["final_response"])  # type: ignore[arg-type]
        assert "jane@example.com" not in out
        assert "«email redacted»" in out

    def test_no_persona_means_no_llm_call(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def boom(system: str, user: str) -> str:
            raise AssertionError("generate must not be called without a persona")

        monkeypatch.setattr(graph_module, "generate", boom)
        state = _state()
        del state["persona"]
        out = str(graph_module.report_generation(state)["final_response"])  # type: ignore[arg-type]
        assert out.startswith("Here is what I found")
