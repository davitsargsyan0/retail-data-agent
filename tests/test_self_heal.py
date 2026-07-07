"""Unit tests for the self-heal routing and graceful failure — no network."""

from __future__ import annotations

from agent.graph import (
    MAX_HEAL_ATTEMPTS,
    _route_after_execute,
    _route_after_guard,
    _route_after_repair,
    graceful_failure,
)
from agent.state import AgentState, HealAttempt


class TestRouting:
    def test_guard_pass_goes_to_execute(self) -> None:
        state: AgentState = {"sql_error": None, "retry_count": 0}
        assert _route_after_guard(state) == "bigquery_execute"

    def test_guard_failure_heals_when_budget_remains(self) -> None:
        state: AgentState = {"sql_error": "Unrecognized name: foo", "retry_count": 0}
        assert _route_after_guard(state) == "sql_repair"
        state["retry_count"] = MAX_HEAL_ATTEMPTS - 1
        assert _route_after_guard(state) == "sql_repair"

    def test_guard_failure_gives_up_at_cap(self) -> None:
        state: AgentState = {"sql_error": "Unrecognized name: foo", "retry_count": 2}
        assert _route_after_guard(state) == "graceful_failure"

    def test_execute_success_goes_to_report(self) -> None:
        state: AgentState = {"sql_error": None, "retry_count": 1}
        assert _route_after_execute(state) == "report_generation"

    def test_empty_result_triggers_heal(self) -> None:
        state: AgentState = {"sql_error": "Query returned zero rows.", "retry_count": 0}
        assert _route_after_execute(state) == "sql_repair"

    def test_empty_result_gives_up_at_cap(self) -> None:
        state: AgentState = {"sql_error": "Query returned zero rows.", "retry_count": 2}
        assert _route_after_execute(state) == "graceful_failure"

    def test_repaired_sql_goes_back_through_guard(self) -> None:
        state: AgentState = {"sql_error": None, "sql": "SELECT 1", "retry_count": 1}
        assert _route_after_repair(state) == "sql_guard"

    def test_repair_llm_failure_exits_gracefully(self) -> None:
        state: AgentState = {"sql_error": "LLM unavailable during repair: boom", "retry_count": 1}
        assert _route_after_repair(state) == "graceful_failure"


class TestGracefulFailure:
    def test_lists_every_attempt(self) -> None:
        attempts = [
            HealAttempt(sql="SELECT bad_col FROM t", error="Unrecognized name: bad_col"),
            HealAttempt(sql="SELECT other FROM t", error="Query returned zero rows."),
        ]
        state: AgentState = {"heal_attempts": attempts, "retry_count": 2}
        message = graceful_failure(state)["final_response"]
        assert "Here's what I tried:" in message
        assert "1. `SELECT bad_col FROM t ...` → Unrecognized name: bad_col" in message
        assert "2. `SELECT other FROM t ...` → Query returned zero rows." in message
        assert "rephrasing" in message

    def test_without_attempts_falls_back_to_reason(self) -> None:
        state: AgentState = {"sql_error": "LLM unavailable: boom", "heal_attempts": []}
        message = graceful_failure(state)["final_response"]
        assert "LLM unavailable: boom" in message
        assert "Here's what I tried" not in message

    def test_never_leaks_a_stack_trace(self) -> None:
        state: AgentState = {"heal_attempts": [], "sql_error": None}
        message = graceful_failure(state)["final_response"]
        assert "Traceback" not in message
