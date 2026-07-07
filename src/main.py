"""CLI chat REPL for the retail data agent.

Run with:
    uv run python src/main.py

Streams graph progress node by node, then prints the final report. Type
``exit`` / ``quit`` (or Ctrl-D) to leave.
"""

from __future__ import annotations

import logging
import sys
import uuid
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Command

from agent.graph import build_graph
from agent.observability import set_debug
from agent.state import AgentState

REPO_ROOT = Path(__file__).resolve().parents[1]

_PROGRESS: dict[str, str] = {
    "load_context": "loading persona",
    "intent_router": "routing intent",
    "polite_refusal": "declining (out of scope)",
    "set_preference": "saving your preference",
    "trio_retrieval": "searching golden bucket",
    "sql_generation": "writing SQL",
    "sql_guard": "validating SQL (dry run)",
    "bigquery_execute": "querying BigQuery",
    "sql_repair": "self-healing SQL",
    "pii_mask": "masking PII",
    "report_generation": "writing report",
    "save_report": "saving report",
    "graceful_failure": "recovering",
    "match_reports": "matching saved reports",
    "confirm_delete": "awaiting confirmation",
    "execute_delete": "deleting reports",
    "cancel_delete": "cancelling",
}


def _print_node_update(node: str, update: dict[str, object]) -> None:
    """Render one streamed node update as CLI progress."""
    label = _PROGRESS.get(node, node)
    if node == "sql_repair":
        label += f" (attempt {update.get('retry_count', '?')})"
    print(f"  · {node} — {label}")
    if node == "intent_router" and update.get("intent"):
        print(f"    intent: {update['intent']}")
    if node == "pii_mask":
        print("    output scrubbed for PII (deterministic)")
    if node == "trio_retrieval":
        trios = update.get("retrieved_trios")
        if isinstance(trios, list) and trios:
            for trio in trios:
                if isinstance(trio, dict):
                    print(f"    retrieved: {trio.get('id')} (similarity {trio.get('score')})")
        else:
            print("    no similar past analyses found — using schema only")
    if node in ("sql_generation", "sql_repair") and update.get("sql"):
        header = "repaired SQL:" if node == "sql_repair" else "generated SQL:"
        print(f"    {header}")
        for line in str(update["sql"]).splitlines():
            print(f"      {line}")
    if update.get("sql_error"):
        print(f"    ! {update['sql_error']}")


def _drive(
    graph: CompiledStateGraph[AgentState, None, AgentState, AgentState],
    graph_input: AgentState | Command[Any],
    config: RunnableConfig,
) -> None:
    """Stream one graph run to completion, pausing at any ``interrupt()``.

    On an interrupt the paused state is already checkpointed; we render the
    preview, collect a typed confirmation, and resume the SAME thread with the
    answer injected — the recursion handles multiple interrupts if they arise.
    """
    final_response = ""
    interrupt_payload: dict[str, object] | None = None
    for chunk in graph.stream(graph_input, config=config, stream_mode="updates"):
        for node, update in chunk.items():
            if node == "__interrupt__":
                interrupts = update
                if isinstance(interrupts, (list, tuple)) and interrupts:
                    value = interrupts[0].value
                    interrupt_payload = value if isinstance(value, dict) else None
                continue
            if not isinstance(update, dict):
                continue
            _print_node_update(node, update)
            if update.get("final_response"):
                final_response = str(update["final_response"])

    if interrupt_payload is not None:
        print()
        print(interrupt_payload.get("message", "Confirm? (type the phrase shown)"))
        try:
            answer = input("confirm> ").strip()
        except (EOFError, KeyboardInterrupt):
            answer = ""
            print()
        _drive(graph, Command(resume=answer), config)
        return

    print()
    print(final_response or "(no response produced)")


def run_turn(
    graph: CompiledStateGraph[AgentState, None, AgentState, AgentState],
    question: str,
) -> None:
    """Run one question through the graph on its own checkpointed thread."""
    config: RunnableConfig = {"configurable": {"thread_id": uuid.uuid4().hex}}
    state: AgentState = {
        "messages": [HumanMessage(content=question)],
        "user_id": "cli-user",
        "trace_id": uuid.uuid4().hex[:8],
    }
    _drive(graph, state, config)


def main() -> int:
    load_dotenv(REPO_ROOT / ".env")
    debug = "--debug" in sys.argv[1:]
    set_debug(debug)
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")

    print("retail-data-agent — ask a question about sales, products, or customers.")
    if debug:
        print("[debug] per-node traces will stream to stderr (also logs/agent.jsonl).")
    print("Type 'exit' to quit.\n")
    graph = build_graph()

    while True:
        try:
            question = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not question:
            continue
        if question.lower() in {"exit", "quit"}:
            break
        try:
            run_turn(graph, question)
        except Exception as exc:  # noqa: BLE001 — the REPL must never crash
            print(f"\nSomething went wrong on my side ({exc}). Please try again.")
        print()
    print("bye")
    return 0


if __name__ == "__main__":
    sys.exit(main())
