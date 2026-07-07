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

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langgraph.graph.state import CompiledStateGraph

from agent.graph import build_graph
from agent.state import AgentState

REPO_ROOT = Path(__file__).resolve().parents[1]

_PROGRESS: dict[str, str] = {
    "load_context": "loading persona",
    "intent_router": "routing intent",
    "trio_retrieval": "searching golden bucket",
    "sql_generation": "writing SQL",
    "sql_guard": "validating SQL (dry run)",
    "bigquery_execute": "querying BigQuery",
    "report_generation": "writing report",
    "graceful_failure": "recovering",
}


def _print_node_update(node: str, update: dict[str, object]) -> None:
    """Render one streamed node update as CLI progress."""
    label = _PROGRESS.get(node, node)
    print(f"  · {node} — {label}")
    if node == "sql_generation" and update.get("sql"):
        print("    generated SQL:")
        for line in str(update["sql"]).splitlines():
            print(f"      {line}")
    if update.get("sql_error"):
        print(f"    ! {update['sql_error']}")


def run_turn(
    graph: CompiledStateGraph[AgentState, None, AgentState, AgentState],
    question: str,
) -> None:
    """Run one question through the graph, streaming progress."""
    state: AgentState = {
        "messages": [HumanMessage(content=question)],
        "user_id": "cli-user",
        "trace_id": uuid.uuid4().hex[:8],
    }
    final_response = ""
    for chunk in graph.stream(state, stream_mode="updates"):
        for node, update in chunk.items():
            if not isinstance(update, dict):
                continue
            _print_node_update(node, update)
            if update.get("final_response"):
                final_response = str(update["final_response"])
    print()
    print(final_response or "(no response produced)")


def main() -> int:
    load_dotenv(REPO_ROOT / ".env")
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")

    print("retail-data-agent — ask a question about sales, products, or customers.")
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
