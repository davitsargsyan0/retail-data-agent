"""LangGraph state machine — milestone 1: the analysis happy path.

Node names follow `docs/architecture.md` §3 exactly:

    load_context → intent_router → trio_retrieval → sql_generation
    → sql_guard (static guard + LIMIT + BigQuery dry-run)
    → bigquery_execute → report_generation

Guard/execution failures route to ``graceful_failure`` so the CLI never sees a
stack trace. The self-heal loop (``sql_repair``), PII masking, and the
report-management branch land in later milestones.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Callable
from pathlib import Path

import pandas as pd
from langchain_core.messages import HumanMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from agent.state import AgentState
from tools.bigquery_client import BigQueryClient, BigQueryClientError
from tools.llm import LLMError, generate

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]
PROMPTS_DIR = REPO_ROOT / "prompts"
SCHEMA_DOC = REPO_ROOT / "docs" / "schema.md"

MAX_REPORT_ROWS = 20

_bq = BigQueryClient()

_CODE_FENCE = re.compile(r"^```[a-zA-Z]*\n|\n?```$", re.MULTILINE)


def _read_text(path: Path, fallback: str = "") -> str:
    """Read a text file, returning ``fallback`` if it does not exist."""
    try:
        return path.read_text(encoding="utf-8")
    except OSError:
        logger.warning("could not read %s — using fallback", path)
        return fallback


def _last_user_question(state: AgentState) -> str:
    """Extract the most recent human message text."""
    for message in reversed(state.get("messages", [])):
        if isinstance(message, HumanMessage) and isinstance(message.content, str):
            return message.content
    return ""


def load_context(state: AgentState) -> AgentState:
    """Hot-load the persona from ``prompts/persona.md`` on every request."""
    persona = _read_text(PROMPTS_DIR / "persona.md", "You are a concise retail data analyst.")
    return {"persona": persona, "retry_count": 0, "sql_error": None}


def intent_router(state: AgentState) -> AgentState:
    """Route the request. Milestone-1 stub: everything is an analysis question."""
    return {"intent": "analysis"}


def trio_retrieval(state: AgentState) -> AgentState:
    """Retrieve golden-bucket trios. Milestone-1 stub: no trios yet."""
    return {"retrieved_trios": []}


def sql_generation(state: AgentState) -> AgentState:
    """Generate a candidate SELECT via Gemini, grounded in ``docs/schema.md``."""
    question = _last_user_question(state)
    schema = _read_text(SCHEMA_DOC)
    if not schema:
        return {"sql": None, "sql_error": "docs/schema.md is missing — cannot ground SQL."}

    trios = state.get("retrieved_trios") or []
    trios_text = (
        "\n\n".join(f"Q: {t['question']}\nSQL:\n{t['sql']}" for t in trios)
        or "(none available)"
    )
    template = _read_text(PROMPTS_DIR / "sql_generation.md")
    prompt = (
        template.replace("{schema}", schema)
        .replace("{trios}", trios_text)
        .replace("{question}", question)
    )
    try:
        raw = generate(
            "You translate business questions into a single BigQuery SELECT statement.",
            prompt,
        )
    except LLMError as exc:
        return {"sql": None, "sql_error": f"LLM unavailable: {exc}"}
    sql = _CODE_FENCE.sub("", raw).strip()
    return {"sql": sql, "sql_error": None}


def sql_guard(state: AgentState) -> AgentState:
    """Static read-only guard + LIMIT injection + zero-cost BigQuery dry-run."""
    sql = state.get("sql")
    if not sql:
        return {"sql_error": state.get("sql_error") or "No SQL was generated."}
    try:
        bytes_estimate = _bq.dry_run(sql)
    except BigQueryClientError as exc:
        return {"sql_error": str(exc)}
    logger.info("sql_guard: dry-run OK, ~%.2f MB would be scanned", bytes_estimate / 1e6)
    return {"sql_error": None}


def bigquery_execute(state: AgentState) -> AgentState:
    """Execute the guarded query and store bounded result rows."""
    sql = state.get("sql")
    if not sql:
        return {"sql_error": "No SQL to execute."}
    try:
        df = _bq.execute(sql)
    except BigQueryClientError as exc:
        return {"sql_error": str(exc), "result_rows": None}
    rows: list[dict[str, object]] = df.to_dict(orient="records")
    if not rows:
        return {"sql_error": "Query returned zero rows.", "result_rows": []}
    return {"result_rows": rows, "sql_error": None}


def report_generation(state: AgentState) -> AgentState:
    """Milestone-1 stub report: plain formatting of the result set."""
    rows = state.get("result_rows") or []
    df = pd.DataFrame(rows)
    table = df.head(MAX_REPORT_ROWS).to_string(
        index=False, float_format=lambda value: f"{value:,.2f}"
    )
    lines = [f"Here is what I found ({len(rows)} row{'s' if len(rows) != 1 else ''}):", "", table]
    if len(rows) > MAX_REPORT_ROWS:
        lines.append(f"... ({len(rows) - MAX_REPORT_ROWS} more rows not shown)")
    return {"final_response": "\n".join(lines)}


def graceful_failure(state: AgentState) -> AgentState:
    """Apologise in plain language — never a stack trace."""
    reason = state.get("sql_error") or "an unexpected problem"
    return {
        "final_response": (
            "I couldn't get a reliable answer to that just now "
            f"(reason: {reason}).\n"
            "Could you try rephrasing the question — for example, naming the "
            "metric and time range you care about?"
        )
    }


def _route_on_error(ok_node: str) -> Callable[[AgentState], str]:
    """Build a conditional-edge router: on ``sql_error`` → graceful_failure."""

    def route(state: AgentState) -> str:
        return "graceful_failure" if state.get("sql_error") else ok_node

    return route


def build_graph() -> CompiledStateGraph[AgentState, None, AgentState, AgentState]:
    """Compile the milestone-1 graph (analysis happy path + graceful failure)."""
    builder = StateGraph(AgentState)
    builder.add_node("load_context", load_context)
    builder.add_node("intent_router", intent_router)
    builder.add_node("trio_retrieval", trio_retrieval)
    builder.add_node("sql_generation", sql_generation)
    builder.add_node("sql_guard", sql_guard)
    builder.add_node("bigquery_execute", bigquery_execute)
    builder.add_node("report_generation", report_generation)
    builder.add_node("graceful_failure", graceful_failure)

    builder.add_edge(START, "load_context")
    builder.add_edge("load_context", "intent_router")
    builder.add_edge("intent_router", "trio_retrieval")
    builder.add_edge("trio_retrieval", "sql_generation")
    builder.add_edge("sql_generation", "sql_guard")
    builder.add_conditional_edges(
        "sql_guard",
        _route_on_error("bigquery_execute"),
        ["bigquery_execute", "graceful_failure"],
    )
    builder.add_conditional_edges(
        "bigquery_execute",
        _route_on_error("report_generation"),
        ["report_generation", "graceful_failure"],
    )
    builder.add_edge("report_generation", END)
    builder.add_edge("graceful_failure", END)
    return builder.compile()
