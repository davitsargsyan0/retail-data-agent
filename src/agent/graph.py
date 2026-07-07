"""LangGraph state machine — milestone 1: the analysis happy path.

Node names follow `docs/architecture.md` §3 exactly:

    load_context → intent_router → trio_retrieval → sql_generation
    → sql_guard (static guard + LIMIT + BigQuery dry-run)
    → bigquery_execute → report_generation

Dry-run failures, execution errors, and empty result sets route to
``sql_repair`` (re-prompt with the verbatim error), capped at
``MAX_HEAL_ATTEMPTS`` (2) via ``retry_count``; on exhaustion,
``graceful_failure`` summarises every attempt — never a stack trace.
PII masking and the report-management branch land in later milestones.
"""

from __future__ import annotations

import logging
import re
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import interrupt

from agent.state import AgentState, HealAttempt, ReportRef
from reports_store.store import ReportStore, parse_delete_request
from safety import intent_gate, pii
from tools.bigquery_client import BigQueryClient, BigQueryClientError
from tools.llm import LLMError, generate
from tools.trio_retrieval import TrioRetriever

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]
PROMPTS_DIR = REPO_ROOT / "prompts"
SCHEMA_DOC = REPO_ROOT / "docs" / "schema.md"
DATA_DIR = REPO_ROOT / "data"

MAX_REPORT_ROWS = 20
MAX_HEAL_ATTEMPTS = 2  # hard self-heal budget (architecture §5.5)

_bq = BigQueryClient()
_retriever = TrioRetriever()
_store = ReportStore(DATA_DIR)

# Maps the intent-gate's three categories onto the state's routing literal.
_INTENT_MAP: dict[str, str] = {
    "analysis": "analysis",
    "report_management": "report_mgmt",
    "out_of_scope": "refuse",
}

_CODE_FENCE = re.compile(r"^```[a-zA-Z]*\n|\n?```$", re.MULTILINE)

# A fully-qualified thelook table reference NOT already wrapped in backticks,
# e.g. "FROM bigquery-public-data.thelook_ecommerce.order_items" — BigQuery
# rejects the unquoted form with "Unrecognized name: bigquery".
_UNQUOTED_TABLE_REF = re.compile(
    r"(?<!`)\b(bigquery-public-data\.thelook_ecommerce\.[A-Za-z_][A-Za-z0-9_]*)\b"
)


def normalize_table_references(sql: str) -> str:
    """Backtick-wrap bare ``bigquery-public-data.thelook_ecommerce.<table>`` refs.

    Deterministic safety net behind the prompt rules: the model is instructed
    to quote and alias every table, but if it still emits a bare dotted path
    (``FROM bigquery-public-data.thelook_ecommerce.order_items``, or a dotted
    column path like ``...order_items.created_at``), the table part is wrapped
    in backticks so the query parses. Already-quoted references are untouched.
    """
    return _UNQUOTED_TABLE_REF.sub(r"`\1`", sql)


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
    return {"persona": persona, "retry_count": 0, "sql_error": None, "heal_attempts": []}


def intent_router(state: AgentState) -> AgentState:
    """Classify the turn (analysis / report_mgmt / refuse) via the intent gate.

    The gate is rule-first (injection and report-library commands are caught
    deterministically); only ambiguous turns reach the LLM. See ``safety``.
    """
    question = _last_user_question(state)
    result = intent_gate.classify(question)
    intent = _INTENT_MAP[result.category]
    logger.info("intent_router: %s (%s) — %s", intent, result.source, result.reason)
    return {"intent": intent, "intent_reason": result.reason}  # type: ignore[typeddict-item]


def polite_refusal(state: AgentState) -> AgentState:
    """Decline out-of-scope / injection / PII-extraction requests, politely.

    The wording is fixed and never echoes the (possibly hostile) user text, so
    injected instructions cannot round-trip through the refusal message.
    """
    message = (
        "I can't help with that request. I'm a retail data-analysis assistant: "
        "I answer aggregate questions about sales, revenue, products, and "
        "customer trends, and I can manage your saved reports. I can't share "
        "personal customer data (emails, phone numbers, addresses) or raw table "
        "dumps. Try asking, for example, \"Which product categories drove the "
        "most revenue last quarter?\""
    )
    return {"final_response": message}


def trio_retrieval(state: AgentState) -> AgentState:
    """Retrieve top-k golden-bucket trios by cosine similarity.

    Degrades, never blocks: on any retrieval failure the agent falls back to
    schema-only SQL generation (architecture §5.5).
    """
    question = _last_user_question(state)
    try:
        trios = _retriever.retrieve(question)
    except Exception as exc:  # noqa: BLE001 — degradation path, never fatal
        logger.warning("trio retrieval degraded to schema-only generation: %s", exc)
        return {"retrieved_trios": []}
    logger.info(
        "trio_retrieval: %s",
        ", ".join(f"{t['id']}={t['score']:.2f}" for t in trios) or "no match above floor",
    )
    return {"retrieved_trios": trios}


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
    sql = normalize_table_references(_CODE_FENCE.sub("", raw).strip())
    return {"sql": sql, "sql_error": None}


def _record_attempt(state: AgentState, sql: str, error: str) -> list[HealAttempt]:
    """Append a failed attempt to the heal history (state lists are replaced,
    not merged, so we return the full list)."""
    return [*state.get("heal_attempts", []), HealAttempt(sql=sql, error=error)]


def sql_guard(state: AgentState) -> AgentState:
    """Read-only guard + PII query-plan denylist + LIMIT + zero-cost dry-run.

    Three checks before any data moves (architecture §5.2, layers 2):
    the ``BigQueryClient`` guard (single read-only SELECT), the **PII column
    denylist** (reject any query selecting ``email``/``phone``/``street_address``
    or ``SELECT *`` on ``users``), and a BigQuery dry-run. A PII-column
    rejection is fed back to the self-heal loop as a repair instruction, so a
    question that merely brushed a PII column still gets answered — just without
    the personal columns.
    """
    sql = state.get("sql")
    if not sql:
        error = state.get("sql_error") or "No SQL was generated."
        return {"sql_error": error, "heal_attempts": _record_attempt(state, "(none)", error)}

    pii_columns = pii.find_pii_columns(sql)
    if pii_columns:
        error = (
            f"Query selects restricted personal columns ({', '.join(pii_columns)}). "
            "Rewrite it to exclude personal data — aggregate over customers or drop "
            "those columns; never select email, phone, or street_address."
        )
        logger.warning("sql_guard: PII denylist blocked columns %s", pii_columns)
        return {"sql_error": error, "heal_attempts": _record_attempt(state, sql, error)}

    try:
        bytes_estimate = _bq.dry_run(sql)
    except BigQueryClientError as exc:
        logger.warning("sql_guard: dry-run failed — %s", exc)
        return {"sql_error": str(exc), "heal_attempts": _record_attempt(state, sql, str(exc))}
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
        logger.warning("bigquery_execute: failed — %s", exc)
        return {
            "sql_error": str(exc),
            "result_rows": None,
            "heal_attempts": _record_attempt(state, sql, str(exc)),
        }
    rows: list[dict[str, object]] = df.to_dict(orient="records")
    if not rows:
        error = "Query returned zero rows."
        logger.warning("bigquery_execute: %s", error)
        return {
            "sql_error": error,
            "result_rows": [],
            "heal_attempts": _record_attempt(state, sql, error),
        }
    return {"result_rows": rows, "sql_error": None}


def pii_mask(state: AgentState) -> AgentState:
    """Deterministic output sweep (architecture §5.2, layer 3).

    Drops denylisted PII columns and regex-masks email/phone/address in every
    string cell, *before* the report LLM sees the rows. Sets ``masked=True`` —
    ``report_generation`` refuses to run without it, so it is structurally
    impossible to generate a report over unmasked data.
    """
    rows = state.get("result_rows") or []
    result = pii.scrub_rows(rows)
    if result.fired():
        logger.info(
            "pii_mask: dropped columns=%s, cells masked=%d",
            result.dropped_columns or "none",
            result.cells_masked,
        )
    return {"result_rows": result.rows, "masked": True}


def report_generation(state: AgentState) -> AgentState:
    """Format the (already-masked) result set into the executive report.

    Refuses to run on unmasked data, and runs a final deterministic PII sweep
    over its own prose — masking is applied after the LLM, never trusted to it.
    """
    if not state.get("masked"):
        logger.error("report_generation reached with masked=False — refusing")
        return {"final_response": "I couldn't prepare a safe answer for that request."}
    rows = state.get("result_rows") or []
    df = pd.DataFrame(rows)
    table = df.head(MAX_REPORT_ROWS).to_string(
        index=False, float_format=lambda value: f"{value:,.2f}"
    )
    lines = [f"Here is what I found ({len(rows)} row{'s' if len(rows) != 1 else ''}):", "", table]
    if len(rows) > MAX_REPORT_ROWS:
        lines.append(f"... ({len(rows) - MAX_REPORT_ROWS} more rows not shown)")
    return {"final_response": pii.mask_text("\n".join(lines))}


def save_report(state: AgentState) -> AgentState:
    """Auto-save every generated report to the store (architecture §5.3).

    The saved body is the already-masked ``final_response``, so the report
    library can never become a PII side-channel. Save failures degrade quietly
    — the user still gets their answer.
    """
    body = state.get("final_response") or ""
    if not body:
        return {}
    question = _last_user_question(state)
    title = (question[:80].strip() or "Analysis") + ("…" if len(question) > 80 else "")
    owner = state.get("user_id") or "cli-user"
    try:
        report = _store.save(owner=owner, title=title, body=body)
        logger.info("save_report: saved %s for %s", report.id, owner)
    except OSError as exc:  # persistence is best-effort, never fatal
        logger.warning("save_report: could not persist report: %s", exc)
    return {}


def sql_repair(state: AgentState) -> AgentState:
    """Self-heal: regenerate the SQL from the question, the failed SQL, and
    the verbatim error. Increments ``retry_count`` (hard cap enforced by the
    routers, architecture §5.5)."""
    question = _last_user_question(state)
    failed_sql = state.get("sql") or "(no SQL was produced)"
    error = state.get("sql_error") or "unknown error"
    attempt = state.get("retry_count", 0) + 1
    logger.warning(
        "sql_repair: heal attempt %d/%d — error: %s", attempt, MAX_HEAL_ATTEMPTS, error
    )
    template = _read_text(PROMPTS_DIR / "sql_repair.md")
    prompt = (
        template.replace("{schema}", _read_text(SCHEMA_DOC))
        .replace("{question}", question)
        .replace("{failed_sql}", failed_sql)
        .replace("{error}", error)
    )
    try:
        raw = generate("You fix broken BigQuery SELECT statements.", prompt)
    except LLMError as exc:
        return {"sql_error": f"LLM unavailable during repair: {exc}", "retry_count": attempt}
    sql = normalize_table_references(_CODE_FENCE.sub("", raw).strip())
    return {"sql": sql, "sql_error": None, "retry_count": attempt}


def graceful_failure(state: AgentState) -> AgentState:
    """Apologise in plain language, listing what was tried — never a stack
    trace."""
    attempts = state.get("heal_attempts") or []
    lines = ["I couldn't complete that analysis reliably, even after retrying."]
    if attempts:
        lines.append("Here's what I tried:")
        for number, attempt in enumerate(attempts, start=1):
            sql_first_line = attempt["sql"].strip().splitlines()[0][:80]
            lines.append(f"  {number}. `{sql_first_line} ...` → {attempt['error']}")
    else:
        lines.append(f"Reason: {state.get('sql_error') or 'an unexpected problem'}.")
    lines.append(
        "Please try rephrasing the question — for example, name the exact "
        "metric and the time range you care about."
    )
    return {"final_response": "\n".join(lines)}


def _heal_or_fail(state: AgentState) -> str:
    """Shared retry decision: heal while budget remains, else fail gracefully."""
    if state.get("retry_count", 0) < MAX_HEAL_ATTEMPTS:
        return "sql_repair"
    return "graceful_failure"


def _route_after_guard(state: AgentState) -> str:
    return _heal_or_fail(state) if state.get("sql_error") else "bigquery_execute"


def _route_after_execute(state: AgentState) -> str:
    return _heal_or_fail(state) if state.get("sql_error") else "pii_mask"


def _route_after_repair(state: AgentState) -> str:
    """Repaired SQL goes back through the guard; a repair-time LLM failure
    exits gracefully instead of burning the remaining budget."""
    return "graceful_failure" if state.get("sql_error") else "sql_guard"


def _route_intent(state: AgentState) -> str:
    """Fan out from the intent router to the three top-level branches."""
    return {
        "analysis": "trio_retrieval",
        "report_mgmt": "match_reports",
        "refuse": "polite_refusal",
    }.get(state.get("intent", "refuse"), "polite_refusal")


# --- Report-management / delete branch (architecture §5.3) ----------------


def match_reports(state: AgentState) -> AgentState:
    """Resolve a report-management command against the user's OWN reports.

    Ownership scoping lives in the store query, never in a prompt. A plain
    "list my reports" (or a delete we couldn't parse) answers with the library;
    a parsed delete resolves the concrete candidate set that the confirmation
    interrupt will show — and that alone will execute.
    """
    owner = state.get("user_id") or "cli-user"
    question = _last_user_question(state)
    request = parse_delete_request(question)

    if request is None:
        reports = _store.list_for_owner(owner)
        if not reports:
            return {"matched_reports": [], "final_response": "You have no saved reports yet."}
        listing = "\n".join(
            f"  {n}. {r.title} (saved {r.created_at[:10]})" for n, r in enumerate(reports, 1)
        )
        return {
            "matched_reports": [],
            "final_response": f"You have {len(reports)} saved report(s):\n{listing}",
        }

    matched = _store.resolve(owner, request)
    if not matched:
        return {
            "matched_reports": [],
            "final_response": f"No saved reports match {request.description} — nothing to delete.",
        }
    refs: list[ReportRef] = [
        ReportRef(id=r.id, title=r.title, created_at=r.created_at) for r in matched
    ]
    return {"matched_reports": refs, "delete_filter": request.description}


def _route_after_match(state: AgentState) -> str:
    """Only enter the confirmation interrupt when there is something to delete."""
    return "confirm_delete" if state.get("matched_reports") else END


def _is_delete_confirmed(answer: str, count: int) -> bool:
    """Confirmation requires the exact typed count phrase — defeats reflexive 'y'."""
    normalised = " ".join(answer.lower().split())
    return normalised in {f"delete {count} report", f"delete {count} reports"}


def confirm_delete(state: AgentState) -> AgentState:
    """Preview the exact matches, then pause the graph for typed confirmation.

    Uses LangGraph ``interrupt()``: state is checkpointed, the client shows the
    preview and collects a typed confirmation, and the graph resumes with that
    answer injected — no bespoke pending-action machinery (ADR-004).
    """
    matched = state.get("matched_reports") or []
    count = len(matched)
    plural = "s" if count != 1 else ""
    listing = "\n".join(
        f"  {n}. {r['title']} (saved {r['created_at'][:10]})" for n, r in enumerate(matched, 1)
    )
    phrase = f"delete {count} report{plural}"
    message = (
        f"These {count} report{plural} will be deleted:\n{listing}\n\n"
        f"Type `{phrase}` to confirm — anything else cancels."
    )
    answer = interrupt(
        {
            "kind": "delete_confirmation",
            "count": count,
            "filter": state.get("delete_filter"),
            "reports": matched,
            "confirm_phrase": phrase,
            "message": message,
        }
    )
    answer_text = answer if isinstance(answer, str) else str(answer)
    return {
        "delete_confirmed": _is_delete_confirmed(answer_text, count),
        "confirmation_text": answer_text,
    }


def _route_after_confirm(state: AgentState) -> str:
    return "execute_delete" if state.get("delete_confirmed") else "cancel_delete"


def execute_delete(state: AgentState) -> AgentState:
    """Delete the confirmed reports (owner-scoped) and write an audit record."""
    owner = state.get("user_id") or "cli-user"
    matched = state.get("matched_reports") or []
    ids = [r["id"] for r in matched]
    deleted = _store.delete(owner, ids)
    _store.append_audit(
        {
            "action": "delete_reports",
            "owner": owner,
            "filter": state.get("delete_filter"),
            "requested_ids": ids,
            "deleted_ids": deleted,
            "confirmation_text": state.get("confirmation_text"),
            "trace_id": state.get("trace_id"),
            "at": datetime.now(UTC).isoformat(),
        }
    )
    deleted_set = set(deleted)
    titles = "\n".join(f"  - {r['title']}" for r in matched if r["id"] in deleted_set)
    plural = "s" if len(deleted) != 1 else ""
    return {
        "final_response": f"Deleted {len(deleted)} report{plural}:\n{titles}",
        "matched_reports": [],
    }


def cancel_delete(state: AgentState) -> AgentState:
    """Clean cancel path — nothing is touched."""
    return {
        "final_response": "Cancelled — nothing was deleted. Your reports are untouched.",
        "delete_confirmed": False,
    }


def build_graph() -> CompiledStateGraph[AgentState, None, AgentState, AgentState]:
    """Compile the full graph: intent routing, analysis + self-heal + masking,
    and the interrupt-guarded report-delete branch.

    Compiled with an in-memory checkpointer (**Prototype** — **Production:** a
    Firestore-backed checkpointer): required for the delete-confirmation
    ``interrupt()`` to persist state across the pause/resume boundary (ADR-004).
    """
    builder = StateGraph(AgentState)
    builder.add_node("load_context", load_context)
    builder.add_node("intent_router", intent_router)
    builder.add_node("polite_refusal", polite_refusal)
    builder.add_node("trio_retrieval", trio_retrieval)
    builder.add_node("sql_generation", sql_generation)
    builder.add_node("sql_guard", sql_guard)
    builder.add_node("bigquery_execute", bigquery_execute)
    builder.add_node("sql_repair", sql_repair)
    builder.add_node("pii_mask", pii_mask)
    builder.add_node("report_generation", report_generation)
    builder.add_node("save_report", save_report)
    builder.add_node("graceful_failure", graceful_failure)
    builder.add_node("match_reports", match_reports)
    builder.add_node("confirm_delete", confirm_delete)
    builder.add_node("execute_delete", execute_delete)
    builder.add_node("cancel_delete", cancel_delete)

    builder.add_edge(START, "load_context")
    builder.add_edge("load_context", "intent_router")
    builder.add_conditional_edges(
        "intent_router",
        _route_intent,
        ["trio_retrieval", "match_reports", "polite_refusal"],
    )

    # Analysis branch
    builder.add_edge("trio_retrieval", "sql_generation")
    builder.add_edge("sql_generation", "sql_guard")
    builder.add_conditional_edges(
        "sql_guard",
        _route_after_guard,
        ["bigquery_execute", "sql_repair", "graceful_failure"],
    )
    builder.add_conditional_edges(
        "bigquery_execute",
        _route_after_execute,
        ["pii_mask", "sql_repair", "graceful_failure"],
    )
    builder.add_conditional_edges(
        "sql_repair",
        _route_after_repair,
        ["sql_guard", "graceful_failure"],
    )
    builder.add_edge("pii_mask", "report_generation")
    builder.add_edge("report_generation", "save_report")
    builder.add_edge("save_report", END)
    builder.add_edge("graceful_failure", END)

    # Report-management / delete branch
    builder.add_conditional_edges("match_reports", _route_after_match, ["confirm_delete", END])
    builder.add_conditional_edges(
        "confirm_delete",
        _route_after_confirm,
        ["execute_delete", "cancel_delete"],
    )
    builder.add_edge("execute_delete", END)
    builder.add_edge("cancel_delete", END)

    builder.add_edge("polite_refusal", END)
    return builder.compile(checkpointer=MemorySaver())
