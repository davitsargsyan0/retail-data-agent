"""End-to-end eval harness — the release gate for requirement 6 (§5.6).

Runs every question in ``golden_questions.yaml`` through the *real* compiled graph
against live BigQuery + Gemini, then checks PROPERTY-BASED assertions (never exact
output): which tables the SQL touched, that a row came back, that a headline number
lands in a plausible band, and — for every question — that zero PII tokens survive.

Design notes:
- The LLM is nondeterministic and BigQuery is live, so checks are tolerant
  (table-set membership, ranges, PII-absence) rather than string equality.
- A single question erroring never aborts the run: it is caught, marked FAIL, and
  the harness moves on.
- Exit code is nonzero if any check fails, so this can gate a deploy.

Run with:  ``uv run python evals/run_evals.py``
"""

from __future__ import annotations

import re
import sys
import traceback
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage

REPO_ROOT = Path(__file__).resolve().parents[1]
# Defensive: works whether or not the project is installed into the venv.
sys.path.insert(0, str(REPO_ROOT / "src"))

from agent.graph import build_graph  # noqa: E402

QUESTIONS_PATH = REPO_ROOT / "evals" / "golden_questions.yaml"

# Independent PII detectors — NOT the module's own regexes, so this is a genuine
# external check that no email/phone/address pattern survived masking.
_EMAIL_RE = re.compile(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}")
_PHONE_RE = re.compile(
    r"(?<!\w)(?:\+?\d{1,3}[\s.\-])?\(?\d{2,4}\)?[\s.\-]\d{2,4}[\s.\-]\d{2,4}(?!\w)"
)
_STREET_RE = re.compile(
    r"\b\d{1,6}\s+(?:[A-Za-z0-9.'\-]+\s+){0,3}"
    r"(?:Street|St|Avenue|Ave|Road|Rd|Boulevard|Blvd|Lane|Ln|Drive|Dr|Court|Ct)\b",
    re.IGNORECASE,
)


def _contains_pii(text: str) -> bool:
    return bool(_EMAIL_RE.search(text) or _PHONE_RE.search(text) or _STREET_RE.search(text))


def _table_referenced(sql: str, table: str) -> bool:
    """Whole-token match so 'orders' does not match inside 'order_items'."""
    pattern = rf"(?<![A-Za-z0-9_]){re.escape(table)}(?![A-Za-z0-9_])"
    return bool(re.search(pattern, sql, re.IGNORECASE))


def _numeric_values(rows: list[dict[str, Any]]) -> list[float]:
    values: list[float] = []
    for row in rows:
        for value in row.values():
            if isinstance(value, bool):
                continue
            if isinstance(value, (int, float)):
                values.append(float(value))
    return values


@dataclass
class QuestionRun:
    """What one graph run produced (for property checks)."""

    sql: str | None = None
    rows: list[dict[str, Any]] | None = None
    final: str = ""
    intent: str | None = None
    interrupted: bool = False


@dataclass
class CheckResult:
    """Per-question outcome: named checks (True/False/None=n.a.) + any error."""

    checks: dict[str, bool | None] = field(default_factory=dict)
    error: str | None = None

    def passed(self) -> bool:
        if self.error is not None:
            return False
        return all(result is not False for result in self.checks.values())


def run_question(graph: Any, question: str) -> QuestionRun:
    """Drive one question end-to-end, capturing SQL, rows, intent, and response."""
    config = {"configurable": {"thread_id": uuid.uuid4().hex}}
    state: dict[str, Any] = {
        "messages": [HumanMessage(content=question)],
        "user_id": "eval-user",
        "trace_id": "eval",
    }
    run = QuestionRun()
    for chunk in graph.stream(state, config=config, stream_mode="updates"):
        for node, update in chunk.items():
            if node == "__interrupt__":
                run.interrupted = True
                continue
            if not isinstance(update, dict):
                continue
            if update.get("sql"):
                run.sql = str(update["sql"])
            if update.get("intent"):
                run.intent = str(update["intent"])
            if update.get("result_rows") is not None:
                run.rows = list(update["result_rows"])
            if update.get("final_response"):
                run.final = str(update["final_response"])
    return run


def check_analysis(spec: dict[str, Any], run: QuestionRun) -> CheckResult:
    checks: dict[str, bool | None] = {}

    expect_tables = spec.get("expect_tables") or []
    if expect_tables:
        sql = run.sql or ""
        checks["tables"] = bool(sql) and all(_table_referenced(sql, t) for t in expect_tables)
    else:
        checks["tables"] = None

    if spec.get("non_empty"):
        checks["non_empty"] = run.rows is not None and len(run.rows) > 0
    else:
        checks["non_empty"] = None

    rng = spec.get("numeric_range")
    if rng:
        values = _numeric_values(run.rows or [])
        checks["numeric"] = any(rng["min"] <= v <= rng["max"] for v in values)
    else:
        checks["numeric"] = None

    checks["zero_pii"] = not _contains_pii(run.final) if spec.get("zero_pii") else None
    return CheckResult(checks=checks)


def check_refusal(spec: dict[str, Any], run: QuestionRun) -> CheckResult:
    checks: dict[str, bool | None] = {
        # Refused: routed to the refusal branch and no query executed.
        "refused": run.intent == "refuse" and not run.rows,
        "zero_pii": not _contains_pii(run.final) if spec.get("zero_pii") else None,
    }
    return CheckResult(checks=checks)


_COLUMNS = ("tables", "non_empty", "numeric", "refused", "zero_pii")


def _cell(value: bool | None) -> str:
    if value is None:
        return "  - "
    return " PASS" if value else " FAIL"


def _print_table(results: list[tuple[str, str, CheckResult]]) -> None:
    header = f"{'id':<26}{'type':<12}" + "".join(f"{c:<11}" for c in _COLUMNS) + "result"
    print(header)
    print("-" * len(header))
    for qid, qtype, result in results:
        cells = "".join(f"{_cell(result.checks.get(c)):<11}" for c in _COLUMNS)
        verdict = "PASS" if result.passed() else "FAIL"
        line = f"{qid:<26}{qtype:<12}{cells}{verdict}"
        print(line)
        if result.error:
            print(f"    error: {result.error}")


def main() -> int:
    load_dotenv(REPO_ROOT / ".env")
    spec = yaml.safe_load(QUESTIONS_PATH.read_text(encoding="utf-8"))
    graph = build_graph()

    results: list[tuple[str, str, CheckResult]] = []

    for entry in spec.get("analysis", []):
        qid = str(entry["id"])
        print(f"running analysis: {qid} ...", flush=True)
        try:
            run = run_question(graph, str(entry["question"]))
            result = check_analysis(entry, run)
        except Exception as exc:  # noqa: BLE001 — one bad question must not abort the run
            result = CheckResult(error=f"{type(exc).__name__}: {exc}")
            traceback.print_exc()
        results.append((qid, "analysis", result))

    for entry in spec.get("adversarial", []):
        qid = str(entry["id"])
        print(f"running adversarial: {qid} ...", flush=True)
        try:
            run = run_question(graph, str(entry["question"]))
            result = check_refusal(entry, run)
        except Exception as exc:  # noqa: BLE001
            result = CheckResult(error=f"{type(exc).__name__}: {exc}")
            traceback.print_exc()
        results.append((qid, "adversarial", result))

    print()
    _print_table(results)

    failed = [qid for qid, _, result in results if not result.passed()]
    print()
    print(f"{len(results) - len(failed)}/{len(results)} passed")
    if failed:
        print("FAILED: " + ", ".join(failed))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
