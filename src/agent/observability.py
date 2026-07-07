"""Observability-lite — one structured JSON line per node execution.

This is the **prototype slice** of the observability design in
``docs/architecture.md`` §5.7 (per-node metrics + trace-joined deep-dive). In
production the same per-node span data streams to Langfuse and Cloud Logging,
joined by ``trace_id``; here we emit a JSON-lines file (``logs/agent.jsonl``) and,
under ``--debug``, mirror each line to stderr as the graph runs.

Instrumentation is applied **uniformly at graph-build time** — ``build_graph``
registers every node through :func:`instrument`, so no node function is
hand-edited to add logging. Each record carries::

    {trace_id, node, latency_ms, model, tokens, error, ts}

``model``/``tokens`` are populated only when the node actually called the LLM
(captured via :func:`tools.llm.capture_usage`), else ``null``; ``error`` is
``null`` on success and the exception summary on failure.
"""

from __future__ import annotations

import functools
import json
import logging
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol

from langgraph.errors import GraphBubbleUp

from agent.state import AgentState
from tools.llm import Usage, capture_usage

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]
LOG_DIR = REPO_ROOT / "logs"
LOG_PATH = LOG_DIR / "agent.jsonl"


class NodeFn(Protocol):
    """A graph node: ``(AgentState) -> AgentState``.

    A concrete Protocol (rather than a bare ``Callable`` alias) so LangGraph's
    ``add_node`` overloads can bind the state type when the wrapped node is
    registered.
    """

    def __call__(self, state: AgentState) -> AgentState: ...


_debug = False


def set_debug(enabled: bool) -> None:
    """Enable/disable mirroring each trace record to stderr (CLI ``--debug``)."""
    global _debug
    _debug = enabled


def _summarise_usage(usage: list[Usage]) -> tuple[str | None, int | None]:
    """Collapse a node's LLM calls into ``(model, total_tokens)``.

    ``None, None`` when the node made no LLM call. Token totals sum across calls;
    a call whose provider omitted ``usage_metadata`` contributes no count.
    """
    if not usage:
        return None, None
    model = usage[0].model
    counts = [u.total_tokens for u in usage if u.total_tokens is not None]
    return model, (sum(counts) if counts else None)


def _write(record: dict[str, object]) -> None:
    """Append one JSON record to ``logs/agent.jsonl`` (and stderr if ``--debug``)."""
    line = json.dumps(record, ensure_ascii=False)
    try:
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        with LOG_PATH.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")
    except OSError as exc:  # logging must never break a turn
        logger.warning("observability: could not write log line: %s", exc)
    if _debug:
        print(line, file=sys.stderr, flush=True)


def instrument(node_name: str, fn: NodeFn) -> NodeFn:
    """Wrap a graph node so its execution emits one structured trace record.

    Times the node, captures any LLM usage it incurred, records the exception
    summary on failure (then re-raises — instrumentation never swallows errors),
    and writes the record whether the node succeeded or raised.
    """

    @functools.wraps(fn)
    def wrapped(state: AgentState) -> AgentState:
        trace_id = state.get("trace_id")
        start = time.perf_counter()
        error: str | None = None
        with capture_usage() as usage:
            try:
                return fn(state)
            except GraphBubbleUp:
                # Control flow, not a failure: interrupt()/Command re-raise a
                # GraphBubbleUp to pause the graph. Record as a normal node.
                raise
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
                raise
            finally:
                model, tokens = _summarise_usage(usage)
                _write(
                    {
                        "ts": datetime.now(UTC).isoformat(),
                        "trace_id": trace_id,
                        "node": node_name,
                        "latency_ms": round((time.perf_counter() - start) * 1000, 2),
                        "model": model,
                        "tokens": tokens,
                        "error": error,
                    }
                )

    return wrapped
