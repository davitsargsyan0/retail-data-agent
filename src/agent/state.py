"""Typed LangGraph state for the retail data agent.

Field names follow `docs/architecture.md` §3 ("State schema") exactly.
"""

from __future__ import annotations

from typing import Annotated, Literal, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

Intent = Literal["analysis", "report_mgmt", "meta", "refuse"]


class Trio(TypedDict):
    """A golden-bucket Question → SQL → Report trio, plus retrieval score."""

    id: str
    question: str
    sql: str
    report: str
    score: float


class ReportRef(TypedDict):
    """Reference to a saved report (delete-branch candidate)."""

    id: str
    title: str
    created_at: str


class HealAttempt(TypedDict):
    """One failed SQL attempt — feeds ``sql_repair`` and the graceful-failure
    "here's what I tried" message."""

    sql: str
    error: str


class AgentState(TypedDict, total=False):
    """Single graph state, checkpointed after every node.

    ``total=False`` lets each node return a partial update.
    """

    messages: Annotated[list[BaseMessage], add_messages]
    user_id: str
    persona: str
    preferences: dict[str, str]
    intent: Intent
    retrieved_trios: list[Trio]
    sql: str | None
    sql_error: str | None
    retry_count: int
    heal_attempts: list[HealAttempt]
    result_rows: list[dict[str, object]] | None
    masked: bool
    matched_reports: list[ReportRef]
    delete_confirmed: bool | None
    final_response: str
    trace_id: str
