"""The single Gemini wrapper — every LLM call in the codebase goes through here.

Provides chat generation and embeddings via ``langchain-google-genai``, with
retry + exponential backoff (with jitter) and one clean error type. Model names
come from the environment (``GEMINI_MODEL`` / ``GEMINI_EMBEDDING_MODEL``) with
sensible defaults. This module is also the seam where the production circuit
breaker / OpenRouter fallback plugs in (see ADR-005).
"""

from __future__ import annotations

import contextvars
import logging
import os
import random
import time
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

logger = logging.getLogger(__name__)

DEFAULT_CHAT_MODEL = "gemini-2.5-flash"
# The architecture doc names text-embedding-004, but that model was retired
# from the Gemini API (404s as of 2026-07); gemini-embedding-001 replaces it.
DEFAULT_EMBEDDING_MODEL = "models/gemini-embedding-001"

_MAX_ATTEMPTS = 3
_BASE_DELAY_SECONDS = 1.0


class LLMError(Exception):
    """Raised when the LLM provider fails after all retries."""


@dataclass(frozen=True)
class Usage:
    """Token accounting for one chat call, lifted from ``response.usage_metadata``."""

    model: str
    input_tokens: int | None
    output_tokens: int | None
    total_tokens: int | None


# Observability seam: a context-local sink. ``capture_usage`` opens a scope, and
# every ``chat`` call inside it appends its ``Usage`` — so the node-instrumentation
# wrapper can attribute tokens/model to a node without touching the LLM signature.
_usage_sink: contextvars.ContextVar[list[Usage] | None] = contextvars.ContextVar(
    "_usage_sink", default=None
)


@contextmanager
def capture_usage() -> Iterator[list[Usage]]:
    """Collect the token usage of every ``chat`` call made within this scope."""
    sink: list[Usage] = []
    token = _usage_sink.set(sink)
    try:
        yield sink
    finally:
        _usage_sink.reset(token)


def _as_int(value: object) -> int | None:
    return value if isinstance(value, int) else None


def _record_usage(response: BaseMessage, model: str) -> None:
    """Append this response's token usage to the active sink, if any."""
    sink = _usage_sink.get()
    if sink is None:
        return
    meta = getattr(response, "usage_metadata", None)
    if isinstance(meta, dict):
        sink.append(
            Usage(
                model=model,
                input_tokens=_as_int(meta.get("input_tokens")),
                output_tokens=_as_int(meta.get("output_tokens")),
                total_tokens=_as_int(meta.get("total_tokens")),
            )
        )
    else:
        sink.append(Usage(model=model, input_tokens=None, output_tokens=None, total_tokens=None))


def chat_model_name() -> str:
    """Chat model name, read from the environment on every call."""
    return os.environ.get("GEMINI_MODEL", DEFAULT_CHAT_MODEL)


def embedding_model_name() -> str:
    """Embedding model name, read from the environment on every call."""
    return os.environ.get("GEMINI_EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL)


def _with_retry[T](operation: Callable[[], T], description: str) -> T:
    """Run ``operation`` with exponential backoff + jitter, max 3 attempts."""
    last_error: Exception | None = None
    for attempt in range(1, _MAX_ATTEMPTS + 1):
        try:
            return operation()
        except Exception as exc:  # noqa: BLE001 — provider raises many types
            last_error = exc
            if attempt < _MAX_ATTEMPTS:
                delay = _BASE_DELAY_SECONDS * (2 ** (attempt - 1)) + random.uniform(0.0, 0.5)
                logger.warning(
                    "%s failed (attempt %d/%d): %s — retrying in %.1fs",
                    description,
                    attempt,
                    _MAX_ATTEMPTS,
                    exc,
                    delay,
                )
                time.sleep(delay)
    raise LLMError(f"{description} failed after {_MAX_ATTEMPTS} attempts: {last_error}") from (
        last_error
    )


def _content_to_text(content: str | list[str | dict[str, object]]) -> str:
    """Coerce a LangChain message content payload into plain text."""
    if isinstance(content, str):
        return content
    parts: list[str] = []
    for block in content:
        if isinstance(block, str):
            parts.append(block)
        elif block.get("type") == "text":
            text = block.get("text", "")
            if isinstance(text, str):
                parts.append(text)
    return "".join(parts)


def chat(messages: Sequence[BaseMessage], *, temperature: float = 0.0) -> str:
    """Send a chat request to Gemini and return the response text."""
    name = chat_model_name()
    model = ChatGoogleGenerativeAI(model=name, temperature=temperature)

    def _invoke() -> str:
        response = model.invoke(list(messages))
        _record_usage(response, name)
        return _content_to_text(response.content)

    return _with_retry(_invoke, f"chat({name})")


def generate(system: str, user: str, *, temperature: float = 0.0) -> str:
    """Convenience: one system + one human message → response text."""
    messages = [SystemMessage(content=system), HumanMessage(content=user)]
    return chat(messages, temperature=temperature)


def embed_texts(texts: Sequence[str]) -> list[list[float]]:
    """Embed a batch of documents (e.g. golden-trio questions)."""
    embedder = GoogleGenerativeAIEmbeddings(model=embedding_model_name())
    return _with_retry(
        lambda: embedder.embed_documents(list(texts)),
        f"embed_documents({embedding_model_name()})",
    )


def embed_query(text: str) -> list[float]:
    """Embed a single query string (the user question at retrieval time)."""
    embedder = GoogleGenerativeAIEmbeddings(model=embedding_model_name())
    return _with_retry(
        lambda: embedder.embed_query(text),
        f"embed_query({embedding_model_name()})",
    )
