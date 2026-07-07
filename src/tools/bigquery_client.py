"""Read-only BigQuery client for `bigquery-public-data.thelook_ecommerce`.

Safety properties (deterministic, not prompt-based):

- ``validate_select`` rejects anything that is not a single ``SELECT``/``WITH``
  statement — no DML/DDL, no multi-statement scripts.
- ``ensure_limit`` injects a hard ``LIMIT`` when the query lacks one.
- ``dry_run`` validates syntax and estimates bytes scanned at zero cost before
  any real execution.
- ``execute`` runs with a ``maximum_bytes_billed`` cap and returns a pandas
  DataFrame.
"""

from __future__ import annotations

import logging
import os
import re

import pandas as pd
from google.api_core import exceptions as gcloud_exceptions
from google.cloud import bigquery

logger = logging.getLogger(__name__)

DATASET = "bigquery-public-data.thelook_ecommerce"
DEFAULT_ROW_LIMIT = 1000
DEFAULT_MAX_BYTES_BILLED = 2 * 1024**3  # 2 GiB — generous for this dataset

_FORBIDDEN_KEYWORDS = re.compile(
    r"\b("
    r"INSERT|UPDATE|DELETE|MERGE|TRUNCATE"  # DML (REPLACE excluded: valid SELECT function)
    r"|CREATE|DROP|ALTER|GRANT|REVOKE"  # DDL / DCL
    r"|CALL|EXECUTE|BEGIN|COMMIT|ROLLBACK|DECLARE|SET|EXPORT|LOAD"  # scripting
    r")\b",
    re.IGNORECASE,
)
_LIMIT_PATTERN = re.compile(r"\bLIMIT\s+\d+", re.IGNORECASE)


class BigQueryClientError(Exception):
    """Base error for the BigQuery tool."""


class SQLGuardError(BigQueryClientError):
    """The candidate SQL was rejected before reaching BigQuery."""


class BigQueryExecutionError(BigQueryClientError):
    """BigQuery rejected or failed the query (syntax error, quota, ...)."""


def strip_comments_and_literals(sql: str) -> str:
    """Return ``sql`` with comments removed and string literals blanked.

    This keeps keyword scanning honest: a ``DROP`` inside a string literal or
    a comment must not trigger (or mask) the guard.
    """
    out: list[str] = []
    i = 0
    n = len(sql)
    while i < n:
        ch = sql[i]
        two = sql[i : i + 2]
        if two == "--" or ch == "#":  # line comment
            while i < n and sql[i] != "\n":
                i += 1
        elif two == "/*":  # block comment
            end = sql.find("*/", i + 2)
            i = n if end == -1 else end + 2
        elif ch in ("'", '"', "`"):  # string literal / quoted identifier
            quote = ch
            i += 1
            while i < n:
                if sql[i] == "\\":
                    i += 2
                    continue
                if sql[i] == quote:
                    i += 1
                    break
                i += 1
            out.append(f"{quote}{quote}")  # keep an empty literal as a placeholder
        else:
            out.append(ch)
            i += 1
    return "".join(out)


def validate_select(sql: str) -> None:
    """Reject anything that is not a single read-only SELECT statement.

    Raises:
        SQLGuardError: on empty input, multi-statement scripts, statements not
            starting with ``SELECT``/``WITH``, or any DML/DDL/scripting keyword.
    """
    stripped = strip_comments_and_literals(sql).strip()
    if not stripped:
        raise SQLGuardError("Empty SQL statement.")

    body = stripped.rstrip(";").strip()
    if ";" in body:
        raise SQLGuardError("Multi-statement SQL is not allowed — submit a single SELECT.")

    first_word = body.split(None, 1)[0].upper() if body else ""
    if first_word not in ("SELECT", "WITH"):
        raise SQLGuardError(
            f"Only SELECT statements are allowed (query starts with {first_word!r})."
        )

    match = _FORBIDDEN_KEYWORDS.search(body)
    if match:
        raise SQLGuardError(
            f"Forbidden keyword {match.group(0).upper()!r} — the database is read-only."
        )


def ensure_limit(sql: str, limit: int = DEFAULT_ROW_LIMIT) -> str:
    """Append a hard ``LIMIT`` if the query does not already contain one."""
    if _LIMIT_PATTERN.search(strip_comments_and_literals(sql)):
        return sql
    return f"{sql.strip().rstrip(';').strip()}\nLIMIT {limit}"


class BigQueryClient:
    """Thin read-only wrapper around ``google.cloud.bigquery.Client``.

    Every query passes through ``validate_select`` + ``ensure_limit`` before it
    is sent, and ``execute`` enforces ``maximum_bytes_billed``.
    """

    def __init__(
        self,
        project: str | None = None,
        max_bytes_billed: int | None = None,
    ) -> None:
        self._project = project or os.environ.get("GOOGLE_CLOUD_PROJECT") or None
        self._max_bytes_billed = max_bytes_billed or int(
            os.environ.get("BQ_MAX_BYTES_BILLED", str(DEFAULT_MAX_BYTES_BILLED))
        )
        self._client: bigquery.Client | None = None

    @property
    def client(self) -> bigquery.Client:
        """Lazily created underlying client (avoids auth at import time)."""
        if self._client is None:
            try:
                self._client = bigquery.Client(project=self._project)
            except Exception as exc:  # pragma: no cover — auth/environment issue
                raise BigQueryClientError(f"Could not create BigQuery client: {exc}") from exc
        return self._client

    def dry_run(self, sql: str) -> int:
        """Validate ``sql`` against BigQuery at zero cost.

        Returns:
            Estimated bytes that the query would process.

        Raises:
            SQLGuardError: if the statement fails the read-only guard.
            BigQueryExecutionError: if BigQuery rejects the query (syntax,
                unknown column, ...). The message is verbatim from BigQuery so
                the self-heal loop can feed it back to the model.
        """
        validate_select(sql)
        job_config = bigquery.QueryJobConfig(dry_run=True, use_query_cache=False)
        try:
            job = self.client.query(sql, job_config=job_config)
        except gcloud_exceptions.GoogleAPICallError as exc:
            raise BigQueryExecutionError(exc.message or str(exc)) from exc
        bytes_processed = int(job.total_bytes_processed or 0)
        logger.info("dry_run ok: %d bytes would be processed", bytes_processed)
        return bytes_processed

    def execute(self, sql: str, row_limit: int = DEFAULT_ROW_LIMIT) -> pd.DataFrame:
        """Run a guarded, LIMIT-capped, bytes-capped query; return a DataFrame.

        Raises:
            SQLGuardError: if the statement fails the read-only guard.
            BigQueryExecutionError: on any BigQuery-side failure.
        """
        validate_select(sql)
        bounded_sql = ensure_limit(sql, row_limit)
        job_config = bigquery.QueryJobConfig(maximum_bytes_billed=self._max_bytes_billed)
        try:
            job = self.client.query(bounded_sql, job_config=job_config)
            rows = [dict(row.items()) for row in job.result()]
        except gcloud_exceptions.GoogleAPICallError as exc:
            raise BigQueryExecutionError(exc.message or str(exc)) from exc
        logger.info("execute ok: %d rows, %s bytes processed", len(rows), job.total_bytes_processed)
        return pd.DataFrame(rows)
