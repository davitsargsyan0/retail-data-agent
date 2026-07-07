"""Deterministic PII masking — defense in depth, layers 2 and 3 (architecture §5.2).

PII (customer email, phone, street address) must **never** reach the user, even
if the SQL retrieved it. Prompting is not a control; this module is pure Python,
unit-tested, and runs *after* the LLM:

- ``find_pii_columns`` / ``sql_selects_pii`` — the **query-plan check**. Parse the
  candidate SQL and report any denylisted PII column it selects, so ``sql_guard``
  can reject/repair the query *before* execution.
- ``scrub_rows`` — the **output sweep** over result rows: drop denylisted columns
  and regex-mask email/phone/address in every string cell, *before* the report
  LLM ever sees the data.
- ``mask_text`` — the same regex sweep over the final report prose, applied after
  generation.

Over-masking (a product code that looks like a phone number) is a cosmetic bug;
under-masking is an incident — so the regexes err toward masking (ADR-003).
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from tools.bigquery_client import strip_comments_and_literals

# Denylisted PII "words". A column is denylisted if its (lowercased) name equals
# or contains one of these — catches aliases like ``customer_email`` and
# ``phone_number`` as well as the bare thelook columns.
_PII_COLUMN_WORDS: tuple[str, ...] = ("email", "phone", "street_address")

# Exact column names for the fast membership path / documentation.
PII_COLUMNS: frozenset[str] = frozenset(
    {"email", "phone", "phone_number", "street_address", "address"}
)

# Deterministic replacement tokens (guillemets are unlikely to occur in data).
EMAIL_MASK = "«email redacted»"
PHONE_MASK = "«phone redacted»"
ADDRESS_MASK = "«address redacted»"
COLUMN_MASK = "«redacted»"

_EMAIL_RE = re.compile(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}")

# A phone number must carry at least one separator, a parenthesised area code,
# or a leading country code — so a bare integer (a revenue figure, an id) is not
# masked. Deliberately conservative on that side (ADR-003).
_PHONE_RE = re.compile(
    r"""(?<!\w)
        (?:\+?\d{1,3}[\s.\-]?)?          # optional country code
        (?:\(\d{2,4}\)[\s.\-]?|\d{2,4}[\s.\-])  # area code w/ paren or separator
        \d{2,4}[\s.\-]?\d{2,4}
        (?!\w)""",
    re.VERBOSE,
)

_STREET_SUFFIX = (
    r"Street|St|Avenue|Ave|Road|Rd|Boulevard|Blvd|Lane|Ln|Drive|Dr|Court|Ct|"
    r"Way|Place|Pl|Terrace|Ter|Circle|Cir|Highway|Hwy|Parkway|Pkwy|Square|Sq"
)
_STREET_RE = re.compile(
    rf"\b\d{{1,6}}\s+(?:[A-Za-z0-9.'\-]+\s+){{0,3}}(?:{_STREET_SUFFIX})\b\.?",
    re.IGNORECASE,
)


def _is_denylisted_column(name: str) -> bool:
    """True if a column name is (or contains) a denylisted PII word."""
    lowered = name.lower()
    return lowered in PII_COLUMNS or any(word in lowered for word in _PII_COLUMN_WORDS)


def find_pii_columns(sql: str) -> list[str]:
    """Return the denylisted PII column tokens referenced by ``sql``.

    Deterministic, conservative query-plan check: comments and string literals
    are stripped first (so ``'drop the email'`` in a literal never triggers),
    then denylisted words are matched as whole identifiers, qualified or not
    (``u.email``, ``users.street_address``). A ``SELECT *`` / ``t.*`` against the
    ``users`` table is also flagged, because the star expands to PII columns.
    Returns a sorted, de-duplicated list; empty means the query is clean.
    """
    cleaned = strip_comments_and_literals(sql)
    hits: set[str] = set()

    word_pattern = "|".join(_PII_COLUMN_WORDS)
    for match in re.finditer(r"\b([A-Za-z_][A-Za-z0-9_.]*)\b", cleaned):
        token = match.group(1)
        column = token.rsplit(".", 1)[-1]  # strip table qualifier
        if _is_denylisted_column(column) and re.search(word_pattern, column, re.IGNORECASE):
            hits.add(column.lower())

    star = re.search(r"(?:^|[\s,(])\*|\b[A-Za-z_][A-Za-z0-9_]*\.\*", cleaned)
    if star and re.search(r"\busers\b", cleaned, re.IGNORECASE):
        hits.add("* (users)")

    return sorted(hits)


def sql_selects_pii(sql: str) -> bool:
    """True if the query references any denylisted PII column."""
    return bool(find_pii_columns(sql))


def mask_text(text: str) -> str:
    """Mask email / phone / street-address patterns in free prose.

    Deterministic and idempotent; applied to the final report after the LLM.
    """
    text = _EMAIL_RE.sub(EMAIL_MASK, text)
    text = _STREET_RE.sub(ADDRESS_MASK, text)
    text = _PHONE_RE.sub(PHONE_MASK, text)
    return text


def _mask_cell(value: Any) -> tuple[Any, bool]:
    """Mask a single cell value; return ``(value, changed)``."""
    if not isinstance(value, str):
        return value, False
    masked = mask_text(value)
    return masked, masked != value


@dataclass(frozen=True)
class ScrubResult:
    """Outcome of scrubbing a result set — observable so nodes can log/demo it."""

    rows: list[dict[str, Any]]
    dropped_columns: list[str] = field(default_factory=list)
    cells_masked: int = 0

    def fired(self) -> bool:
        """True if any PII control triggered (a column dropped or a cell masked)."""
        return bool(self.dropped_columns) or self.cells_masked > 0


def scrub_rows(rows: Sequence[Mapping[str, Any]]) -> ScrubResult:
    """Scrub result rows: drop denylisted columns, regex-mask string cells.

    Runs *between* BigQuery and the report LLM, so the model never sees raw PII.
    Returns a :class:`ScrubResult` carrying the cleaned rows plus what fired.
    """
    if not rows:
        return ScrubResult(rows=[])

    all_columns = {key for row in rows for key in row}
    dropped = sorted(col for col in all_columns if _is_denylisted_column(col))
    dropped_set = set(dropped)

    cleaned: list[dict[str, Any]] = []
    cells_masked = 0
    for row in rows:
        new_row: dict[str, Any] = {}
        for key, value in row.items():
            if key in dropped_set:
                new_row[key] = COLUMN_MASK
                continue
            masked_value, changed = _mask_cell(value)
            if changed:
                cells_masked += 1
            new_row[key] = masked_value
        cleaned.append(new_row)

    return ScrubResult(rows=cleaned, dropped_columns=dropped, cells_masked=cells_masked)
