"""Saved-reports store — JSON files with metadata (architecture §5.3).

**Prototype:** each report is a JSON file under ``data/reports/`` and the audit
trail is appended to ``data/audit.jsonl``. **Production:** a Firestore ``reports``
collection with the same document shape; ownership scoping and audit writes move
to Firestore queries unchanged.

The store owns two safety-critical properties:

- **Ownership scoping is enforced here, in the query — never in a prompt.** Every
  read and every delete filters on ``owner``, so a manipulated LLM can neither
  see nor delete another manager's reports.
- **Deletes are explicit and auditable.** ``delete`` only removes reports the
  owner actually holds and returns exactly what it removed; the delete node pairs
  it with an :meth:`ReportStore.append_audit` record.

``parse_delete_request`` turns a natural-language delete command
("delete all reports mentioning Client X", "delete today's reports") into a
structured, resolvable filter — deterministically, so the resolved candidate set
shown at the confirmation interrupt is exactly what executes.
"""

from __future__ import annotations

import json
import re
import uuid
from dataclasses import asdict, dataclass, field
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any, Literal

# Words we treat as proper-noun entities when auto-extracting from a report.
_STOPWORDS = frozenset(
    {
        "the", "a", "an", "and", "or", "of", "for", "to", "in", "on", "by", "with",
        "here", "what", "who", "why", "how", "which", "these", "this", "that",
        "i", "you", "we", "report", "reports", "analysis", "found", "total",
        "top", "customers", "customer", "revenue", "orders", "order", "products",
        "product", "monthly", "average",
    }
)

_DELETE_VERB = re.compile(r"\b(delete|remove|purge|clear|discard|trash|erase)\b", re.I)
_TODAY = re.compile(r"\btoday'?s?\b", re.I)
_ENTITY_CLAUSE = re.compile(
    r"\b(?:mentioning|about|regarding|referencing|containing|that\s+mentions?|which\s+mention)\s+(.+)",
    re.I,
)
_ALL_OR_MINE = re.compile(r"\b(all|every|my)\b", re.I)


def extract_entities(text: str) -> list[str]:
    """Best-effort deterministic entity extraction for report metadata.

    Pulls quoted strings and capitalised word-sequences (brand/category/client
    names) out of a report, dropping common words. Used only to enrich search;
    delete-matching also scans the title and body, so a miss here is harmless.
    """
    entities: list[str] = []
    seen: set[str] = set()

    def _add(value: str) -> None:
        cleaned = value.strip().strip("\"'“”").strip()
        key = cleaned.lower()
        if cleaned and key not in seen and key not in _STOPWORDS:
            seen.add(key)
            entities.append(cleaned)

    for quoted in re.findall(r"[\"“]([^\"”]{2,40})[\"”]", text):
        _add(quoted)
    for match in re.finditer(r"\b([A-Z][A-Za-z0-9&]+(?:\s+[A-Z][A-Za-z0-9&]+)*)", text):
        phrase = match.group(1)
        # Keep multi-word proper nouns; for single words, skip common stopwords.
        if " " in phrase or phrase.lower() not in _STOPWORDS:
            _add(phrase)
    return entities


@dataclass
class Report:
    """A saved report and its metadata."""

    id: str
    owner: str
    title: str
    body: str
    created_at: str  # ISO-8601, UTC
    mentioned_entities: list[str] = field(default_factory=list)


DeleteKind = Literal["entity", "date", "all"]


@dataclass(frozen=True)
class DeleteRequest:
    """A parsed, resolvable delete command."""

    kind: DeleteKind
    description: str  # human-readable, shown in the confirmation preview
    entity: str | None = None
    day: date | None = None


def _clean_entity(raw: str) -> str:
    """Trim a captured entity phrase ('Client X reports.' -> 'Client X')."""
    text = raw.strip().strip(".!?\"'“” ")
    text = re.sub(r"\breports?\b.*$", "", text, flags=re.I).strip()
    return text.strip(".!?\"'“” ")


def parse_delete_request(text: str) -> DeleteRequest | None:
    """Parse a natural-language delete command; ``None`` if it isn't one.

    Recognises entity scope ("...mentioning Client X"), date scope
    ("...today's reports"), and a bare "delete all/my reports". Deterministic:
    the same text always resolves to the same candidate set.
    """
    if not _DELETE_VERB.search(text):
        return None

    if _TODAY.search(text):
        return DeleteRequest("date", "created today", day=datetime.now(UTC).date())

    match = _ENTITY_CLAUSE.search(text)
    if match:
        entity = _clean_entity(match.group(1))
        if entity:
            return DeleteRequest("entity", f"mentioning “{entity}”", entity=entity)

    # A delete verb with no narrower scope -> all of the user's reports. The
    # confirmation interrupt still shows the full list before anything happens.
    if _ALL_OR_MINE.search(text) or "report" in text.lower():
        return DeleteRequest("all", "all of your saved reports")
    return None


class ReportStore:
    """File-backed saved-reports store, scoped by ``owner`` on every access."""

    def __init__(self, root: Path) -> None:
        self._root = root
        self._dir = root / "reports"
        self._audit = root / "audit.jsonl"

    # -- writes -----------------------------------------------------------
    def save(
        self,
        owner: str,
        title: str,
        body: str,
        mentioned_entities: list[str] | None = None,
    ) -> Report:
        """Persist a report and return it. Entities are auto-extracted if omitted."""
        entities = (
            mentioned_entities
            if mentioned_entities is not None
            else extract_entities(f"{title}\n{body}")
        )
        report = Report(
            id=uuid.uuid4().hex[:12],
            owner=owner,
            title=title.strip() or "Untitled analysis",
            body=body,
            created_at=datetime.now(UTC).isoformat(),
            mentioned_entities=entities,
        )
        self._dir.mkdir(parents=True, exist_ok=True)
        (self._dir / f"{report.id}.json").write_text(
            json.dumps(asdict(report), indent=2), encoding="utf-8"
        )
        return report

    def delete(self, owner: str, ids: list[str]) -> list[str]:
        """Delete only reports that ``owner`` holds and whose id is requested.

        Ownership is re-checked here (not trusted from the caller), so this is
        the single choke point that makes cross-owner deletion impossible.
        """
        wanted = set(ids)
        deleted: list[str] = []
        for report in self.list_for_owner(owner):
            if report.id in wanted:
                try:
                    (self._dir / f"{report.id}.json").unlink()
                except OSError:
                    continue
                deleted.append(report.id)
        return deleted

    def append_audit(self, record: dict[str, Any]) -> None:
        """Append an immutable audit line (who/when/what/confirmation)."""
        self._root.mkdir(parents=True, exist_ok=True)
        with self._audit.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")

    # -- reads (all owner-scoped) ----------------------------------------
    def _load_all(self) -> list[Report]:
        if not self._dir.exists():
            return []
        reports: list[Report] = []
        for path in sorted(self._dir.glob("*.json")):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                reports.append(Report(**data))
            except (OSError, json.JSONDecodeError, TypeError):
                continue
        return reports

    def list_for_owner(self, owner: str) -> list[Report]:
        """All of ``owner``'s reports, newest first."""
        owned = [report for report in self._load_all() if report.owner == owner]
        return sorted(owned, key=lambda report: report.created_at, reverse=True)

    def match_by_entity(self, owner: str, needle: str) -> list[Report]:
        """Owner's reports mentioning ``needle`` (title, body, or entities)."""
        query = needle.lower()
        return [
            report
            for report in self.list_for_owner(owner)
            if query in report.title.lower()
            or query in report.body.lower()
            or any(query in entity.lower() for entity in report.mentioned_entities)
        ]

    def match_by_date(self, owner: str, day: date) -> list[Report]:
        """Owner's reports created on ``day`` (UTC)."""
        return [
            report
            for report in self.list_for_owner(owner)
            if _created_date(report) == day
        ]

    def resolve(self, owner: str, request: DeleteRequest) -> list[Report]:
        """Resolve a parsed delete request into a concrete, owner-scoped set."""
        if request.kind == "entity" and request.entity is not None:
            return self.match_by_entity(owner, request.entity)
        if request.kind == "date" and request.day is not None:
            return self.match_by_date(owner, request.day)
        return self.list_for_owner(owner)


def _created_date(report: Report) -> date | None:
    try:
        return datetime.fromisoformat(report.created_at).date()
    except ValueError:
        return None
