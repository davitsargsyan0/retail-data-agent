"""Unit tests for the saved-reports store and delete parsing — no network."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from reports_store.store import (
    ReportStore,
    extract_entities,
    parse_delete_request,
)


def _store(tmp_path: Path) -> ReportStore:
    return ReportStore(tmp_path)


class TestSaveAndList:
    def test_save_then_list(self, tmp_path: Path) -> None:
        store = _store(tmp_path)
        report = store.save("alice", "Top customers", "Body text")
        listed = store.list_for_owner("alice")
        assert [r.id for r in listed] == [report.id]
        assert listed[0].created_at  # ISO timestamp present

    def test_auto_extracts_entities(self, tmp_path: Path) -> None:
        store = _store(tmp_path)
        report = store.save("alice", "Revenue for Nike and Levi's", "The brand Nike led.")
        assert any("Nike" in e for e in report.mentioned_entities)


class TestOwnerScoping:
    def test_list_is_owner_scoped(self, tmp_path: Path) -> None:
        store = _store(tmp_path)
        store.save("alice", "A", "x")
        store.save("bob", "B", "y")
        assert [r.owner for r in store.list_for_owner("alice")] == ["alice"]

    def test_cannot_delete_another_owners_report(self, tmp_path: Path) -> None:
        store = _store(tmp_path)
        bob_report = store.save("bob", "B", "y")
        # Alice tries to delete Bob's report id — must delete nothing.
        deleted = store.delete("alice", [bob_report.id])
        assert deleted == []
        assert [r.id for r in store.list_for_owner("bob")] == [bob_report.id]

    def test_delete_removes_only_requested_owned(self, tmp_path: Path) -> None:
        store = _store(tmp_path)
        keep = store.save("alice", "Keep", "x")
        drop = store.save("alice", "Drop", "y")
        deleted = store.delete("alice", [drop.id])
        assert deleted == [drop.id]
        assert [r.id for r in store.list_for_owner("alice")] == [keep.id]


class TestMatching:
    def test_match_by_entity_across_title_body(self, tmp_path: Path) -> None:
        store = _store(tmp_path)
        store.save("alice", "Jeans revenue", "Category breakdown")
        store.save("alice", "Shoes revenue", "mentions Jeans in body")
        store.save("alice", "Socks revenue", "unrelated")
        matched = store.match_by_entity("alice", "jeans")
        assert len(matched) == 2

    def test_match_by_date_today(self, tmp_path: Path) -> None:
        store = _store(tmp_path)
        store.save("alice", "Today", "x")
        today = datetime.now(UTC).date()
        assert len(store.match_by_date("alice", today)) == 1


class TestAudit:
    def test_append_audit_writes_line(self, tmp_path: Path) -> None:
        store = _store(tmp_path)
        store.append_audit({"action": "delete_reports", "deleted_ids": ["abc"]})
        audit = (tmp_path / "audit.jsonl").read_text(encoding="utf-8").strip()
        assert '"delete_reports"' in audit


class TestParseDeleteRequest:
    def test_not_a_delete_returns_none(self) -> None:
        assert parse_delete_request("show me my reports") is None
        assert parse_delete_request("what is total revenue") is None

    def test_today_scope(self) -> None:
        request = parse_delete_request("delete today's reports")
        assert request is not None and request.kind == "date"
        assert request.day == datetime.now(UTC).date()

    def test_entity_scope(self) -> None:
        request = parse_delete_request("delete all reports mentioning Client X")
        assert request is not None and request.kind == "entity"
        assert request.entity == "Client X"

    def test_entity_scope_strips_trailing_reports(self) -> None:
        request = parse_delete_request("remove reports about Jeans reports")
        assert request is not None and request.entity == "Jeans"

    def test_all_scope(self) -> None:
        request = parse_delete_request("delete all my reports")
        assert request is not None and request.kind == "all"


class TestExtractEntities:
    def test_pulls_quoted_and_capitalised(self) -> None:
        entities = extract_entities('Report about "Client X" and Nike')
        assert "Client X" in entities
        assert any("Nike" in e for e in entities)

    def test_drops_common_words(self) -> None:
        assert "The" not in extract_entities("The revenue was high")
