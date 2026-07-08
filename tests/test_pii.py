"""Unit tests for deterministic PII masking (architecture §5.2) — no network."""

from __future__ import annotations

from agent.graph import sql_guard
from agent.state import AgentState
from safety import pii


class TestMaskText:
    def test_masks_email(self) -> None:
        out = pii.mask_text("Reach Jane at jane.doe@example.com today.")
        assert "jane.doe@example.com" not in out
        assert pii.EMAIL_MASK in out

    def test_masks_phone_with_separators(self) -> None:
        assert "555-123-4567" not in pii.mask_text("Call 555-123-4567 now")
        assert "(555) 123-4567" not in pii.mask_text("Call (555) 123-4567 now")

    def test_masks_street_address(self) -> None:
        out = pii.mask_text("Ship to 123 Main Street, apt 4.")
        assert "123 Main Street" not in out
        assert pii.ADDRESS_MASK in out

    def test_does_not_mask_plain_integer_revenue(self) -> None:
        # A bare integer must not be mistaken for a phone number.
        assert pii.mask_text("Total revenue was 5551234567") == "Total revenue was 5551234567"

    def test_does_not_mask_currency(self) -> None:
        text = "Revenue: 1,234,567.89 across 42 orders"
        assert pii.mask_text(text) == text

    def test_idempotent(self) -> None:
        once = pii.mask_text("a@b.com and 555-123-4567")
        assert pii.mask_text(once) == once

    def test_does_not_bridge_amounts_across_table_lines(self) -> None:
        # Regression: in a rendered month/revenue table, "004.60\n2025-02" is a
        # phone-shaped digit run — the mask must never join digits across lines.
        table = (
            "  month  total_revenue\n"
            "2025-01     141,004.60\n"
            "2025-02     143,007.91\n"
            "2025-03     147,553.11"
        )
        assert pii.mask_text(table) == table

    def test_does_not_mask_amount_followed_by_date_on_one_line(self) -> None:
        # The tail of a formatted amount ("...,004.60") must not seed a match
        # even when a date-like digit group follows on the same line.
        text = "revenue was 141,004.60 2025-02 was stronger"
        assert pii.mask_text(text) == text

    def test_still_masks_phone_after_comma_with_space(self) -> None:
        out = pii.mask_text("Reach Jane at, 555-123-4567.")
        assert "555-123-4567" not in out
        assert pii.PHONE_MASK in out


class TestFindPiiColumns:
    def test_flags_qualified_email(self) -> None:
        sql = "SELECT u.first_name, u.email FROM users AS u"
        assert "email" in pii.find_pii_columns(sql)

    def test_flags_street_address_and_phone(self) -> None:
        cols = pii.find_pii_columns("SELECT street_address, phone FROM users")
        assert "street_address" in cols and "phone" in cols

    def test_flags_alias_customer_email(self) -> None:
        assert "customer_email" in pii.find_pii_columns("SELECT customer_email FROM t")

    def test_flags_select_star_on_users(self) -> None:
        assert "* (users)" in pii.find_pii_columns("SELECT * FROM users")

    def test_ignores_pii_word_in_string_literal(self) -> None:
        # Comments/literals are stripped before scanning.
        assert pii.find_pii_columns("SELECT 'send email to boss' AS note FROM orders") == []

    def test_clean_analysis_query_passes(self) -> None:
        sql = (
            "SELECT u.first_name, u.last_name, SUM(oi.sale_price) AS spend "
            "FROM order_items oi JOIN users u ON oi.user_id = u.id GROUP BY 1,2"
        )
        assert pii.find_pii_columns(sql) == []
        assert pii.sql_selects_pii(sql) is False


class TestScrubRows:
    def test_drops_denylisted_column(self) -> None:
        rows = [{"first_name": "Jane", "email": "jane@example.com", "spend": 100.0}]
        result = pii.scrub_rows(rows)
        assert result.rows[0]["email"] == pii.COLUMN_MASK
        assert result.rows[0]["first_name"] == "Jane"
        assert "email" in result.dropped_columns
        assert result.fired() is True

    def test_masks_pii_inside_free_text_cell(self) -> None:
        rows = [{"note": "contact jane@example.com or 555-123-4567"}]
        result = pii.scrub_rows(rows)
        assert "jane@example.com" not in result.rows[0]["note"]
        assert result.cells_masked == 1
        assert result.fired() is True

    def test_clean_rows_untouched(self) -> None:
        rows = [{"category": "Jeans", "revenue": 1234.5}]
        result = pii.scrub_rows(rows)
        assert result.rows == rows
        assert result.fired() is False

    def test_empty_rows(self) -> None:
        assert pii.scrub_rows([]).rows == []


class TestSqlGuardPiiDenylist:
    """The query-plan check (layer 2) rejects PII-selecting SQL before it runs,
    feeding the self-heal loop — so no BigQuery call is made here."""

    def test_pii_selecting_query_is_blocked_before_execution(self) -> None:
        state: AgentState = {
            "sql": "SELECT u.email FROM `bigquery-public-data.thelook_ecommerce.users` u"
        }
        result = sql_guard(state)
        assert result["sql_error"] is not None
        assert "email" in result["sql_error"]
        assert result["heal_attempts"]  # recorded for the "here's what I tried" trail
