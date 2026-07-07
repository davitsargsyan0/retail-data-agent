"""Extended PII-masking edge cases (architecture §5.2) — no network.

Complements ``test_pii.py`` with the harder cases the assignment's requirement 2
hinges on: multiple PII tokens per cell, partial/adjacent matches, international
phone shapes, address abbreviations, the column denylist for aliases and exact
names, ``t.*`` star expansion, and already-safe text left untouched.
"""

from __future__ import annotations

import pytest

from safety import pii


class TestMaskTextMultipleAndCombined:
    def test_masks_every_email_in_a_cell(self) -> None:
        out = pii.mask_text("Contact a@b.com or c@d.org for details.")
        assert "a@b.com" not in out
        assert "c@d.org" not in out
        assert out.count(pii.EMAIL_MASK) == 2

    def test_masks_email_and_phone_together(self) -> None:
        out = pii.mask_text("Reach jane@example.com or call 555-123-4567.")
        assert "jane@example.com" not in out
        assert "555-123-4567" not in out
        assert pii.EMAIL_MASK in out and pii.PHONE_MASK in out

    def test_masks_email_phone_and_address_all_present(self) -> None:
        # Separated by prose so the three regexes do not overlap.
        text = "Email: jane@example.com. Phone: 555-123-4567. Address: 742 Evergreen Terrace."
        out = pii.mask_text(text)
        assert "jane@example.com" not in out
        assert "555-123-4567" not in out
        assert "742 Evergreen Terrace" not in out

    def test_international_phone_with_country_code(self) -> None:
        assert "+1 555 123 4567" not in pii.mask_text("Call +1 555 123 4567 now")

    def test_dotted_phone_separator(self) -> None:
        assert "555.123.4567" not in pii.mask_text("num 555.123.4567 here")

    @pytest.mark.parametrize(
        "address",
        [
            "742 Evergreen Ave",
            "10 Downing St",
            "1600 Pennsylvania Avenue",
            "221 Baker Road",
            "5 Elm Boulevard",
        ],
    )
    def test_masks_common_street_suffixes(self, address: str) -> None:
        assert address not in pii.mask_text(f"Ship to {address} today")


class TestMaskTextSafeText:
    def test_already_safe_text_unchanged(self) -> None:
        text = "Category Jeans led with 4,321 units and 12.5% margin."
        assert pii.mask_text(text) == text

    def test_product_sku_like_integer_untouched(self) -> None:
        text = "SKU 1234567890 sold 42 units"
        assert pii.mask_text(text) == text

    def test_empty_string(self) -> None:
        assert pii.mask_text("") == ""

    def test_double_masking_is_idempotent(self) -> None:
        once = pii.mask_text("x@y.com / 555-123-4567 / 742 Evergreen Ter")
        assert pii.mask_text(once) == once


class TestColumnDenylist:
    @pytest.mark.parametrize(
        "column",
        ["email", "phone", "phone_number", "street_address", "address"],
    )
    def test_exact_pii_column_names_flagged(self, column: str) -> None:
        assert pii._is_denylisted_column(column) is True

    @pytest.mark.parametrize(
        "column",
        ["customer_email", "user_phone", "billing_street_address", "EMAIL", "Phone_Number"],
    )
    def test_aliased_pii_columns_flagged(self, column: str) -> None:
        assert pii._is_denylisted_column(column) is True

    @pytest.mark.parametrize(
        "column",
        ["first_name", "revenue", "order_id", "category", "sale_price", "created_at"],
    )
    def test_benign_columns_not_flagged(self, column: str) -> None:
        assert pii._is_denylisted_column(column) is False

    def test_star_alias_on_users_flagged(self) -> None:
        assert "* (users)" in pii.find_pii_columns("SELECT t.* FROM users t")

    def test_star_without_users_not_flagged(self) -> None:
        assert pii.find_pii_columns("SELECT * FROM orders") == []

    def test_multiple_pii_columns_all_reported(self) -> None:
        cols = pii.find_pii_columns("SELECT email, phone, street_address FROM users")
        assert set(cols) >= {"email", "phone", "street_address"}


class TestScrubRowsEdge:
    def test_drops_multiple_denylisted_columns(self) -> None:
        rows = [{"customer_email": "a@b.com", "phone_number": "555-1234", "name": "Jane", "x": 5}]
        result = pii.scrub_rows(rows)
        assert result.rows[0]["customer_email"] == pii.COLUMN_MASK
        assert result.rows[0]["phone_number"] == pii.COLUMN_MASK
        assert result.rows[0]["name"] == "Jane"
        assert result.rows[0]["x"] == 5
        assert set(result.dropped_columns) == {"customer_email", "phone_number"}

    def test_masks_free_text_pii_when_column_name_is_benign(self) -> None:
        # A benign column ("note") can still carry PII in its value.
        rows = [{"note": "email boss at boss@corp.com"}]
        result = pii.scrub_rows(rows)
        assert "boss@corp.com" not in result.rows[0]["note"]
        assert result.cells_masked == 1

    def test_non_string_cells_are_preserved(self) -> None:
        rows = [{"revenue": 8_053_624.54, "orders": 125226, "active": True}]
        result = pii.scrub_rows(rows)
        assert result.rows == rows
        assert result.fired() is False

    def test_column_dropped_across_heterogeneous_rows(self) -> None:
        # Denylist is computed over the union of columns across all rows.
        rows = [{"name": "A"}, {"name": "B", "email": "b@x.com"}]
        result = pii.scrub_rows(rows)
        assert "email" in result.dropped_columns
        assert all(row.get("email", pii.COLUMN_MASK) == pii.COLUMN_MASK for row in result.rows)
