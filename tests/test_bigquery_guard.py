"""Unit tests for the read-only SQL guard and LIMIT injection — no network."""

from __future__ import annotations

import pytest

from tools.bigquery_client import (
    SQLGuardError,
    ensure_limit,
    strip_comments_and_literals,
    validate_select,
)


class TestValidateSelect:
    def test_plain_select_passes(self) -> None:
        validate_select("SELECT order_id FROM `bigquery-public-data.thelook_ecommerce.orders`")

    def test_with_cte_passes(self) -> None:
        validate_select(
            "WITH revenue AS (SELECT SUM(sale_price) AS total FROM "
            "`bigquery-public-data.thelook_ecommerce.order_items`) SELECT * FROM revenue"
        )

    def test_trailing_semicolon_passes(self) -> None:
        validate_select("SELECT 1;")

    def test_lowercase_select_passes(self) -> None:
        validate_select("select 1")

    def test_replace_function_passes(self) -> None:
        validate_select("SELECT REPLACE(name, 'a', 'b') FROM products")

    def test_created_at_column_passes(self) -> None:
        validate_select("SELECT created_at FROM orders")

    @pytest.mark.parametrize(
        "sql",
        [
            "DELETE FROM orders WHERE 1=1",
            "INSERT INTO orders VALUES (1)",
            "UPDATE orders SET status = 'x'",
            "DROP TABLE orders",
            "CREATE TABLE t AS SELECT 1",
            "TRUNCATE TABLE orders",
            "MERGE orders USING x ON true WHEN MATCHED THEN DELETE",
        ],
    )
    def test_dml_ddl_rejected(self, sql: str) -> None:
        with pytest.raises(SQLGuardError):
            validate_select(sql)

    def test_multi_statement_rejected(self) -> None:
        with pytest.raises(SQLGuardError, match="Multi-statement"):
            validate_select("SELECT 1; SELECT 2")

    def test_select_then_dml_rejected(self) -> None:
        with pytest.raises(SQLGuardError):
            validate_select("SELECT 1; DROP TABLE orders")

    def test_empty_rejected(self) -> None:
        with pytest.raises(SQLGuardError, match="Empty"):
            validate_select("   ")

    def test_comment_only_rejected(self) -> None:
        with pytest.raises(SQLGuardError, match="Empty"):
            validate_select("-- just a comment")

    def test_forbidden_keyword_inside_string_literal_is_fine(self) -> None:
        validate_select("SELECT 'please DROP TABLE orders' AS note")

    def test_semicolon_inside_string_literal_is_fine(self) -> None:
        validate_select("SELECT 'a;b' AS note")

    def test_dml_hidden_behind_comment_rejected(self) -> None:
        with pytest.raises(SQLGuardError):
            validate_select("/* harmless */ DELETE FROM orders")

    def test_non_select_scripting_rejected(self) -> None:
        with pytest.raises(SQLGuardError):
            validate_select("BEGIN SELECT 1; END")


class TestEnsureLimit:
    def test_appends_limit_when_missing(self) -> None:
        assert ensure_limit("SELECT 1", limit=50) == "SELECT 1\nLIMIT 50"

    def test_strips_trailing_semicolon_before_appending(self) -> None:
        assert ensure_limit("SELECT 1;", limit=10) == "SELECT 1\nLIMIT 10"

    def test_keeps_existing_limit(self) -> None:
        sql = "SELECT 1 LIMIT 5"
        assert ensure_limit(sql) == sql

    def test_existing_lowercase_limit_kept(self) -> None:
        sql = "select 1 limit 5"
        assert ensure_limit(sql) == sql

    def test_limit_inside_string_literal_does_not_count(self) -> None:
        sql = "SELECT 'no LIMIT 5 here' AS note"
        assert ensure_limit(sql, limit=7).endswith("LIMIT 7")


class TestStripCommentsAndLiterals:
    def test_line_comment_removed(self) -> None:
        assert "secret" not in strip_comments_and_literals("SELECT 1 -- secret")

    def test_block_comment_removed(self) -> None:
        assert "secret" not in strip_comments_and_literals("SELECT /* secret */ 1")

    def test_literals_blanked_but_quotes_kept(self) -> None:
        assert strip_comments_and_literals("SELECT 'abc'") == "SELECT ''"

    def test_backtick_identifiers_blanked(self) -> None:
        result = strip_comments_and_literals("SELECT x FROM `proj.ds.table`")
        assert "proj" not in result
