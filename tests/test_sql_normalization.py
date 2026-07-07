"""Unit tests for the table-reference normalizer — no network needed."""

from __future__ import annotations

from agent.graph import normalize_table_references

# The exact failing shape from live use: unquoted fully-qualified table in
# FROM, plus dotted full-path column references. BigQuery dry-run failed with
# "Unrecognized name: bigquery at [7:5]".
FAILING_SQL = """\
SELECT
    FORMAT_TIMESTAMP('%Y-%m',
        bigquery-public-data.thelook_ecommerce.order_items.created_at) AS month,
    ROUND(SUM(bigquery-public-data.thelook_ecommerce.order_items.sale_price), 2) AS total_revenue
FROM
    bigquery-public-data.thelook_ecommerce.order_items
WHERE
    bigquery-public-data.thelook_ecommerce.order_items.status NOT IN ('Cancelled', 'Returned')
    AND EXTRACT(YEAR FROM bigquery-public-data.thelook_ecommerce.order_items.created_at) = 2024
GROUP BY month
ORDER BY month
LIMIT 12"""


class TestNormalizeTableReferences:
    def test_bare_from_clause_gets_backticks(self) -> None:
        result = normalize_table_references(FAILING_SQL)
        assert "FROM\n    `bigquery-public-data.thelook_ecommerce.order_items`" in result

    def test_dotted_column_paths_get_table_backticked(self) -> None:
        result = normalize_table_references(FAILING_SQL)
        assert "`bigquery-public-data.thelook_ecommerce.order_items`.created_at" in result
        assert "`bigquery-public-data.thelook_ecommerce.order_items`.sale_price" in result
        assert "`bigquery-public-data.thelook_ecommerce.order_items`.status" in result

    def test_no_unquoted_references_remain(self) -> None:
        result = normalize_table_references(FAILING_SQL)
        for line in result.splitlines():
            assert "bigquery-public-data" not in line.replace(
                "`bigquery-public-data.thelook_ecommerce.order_items`", ""
            )

    def test_already_quoted_reference_is_untouched(self) -> None:
        sql = (
            "SELECT oi.sale_price\n"
            "FROM `bigquery-public-data.thelook_ecommerce.order_items` AS oi\n"
            "LIMIT 10"
        )
        assert normalize_table_references(sql) == sql

    def test_idempotent(self) -> None:
        once = normalize_table_references(FAILING_SQL)
        assert normalize_table_references(once) == once

    def test_multiple_tables_all_wrapped(self) -> None:
        sql = (
            "SELECT o.order_id, u.state\n"
            "FROM bigquery-public-data.thelook_ecommerce.orders AS o\n"
            "JOIN bigquery-public-data.thelook_ecommerce.users AS u ON o.user_id = u.id\n"
            "LIMIT 5"
        )
        result = normalize_table_references(sql)
        assert "FROM `bigquery-public-data.thelook_ecommerce.orders` AS o" in result
        assert "JOIN `bigquery-public-data.thelook_ecommerce.users` AS u" in result

    def test_sql_without_dataset_references_unchanged(self) -> None:
        sql = "SELECT 1 AS x"
        assert normalize_table_references(sql) == sql
