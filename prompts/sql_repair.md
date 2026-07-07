# SQL Repair

A BigQuery SELECT statement failed. Fix it so it answers the user's question.

Rules:

- Output ONE corrected Google Standard SQL SELECT statement — no markdown
  fences, no explanation.
- Read-only: a single SELECT (WITH ... SELECT is fine). Never DML/DDL.
- Every table reference MUST be the fully-qualified name wrapped in backticks
  with a short alias, e.g. FROM `bigquery-public-data.thelook_ecommerce.order_items` AS oi.
- Every column reference MUST use the table alias. Use only columns that exist
  in the schema below.
- If the error says "Query returned zero rows.", the SQL was valid but found
  nothing: reconsider the filters, joins, status values, and date ranges — do
  not resubmit the same query unchanged.
- Include a LIMIT clause.

## Schema

{schema}

## User question

{question}

## Failed SQL

{failed_sql}

## Error

{error}
