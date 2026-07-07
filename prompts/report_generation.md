# Report Generation

Write a short executive summary of the query results below, answering the
user's question. The summary sits above a data table, so do not repeat every
row — state the headline finding and what it means.

Rules:

- 2–4 sentences. Lead with the headline number or finding.
- Use ONLY values present in the result rows — never invent, extrapolate, or
  round beyond normal presentation (e.g. $8.05M for 8049837.34).
- No SQL, schema, or column-name jargon.
- Some cells may contain redaction tokens like «email redacted» — never mention
  them and never guess what was redacted.
- Output plain prose only — no headings, no bullet lists, no markdown fences.

## User question

{question}

## Result rows (already PII-masked, JSON)

{rows}

## Style notes from past analyst reports on similar questions

{style_notes}
