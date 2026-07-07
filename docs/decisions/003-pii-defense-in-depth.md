# ADR-003 — Deterministic, layered PII defense instead of prompt-level instructions

**Status:** Accepted

## Context

Requirement 2: customer emails, phones, and street addresses must **never** appear in final output, *even if the SQL query retrieves them*, and the agent must resist malicious users. Prompt instructions alone are probabilistic and jailbreakable — they cannot be the control for a hard guarantee. CLAUDE.md mandates a deterministic masking layer.

## Decision

Three independent layers, each sufficient alone:

1. **IAM / policy tags (production only):** the agent's read-only BigQuery service account is denied PII columns via BigQuery column-level security, so offending queries fail at the warehouse. Not implementable against the public dataset, so design-only in the prototype.
2. **Pre-execution query guard (implemented):** `sql_guard` rejects non-single-SELECT statements and any query referencing denylisted columns (including `SELECT *` on `users`), and enforces `LIMIT` plus a `maximum_bytes_billed` cap. Rejections feed the self-heal loop with a repair instruction.
3. **Deterministic output mask (implemented):** pure-Python column-name sweep plus content regexes (email/phone/address patterns) applied to result rows **before** the report LLM sees them, and again to the final prose. A `masked` state flag structurally blocks report generation on unmasked data.

The intent router additionally refuses out-of-scope and injection-style requests, but is treated as UX, not as a security control.

## Consequences

- PII safety does not depend on any LLM behaving: a fully jailbroken model still cannot emit data it never received and that the output regex would scrub.
- Layer 3 is testable with plain pytest (no LLM, no network), making the guarantee CI-enforceable.
- Regex masking can over-mask (false positives on e.g. product codes resembling phone numbers) — an acceptable trade: over-masking is a cosmetic bug, under-masking is an incident.
- The denylist is config, so new sources/columns inherit protection by listing them.
