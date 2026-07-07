# ADR-004 — LangGraph `interrupt()` for delete confirmation

**Status:** Accepted

## Context

Requirement 3: bulk deletion of saved reports ("delete all reports mentioning Client X") is destructive and needs a strict confirmation flow **without breaking UX**; users may delete only their own reports. Options considered: (a) a bespoke two-turn protocol with pending-action flags in application state, (b) client-side confirmation dialogs, (c) LangGraph's native `interrupt()` + checkpointer.

## Decision

Use `interrupt()`. The delete branch resolves matches (query hard-scoped to `owner_id == user_id` at the store layer — never left to the prompt), previews count/titles/dates, then interrupts. The graph checkpoints and the client renders one question requiring a **typed confirmation that echoes the count** (e.g. `delete 3 reports`). On resume, exact match executes the delete and writes an immutable audit record (who, when, which IDs, filter, confirmation text); any other input cancels.

## Consequences

- Confirmation state survives process restarts and works identically for CLI (prototype, in-memory checkpointer) and the production API (Firestore checkpointer) — no client-specific session code.
- The typed count-echo defeats reflexive "y" confirmations while keeping the flow to a single extra exchange, satisfying the "without breaking UX" constraint.
- Ownership scoping in the store query means even a manipulated LLM cannot match another manager's reports.
- The same pattern is reusable verbatim for future high-stakes tools (e.g. emailing reports externally), making "add oversight" a graph-edge decision rather than new machinery.
- Cost: requires a checkpointer everywhere the graph runs, and tests must script the interrupt/resume cycle.
