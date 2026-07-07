# ADR-001 — LangGraph state machine over plain LangChain chains

**Status:** Accepted

## Context

The agent needs branching intent routing, a bounded SQL self-heal loop, a human-in-the-loop pause for delete confirmation, and resumable per-conversation state. Plain LangChain chains (LCEL) express linear pipelines well but make cycles, conditional branches, and mid-run human pauses awkward — they would have to be hand-rolled with custom control flow around the chain. The assignment also explicitly prefers LangGraph/LangChain v1, and CLAUDE.md mandates a LangGraph state machine.

## Decision

Model the entire agent as a LangGraph `StateGraph` over a single `TypedDict` state. Nodes are pure-ish functions (router, retrieval, SQL gen, guard, execute, repair, mask, report, delete branch); edges — including the self-heal cycle capped by `retry_count` and the conditional delete branch — are explicit graph edges. Checkpointing uses LangGraph's checkpointer interface: in-memory for the prototype, Firestore-backed in production, keyed by `conversation_id`.

## Consequences

- The self-heal loop and delete `interrupt()` come from framework primitives instead of bespoke code; the retry cap is a graph edge condition, trivially testable.
- Every node transition is an observable event — per-node tracing and metrics (§5.7 of the architecture doc) fall out naturally.
- Slightly more boilerplate than a single chain, and node signatures must stay disciplined about which state fields they read/write.
- Swapping the checkpointer (memory → Firestore) is the only change needed to move conversation persistence to production.
