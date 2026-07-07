# ADR-005 — Gemini as the primary model behind one wrapper, with OpenRouter as designed fallback

**Status:** Accepted

## Context

The assignment prefers a recent Gemini model (free AI Studio tier, generous BigQuery adjacency on GCP) and explicitly names OpenRouter/Ollama as alternatives for rate-limit problems. Requirement 5 demands resilience to third-party outages. CLAUDE.md mandates Gemini via `langchain-google-genai` with **all** LLM calls behind one thin wrapper in `src/tools/llm.py`.

## Decision

Gemini (Flash-class for routing/repair, Pro-class for report generation — both selectable per call site) is the sole provider in the prototype, called exclusively through `src/tools/llm.py`, which owns retry with exponential backoff + jitter (3 attempts on 429/5xx/timeout). **Production** adds, inside the same wrapper: a circuit breaker (open after N consecutive failures, cool-down, half-open probes) that shifts traffic to an equivalent-tier model **via OpenRouter**; keys live in Secret Manager. Embeddings follow the same path (Vertex AI in production). The fallback is design-only in the prototype — the wrapper is the seam where it plugs in without touching any node.

## Consequences

- A single choke point means resilience, tracing (Langfuse spans), token/cost accounting, and provider swaps are one-file changes; no node knows which provider answered.
- Gemini keeps the stack aligned with GCP (Vertex AI IAM, billing, data residency) and the assignment's preference.
- Fallback answers may differ stylistically from Gemini's; acceptable during an outage, and the persona prompt travels with the request regardless of provider.
- Not implementing the fallback in the prototype keeps deps lean (no OpenRouter client), at the cost of the prototype being unavailable during a Gemini outage — mitigated by backoff and clearly out of the 6–12h scope.
