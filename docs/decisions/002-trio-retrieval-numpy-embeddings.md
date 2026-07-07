# ADR-002 — Trio retrieval with numpy over Gemini embeddings; no vector DB in the prototype

**Status:** Accepted

## Context

Requirement 1 (Hybrid Intelligence) demands retrieving relevant golden Trios at query time. The obvious pattern is a vector database, but the golden bucket is a *curated expert library* — realistically hundreds to a few thousand entries, not millions. CLAUDE.md mandates lean dependencies and explicitly forbids a vector DB for the prototype. The assignment budget is 6–12 hours.

## Decision

**Prototype:** store Trios in a local JSON file (`data/golden_trios.json`), each with a pre-computed Gemini embedding (`text-embedding-004`, via the single LLM wrapper). Retrieval is cosine similarity computed as one numpy matrix–vector product over the normalized embedding matrix, returning top-k (k=3) above a 0.60 similarity floor.

**Production:** the same trio JSON objects live in a versioned GCS prefix; the embedding index moves to Vertex AI Vector Search only because of operational concerns (concurrent writers, index refresh, IAM) — not scale. The retrieval function signature (`question -> list[Trio]`) is the stable interface; the backend is swappable.

## Consequences

- At this corpus size a brute-force scan is sub-millisecond — a vector DB would add a service, credentials, and failure mode for zero user-visible benefit.
- Embeddings are pre-computed and stamped with `embedding_model_version`; changing the embedding model requires a re-embed pass (a scripted, nightly job in production).
- If the bucket ever grows past ~100k trios, the numpy path degrades gracefully (linear) and the production Vector Search path already exists.
- One extra offline step: new trios must be embedded before they are retrievable (handled by the curation pipeline / a small script in the prototype).
