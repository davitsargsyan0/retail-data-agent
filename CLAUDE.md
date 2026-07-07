# CLAUDE.md

## Context

Technical assignment — data-analysis chat agent for retail executives.
Full spec in `assignment/assignment.md`. **Read it before any design or code decision.**
Grading focus: system design, technical explanation, elegant prototype. 6–12h budget.

## Hard rules

- LangGraph state machine, CLI interface only. No web UI.
- BigQuery dataset: `bigquery-public-data.thelook_ecommerce` (`orders`, `order_items`, `products`, `users`). **READ-ONLY.**
- LLM: Gemini via `langchain-google-genai`. All LLM calls behind one thin wrapper in `src/tools/llm.py`.
- PII (email, phone, street address) must **NEVER** appear in final output — enforce with a deterministic masking layer, not prompting alone.
- Prototype requirements implemented: PII Masking, High-Stakes Oversight (delete confirmation via LangGraph interrupt), Resilience/self-heal (max 2 SQL retries).
- Persona lives in `prompts/persona.md`, hot-loaded per request.
- Lean: stdlib + declared deps only. No vector DB — numpy over Gemini embeddings for trio retrieval.
- Every architectural choice gets a short ADR in `docs/decisions/`.
- Style: uv, ruff, mypy strict on `src/`, pytest. Type hints everywhere.
- Never invent BigQuery schema — use `INFORMATION_SCHEMA` or the documented thelook schema in `docs/schema.md`.

## Workflow

- Commit after each working increment with conventional commits.
- Run pytest before claiming anything works.
