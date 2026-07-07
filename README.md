# retail-data-agent

Conversational data agent for a retail company — technical assignment.

## Setup

Requires Python 3.12 and [uv](https://docs.astral.sh/uv/).

```sh
uv sync
cp .env.example .env  # then fill in the values
```

## Layout

| Path | Purpose |
| --- | --- |
| `docs/` | Architecture notes and ADRs (`docs/decisions/`) |
| `src/agent/` | LangGraph graph, nodes, state |
| `src/tools/` | BigQuery client, trio retrieval |
| `src/safety/` | Intent gate, PII masking |
| `src/reports_store/` | Saved reports library and delete flow |
| `golden_bucket/trios/` | Golden question/SQL/report trios |
| `prompts/` | Persona, SQL generation, report generation prompts |
| `evals/` | Evaluation harness |
| `tests/` | Tests |
| `assignment/` | Assignment brief |
