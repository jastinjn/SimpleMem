# marklymem — CLAUDE.md

FastAPI server exposing the evolver memory engine as a REST API. Backed by PostgreSQL (FTS + pgvector).

## Stack

FastAPI + SQLAlchemy (async) + PostgreSQL. Full-text search via `websearch_to_tsquery`; vector search via pgvector. OpenAI for LLM extraction (`ingestion_mode=llm`) and embeddings (`embedder_mode=semantic`). Pydantic-settings for config. Alembic for migrations. Dependencies managed with uv.

## Setup

```bash
cp .env.example .env        # fill in DATABASE_URL and OPENAI_API_KEY
uv sync                     # installs deps into .venv
uv run alembic upgrade head # apply all migrations
uv run python run.py        # starts server at localhost:8100
uv run python run.py --reload  # with auto-reload
```

## Environment

All settings are configured via `.env` (see `.env.example`).

## Ingestion modes

**`llm`** (default): sliding-window LLM extraction via `llm_extractor.py`. Sends windows of up to 15 turns to OpenAI using structured outputs (Pydantic `client.responses.parse`). Infers `memory_type`, `importance`, and `confidence` per unit. Supports all four assignable types: `preference`, `procedural_observation`, `semantic`, `episodic`.

**`pattern`**: per-turn regex/keyword extraction. Fast and deterministic, no API key required. Produces `EPISODIC`, `SEMANTIC`, and `PREFERENCE` units based on keyword matching. Use when extraction latency or API cost is a priority over quality.

## Migrations

```bash
cd marklymem
uv run alembic upgrade head
uv run alembic revision --autogenerate -m "description"
```

## Tests

```bash
cd marklymem
uv run pytest evolver/tests/ -v   # unit tests (require Postgres)
uv run pytest tests/ -v           # API integration tests (require Postgres)
uv run ruff check .               # lint
uv run pyright .                  # type check
```

## Architecture

**Tenant isolation** — every request requires `user_id`. Optional `namespace` narrows to a sub-context. No namespace filter = queries across all namespaces for that user.

**Write pipeline** — `MemoryManager.ingest_session_turns()` → extraction (LLM or pattern) → pre-ingestion dedup against store → local conflict detection → optional embedding → `MemoryStore.add_memories()` → auto-consolidation (dedup + decay) → returns `{added, superseded, decayed, reinforced}`.

**Retrieval** — `MemoryRetriever` dispatches to keyword (Postgres FTS via `websearch_to_tsquery`), embedding (pgvector cosine), or hybrid. Results scored by IDF + importance + recency + reinforcement + type boost + confidence factor.

**Consolidation** — runs automatically after every ingest. Supersedes exact duplicates and near-duplicates (Jaccard ≥ 0.80). Applies importance decay to memories not accessed in 30+ days. Boosts reinforcement score for memories sharing entities.

## Observability

`marklymem/telemetry.py` is a context-manager facade over OpenTelemetry that exports traces to a self-hosted Langfuse instance via OTLP/HTTP. Tracing is opt-in (`OTEL_ENABLED=true` + Langfuse host/keys). When disabled, every facade call is an OTel no-op — zero overhead.

Two operations are traced end-to-end:

- **`memory.ingest`**: `extract.session` → per-window `extract.window` generation spans (LLM, tokens) → `embedding` batch span with per-chunk `embedding.chunk` generation spans → `consolidate` span (output = content of superseded memories).
- **`memory.retrieve`**: `embedding` span (model, token count) → output = retrieved memories with scores and content.

Call sites use `telemetry.trace()` (root), `telemetry.span()` (child), `telemetry.generation()` (LLM/embedding API call). SDK imports are local to `setup_telemetry()` — no hard SDK dep at import time. `setup_telemetry(settings)` is called in the FastAPI lifespan; `shutdown_telemetry()` flushes the batch processor on exit.

## Key Constraints

- `user_id` is required on every store/retriever call — never omit it.
- `namespace=None` on reads means no namespace filter (all namespaces returned); on writes it targets the null namespace. Reads are hierarchical — `"proj"` matches `"proj/api"`, `"proj/api/auth"`, etc. Writes and consolidation always target the exact namespace.
- Self-evolution, policy optimisation, and benchmark modules have been removed — the engine is inference-only.
