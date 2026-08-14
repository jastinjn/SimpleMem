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

**`pattern`** (default fallback, no API key needed): per-turn regex/keyword extraction. Fast and deterministic. Produces `EPISODIC`, `SEMANTIC`, and `PREFERENCE` units based on keyword matching.

**`llm`** (recommended): sliding-window LLM extraction via `llm_extractor.py`. Sends windows of up to 15 turns to OpenAI using structured outputs (Pydantic `client.responses.parse`). Infers `memory_type`, `importance`, and `confidence` per unit. Supports all four assignable types: `preference`, `procedural_observation`, `semantic`, `episodic`.

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

**Tenant isolation** — every request requires `user_id`. Optional `scope_id` narrows to a sub-context. No scope filter = queries across all scopes for that user.

**Write pipeline** — `MemoryManager.ingest_session_turns()` → extraction (LLM or pattern) → pre-ingestion dedup against store → local conflict detection → optional embedding → `MemoryStore.add_memories()` → auto-consolidation (dedup + decay) → returns `{added, superseded, decayed, reinforced}`.

**Retrieval** — `MemoryRetriever` dispatches to keyword (Postgres FTS via `websearch_to_tsquery`), embedding (pgvector cosine), or hybrid. Results scored by IDF + importance + recency + reinforcement + type boost + confidence factor.

**Consolidation** — runs automatically after every ingest. Supersedes exact duplicates and near-duplicates (Jaccard ≥ 0.80). Applies importance decay to memories not accessed in 30+ days. Boosts reinforcement score for memories sharing entities.

## Key Constraints

- `user_id` is required on every store/retriever call — never omit it.
- `scope_id=None` means no scope filter (all scopes), not a default scope.
- Self-evolution, policy optimisation, telemetry, and benchmark modules have been removed — the engine is inference-only.
