# evolver_server — CLAUDE.md

FastAPI server exposing the evolver memory engine as a REST API. Backed by PostgreSQL (FTS + pgvector). No LLM required — all extraction, retrieval, and consolidation is deterministic.

## Layout

```
evolver_server/
  app.py          # FastAPI app, lifespan, route handlers
  config.py       # Settings (pydantic-settings, loads from .env)
  models.py       # Pydantic request/response models
  run.py          # Uvicorn launcher (uv run python run.py)
  evolver/        # Core memory engine (MemoryManager, MemoryStore, etc.)
    manager.py    # Facade: ingest, retrieve, consolidate
    store.py      # Async PostgreSQL store (SQLAlchemy + pgvector)
    retriever.py  # Keyword (FTS5) + embedding + hybrid retrieval
    consolidator.py # Dedup, near-dup merge, decay, entity reinforcement
    models.py     # MemoryUnit, MemoryType, MemoryQuery, etc.
    schema.py     # SQLAlchemy ORM models
    db.py         # Engine + sessionmaker factory
    embeddings.py # HashingEmbedder + sentence-transformer wrapper
    config.py     # EvolveMemConfig
    tests/        # Unit + integration tests (require Postgres)
  alembic/        # DB migrations
  tests/          # API-level integration tests (require Postgres)
```

## Setup

```bash
cp .env.example .env        # fill in DATABASE_URL
uv sync                     # installs deps into .venv
uv run python run.py        # starts server at localhost:8100
uv run python run.py --reload  # with auto-reload
```

## Environment

Key variables (see `.env.example`):

| Variable | Default | Description |
|---|---|---|
| `FASTAPI_HOST` | `localhost` | Bind address |
| `FASTAPI_PORT` | `8100` | Bind port |
| `DATABASE_URL` | — | PostgreSQL async DSN (`postgresql+asyncpg://...`) |

Other settings (`embedding_dim`, `retrieval_mode`, `db_pool_size`, etc.) are configured directly in `config.py`.

## Migrations

```bash
cd evolver_server
uv run alembic upgrade head   # apply all migrations
uv run alembic revision --autogenerate -m "description"  # new migration
```

## Tests

```bash
cd evolver_server
uv run pytest evolver/tests/ -v   # unit tests (require Postgres)
uv run pytest tests/ -v           # API integration tests (require Postgres)
```

## Architecture

**Tenant isolation** — every request requires `user_id`. Optional `scope_id` narrows to a sub-context (e.g. a class or assignment). No scope filter = queries across all scopes for that user.

**Memory pipeline on write** — `MemoryManager.ingest_session_turns()` → regex pattern extraction → optional embedding → `MemoryStore.add_memories()` → auto-consolidation (dedup + decay).

**Retrieval** — `MemoryManager.retrieve_for_prompt()` dispatches to keyword (Postgres FTS via `websearch_to_tsquery`), embedding (pgvector `<=>` cosine), or hybrid. Results ranked by IDF score + importance + reinforcement score.

**Consolidation** — runs automatically after every ingest. Supersedes stale working summaries, exact duplicates, and near-duplicates (Jaccard ≥ 0.80). Applies importance decay to memories not accessed in 30+ days. Boosts reinforcement score for memories sharing entities.

## Key Constraints

- `evolver/` is the memory engine — do not import from `evolver_server.app` inside it.
- `user_id` is required on every store/retriever call — never omit it.
- `scope_id=None` means no scope filter (all scopes), not a default scope.
- Embeddings are generated at write time if an embedder is configured; retrieval re-encodes the query on the fly.
