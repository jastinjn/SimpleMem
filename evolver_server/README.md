# evolver_server

FastAPI REST API for the evolver memory engine. Stores and retrieves memories for LLM agents using PostgreSQL (full-text search + pgvector). No LLM or API key required.

## Quickstart

```bash
cp .env.example .env
# edit .env — set DATABASE_URL to your Postgres instance

uv sync
uv run python run.py
```

Server starts at `http://localhost:8100`. Interactive docs at `http://localhost:8100/docs`.

## Endpoints

| Method | Path | Description |
|---|---|---|
| `POST` | `/memory/add` | Ingest a single conversation turn |
| `POST` | `/memory/add_batch` | Ingest multiple turns at once |
| `POST` | `/memory/retrieve` | Retrieve relevant memories for a query |
| `POST` | `/memory/clear` | Archive all memories for a user/scope |
| `POST` | `/memory/stats` | Memory counts and type breakdown |
| `GET` | `/health` | Health check |

## Request model

All write and query endpoints require `user_id`. `scope_id` is optional — use it to namespace memories within a user (e.g. by class or assignment).

```json
{
  "user_id": "teacher_abc",
  "scope_id": "year10-english",
  "session_id": "session_1",
  "prompt_text": "always deduct marks for missing citations",
  "response_text": ""
}
```

## Environment

| Variable | Description |
|---|---|
| `FASTAPI_HOST` | Bind address (default `localhost`) |
| `FASTAPI_PORT` | Bind port (default `8100`) |
| `DATABASE_URL` | PostgreSQL async DSN e.g. `postgresql+asyncpg://user:pass@localhost:5432/simplemem` |

## Database

Run migrations before first use:

```bash
uv run alembic upgrade head
```
