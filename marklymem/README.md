# marklymem

FastAPI REST API for the evolver memory engine. Stores and retrieves typed memories for LLM agents using PostgreSQL (full-text search + pgvector).

## Quickstart

```bash
cp .env.example .env
# edit .env — set DATABASE_URL and OPENAI_API_KEY

uv sync
uv run alembic upgrade head
uv run python run.py
```

Server starts at `http://localhost:8100`. Interactive docs at `http://localhost:8100/docs`.

## Endpoints

| Method | Path | Description |
|---|---|---|
| `POST` | `/memory/add` | Ingest a single conversation turn |
| `POST` | `/memory/add_batch` | Ingest multiple turns (max 50) |
| `POST` | `/memory/retrieve` | Retrieve relevant memories for a query |
| `POST` | `/memory/clear` | Archive all memories for a user/scope |
| `POST` | `/memory/stats` | Memory counts and type breakdown |
| `GET` | `/health` | Health check |

## Request format

All endpoints require `user_id`. `scope_id` and `session_id` are optional.

```json
{
  "user_id": "teacher_abc",
  "scope_id": "year10-english",
  "session_id": "session_1",
  "prompt_text": "Always deduct marks for missing citations",
  "response_text": ""
}
```

Batch ingest:

```json
{
  "user_id": "teacher_abc",
  "scope_id": "year10-english",
  "turns": [
    {"prompt_text": "Always deduct marks for missing citations", "response_text": ""},
    {"prompt_text": "Students should use Harvard referencing", "response_text": "Got it."}
  ]
}
```

## Response format

Add routes return:

```json
{
  "user_id": "teacher_abc",
  "scope_id": "year10-english",
  "session_id": null,
  "units_added": 3,
  "units_consolidated": 0
}
```

`units_consolidated` is the number of older memories superseded by consolidation after this ingest.

## Ingestion modes

Set via `ingestion_mode` in `.env`:

- **`llm`** (default): sliding-window LLM extraction using OpenAI structured outputs. Infers memory type, importance, and confidence per unit. Requires `OPENAI_API_KEY`.
- **`pattern`**: per-turn regex/keyword extraction. No API key needed. Faster but less precise.

## Memory types

| Type | Description |
|---|---|
| `preference` | Stable likes, dislikes, conventions |
| `procedural_observation` | Rules and workflows ("always...", "never...") |
| `semantic` | Durable facts and knowledge |
| `episodic` | Specific events tied to a session |

## Environment

| Variable | Default | Description |
|---|---|---|
| `DATABASE_URL` | — | PostgreSQL async DSN |
| `OPENAI_API_KEY` | `""` | Required for `ingestion_mode=llm` or `embedder_mode=semantic` |
| `OPENAI_EMBEDDING_MODEL` | `text-embedding-3-small` | Embedding model |
| `ingestion_mode` | `llm` | `"llm"` or `"pattern"` |
| `retrieval_mode` | `hybrid` | `"keyword"`, `"embedding"`, `"hybrid"`, or `"auto"` |
| `embedder_mode` | `semantic` | `"semantic"` (OpenAI) or `"hashing"` (no API key) |
| `FASTAPI_HOST` | `localhost` | Bind address |
| `FASTAPI_PORT` | `8100` | Bind port |

## Database

Migrations are managed with Alembic:

```bash
uv run alembic upgrade head
```
