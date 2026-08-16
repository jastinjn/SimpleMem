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

| Method | Path                   | Description                                 |
| ------ | ---------------------- | ------------------------------------------- |
| `POST` | `/memory/add`          | Ingest a single conversation turn           |
| `POST` | `/memory/add_dialogue` | Ingest multiple turns (max 50)              |
| `POST` | `/memory/retrieve`     | Retrieve relevant memories for a query      |
| `POST` | `/memory/clear`        | Archive all memories for a user/namespace       |
| `POST` | `/memory/stats`        | Memory counts and type breakdown            |
| `POST` | `/memory/clone_namespace`  | Copy all memories from one namespace to another |
| `GET`  | `/health`              | Health check                                |

## Scope IDs

`namespace` is an optional hierarchical namespace that narrows reads and writes to a sub-context within a user's memory. It follows the AWS AgentCore convention — leading slashes are allowed and segments are separated by `/`:

```
/retail-agent/customer-123/preferences
/support-agent/customer-123/case-summaries/session-001
year10-english
year10-english/term2
```

**Reads are hierarchical** — querying `namespace = "year10-english"` returns memories stored under `year10-english`, `year10-english/term2`, `year10-english/term2/week3`, etc. Omitting `namespace` returns memories from all scopes for that user.

**Writes and consolidation target the exact namespace** — ingesting into `year10-english/term1` never touches `year10-english/term2`. Omitting `namespace` writes to the global namespace (`namespace = null` in the DB).

## Ingestion modes

Set via `ingestion_mode` in `.env`:

- **`llm`** (default): sliding-window LLM extraction using OpenAI structured outputs. Infers memory type, importance, and confidence per unit. Requires `OPENAI_API_KEY`.
- **`pattern`**: per-turn regex/keyword extraction. No API key needed. Faster but less precise.

## Memory types

| Type                     | Description                                   |
| ------------------------ | --------------------------------------------- |
| `preference`             | Stable likes, dislikes, conventions           |
| `procedural_observation` | Rules and workflows ("always...", "never...") |
| `semantic`               | Durable facts and knowledge                   |
| `episodic`               | Specific events tied to a session             |

## Observability

marklymem traces memory operations to a self-hosted [Langfuse](https://langfuse.com) instance via OpenTelemetry (OTLP/HTTP). Tracing is off by default — set `OTEL_ENABLED=true` and provide the three Langfuse settings to enable it.

Each operation emits one trace:

- **`memory.ingest`** — extraction (`extract.session` → per-window `extract.window` generation spans with token usage) → `embedding` batch span with per-chunk `embedding.chunk` generation spans → `consolidate` span (output: content of every superseded memory).
- **`memory.retrieve`** — `embedding` span (model, token count) → output: retrieved memories with scores and content.

Traces carry `session_id`, `user_id`, and `namespace` so every operation can be correlated to its originating conversation in the Langfuse session view. Content capture (dialogue, memory text, retrieved hits) is always on when tracing is enabled — use a self-hosted Langfuse instance for data residency requirements.

## Environment

| Variable                 | Default                  | Description                                                   |
| ------------------------ | ------------------------ | ------------------------------------------------------------- |
| `DATABASE_URL`           | —                        | PostgreSQL async DSN                                          |
| `OPENAI_API_KEY`         | `""`                     | Required for `ingestion_mode=llm` or `embedder_mode=semantic` |
| `OPENAI_EMBEDDING_MODEL` | `text-embedding-3-small` | Embedding model                                               |
| `ingestion_mode`         | `llm`                    | `"llm"` or `"pattern"`                                        |
| `retrieval_mode`         | `hybrid`                 | `"keyword"`, `"embedding"`, `"hybrid"`, or `"auto"`           |
| `embedder_mode`          | `semantic`               | `"semantic"` (OpenAI) or `"hashing"` (no API key)             |
| `FASTAPI_HOST`           | `localhost`              | Bind address                                                  |
| `FASTAPI_PORT`           | `8100`                   | Bind port                                                     |
| `OTEL_ENABLED`           | `false`                  | Enable OTel tracing to Langfuse                               |
| `LANGFUSE_HOST`          | `""`                     | Langfuse base URL, e.g. `http://localhost:3000`               |
| `LANGFUSE_PUBLIC_KEY`    | `""`                     | Langfuse project public key                                   |
| `LANGFUSE_SECRET_KEY`    | `""`                     | Langfuse project secret key                                   |

## Database

Migrations are managed with Alembic:

```bash
uv run alembic upgrade head
```
