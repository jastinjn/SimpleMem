# SimpleMem — CLAUDE.md

SimpleMem (v0.3.0, Apache 2.0) is a lifelong memory system for LLM agents. Dialogue is compressed into atomic facts and retrieved via a three-stage hybrid pipeline (semantic + lexical + symbolic).

## Repo Layout

```
simplemem/       # Installable package
  core/          # MemoryBuilder, HybridRetriever, AnswerGenerator (text runtime)
  text/          # SimpleMemSystem — wires core into the text backend
  evolver/       # MemoryManager runtime + offline EvolutionEngine
  evolver/tests/ # Deterministic unit + integration tests (no API key required)
  multimodal/    # OmniSimpleMem: image/audio/video memory
cross/           # SimpleMem-Cross: persistent cross-conversation memory
MCP/             # Production MCP + HTTP server (FastAPI, port 8000)
EvolveMem/       # Upstream research subproject — mirrored into simplemem/evolver/
OmniSimpleMem/   # Upstream research subproject — mirrored into simplemem/multimodal/
```

## Setup

```bash
cp config.py.example config.py   # set OPENAI_API_KEY; never commit config.py

# Install uv once (https://docs.astral.sh/uv/getting-started/installation/)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install the project — uv creates .venv automatically, no activate needed
uv sync --extra server --extra dev   # core + MCP server + test deps

# Run anything in the managed env
uv run python MCP/run.py             # MCP server (localhost:8000)
uv run python test_locomo10.py       # LoCoMo-10 benchmark (needs API key)

# Add / upgrade a dependency
uv add some-package
uv add --optional server some-server-dep
```

Settings resolve: `config.py` → env var → built-in default. Key vars: `OPENAI_API_KEY`, `OPENAI_BASE_URL`, `LLM_MODEL` (default `gpt-4.1-mini`), `EMBEDDING_MODEL` (default `Qwen/Qwen3-Embedding-0.6B`, local), `LANCEDB_PATH` (default `./lancedb_data`). For the MCP server copy `.env.example` to `.env`.

The evolver layer does not require an API key — it defaults to `HashingEmbedder` (SHA-256, deterministic, no network).

## Tests & Running

```bash
uv run pytest tests/                        # vector store unit tests (no API key)
uv run pytest cross/tests/                  # SimpleMem-Cross tests (no API key, real SQLite)
uv run pytest simplemem/evolver/tests/ -v   # evolver unit + integration tests (no API key)

uv run python MCP/run.py                    # MCP server (localhost:8000)
uv run python test_locomo10.py              # LoCoMo-10 benchmark (needs API key)
cd EvolveMem && python run_evolution.py ... # paper-faithful evolution (own requirements.txt)
```

## Architecture

**Layer 1 — Text runtime** (`simplemem/text/`, `simplemem/core/`): runs every conversation. `SimpleMemSystem` wires `MemoryBuilder` (extraction + consolidation) → `HybridRetriever` (3-view retrieval) → `AnswerGenerator`. Backed by LanceDB. Public entry point: `simplemem.SimpleMem`.

**Layer 2 — Evolver runtime** (`simplemem/evolver/`): a parallel SQLite-backed memory system (`~/.simplemem/store.jsonl`). `MemoryManager` wraps `MemoryStore` (SQLite + FTS5), `MemoryRetriever`, `MemoryConsolidator`, and `MemoryPolicy`. Independent of Layer 1's LanceDB store.

**Layer 3 — Offline evolution** (`simplemem/evolver/evolution.py`): `EvolutionEngine` runs a closed-loop `Extract → Evaluate → Diagnose → Adjust` cycle to tune a `RetrievalConfig`. Only ever invoked via `simplemem.optimize(mem, dev_questions)` or `EvolveMem/run_evolution.py` — never at inference time.

**SimpleMem-Cross** (`cross/`): wraps `simplemem` without modifying it. Adds session lifecycle (`start → record → stop → end`), SQLite session storage, cross-session LanceDB vector search, a FastAPI router, and MCP tools. Entry point: `CrossMemOrchestrator`. Data at `~/.simplemem-cross/`.

## Key Constraints

- `simplemem/core/` has no knowledge of the evolver. Dependency is one-directional: evolver → core.
- `simplemem.optimize()` lazy-imports the evolver; it is not loaded at package import time.
- The MCP server's SQLite (`MCP/server/database/user_store.py`) is for auth only — unrelated to `MemoryStore`.
- `simplemem/multimodal/evolution/` (MetaController, StrategyOptimizer) is a separate online evolution system from OmniSimpleMem, unrelated to `EvolutionEngine`.
- Do not edit `EvolveMem/` or `OmniSimpleMem/` directly — they are upstream mirrors.
- Packaging: `pyproject.toml` (root) is the sole packaging source; `setup.py` has been removed. Bump `version` in both `pyproject.toml` and `simplemem/__init__.py` on release.
