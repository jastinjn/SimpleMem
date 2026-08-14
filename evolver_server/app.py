"""FastAPI app exposing the evolver memory engine as a REST API.

One shared ``MemoryStore`` + one process-wide ``MemoryManager`` are created in the
lifespan handler. Tenant isolation is via the ``user_id`` (required) and optional
``scope_id`` passed on every call — a single shared PostgreSQL DB with ``user_id`` and
``scope_id`` columns (the evolver's native model).
"""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from evolver_server.evolver.db import build_engine, build_sessionmaker
from evolver_server.evolver.embeddings import create_embedder
from evolver_server.evolver.llm_extractor import create_llm_extractor
from evolver_server.evolver.manager import MemoryManager
from evolver_server.evolver.models import MemoryQuery
from evolver_server.evolver.store import MemoryStore

from .config import get_settings
from .models import (
    AddBatchRequest,
    AddRequest,
    AddResponse,
    ClearRequest,
    ClearResponse,
    MemoryHit,
    RetrieveRequest,
    RetrieveResponse,
    StatsRequest,
    StatsResponse,
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    engine = build_engine(
        settings.DATABASE_URL,
        pool_size=settings.db_pool_size,
        max_overflow=settings.db_max_overflow,
    )
    sm = build_sessionmaker(engine)
    if settings.embedder_mode == "semantic" and not settings.OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY must be set in .env when embedder_mode=semantic")

    store = MemoryStore(sm)
    embedder = create_embedder(mode=settings.embedder_mode, dimensions=settings.embedding_dim)

    llm_extractor = None
    if settings.ingestion_mode == "llm":
        llm_extractor = create_llm_extractor(settings)
        if llm_extractor is None:
            raise RuntimeError(
                "ingestion_mode=llm requires OPENAI_API_KEY to be set in .env"
            )

    mgr = MemoryManager(
        store=store,
        retrieval_mode=settings.retrieval_mode,
        auto_consolidate=True,
        embedder=embedder,
        ingestion_mode=settings.ingestion_mode,
        llm_extractor=llm_extractor,
    )
    app.state.store = store
    app.state.mgr = mgr
    print(
        f"[EvolverAPI] ready — db={settings.DATABASE_URL!r} "
        f"retrieval_mode={settings.retrieval_mode} embedder={settings.embedder_mode} "
        f"ingestion_mode={settings.ingestion_mode} auto_consolidate=True"
    )
    yield
    await engine.dispose()


settings = get_settings()

app = FastAPI(
    title="SimpleMem Evolver API",
    description="Standalone REST API over the evolver hybrid memory engine. No auth; "
    "every endpoint requires a user_id.",
    version="0.1.0",
    lifespan=lifespan,
)

# CORS: never combine wildcard origin with credentials (invalid per the spec).
if settings.cors_allowed_origins.strip() == "*":
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )
else:
    origins = [o.strip() for o in settings.cors_allowed_origins.split(",") if o.strip()]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )


def _mgr(app: FastAPI) -> MemoryManager:
    return app.state.mgr


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/memory/add", response_model=AddResponse)
async def memory_add(req: AddRequest) -> AddResponse:
    """Ingest a single dialogue turn into the caller's scope.

    Consolidation runs automatically afterwards (auto_consolidate=True).
    """
    try:
        added = await _mgr(app).ingest_session_turns(
            req.session_id,
            [{"prompt_text": req.prompt_text, "response_text": req.response_text}],
            user_id=req.user_id,
            scope_id=req.scope_id,
        )
    except Exception as e:  # noqa: BLE001 - surface as 500 for the caller
        raise HTTPException(status_code=500, detail=str(e)) from e
    return AddResponse(
        user_id=req.user_id,
        scope_id=req.scope_id,
        session_id=req.session_id,
        units_added=added,
    )


@app.post("/memory/add_batch", response_model=AddResponse)
async def memory_add_batch(req: AddBatchRequest) -> AddResponse:
    """Ingest multiple dialogue turns into the caller's scope (windowed extraction)."""
    try:
        added = await _mgr(app).ingest_session_turns(
            req.session_id,
            [t.model_dump() for t in req.turns],
            user_id=req.user_id,
            scope_id=req.scope_id,
        )
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=str(e)) from e
    return AddResponse(
        user_id=req.user_id,
        scope_id=req.scope_id,
        session_id=req.session_id,
        units_added=added,
    )


@app.post("/memory/retrieve", response_model=RetrieveResponse)
async def memory_retrieve(req: RetrieveRequest) -> RetrieveResponse:
    """Hybrid (semantic + lexical) search within the caller's scope."""
    try:
        query = MemoryQuery(
            user_id=req.user_id,
            scope_id=req.scope_id,
            query_text=req.query,
            top_k=req.top_k,
        )
        hits = await _mgr(app).retriever.retrieve(query)
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=str(e)) from e

    results = [
        MemoryHit(
            memory_id=h.unit.memory_id,
            content=h.unit.content,
            memory_type=h.unit.memory_type.value,
            importance=h.unit.importance,
            score=round(float(h.score), 4),
            matched_terms=h.matched_terms,
            entities=h.unit.entities,
            topics=h.unit.topics,
            updated_at=h.unit.updated_at,
        )
        for h in hits
    ]
    return RetrieveResponse(
        user_id=req.user_id,
        scope_id=req.scope_id,
        query=req.query,
        results=results,
        total=len(results),
    )


@app.post("/memory/clear", response_model=ClearResponse)
async def memory_clear(req: ClearRequest) -> ClearResponse:
    """Soft-clear the caller's scope (archive all non-pinned active memories)."""
    try:
        result = await _mgr(app).archive_scope(req.user_id, req.scope_id)
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=str(e)) from e
    return ClearResponse(
        user_id=req.user_id,
        scope_id=req.scope_id,
        archived=int(result.get("archived", 0)),
        pinned_kept=int(result.get("pinned_kept", 0)),
        total_before=int(result.get("total_before", 0)),
    )


@app.post("/memory/stats", response_model=StatsResponse)
async def memory_stats(req: StatsRequest) -> StatsResponse:
    """Return memory counts for the caller's scope."""
    try:
        stats = await _mgr(app).get_scope_stats(req.user_id, req.scope_id)
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=str(e)) from e
    return StatsResponse(
        user_id=req.user_id,
        scope_id=req.scope_id,
        entry_count=int(stats.get("active", 0)),
        total=int(stats.get("total", 0)),
        superseded=int(stats.get("superseded", 0)),
        active_by_type=stats.get("active_by_type", {}),
        type_count=int(stats.get("type_count", 0)),
        dominant_type=stats.get("dominant_type", ""),
    )
