"""FastAPI app exposing the evolver memory engine as a REST API.

One shared ``MemoryStore`` + one process-wide ``MemoryManager`` are created in the
lifespan handler. Tenant isolation is via the ``user_id`` (required) and optional
``scope_id`` passed on every call — a single shared PostgreSQL DB with ``user_id`` and
``scope_id`` columns (the evolver's native model).
"""

from __future__ import annotations

import logging
import time
import uuid
from contextlib import asynccontextmanager

from fastapi import APIRouter, Depends, FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

from marklymem.evolver.db import build_engine, build_sessionmaker
from marklymem.evolver.embeddings import create_embedder
from marklymem.evolver.llm_extractor import create_llm_extractor
from marklymem.evolver.manager import MemoryManager
from marklymem.evolver.models import MemoryQuery
from marklymem.evolver.store import MemoryStore

from .config import get_settings
from .models import (
    AddDialogueRequest,
    AddResponse,
    ClearResponse,
    MemoryHit,
    RetrieveRequest,
    RetrieveResponse,
    ScopedRequest,
    StatsResponse,
)
from .utils import telemetry
from .utils.auth import verify_internal_api_key

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    tracing_enabled = telemetry.setup_telemetry(settings)
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

    if settings.APP_ENV != "local":
        if not settings.INTERNAL_API_KEY:
            raise RuntimeError("INTERNAL_API_KEY must be set when APP_ENV is not 'local'")
        if len(settings.INTERNAL_API_KEY) < 32:
            raise RuntimeError("INTERNAL_API_KEY must be at least 32 characters")

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
        f"[marklymem] ready — routes=/api/ db={settings.DATABASE_URL!r} "
        f"retrieval_mode={settings.retrieval_mode} embedder={settings.embedder_mode} "
        f"ingestion_mode={settings.ingestion_mode} auto_consolidate=True "
        f"tracing={'on' if tracing_enabled else 'off'}"
    )
    yield
    try:
        await engine.dispose()
    finally:
        await telemetry.shutdown_telemetry()


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


def _set_request_context(request: Request, req: ScopedRequest) -> None:
    request.state.user_id = req.user_id
    request.state.scope_id = req.scope_id


router = APIRouter(prefix="/api", dependencies=[Depends(verify_internal_api_key)])


@app.middleware("http")
async def log_requests(request: Request, call_next):
    if request.url.path == "/health":
        return await call_next(request)
    req_id = uuid.uuid4().hex[:8]
    request.state.req_id = req_id
    start = time.perf_counter()
    response = await call_next(request)
    duration_ms = (time.perf_counter() - start) * 1000
    user_id = getattr(request.state, "user_id", None)
    scope_id = getattr(request.state, "scope_id", None)
    log = logger.error if response.status_code >= 500 else logger.info
    log(
        "%s %s %d %.1fms req_id=%s user_id=%s scope_id=%s",
        request.method, request.url.path, response.status_code, duration_ms,
        req_id, user_id, scope_id,
    )
    return response


def _mgr(app: FastAPI) -> MemoryManager:
    return app.state.mgr


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@router.post("/memory/add_dialogue", response_model=AddResponse)
async def memory_add_dialogue(req: AddDialogueRequest, request: Request) -> AddResponse:
    """Ingest multiple dialogue turns into the caller's scope (max 50 turns)."""
    _set_request_context(request, req)
    result = await _mgr(app).ingest_session_turns(
        req.session_id,
        [t.model_dump() for t in req.turns],
        user_id=req.user_id,
        scope_id=req.scope_id,
    )
    return AddResponse(
        user_id=req.user_id,
        scope_id=req.scope_id,
        session_id=req.session_id,
        units_added=result["added"],
        units_consolidated=result["superseded"],
    )


@router.post("/memory/retrieve", response_model=RetrieveResponse)
async def memory_retrieve(req: RetrieveRequest, request: Request) -> RetrieveResponse:
    """Hybrid (semantic + lexical) search within the caller's scope."""
    _set_request_context(request, req)
    query = MemoryQuery(
        user_id=req.user_id,
        scope_id=req.scope_id,
        session_id=req.session_id,
        query_text=req.query,
        top_k=req.top_k,
    )
    hits = await _mgr(app).retriever.retrieve(query)
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


@router.post("/memory/clear", response_model=ClearResponse)
async def memory_clear(req: ScopedRequest, request: Request) -> ClearResponse:
    """Soft-clear the caller's scope (archive all non-pinned active memories)."""
    _set_request_context(request, req)
    result = await _mgr(app).archive_scope(req.user_id, req.scope_id)
    return ClearResponse(
        user_id=req.user_id,
        scope_id=req.scope_id,
        archived=int(result.get("archived", 0)),
        pinned_kept=int(result.get("pinned_kept", 0)),
        total_before=int(result.get("total_before", 0)),
    )


@router.post("/memory/stats", response_model=StatsResponse)
async def memory_stats(req: ScopedRequest, request: Request) -> StatsResponse:
    """Return memory counts for the caller's scope."""
    _set_request_context(request, req)
    stats = await _mgr(app).get_scope_stats(req.user_id, req.scope_id)
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


app.include_router(router)
