"""Two-stage conflict resolver: cosine recall → batched LLM verification.

Replaces the ingest-time Jaccard-based ``auto_resolve_conflicts`` with a pipeline
that is both more precise (LLM verdict, not token overlap) and more recall-friendly
(cosine similarity with no type gate).

Public surface
--------------
- :class:`ConflictRelationship` — ``CONTRADICTION`` / ``DUPLICATE`` / ``INDEPENDENT``
- :class:`PairVerdict` / :class:`BatchVerdict` — Pydantic structured-output schemas
- :class:`CandidatePair` — dataclass carrying the two units under consideration
- :class:`ResolverConfig` — tunable thresholds and batching parameters
- :class:`ConflictResolver` — the stateful resolver wired to a store + embedder
- :func:`create_conflict_resolver` — factory; returns ``None`` when no API key
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from enum import Enum

from pydantic import BaseModel

from marklymem.utils import telemetry

from .embeddings import BaseEmbedder, cosine_similarity
from .models import MemoryStatus, MemoryUnit, utc_now_iso
from .store import MemoryStore

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

class ConflictRelationship(str, Enum):
    CONTRADICTION = "CONTRADICTION"
    DUPLICATE = "DUPLICATE"
    INDEPENDENT = "INDEPENDENT"


class PairVerdict(BaseModel):
    relationship: ConflictRelationship


class BatchVerdict(BaseModel):
    verdicts: list[PairVerdict]


@dataclass
class CandidatePair:
    new_unit: MemoryUnit
    existing_unit: MemoryUnit
    score: float


VerifyBatchCall = Callable[[list[CandidatePair]], Awaitable[BatchVerdict | None]]


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class ResolverConfig:
    cosine_threshold: float = 0.65
    jaccard_threshold: float = 0.5
    model: str = "gpt-4.1-mini"
    max_output_tokens: int = 1024
    max_candidates_per_unit: int = 3
    max_verified_pairs: int = 50
    batch_size: int = 10
    max_parallel: int = 4
    max_retries: int = 3


# ---------------------------------------------------------------------------
# LLM instructions
# ---------------------------------------------------------------------------

VERIFY_INSTRUCTIONS = """\
You are a memory conflict detector. For each pair below, decide whether the two \
memory statements CONTRADICT each other, are DUPLICATE (same fact stated \
differently), or are INDEPENDENT (compatible — both can be true simultaneously).

Rules:
- CONTRADICTION: one statement directly negates or reverses the other \
(e.g. "never do X" vs "always do X", or weight 25% vs weight 40%).
- DUPLICATE: both statements express the same fact, even if worded differently.
- INDEPENDENT: the statements are compatible — both can be true at the same time.

Return exactly one verdict per pair in the same order as the pairs are listed. \
Do not skip or reorder any pair.
"""


# ---------------------------------------------------------------------------
# Resolver
# ---------------------------------------------------------------------------

class ConflictResolver:
    """Resolve conflicts between newly ingested memories and the active pool.

    Stage 1 — *recall*: ``_recall`` finds candidate pairs using cosine similarity
    (or Jaccard fallback when the embedder is absent).  Only pairs involving at
    least one unit from ``new_units`` are generated.

    Stage 2 — *verify*: pairs are chunked into batches of ``config.batch_size``
    and sent concurrently to the injected ``verify_batch_call`` under a semaphore,
    mirroring the windowed ``asyncio.gather`` pattern in ``llm_extractor.py``.
    Each batch has its own retry loop (up to ``config.max_retries`` attempts) that
    mirrors ``_extract_window``.

    Stage 3 — *act*: verdicts are applied sequentially; the older of each
    conflicting pair is superseded.
    """

    def __init__(
        self,
        store: MemoryStore,
        embedder: BaseEmbedder | None,
        verify_batch_call: VerifyBatchCall,
        config: ResolverConfig | None = None,
        notify: Callable | None = None,
    ):
        self.store = store
        self.embedder = embedder
        self.verify_batch_call = verify_batch_call
        self.config = config or ResolverConfig()
        self.notify = notify

    async def detect_conflicts(
        self,
        user_id: str,
        namespace: str | None,
        new_units: list[MemoryUnit],
    ) -> list[CandidatePair]:
        """Return candidate conflict pairs involving ``new_units`` against the active pool.

        Fetches the full active pool (including ``new_units`` themselves so
        new×new pairs within the same ingest are candidates), runs
        ``_similarity_conflict`` per new unit, and deduplicates A–B / B–A via
        a frozenset ``seen`` set.
        Old×old pairs are never generated — the pool is internally consistent
        by induction from prior ingests.
        """
        pool = await self.store.list_active(user_id, namespace, limit=500)
        if not pool or not new_units:
            return []

        seen: set[tuple[str, str]] = set()
        pairs: list[CandidatePair] = []

        for new_unit in new_units:
            for existing, score in self._similarity_conflict(new_unit, pool):
                pair_key = (
                    min(new_unit.memory_id, existing.memory_id),
                    max(new_unit.memory_id, existing.memory_id),
                )
                if pair_key in seen:
                    continue
                seen.add(pair_key)
                pairs.append(CandidatePair(new_unit=new_unit, existing_unit=existing, score=score))

        pairs.sort(key=lambda p: p.score, reverse=True)
        return pairs[:self.config.max_verified_pairs]

    async def resolve(
        self,
        user_id: str,
        namespace: str | None,
        new_units: list[MemoryUnit],
    ) -> dict:
        """Detect and resolve conflicts involving ``new_units`` in the active pool.

        Returns ``{"resolved", "candidates", "verified", "dropped": [...]}``.
        """
        pairs = await self.detect_conflicts(user_id, namespace, new_units)

        if not pairs:
            return {"resolved": 0, "candidates": 0, "verified": 0, "dropped": []}

        cfg = self.config
        chunks = [pairs[i:i + cfg.batch_size] for i in range(0, len(pairs), cfg.batch_size)]
        sem = asyncio.Semaphore(max(1, cfg.max_parallel))

        raw = await asyncio.gather(
            *(self._verify_batch(chunk, sem) for chunk in chunks),
            return_exceptions=True,
        )

        verdicts: list[tuple[CandidatePair, PairVerdict]] = []
        for chunk, result in zip(chunks, raw):
            if isinstance(result, BaseException):
                logger.warning("[Resolver] batch skipped after retries: %s", result)
                continue
            if result is None:
                continue
            for pair, pv in zip(chunk, result.verdicts):
                verdicts.append((pair, pv))

        now = utc_now_iso()
        resolved = 0
        dropped: list[dict] = []

        for pair, pv in verdicts:
            if pv.relationship == ConflictRelationship.INDEPENDENT:
                continue
            a = await self.store.get_by_id(pair.new_unit.memory_id)
            b = await self.store.get_by_id(pair.existing_unit.memory_id)
            if a is None or b is None:
                continue
            if a.status != MemoryStatus.ACTIVE or b.status != MemoryStatus.ACTIVE:
                continue
            if a.importance >= 0.99 or b.importance >= 0.99:
                continue
            drop, keep = (a, b) if a.created_at <= b.created_at else (b, a)
            await self.store.supersede(drop.memory_id, keep.memory_id, now)
            dropped.append({
                "dropped_id": drop.memory_id,
                "kept_id": keep.memory_id,
                "relationship": pv.relationship.value,
                "dropped_content": drop.content,
                "kept_content": keep.content,
            })
            resolved += 1

        if self.notify and dropped:
            self.notify("conflict_resolution", resolved=resolved, dropped=dropped)

        return {
            "resolved": resolved,
            "candidates": len(pairs),
            "verified": len(verdicts),
            "dropped": dropped,
        }

    def _similarity_conflict(
        self,
        new_unit: MemoryUnit,
        pool: list[MemoryUnit],
    ) -> list[tuple[MemoryUnit, float]]:
        """Return candidate (unit, score) pairs for ``new_unit`` from ``pool``.

        Returns candidate (unit, score) pairs for ``new_unit`` from ``pool``.

        Uses cosine similarity when embeddings are available; falls back to
        Jaccard over topics+entities with a same-type gate when they are not.
        Results are sorted by score descending and capped at
        ``config.max_candidates_per_unit``.
        """
        cfg = self.config
        results: list[tuple[MemoryUnit, float]] = []

        a_terms = {t.lower() for t in new_unit.topics + new_unit.entities}
        use_cosine = self.embedder is not None and bool(new_unit.embedding)

        for unit in pool:
            if unit.memory_id == new_unit.memory_id:
                continue
            if use_cosine and unit.embedding:
                sim = cosine_similarity(new_unit.embedding, unit.embedding)
                if sim >= cfg.cosine_threshold:
                    results.append((unit, sim))
            else:
                # Jaccard fallback: per-unit when embedding unavailable, with type gate
                if unit.memory_type != new_unit.memory_type:
                    continue
                b_terms = {t.lower() for t in unit.topics + unit.entities}
                if not a_terms or not b_terms:
                    continue
                jac = len(a_terms & b_terms) / float(len(a_terms | b_terms))
                if jac >= cfg.jaccard_threshold:
                    results.append((unit, jac))

        results.sort(key=lambda x: x[1], reverse=True)
        return results[:cfg.max_candidates_per_unit]

    async def _verify_batch(
        self,
        group: list[CandidatePair],
        sem: asyncio.Semaphore,
    ) -> BatchVerdict | None:
        """Verify one batch of candidate pairs under ``sem``, with retry.

        Mirrors ``_extract_window`` in ``llm_extractor.py``: up to
        ``config.max_retries`` attempts with exponential back-off; re-raises on
        the final attempt so ``asyncio.gather(return_exceptions=True)`` can
        surface it as a skippable exception.
        """
        for attempt in range(self.config.max_retries):
            try:
                async with sem:
                    return await self.verify_batch_call(group)
            except Exception as exc:  # noqa: BLE001
                logger.debug(
                    "[Resolver] batch attempt %d/%d failed: %s",
                    attempt + 1, self.config.max_retries, exc,
                )
                if attempt == self.config.max_retries - 1:
                    raise
                await asyncio.sleep(2 * (attempt + 1))
        return None  # unreachable; satisfies type checker



# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def create_conflict_resolver(
    settings,
    store: MemoryStore,
    embedder: BaseEmbedder | None = None,
) -> ConflictResolver | None:
    """Build a :class:`ConflictResolver` backed by AsyncOpenAI.

    Returns ``None`` when ``OPENAI_API_KEY`` is absent — the manager then skips
    conflict resolution entirely.
    """
    if not getattr(settings, "OPENAI_API_KEY", ""):
        return None

    try:
        from openai import AsyncOpenAI
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("openai package not installed") from exc

    client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)
    config = ResolverConfig()

    async def _verify_batch_call(pairs: list[CandidatePair]) -> BatchVerdict | None:
        rendered = "\n\n".join(
            f"Pair {i}: A={p.new_unit.content}\n"
            f"         B={p.existing_unit.content}"
            for i, p in enumerate(pairs)
        )
        with telemetry.generation("resolve.verify_batch", model=config.model) as gen:
            telemetry.set_input(gen, rendered)
            response = await client.responses.parse(
                model=config.model,
                instructions=VERIFY_INSTRUCTIONS,
                input=rendered,
                text_format=BatchVerdict,
                temperature=0,
                max_output_tokens=config.max_output_tokens,
            )
            usage = getattr(response, "usage", None)
            telemetry.record_usage(
                gen,
                input_tokens=getattr(usage, "input_tokens", None),
                output_tokens=getattr(usage, "output_tokens", None),
            )
            parsed = response.output_parsed
            if parsed is None:
                raise ValueError("output_parsed is None — LLM returned unparseable output")
            if len(parsed.verdicts) != len(pairs):
                raise ValueError(
                    f"verdict count mismatch: expected {len(pairs)}, got {len(parsed.verdicts)}"
                )
            telemetry.set_output(gen, parsed.model_dump())
            return parsed

    return ConflictResolver(store, embedder, _verify_batch_call, config)
