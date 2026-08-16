# pyright: reportMissingImports=false
"""Integration tests for ConflictResolver.

All tests use real Postgres via the `store` fixture, HashingEmbedder for
deterministic embeddings, and injected fake verify callables so no API key
is required.

Embedding boundary tests use hand-set unit vectors (dimension 1024) so
cosine similarity is exact and independent of the hashing scheme.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from marklymem.evolver.embeddings import HashingEmbedder
from marklymem.evolver.models import MemoryStatus, MemoryType
from marklymem.evolver.resolver import (
    BatchVerdict,
    CandidatePair,
    ConflictRelationship,
    ConflictResolver,
    PairVerdict,
    ResolverConfig,
    create_conflict_resolver,
)

from .conftest import UID, _make_unit

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

DIM = 1024


def _unit_vec(*positions: int) -> list[float]:
    """Unit vector with 1.0 at each given position, 0.0 elsewhere (L2-normalised)."""
    v = [0.0] * DIM
    for p in positions:
        v[p] = 1.0
    norm = sum(x * x for x in v) ** 0.5
    return [x / norm for x in v]


def make_fake_verify(rules: dict[str, ConflictRelationship]):
    """Return an async batch callable that assigns relationships by substring match.

    For each pair the first matching rule key found in either unit's content
    wins; defaults to INDEPENDENT.
    """
    async def _fake(pairs: list[CandidatePair]) -> BatchVerdict:
        verdicts = []
        for pair in pairs:
            rel = ConflictRelationship.INDEPENDENT
            combined = pair.new_unit.content + " " + pair.existing_unit.content
            for keyword, relationship in rules.items():
                if keyword.lower() in combined.lower():
                    rel = relationship
                    break
            verdicts.append(PairVerdict(relationship=rel))
        return BatchVerdict(verdicts=verdicts)
    return _fake


def _resolver(
    store,
    verify_call=None,
    embedder=None,
    config: ResolverConfig | None = None,
) -> ConflictResolver:
    if verify_call is None:
        verify_call = make_fake_verify({})
    return ConflictResolver(store, embedder, verify_call, config=config)


# ---------------------------------------------------------------------------
# _similarity_conflict — cosine path
# ---------------------------------------------------------------------------

class TestSimilarityConflictCosine:
    async def test_generates_candidates_above_threshold(self, store):
        embedder = HashingEmbedder(dimensions=DIM)
        new_unit = _make_unit(memory_id="n-001", namespace="test")
        existing = _make_unit(memory_id="e-001", namespace="test")
        new_unit.embedding = _unit_vec(0)
        existing.embedding = _unit_vec(0)  # cosine = 1.0
        await store.add_memories([existing])

        r = _resolver(store, embedder=embedder)
        candidates = r._similarity_conflict(new_unit, [existing])
        assert len(candidates) == 1
        assert candidates[0][0].memory_id == "e-001"

    async def test_excludes_dissimilar_pairs(self, store):
        embedder = HashingEmbedder(dimensions=DIM)
        new_unit = _make_unit(memory_id="n-001", namespace="test")
        existing = _make_unit(memory_id="e-001", namespace="test")
        new_unit.embedding = _unit_vec(0)
        existing.embedding = _unit_vec(1)  # cosine = 0.0
        await store.add_memories([existing])

        r = _resolver(store, embedder=embedder)
        candidates = r._similarity_conflict(new_unit, [existing])
        assert candidates == []

    async def test_excludes_self_pair(self, store):
        embedder = HashingEmbedder(dimensions=DIM)
        unit = _make_unit(memory_id="u-001", namespace="test")
        unit.embedding = _unit_vec(0)
        await store.add_memories([unit])

        r = _resolver(store, embedder=embedder)
        candidates = r._similarity_conflict(unit, [unit])
        assert candidates == []


# ---------------------------------------------------------------------------
# _similarity_conflict — Jaccard fallback
# ---------------------------------------------------------------------------

class TestSimilarityConflictJaccard:
    async def test_fallback_when_no_embedder(self, store):
        new_unit = _make_unit(
            memory_id="n-001", namespace="test",
            memory_type=MemoryType.PREFERENCE,
            topics=["marking", "feedback"], entities=[],
        )
        existing = _make_unit(
            memory_id="e-001", namespace="test",
            memory_type=MemoryType.PREFERENCE,
            topics=["marking", "feedback"], entities=[],
        )
        await store.add_memories([existing])

        r = _resolver(store, embedder=None)
        candidates = r._similarity_conflict(new_unit, [existing])
        assert len(candidates) == 1
        assert candidates[0][1] == pytest.approx(1.0)

    async def test_different_type_not_recalled(self, store):
        new_unit = _make_unit(
            memory_id="n-001", namespace="test",
            memory_type=MemoryType.PREFERENCE,
            topics=["marking", "feedback"], entities=[],
        )
        existing = _make_unit(
            memory_id="e-001", namespace="test",
            memory_type=MemoryType.SEMANTIC,
            topics=["marking", "feedback"], entities=[],
        )
        await store.add_memories([existing])

        r = _resolver(store, embedder=None)
        candidates = r._similarity_conflict(new_unit, [existing])
        assert candidates == []

    async def test_below_threshold_excluded(self, store):
        new_unit = _make_unit(
            memory_id="n-001", namespace="test",
            memory_type=MemoryType.PREFERENCE,
            topics=["marking"], entities=[],
        )
        existing = _make_unit(
            memory_id="e-001", namespace="test",
            memory_type=MemoryType.PREFERENCE,
            topics=["feedback", "rubric", "essay"], entities=[],
        )
        await store.add_memories([existing])

        r = _resolver(store, embedder=None)
        candidates = r._similarity_conflict(new_unit, [existing])
        assert candidates == []


# ---------------------------------------------------------------------------
# detect_conflicts — pair generation
# ---------------------------------------------------------------------------

class TestDetectConflicts:
    async def test_fanout_capped_at_max_candidates_per_unit(self, store):
        embedder = HashingEmbedder(dimensions=DIM)
        cfg = ResolverConfig(max_candidates_per_unit=3, max_verified_pairs=50)
        new_unit = _make_unit(memory_id="n-001", namespace="test")
        new_unit.embedding = _unit_vec(0)

        pool = []
        for i in range(6):
            u = _make_unit(memory_id=f"e-{i:03d}", namespace="test")
            u.embedding = _unit_vec(0)
            pool.append(u)
        await store.add_memories(pool + [new_unit])

        r = _resolver(store, embedder=embedder, config=cfg)
        pairs = await r.detect_conflicts(UID, "test", [new_unit])
        assert len(pairs) <= cfg.max_candidates_per_unit

    async def test_max_verified_pairs_cap(self, store):
        embedder = HashingEmbedder(dimensions=DIM)
        cfg = ResolverConfig(max_candidates_per_unit=3, max_verified_pairs=2)

        new_units = []
        for i in range(3):
            nu = _make_unit(memory_id=f"n-{i:03d}", namespace="test")
            nu.embedding = _unit_vec(0)
            new_units.append(nu)
        pool = []
        for i in range(4):
            u = _make_unit(memory_id=f"e-{i:03d}", namespace="test")
            u.embedding = _unit_vec(0)
            pool.append(u)
        await store.add_memories(pool + new_units)

        r = _resolver(store, embedder=embedder, config=cfg)
        pairs = await r.detect_conflicts(UID, "test", new_units)
        assert len(pairs) <= cfg.max_verified_pairs

    async def test_new_vs_new_pair_generated(self, store):
        embedder = HashingEmbedder(dimensions=DIM)
        unit_a = _make_unit(memory_id="n-001", namespace="test",
                            created_at="2025-01-01T00:00:00+00:00",
                            updated_at="2025-01-01T00:00:00+00:00")
        unit_b = _make_unit(memory_id="n-002", namespace="test",
                            created_at="2025-01-02T00:00:00+00:00",
                            updated_at="2025-01-02T00:00:00+00:00")
        unit_a.embedding = _unit_vec(0)
        unit_b.embedding = _unit_vec(0)
        await store.add_memories([unit_a, unit_b])

        r = _resolver(store, embedder=embedder)
        pairs = await r.detect_conflicts(UID, "test", [unit_a, unit_b])
        pair_ids = {frozenset({p.new_unit.memory_id, p.existing_unit.memory_id}) for p in pairs}
        assert frozenset({"n-001", "n-002"}) in pair_ids

    async def test_no_old_vs_old_pairs(self, store):
        embedder = HashingEmbedder(dimensions=DIM)
        old_a = _make_unit(memory_id="o-001", namespace="test")
        old_b = _make_unit(memory_id="o-002", namespace="test")
        old_a.embedding = _unit_vec(0)
        old_b.embedding = _unit_vec(0)
        new_unit = _make_unit(memory_id="n-001", namespace="test")
        new_unit.embedding = _unit_vec(1)  # dissimilar — won't pair with old units
        await store.add_memories([old_a, old_b, new_unit])

        r = _resolver(store, embedder=embedder)
        pairs = await r.detect_conflicts(UID, "test", [new_unit])
        pair_ids = {frozenset({p.new_unit.memory_id, p.existing_unit.memory_id}) for p in pairs}
        assert frozenset({"o-001", "o-002"}) not in pair_ids


# ---------------------------------------------------------------------------
# resolve — act step
# ---------------------------------------------------------------------------

class TestResolve:
    async def test_contradiction_supersedes_existing(self, store):
        existing = _make_unit(
            memory_id="e-001", namespace="test",
            content="always include mark in feedback",
            created_at="2025-01-01T00:00:00+00:00",
            updated_at="2025-01-01T00:00:00+00:00",
        )
        new_unit = _make_unit(
            memory_id="n-001", namespace="test",
            content="never include mark in feedback",
            created_at="2025-02-01T00:00:00+00:00",
            updated_at="2025-02-01T00:00:00+00:00",
        )
        existing.embedding = _unit_vec(0)
        new_unit.embedding = _unit_vec(0)
        await store.add_memories([existing, new_unit])

        embedder = HashingEmbedder(dimensions=DIM)
        r = _resolver(store, embedder=embedder,
                      verify_call=make_fake_verify({"mark": ConflictRelationship.CONTRADICTION}))
        result = await r.resolve(UID, "test", [new_unit])

        assert result["resolved"] == 1
        fetched = (await store.get_by_ids(["e-001"]))[0]
        assert fetched.status == MemoryStatus.SUPERSEDED
        assert fetched.superseded_by == "n-001"

    async def test_independent_keeps_both(self, store):
        existing = _make_unit(memory_id="e-001", namespace="test", content="use blue pen")
        new_unit = _make_unit(memory_id="n-001", namespace="test", content="use blue pen")
        existing.embedding = _unit_vec(0)
        new_unit.embedding = _unit_vec(0)
        await store.add_memories([existing, new_unit])

        embedder = HashingEmbedder(dimensions=DIM)
        r = _resolver(store, embedder=embedder, verify_call=make_fake_verify({}))
        result = await r.resolve(UID, "test", [new_unit])

        assert result["resolved"] == 0
        active_ids = [u.memory_id for u in await store.list_active(UID, "test")]
        assert "e-001" in active_ids and "n-001" in active_ids

    async def test_duplicate_supersedes_older(self, store):
        existing = _make_unit(
            memory_id="e-001", namespace="test", content="mark goes at top",
            created_at="2025-01-01T00:00:00+00:00",
            updated_at="2025-01-01T00:00:00+00:00",
        )
        new_unit = _make_unit(
            memory_id="n-001", namespace="test", content="mark goes at top",
            created_at="2025-02-01T00:00:00+00:00",
            updated_at="2025-02-01T00:00:00+00:00",
        )
        existing.embedding = _unit_vec(0)
        new_unit.embedding = _unit_vec(0)
        await store.add_memories([existing, new_unit])

        embedder = HashingEmbedder(dimensions=DIM)
        r = _resolver(store, embedder=embedder,
                      verify_call=make_fake_verify({"mark": ConflictRelationship.DUPLICATE}))
        result = await r.resolve(UID, "test", [new_unit])

        assert result["resolved"] == 1
        assert (await store.get_by_ids(["e-001"]))[0].status == MemoryStatus.SUPERSEDED

    async def test_cross_type_contradiction_caught(self, store):
        existing = _make_unit(
            memory_id="e-001", namespace="test",
            memory_type=MemoryType.PROCEDURAL_OBSERVATION,
            content="always write mark at top",
            created_at="2025-01-01T00:00:00+00:00",
            updated_at="2025-01-01T00:00:00+00:00",
        )
        new_unit = _make_unit(
            memory_id="n-001", namespace="test",
            memory_type=MemoryType.PREFERENCE,
            content="never write mark at top",
            created_at="2025-02-01T00:00:00+00:00",
            updated_at="2025-02-01T00:00:00+00:00",
        )
        existing.embedding = _unit_vec(0)
        new_unit.embedding = _unit_vec(0)
        await store.add_memories([existing, new_unit])

        embedder = HashingEmbedder(dimensions=DIM)
        r = _resolver(store, embedder=embedder,
                      verify_call=make_fake_verify({"mark": ConflictRelationship.CONTRADICTION}))
        result = await r.resolve(UID, "test", [new_unit])

        assert result["resolved"] == 1
        assert (await store.get_by_ids(["e-001"]))[0].status == MemoryStatus.SUPERSEDED

    async def test_pinned_unit_not_superseded(self, store):
        existing = _make_unit(
            memory_id="e-001", namespace="test", content="always include mark",
            importance=0.99,
            created_at="2025-01-01T00:00:00+00:00",
            updated_at="2025-01-01T00:00:00+00:00",
        )
        new_unit = _make_unit(
            memory_id="n-001", namespace="test", content="never include mark",
            created_at="2025-02-01T00:00:00+00:00",
            updated_at="2025-02-01T00:00:00+00:00",
        )
        existing.embedding = _unit_vec(0)
        new_unit.embedding = _unit_vec(0)
        await store.add_memories([existing, new_unit])

        embedder = HashingEmbedder(dimensions=DIM)
        r = _resolver(store, embedder=embedder,
                      verify_call=make_fake_verify({"mark": ConflictRelationship.CONTRADICTION}))
        result = await r.resolve(UID, "test", [new_unit])

        assert result["resolved"] == 0
        assert (await store.get_by_ids(["e-001"]))[0].status == MemoryStatus.ACTIVE

    async def test_new_vs_new_earlier_superseded(self, store):
        unit_a = _make_unit(
            memory_id="n-001", namespace="test", content="always include mark",
            created_at="2025-01-01T00:00:00+00:00",
            updated_at="2025-01-01T00:00:00+00:00",
        )
        unit_b = _make_unit(
            memory_id="n-002", namespace="test", content="never include mark",
            created_at="2025-02-01T00:00:00+00:00",
            updated_at="2025-02-01T00:00:00+00:00",
        )
        unit_a.embedding = _unit_vec(0)
        unit_b.embedding = _unit_vec(0)
        await store.add_memories([unit_a, unit_b])

        embedder = HashingEmbedder(dimensions=DIM)
        r = _resolver(store, embedder=embedder,
                      verify_call=make_fake_verify({"mark": ConflictRelationship.CONTRADICTION}))
        result = await r.resolve(UID, "test", [unit_a, unit_b])

        assert result["resolved"] == 1
        assert (await store.get_by_ids(["n-001"]))[0].status == MemoryStatus.SUPERSEDED
        assert (await store.get_by_ids(["n-002"]))[0].status == MemoryStatus.ACTIVE


# ---------------------------------------------------------------------------
# Batching and retry
# ---------------------------------------------------------------------------

class TestBatching:
    async def test_verify_called_in_batches(self, store):
        embedder = HashingEmbedder(dimensions=DIM)
        cfg = ResolverConfig(max_candidates_per_unit=3, max_verified_pairs=50, batch_size=2)

        new_unit = _make_unit(memory_id="n-001", namespace="test",
                              created_at="2025-02-01T00:00:00+00:00",
                              updated_at="2025-02-01T00:00:00+00:00")
        new_unit.embedding = _unit_vec(0)

        pool = []
        for i in range(3):
            u = _make_unit(memory_id=f"e-{i:03d}", namespace="test",
                           created_at=f"2025-01-{i+1:02d}T00:00:00+00:00",
                           updated_at=f"2025-01-{i+1:02d}T00:00:00+00:00")
            u.embedding = _unit_vec(0)
            pool.append(u)
        await store.add_memories(pool + [new_unit])

        batch_sizes: list[int] = []

        async def spy(pairs: list[CandidatePair]) -> BatchVerdict:
            batch_sizes.append(len(pairs))
            return BatchVerdict(verdicts=[
                PairVerdict(relationship=ConflictRelationship.INDEPENDENT)
                for _ in pairs
            ])

        r = _resolver(store, embedder=embedder, verify_call=spy, config=cfg)
        await r.resolve(UID, "test", [new_unit])

        assert all(s <= cfg.batch_size for s in batch_sizes)
        assert sum(batch_sizes) == 3

    async def test_batch_retries_then_succeeds(self, store, monkeypatch):
        embedder = HashingEmbedder(dimensions=DIM)
        cfg = ResolverConfig(max_retries=3)

        existing = _make_unit(
            memory_id="e-001", namespace="test", content="always include mark",
            created_at="2025-01-01T00:00:00+00:00",
            updated_at="2025-01-01T00:00:00+00:00",
        )
        new_unit = _make_unit(
            memory_id="n-001", namespace="test", content="never include mark",
            created_at="2025-02-01T00:00:00+00:00",
            updated_at="2025-02-01T00:00:00+00:00",
        )
        existing.embedding = _unit_vec(0)
        new_unit.embedding = _unit_vec(0)
        await store.add_memories([existing, new_unit])

        call_count = 0

        async def flaky(pairs: list[CandidatePair]) -> BatchVerdict:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise RuntimeError("transient error")
            return BatchVerdict(verdicts=[
                PairVerdict(relationship=ConflictRelationship.CONTRADICTION)
                for _ in pairs
            ])

        monkeypatch.setattr("marklymem.evolver.resolver.asyncio.sleep", AsyncMock())
        r = _resolver(store, embedder=embedder, verify_call=flaky, config=cfg)
        result = await r.resolve(UID, "test", [new_unit])

        assert call_count == 2
        assert result["resolved"] == 1

    async def test_exhausted_retries_skips_batch(self, store, monkeypatch):
        embedder = HashingEmbedder(dimensions=DIM)
        cfg = ResolverConfig(max_retries=3)

        existing = _make_unit(
            memory_id="e-001", namespace="test", content="always include mark",
            created_at="2025-01-01T00:00:00+00:00",
            updated_at="2025-01-01T00:00:00+00:00",
        )
        new_unit = _make_unit(
            memory_id="n-001", namespace="test", content="never include mark",
            created_at="2025-02-01T00:00:00+00:00",
            updated_at="2025-02-01T00:00:00+00:00",
        )
        existing.embedding = _unit_vec(0)
        new_unit.embedding = _unit_vec(0)
        await store.add_memories([existing, new_unit])

        call_count = 0

        async def always_fails(pairs: list[CandidatePair]) -> BatchVerdict:
            nonlocal call_count
            call_count += 1
            raise RuntimeError("permanent error")

        monkeypatch.setattr("marklymem.evolver.resolver.asyncio.sleep", AsyncMock())
        r = _resolver(store, embedder=embedder, verify_call=always_fails, config=cfg)
        result = await r.resolve(UID, "test", [new_unit])

        assert call_count == cfg.max_retries
        assert result["resolved"] == 0
        assert (await store.get_by_ids(["e-001"]))[0].status == MemoryStatus.ACTIVE


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

class TestFactory:
    def test_returns_none_without_api_key(self, store):
        class FakeSettings:
            OPENAI_API_KEY = ""

        assert create_conflict_resolver(FakeSettings(), store) is None

    def test_returns_resolver_with_api_key(self, store):
        class FakeSettings:
            OPENAI_API_KEY = "sk-test"

        assert isinstance(create_conflict_resolver(FakeSettings(), store), ConflictResolver)


# ---------------------------------------------------------------------------
# Manager integration
# ---------------------------------------------------------------------------

class TestManagerIntegration:
    async def test_d1_d2_reversal_resolved(self, store, monkeypatch, fake_uuid):
        from marklymem.evolver.manager import MemoryManager
        from marklymem.evolver.policy import MemoryPolicy

        FIXED_TS = "2025-01-15T14:00:00+00:00"
        monkeypatch.setattr("marklymem.evolver.manager.utc_now_iso", lambda: FIXED_TS)

        embedder = HashingEmbedder(dimensions=DIM)
        resolver = ConflictResolver(
            store, embedder,
            make_fake_verify({"mark": ConflictRelationship.CONTRADICTION}),
            config=ResolverConfig(cosine_threshold=0.0),
        )

        mgr = MemoryManager(
            store=store,
            policy=MemoryPolicy(recency_weight=0.0),
            user_id=UID,
            namespace="test",
            auto_consolidate=False,
            auto_resolve=True,
            embedder=embedder,
            ingestion_mode="pattern",
            resolution_mode="llm",
            resolver=resolver,
        )

        await mgr.ingest_session_turns("sess-d1", [
            {"prompt_text": "Always include the mark in the feedback.", "response_text": "Got it."},
        ])
        await mgr.ingest_session_turns("sess-d2", [
            {"prompt_text": "Never include the mark in the feedback.", "response_text": "Understood."},
        ])

        active = await store.list_active(UID, "test")
        mark_units = [u for u in active if "mark" in u.content.lower()]
        assert len(mark_units) <= 1
