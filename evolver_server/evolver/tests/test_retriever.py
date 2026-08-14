# pyright: reportMissingImports=false
"""Tests for MemoryRetriever.

recency_weight=0 by default to eliminate time-based nondeterminism.
Scoring tests use pytest.approx for float comparisons.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from evolver_server.evolver.embeddings import HashingEmbedder
from evolver_server.evolver.models import MemoryQuery, MemoryType
from evolver_server.evolver.policy import MemoryPolicy
from evolver_server.evolver.retriever import MemoryRetriever

from .conftest import UID, _make_unit, create_test_units


def _policy(recency_weight: float = 0.0, **kwargs) -> MemoryPolicy:
    return MemoryPolicy(recency_weight=recency_weight, **kwargs)


def _query(query_text: str, scope_id: str = "test", top_k: int = 6, **kwargs) -> MemoryQuery:
    return MemoryQuery(user_id=UID, scope_id=scope_id, query_text=query_text, top_k=top_k, **kwargs)


# ---------------------------------------------------------------------------
# Keyword mode
# ---------------------------------------------------------------------------

class TestKeywordRetrieval:
    async def test_matching_unit_returned(self, store):
        await store.add_memories(create_test_units())
        r = MemoryRetriever(store, policy=_policy(), retrieval_mode="keyword")
        hits = await r.retrieve(_query("PostgreSQL"))
        assert any(h.unit.memory_id == "unit-001" for h in hits)

    async def test_no_match_returns_empty(self, store):
        await store.add_memories(create_test_units())
        r = MemoryRetriever(store, policy=_policy(), retrieval_mode="keyword")
        hits = await r.retrieve(_query("xyzzy_no_match_token"))
        assert hits == []

    async def test_limit_from_min_top_k_and_policy(self, store):
        await store.add_memories(create_test_units())
        policy = _policy(max_injected_units=2)
        r = MemoryRetriever(store, policy=policy, retrieval_mode="keyword")
        hits = await r.retrieve(_query("the", top_k=10))
        assert len(hits) <= 2

    async def test_scores_are_positive(self, store):
        await store.add_memories(create_test_units())
        r = MemoryRetriever(store, policy=_policy(), retrieval_mode="keyword")
        hits = await r.retrieve(_query("database"))
        assert all(h.score > 0 for h in hits)


# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Embedding mode
# ---------------------------------------------------------------------------

class TestEmbeddingRetrieval:
    async def test_similar_content_returned(self, store):
        embedder = HashingEmbedder(dimensions=1024)
        units = create_test_units()
        for u in units:
            u.embedding = embedder.encode(u.content)
        await store.add_memories(units)
        r = MemoryRetriever(store, policy=_policy(), retrieval_mode="embedding", embedder=embedder)
        hits = await r.retrieve(_query("PostgreSQL database"))
        assert len(hits) > 0
        assert any(h.unit.memory_id == "unit-001" for h in hits)

    async def test_empty_embedding_units_skipped(self, store):
        embedder = HashingEmbedder(dimensions=1024)
        u = _make_unit(memory_id="no-emb-001", scope_id="emb")
        u.embedding = []
        u2 = _make_unit(
            memory_id="has-emb-001", scope_id="emb",
            content="database PostgreSQL server",
            updated_at="2025-01-01T00:00:02+00:00",
        )
        u2.embedding = embedder.encode(u2.content)
        await store.add_memories([u, u2])
        r = MemoryRetriever(store, policy=_policy(), retrieval_mode="embedding", embedder=embedder)
        hits = await r.retrieve(_query("database", scope_id="emb"))
        ids = {h.unit.memory_id for h in hits}
        assert "no-emb-001" not in ids
        assert "has-emb-001" in ids

    async def test_no_embedder_returns_empty(self, store):
        await store.add_memories(create_test_units())
        r = MemoryRetriever(store, policy=_policy(), retrieval_mode="embedding", embedder=None)
        hits = await r.retrieve(_query("database"))
        assert hits == []

    async def test_score_formula(self, store):
        embedder = HashingEmbedder(dimensions=1024)
        policy = MemoryPolicy(
            importance_weight=0.5,
            recency_weight=0.0,
            type_boosts={"semantic": 1.0},
        )
        u = _make_unit(
            memory_id="score-001", scope_id="score_scope",
            memory_type=MemoryType.SEMANTIC,
            content="PostgreSQL database server",
            importance=0.8,
            confidence=0.9,
            reinforcement_score=0.0,
            updated_at="2025-01-01T00:00:01+00:00",
        )
        u.embedding = embedder.encode(u.content)
        await store.add_memories([u])
        r = MemoryRetriever(store, policy=policy, retrieval_mode="embedding", embedder=embedder)
        hits = await r.retrieve(_query("PostgreSQL database server", scope_id="score_scope"))
        assert len(hits) == 1
        from evolver_server.evolver.embeddings import cosine_similarity
        sim = cosine_similarity(embedder.encode("PostgreSQL database server"), u.embedding)
        expected = (sim + 0.5 * 0.8 + 0.0) * 1.0 * (0.8 + 0.2 * 0.9)
        assert hits[0].score == pytest.approx(expected, abs=1e-4)


# ---------------------------------------------------------------------------
# Hybrid mode
# ---------------------------------------------------------------------------

class TestHybridRetrieval:
    async def test_returns_hits(self, store):
        embedder = HashingEmbedder(dimensions=1024)
        units = create_test_units()
        for u in units:
            u.embedding = embedder.encode(u.content)
        await store.add_memories(units)
        r = MemoryRetriever(store, policy=_policy(), retrieval_mode="hybrid", embedder=embedder)
        hits = await r.retrieve(_query("PostgreSQL database"))
        assert len(hits) > 0

    async def test_include_types_filters(self, store):
        embedder = HashingEmbedder(dimensions=1024)
        units = create_test_units()
        for u in units:
            u.embedding = embedder.encode(u.content)
        await store.add_memories(units)
        r = MemoryRetriever(store, policy=_policy(), retrieval_mode="hybrid", embedder=embedder)
        hits = await r.retrieve(_query(
            "the database authentication deployment",
            include_types=[MemoryType.SEMANTIC],
        ))
        assert all(h.unit.memory_type == MemoryType.SEMANTIC for h in hits)


# ---------------------------------------------------------------------------
# Auto mode
# ---------------------------------------------------------------------------

class TestAutoMode:
    async def test_short_query_uses_keyword(self, store):
        await store.add_memories(create_test_units())
        embedder = HashingEmbedder(dimensions=1024)
        r = MemoryRetriever(store, policy=_policy(), retrieval_mode="auto", embedder=embedder)
        q = MemoryQuery(user_id=UID, scope_id="test", query_text="db key")
        mode = r._auto_select_mode(q)
        assert mode == "keyword"

    async def test_long_query_with_embedder_uses_hybrid(self, store):
        embedder = HashingEmbedder(dimensions=1024)
        r = MemoryRetriever(store, policy=_policy(), retrieval_mode="auto", embedder=embedder)
        q = MemoryQuery(user_id=UID, scope_id="test", query_text="the project uses PostgreSQL database backend")
        mode = r._auto_select_mode(q)
        assert mode == "hybrid"

    async def test_long_query_without_embedder_uses_keyword(self, store):
        r = MemoryRetriever(store, policy=_policy(), retrieval_mode="auto", embedder=None)
        q = MemoryQuery(user_id=UID, scope_id="test", query_text="the project uses PostgreSQL database backend")
        mode = r._auto_select_mode(q)
        assert mode == "keyword"


# ---------------------------------------------------------------------------
# Tag boost
# ---------------------------------------------------------------------------

class TestTagBoost:
    async def test_tag_matching_boosts_score(self, store):
        await store.add_memories(create_test_units())
        r = MemoryRetriever(store, policy=_policy(), retrieval_mode="keyword")
        hits_no_tags = await r.retrieve(_query("the"))
        hits_db_tag = await r.retrieve(_query("the", context_tags=["db"]))
        score_no_tag = next((h.score for h in hits_no_tags if h.unit.memory_id == "unit-001"), None)
        score_with_tag = next((h.score for h in hits_db_tag if h.unit.memory_id == "unit-001"), None)
        if score_no_tag is not None and score_with_tag is not None:
            assert score_with_tag > score_no_tag

    async def test_boost_capped_at_50_percent(self, store):
        u = _make_unit(
            memory_id="tag-001", scope_id="tagtest",
            content="database authentication",
            tags=["t1", "t2", "t3", "t4", "t5"],
            updated_at="2025-01-01T00:00:01+00:00",
        )
        await store.add_memories([u])
        r = MemoryRetriever(store, policy=_policy(), retrieval_mode="keyword")
        hits_no_boost = await r.retrieve(_query("database", scope_id="tagtest"))
        hits_boosted = await r.retrieve(_query("database", scope_id="tagtest",
                                               context_tags=["t1", "t2", "t3", "t4", "t5", "t6"]))
        if hits_no_boost and hits_boosted:
            ratio = hits_boosted[0].score / hits_no_boost[0].score
            assert ratio == pytest.approx(1.5, abs=0.01)


# ---------------------------------------------------------------------------
# Recency bonus (monkeypatched)
# ---------------------------------------------------------------------------

class TestRecencyBonus:
    async def test_recent_unit_gets_bonus(self, store, frozen_retriever_clock):
        embedder = HashingEmbedder(dimensions=1024)
        recent_ts = "2025-02-14T23:00:00+00:00"
        old_ts = "2024-01-01T00:00:00+00:00"

        u_recent = _make_unit(
            memory_id="recent-001", scope_id="rec",
            content="authentication system service",
            importance=0.5,
            updated_at=recent_ts,
            created_at=recent_ts,
        )
        u_recent.embedding = embedder.encode(u_recent.content)
        u_old = _make_unit(
            memory_id="old-001", scope_id="rec",
            content="authentication system service",
            importance=0.5,
            updated_at=old_ts,
            created_at=old_ts,
        )
        u_old.embedding = embedder.encode(u_old.content)
        await store.add_memories([u_recent, u_old])
        policy = MemoryPolicy(recency_weight=1.0, recent_bonus_hours=24,
                              keyword_weight=0.0, metadata_weight=0.0,
                              importance_weight=0.0)
        r = MemoryRetriever(store, policy=policy, retrieval_mode="hybrid", embedder=embedder)
        hits = await r.retrieve(_query("authentication system", scope_id="rec"))
        scores = {h.unit.memory_id: h.score for h in hits}
        if "recent-001" in scores and "old-001" in scores:
            assert scores["recent-001"] > scores["old-001"]

    def test_recency_zero_at_boundary(self):
        from evolver_server.evolver.retriever import _estimate_recency_bonus
        bonus = _estimate_recency_bonus("2000-01-01T00:00:00+00:00", recent_bonus_hours=72)
        assert bonus == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Sort determinism
# ---------------------------------------------------------------------------

class TestSortDeterminism:
    async def test_distinct_timestamps_give_stable_order(self, store):
        await store.add_memories(create_test_units())
        r = MemoryRetriever(store, policy=_policy(), retrieval_mode="keyword")
        hits1 = [h.unit.memory_id for h in await r.retrieve(_query("the"))]
        hits2 = [h.unit.memory_id for h in await r.retrieve(_query("the"))]
        assert hits1 == hits2
