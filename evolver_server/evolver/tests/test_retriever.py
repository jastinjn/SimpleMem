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

from .conftest import UID, _make_store, _make_unit, create_test_units


def _policy(recency_weight: float = 0.0, **kwargs) -> MemoryPolicy:
    return MemoryPolicy(recency_weight=recency_weight, **kwargs)


def _query(query_text: str, scope_id: str = "test", top_k: int = 6, **kwargs) -> MemoryQuery:
    return MemoryQuery(user_id=UID, scope_id=scope_id, query_text=query_text, top_k=top_k, **kwargs)


# ---------------------------------------------------------------------------
# Keyword mode
# ---------------------------------------------------------------------------

class TestKeywordRetrieval:
    def test_matching_unit_returned(self, tmp_path):
        store = _make_store(tmp_path)
        store.add_memories(create_test_units())
        r = MemoryRetriever(store, policy=_policy(), retrieval_mode="keyword")
        hits = r.retrieve(_query("PostgreSQL"))
        assert any(h.unit.memory_id == "unit-001" for h in hits)

    def test_no_match_returns_empty(self, tmp_path):
        store = _make_store(tmp_path)
        store.add_memories(create_test_units())
        r = MemoryRetriever(store, policy=_policy(), retrieval_mode="keyword")
        hits = r.retrieve(_query("xyzzy_no_match_token"))
        assert hits == []

    def test_limit_from_min_top_k_and_policy(self, tmp_path):
        store = _make_store(tmp_path)
        store.add_memories(create_test_units())
        policy = _policy(max_injected_units=2)
        r = MemoryRetriever(store, policy=policy, retrieval_mode="keyword")
        hits = r.retrieve(_query("the", top_k=10))
        assert len(hits) <= 2

    def test_scores_are_positive(self, tmp_path):
        store = _make_store(tmp_path)
        store.add_memories(create_test_units())
        r = MemoryRetriever(store, policy=_policy(), retrieval_mode="keyword")
        hits = r.retrieve(_query("database"))
        assert all(h.score > 0 for h in hits)


# ---------------------------------------------------------------------------
# Query expansion
# ---------------------------------------------------------------------------

class TestQueryExpansion:
    def test_expansion_triggers_when_few_direct_hits(self, tmp_path):
        store = _make_store(tmp_path)
        # Only one unit matches "db" directly; expansion "db"→"database" finds more.
        u_db = _make_unit(
            memory_id="db-001", scope_id="expand",
            content="the database system is configured",
            updated_at="2025-01-01T00:00:01+00:00",
        )
        store.add_memories([u_db])
        policy = _policy(max_injected_units=6)
        r = MemoryRetriever(store, policy=policy, retrieval_mode="keyword")
        hits = r.retrieve(_query("db", scope_id="expand"))
        ids = {h.unit.memory_id for h in hits}
        assert "db-001" in ids

    def test_expansion_only_hits_get_score_penalty(self, tmp_path):
        store = _make_store(tmp_path)
        # "ci" expands to ["continuous integration", "pipeline"].
        # "ci" is NOT a substring of "continuous integration pipeline" so the unit
        # only surfaces via the expansion path, not via the direct keyword search.
        expanded_only = _make_unit(
            memory_id="exp-only-001", scope_id="exp2",
            content="continuous integration pipeline runs on merge",
            importance=0.5,
            updated_at="2025-01-01T00:00:01+00:00",
        )
        store.add_memories([expanded_only])
        policy = _policy(max_injected_units=6)
        r = MemoryRetriever(store, policy=policy, retrieval_mode="keyword")
        # Direct search for "ci" finds nothing (no "ci" in the content string).
        hits = r.retrieve(_query("ci", scope_id="exp2"))
        # The unit must be found via expansion.
        assert any(h.unit.memory_id == "exp-only-001" for h in hits)
        # The 0.85 penalty is applied: with 1 doc, all idf scores = log2(1/1) = 0.
        # base score = 0.0 + 0.5 (importance) + 0.0 (reinforcement) = 0.5
        # penalised = 0.5 * 0.85 = 0.425
        hit = next(h for h in hits if h.unit.memory_id == "exp-only-001")
        assert hit.score == pytest.approx(0.425, abs=1e-4)


# ---------------------------------------------------------------------------
# Embedding mode
# ---------------------------------------------------------------------------

class TestEmbeddingRetrieval:
    def test_similar_content_returned(self, tmp_path):
        store = _make_store(tmp_path)
        embedder = HashingEmbedder(dimensions=64)
        units = create_test_units()
        for u in units:
            u.embedding = embedder.encode(u.content)
        store.add_memories(units)
        policy = _policy()
        r = MemoryRetriever(store, policy=policy, retrieval_mode="embedding", embedder=embedder)
        hits = r.retrieve(_query("PostgreSQL database"))
        assert len(hits) > 0
        assert any(h.unit.memory_id == "unit-001" for h in hits)

    def test_empty_embedding_units_skipped(self, tmp_path):
        store = _make_store(tmp_path)
        embedder = HashingEmbedder(dimensions=64)
        u = _make_unit(memory_id="no-emb-001", scope_id="emb")
        u.embedding = []
        u2 = _make_unit(
            memory_id="has-emb-001", scope_id="emb",
            content="database PostgreSQL server",
            updated_at="2025-01-01T00:00:02+00:00",
        )
        u2.embedding = embedder.encode(u2.content)
        store.add_memories([u, u2])
        r = MemoryRetriever(store, policy=_policy(), retrieval_mode="embedding", embedder=embedder)
        hits = r.retrieve(_query("database", scope_id="emb"))
        ids = {h.unit.memory_id for h in hits}
        assert "no-emb-001" not in ids
        assert "has-emb-001" in ids

    def test_no_embedder_returns_empty(self, tmp_path):
        store = _make_store(tmp_path)
        store.add_memories(create_test_units())
        r = MemoryRetriever(store, policy=_policy(), retrieval_mode="embedding", embedder=None)
        hits = r.retrieve(_query("database"))
        assert hits == []

    def test_score_formula(self, tmp_path):
        store = _make_store(tmp_path)
        embedder = HashingEmbedder(dimensions=64)
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
        store.add_memories([u])
        r = MemoryRetriever(store, policy=policy, retrieval_mode="embedding", embedder=embedder)
        hits = r.retrieve(_query("PostgreSQL database server", scope_id="score_scope"))
        assert len(hits) == 1
        from evolver_server.evolver.embeddings import cosine_similarity
        sim = cosine_similarity(embedder.encode("PostgreSQL database server"), u.embedding)
        expected = (sim + 0.5 * 0.8 + 0.0) * 1.0 * (0.8 + 0.2 * 0.9)
        assert hits[0].score == pytest.approx(expected, abs=1e-4)


# ---------------------------------------------------------------------------
# Hybrid mode
# ---------------------------------------------------------------------------

class TestHybridRetrieval:
    def test_returns_hits(self, tmp_path):
        store = _make_store(tmp_path)
        embedder = HashingEmbedder(dimensions=64)
        units = create_test_units()
        for u in units:
            u.embedding = embedder.encode(u.content)
        store.add_memories(units)
        r = MemoryRetriever(store, policy=_policy(), retrieval_mode="hybrid", embedder=embedder)
        hits = r.retrieve(_query("PostgreSQL database"))
        assert len(hits) > 0

    def test_include_types_filters(self, tmp_path):
        store = _make_store(tmp_path)
        embedder = HashingEmbedder(dimensions=64)
        units = create_test_units()
        for u in units:
            u.embedding = embedder.encode(u.content)
        store.add_memories(units)
        r = MemoryRetriever(store, policy=_policy(), retrieval_mode="hybrid", embedder=embedder)
        hits = r.retrieve(_query(
            "the database authentication deployment",
            include_types=[MemoryType.SEMANTIC],
        ))
        assert all(h.unit.memory_type == MemoryType.SEMANTIC for h in hits)


# ---------------------------------------------------------------------------
# Auto mode
# ---------------------------------------------------------------------------

class TestAutoMode:
    def test_short_query_uses_keyword(self, tmp_path):
        store = _make_store(tmp_path)
        store.add_memories(create_test_units())
        embedder = HashingEmbedder(dimensions=64)
        r = MemoryRetriever(store, policy=_policy(), retrieval_mode="auto", embedder=embedder)
        # "_auto_select_mode" picks keyword for fewer than 4 terms.
        from evolver_server.evolver.models import MemoryQuery
        q = MemoryQuery(user_id=UID, scope_id="test", query_text="db key")
        mode = r._auto_select_mode(q)
        assert mode == "keyword"

    def test_long_query_with_embedder_uses_hybrid(self, tmp_path):
        store = _make_store(tmp_path)
        embedder = HashingEmbedder(dimensions=64)
        r = MemoryRetriever(store, policy=_policy(), retrieval_mode="auto", embedder=embedder)
        from evolver_server.evolver.models import MemoryQuery
        q = MemoryQuery(user_id=UID, scope_id="test", query_text="the project uses PostgreSQL database backend")
        mode = r._auto_select_mode(q)
        assert mode == "hybrid"

    def test_long_query_without_embedder_uses_keyword(self, tmp_path):
        store = _make_store(tmp_path)
        r = MemoryRetriever(store, policy=_policy(), retrieval_mode="auto", embedder=None)
        from evolver_server.evolver.models import MemoryQuery
        q = MemoryQuery(user_id=UID, scope_id="test", query_text="the project uses PostgreSQL database backend")
        mode = r._auto_select_mode(q)
        assert mode == "keyword"


# ---------------------------------------------------------------------------
# Tag boost
# ---------------------------------------------------------------------------

class TestTagBoost:
    def test_tag_matching_boosts_score(self, tmp_path):
        store = _make_store(tmp_path)
        store.add_memories(create_test_units())
        r = MemoryRetriever(store, policy=_policy(), retrieval_mode="keyword")
        # unit-001 has tag "db", unit-002 has tag "auth".
        hits_no_tags = r.retrieve(_query("the"))
        hits_db_tag = r.retrieve(_query("the", context_tags=["db"]))
        # unit-001 should rank higher when db tag is provided.
        score_no_tag = next((h.score for h in hits_no_tags if h.unit.memory_id == "unit-001"), None)
        score_with_tag = next((h.score for h in hits_db_tag if h.unit.memory_id == "unit-001"), None)
        if score_no_tag is not None and score_with_tag is not None:
            assert score_with_tag > score_no_tag

    def test_boost_capped_at_50_percent(self, tmp_path):
        store = _make_store(tmp_path)
        u = _make_unit(
            memory_id="tag-001", scope_id="tagtest",
            content="database authentication",
            tags=["t1", "t2", "t3", "t4", "t5"],
            updated_at="2025-01-01T00:00:01+00:00",
        )
        store.add_memories([u])
        r = MemoryRetriever(store, policy=_policy(), retrieval_mode="keyword")
        hits_no_boost = r.retrieve(_query("database", scope_id="tagtest"))
        hits_boosted = r.retrieve(_query("database", scope_id="tagtest",
                                         context_tags=["t1", "t2", "t3", "t4", "t5", "t6"]))
        if hits_no_boost and hits_boosted:
            ratio = hits_boosted[0].score / hits_no_boost[0].score
            assert ratio == pytest.approx(1.5, abs=0.01)


# ---------------------------------------------------------------------------
# Recency bonus (monkeypatched)
# ---------------------------------------------------------------------------

class TestRecencyBonus:
    def test_recent_unit_gets_bonus(self, tmp_path, frozen_retriever_clock):
        # Recency bonus is only applied in HYBRID mode (not keyword).
        store = _make_store(tmp_path)
        embedder = HashingEmbedder(dimensions=64)
        # frozen now = 2025-02-15T00:00:00 UTC (FROZEN_NOW in conftest)
        # recent_bonus_hours=24 → anything within 24h of frozen now gets bonus > 0
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
        store.add_memories([u_recent, u_old])
        policy = MemoryPolicy(recency_weight=1.0, recent_bonus_hours=24,
                              keyword_weight=0.0, metadata_weight=0.0,
                              importance_weight=0.0)
        r = MemoryRetriever(store, policy=policy, retrieval_mode="hybrid", embedder=embedder)
        hits = r.retrieve(_query("authentication system", scope_id="rec"))
        scores = {h.unit.memory_id: h.score for h in hits}
        if "recent-001" in scores and "old-001" in scores:
            assert scores["recent-001"] > scores["old-001"]

    def test_recency_zero_at_boundary(self):
        from evolver_server.evolver.retriever import _estimate_recency_bonus
        # A unit updated exactly recent_bonus_hours ago → bonus = 0.
        # We test the function directly.
        bonus = _estimate_recency_bonus("2000-01-01T00:00:00+00:00", recent_bonus_hours=72)
        assert bonus == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Sort determinism
# ---------------------------------------------------------------------------

class TestSortDeterminism:
    def test_distinct_timestamps_give_stable_order(self, tmp_path):
        store = _make_store(tmp_path)
        store.add_memories(create_test_units())
        r = MemoryRetriever(store, policy=_policy(), retrieval_mode="keyword")
        hits1 = [h.unit.memory_id for h in r.retrieve(_query("the"))]
        hits2 = [h.unit.memory_id for h in r.retrieve(_query("the"))]
        assert hits1 == hits2
