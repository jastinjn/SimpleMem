# pyright: reportMissingImports=false
"""Integration tests for MemoryManager.

Uses real MemoryStore, MemoryRetriever, and MemoryConsolidator.
Monkeypatches manager.uuid and manager.utc_now_iso to remove nondeterminism.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from marklymem.evolver.embeddings import HashingEmbedder, OpenAIEmbedder
from marklymem.evolver.manager import MemoryManager
from marklymem.evolver.models import MemoryStatus, MemoryType
from marklymem.evolver.policy import MemoryPolicy
from marklymem.evolver.store import MemoryStore

from .conftest import UID, UID2, _make_unit

FIXED_TS = "2025-01-15T14:00:00+00:00"


def _manager(
    store: MemoryStore,
    *,
    auto_consolidate: bool = False,
    auto_resolve: bool = True,
    retrieval_mode: str = "keyword",
    embedder=None,
    user_id: str = UID,
    namespace: str = "test",
    ingestion_mode: str = "pattern",
    llm_extractor=None,
    resolution_mode: str = "jaccard",
    resolver=None,
) -> MemoryManager:
    policy = MemoryPolicy(recency_weight=0.0)
    return MemoryManager(
        store=store,
        policy=policy,
        user_id=user_id,
        namespace=namespace,
        auto_consolidate=auto_consolidate,
        auto_resolve=auto_resolve,
        retrieval_mode=retrieval_mode,
        embedder=embedder,
        ingestion_mode=ingestion_mode,
        llm_extractor=llm_extractor,
        resolution_mode=resolution_mode,
        resolver=resolver,
    )


def _patch_time(monkeypatch):
    monkeypatch.setattr("marklymem.evolver.manager.utc_now_iso", lambda: FIXED_TS)


SAMPLE_TURNS = [
    {
        "prompt_text": "The project uses PostgreSQL as the primary database",
        "response_text": "Understood, I will keep that in mind.",
    },
    {
        "prompt_text": "We use Kubernetes for container orchestration and deployment",
        "response_text": "Got it, the deployment is handled by Kubernetes.",
    },
]


# ---------------------------------------------------------------------------
# ingest_session_turns
# ---------------------------------------------------------------------------

class TestIngestSessionTurns:
    async def test_returns_positive_count(self, store, monkeypatch, fake_uuid):
        _patch_time(monkeypatch)
        mgr = _manager(store)
        result = await mgr.ingest_session_turns("sess-001", SAMPLE_TURNS)
        assert result["added"] > 0

    async def test_empty_turns_produces_no_units(self, store, monkeypatch, fake_uuid):
        _patch_time(monkeypatch)
        mgr = _manager(store)
        result = await mgr.ingest_session_turns("sess-001", [])
        assert result["added"] == 0
        assert await store.list_active(UID, "test") == []

    async def test_short_content_filtered(self, store, monkeypatch, fake_uuid):
        _patch_time(monkeypatch)
        mgr = _manager(store)
        short_turns = [{"prompt_text": "ok", "response_text": ""}]
        await mgr.ingest_session_turns("sess-short", short_turns)
        for u in await store.list_active(UID, "test"):
            assert len(u.content.strip()) >= 3

    async def test_dedup_skips_existing_content(self, store, monkeypatch, fake_uuid):
        _patch_time(monkeypatch)
        mgr = _manager(store)
        first = await mgr.ingest_session_turns("sess-001", SAMPLE_TURNS)
        second = await mgr.ingest_session_turns("sess-002", SAMPLE_TURNS)
        assert second["added"] <= first["added"]

    async def test_auto_consolidate_false_does_not_consolidate(self, store, monkeypatch, fake_uuid):
        _patch_time(monkeypatch)
        mgr = _manager(store, auto_consolidate=False)
        dup1 = _make_unit(memory_id="dup-001", namespace="test", content="Duplicate content for consolidation test")
        dup2 = _make_unit(memory_id="dup-002", namespace="test", content="Duplicate content for consolidation test")
        await store.add_memories([dup1, dup2])
        await mgr.ingest_session_turns("sess-001", SAMPLE_TURNS)
        active_ids = {u.memory_id for u in await store.list_active(UID, "test")}
        assert "dup-001" in active_ids
        assert "dup-002" in active_ids

    async def test_auto_consolidate_true_consolidates(self, store, monkeypatch, fake_uuid):
        _patch_time(monkeypatch)
        mgr = _manager(store, auto_consolidate=True)
        dup1 = _make_unit(memory_id="dup-001", namespace="test", content="Duplicate content for consolidation test")
        dup2 = _make_unit(memory_id="dup-002", namespace="test", content="Duplicate content for consolidation test")
        await store.add_memories([dup1, dup2])
        await mgr.ingest_session_turns("sess-001", SAMPLE_TURNS)
        active_ids = {u.memory_id for u in await store.list_active(UID, "test")}
        assert not ("dup-001" in active_ids and "dup-002" in active_ids)

    async def test_embedder_computes_embeddings(self, store, monkeypatch, fake_uuid):
        _patch_time(monkeypatch)
        embedder = HashingEmbedder(dimensions=1024)
        mgr = _manager(store, embedder=embedder)
        await mgr.ingest_session_turns("sess-001", SAMPLE_TURNS)
        units = await store.list_active(UID, "test")
        assert all(len(u.embedding) == 1024 for u in units)

    async def test_no_embedder_produces_empty_embeddings(self, store, monkeypatch, fake_uuid):
        _patch_time(monkeypatch)
        mgr = _manager(store, embedder=None)
        await mgr.ingest_session_turns("sess-001", SAMPLE_TURNS)
        units = await store.list_active(UID, "test")
        assert all(u.embedding == [] for u in units)

    async def test_custom_namespace_used(self, store, monkeypatch, fake_uuid):
        _patch_time(monkeypatch)
        mgr = _manager(store, namespace="default")
        await mgr.ingest_session_turns("sess-001", SAMPLE_TURNS, user_id=UID, namespace="custom_scope")
        assert (await store.get_stats(UID, "custom_scope"))["active"] > 0
        assert (await store.get_stats(UID, "default"))["active"] == 0

    async def test_local_conflict_drops_earlier_turn_unit(self, store, monkeypatch, fake_uuid):
        _patch_time(monkeypatch)
        mgr = _manager(store)
        # Turn 1 and turn 3 both match the PREFERENCE pattern "I prefer ...".
        # They share identical topics so Jaccard = 1.0 > threshold, but have
        # different content — a genuine within-session contradiction. The earlier
        # unit (turn 1) should be dropped; only the later one (turn 3) persists.
        turns = [
            {
                "prompt_text": "I prefer giving feedback marks for marking student work",
                "response_text": "",
            },
            {
                "prompt_text": "Some unrelated content about today's class",
                "response_text": "",
            },
            {
                "prompt_text": "I prefer not giving feedback marks for marking student work",
                "response_text": "",
            },
        ]
        await mgr.ingest_session_turns("sess-conflict", turns)
        active = await store.list_active(UID, "test")
        preference_units = [u for u in active if u.memory_type == MemoryType.PREFERENCE]
        assert len(preference_units) == 1
        assert "not giving" in preference_units[0].content

    async def test_consolidation_supersedes_near_duplicate_of_existing(self, store, monkeypatch, fake_uuid):
        _patch_time(monkeypatch)
        # A pre-existing near-duplicate of what the turn extracts. The ingested
        # unit differs by a single trailing token ("promptly"), so it survives
        # pre-store dedup (which keys on exact content) but the consolidator's
        # near-duplicate pass (content-token Jaccard 0.9 ≥ 0.80) then supersedes
        # one of the pair. Resolve is off to isolate consolidation.
        existing = _make_unit(
            memory_id="cons-001", namespace="test",
            memory_type=MemoryType.PREFERENCE,
            content="User preference: giving detailed written feedback on every essay promptly.",
            importance=0.5,
            created_at="2025-01-01T00:00:00+00:00",
            updated_at="2025-01-01T00:00:00+00:00",
        )
        await store.add_memories([existing])
        mgr = _manager(store, auto_consolidate=True, auto_resolve=False)
        turns = [{"prompt_text": "I prefer giving detailed written feedback on every essay", "response_text": ""}]
        result = await mgr.ingest_session_turns("sess-cons", turns)
        assert result["superseded"] >= 1
        active_prefs = [u for u in await store.list_active(UID, "test") if u.memory_type == MemoryType.PREFERENCE]
        assert len(active_prefs) == 1

    async def test_conflict_resolution_supersedes_contradicting_existing(self, store, monkeypatch, fake_uuid):
        _patch_time(monkeypatch)
        from marklymem.evolver.embeddings import HashingEmbedder
        from marklymem.evolver.resolver import (
            BatchVerdict,
            CandidatePair,
            ConflictRelationship,
            ConflictResolver,
            PairVerdict,
            ResolverConfig,
        )

        embedder = HashingEmbedder(dimensions=1024)

        async def _always_contradiction(pairs: list[CandidatePair]) -> BatchVerdict:
            return BatchVerdict(verdicts=[
                PairVerdict(relationship=ConflictRelationship.CONTRADICTION)
                for _ in pairs
            ])

        resolver = ConflictResolver(
            store, embedder, _always_contradiction,
            config=ResolverConfig(cosine_threshold=0.0),
        )

        existing = _make_unit(
            memory_id="conf-001", namespace="test",
            memory_type=MemoryType.PREFERENCE,
            content="User preference: light minimal marking only.",
            importance=0.5,
            created_at="2025-01-01T00:00:00+00:00",
            updated_at="2025-01-01T00:00:00+00:00",
        )
        existing.embedding = embedder.encode(existing.content)
        await store.add_memories([existing])
        mgr = _manager(store, auto_consolidate=False, auto_resolve=True,
                       embedder=embedder, resolution_mode="llm", resolver=resolver)
        turns = [{"prompt_text": "I prefer heavy marking with detailed grading comments", "response_text": ""}]
        result = await mgr.ingest_session_turns("sess-conf", turns)
        assert result["superseded"] >= 1
        refetched = (await store.get_by_ids(["conf-001"]))[0]
        assert refetched.status == MemoryStatus.SUPERSEDED
        assert refetched.superseded_by is not None

    async def test_openai_embedder_called_during_ingestion(self, store, monkeypatch, fake_uuid):
        _patch_time(monkeypatch)
        from unittest.mock import AsyncMock, MagicMock

        DIM = 1024
        fake_emb = MagicMock()
        fake_emb.embedding = [0.1] * DIM
        fake_emb.index = 0
        fake_response = MagicMock()
        fake_response.data = [fake_emb]

        mock_sync_client = MagicMock()
        mock_async_client = MagicMock()
        mock_async_client.embeddings.create = AsyncMock(return_value=fake_response)

        monkeypatch.setattr("openai.OpenAI", MagicMock(return_value=mock_sync_client))
        monkeypatch.setattr("openai.AsyncOpenAI", MagicMock(return_value=mock_async_client))
        fake_settings = MagicMock()
        fake_settings.OPENAI_API_KEY = "sk-test"
        fake_settings.OPENAI_EMBEDDING_MODEL = "text-embedding-3-small"
        monkeypatch.setattr("marklymem.config.get_settings", lambda: fake_settings)

        embedder = OpenAIEmbedder(dimensions=DIM)
        mgr = _manager(store, embedder=embedder)
        await mgr.ingest_session_turns("sess-openai", SAMPLE_TURNS)

        assert mock_async_client.embeddings.create.called
        active = await store.list_active(UID, "test")
        assert all(len(u.embedding) == DIM for u in active if u.embedding)


# ---------------------------------------------------------------------------
# ingest_session_turns — LLM ingestion mode
# ---------------------------------------------------------------------------

class TestIngestSessionTurnsLLMMode:
    async def test_llm_path_creates_memory_units(self, store, monkeypatch, fake_uuid):
        _patch_time(monkeypatch)
        from unittest.mock import AsyncMock

        llm_unit = _make_unit(
            namespace="test",
            memory_type=MemoryType.SEMANTIC,
            content="A fact extracted by the LLM ingestion path",
            updated_at="2025-01-01T00:00:01+00:00",
        )
        llm_extractor = AsyncMock()
        llm_extractor.extract_session = AsyncMock(return_value=[llm_unit])

        mgr = _manager(store, ingestion_mode="llm", llm_extractor=llm_extractor)
        result = await mgr.ingest_session_turns("sess-llm", SAMPLE_TURNS)

        llm_extractor.extract_session.assert_awaited_once_with(
            turns=SAMPLE_TURNS,
            user_id=UID,
            namespace="test",
            session_id="sess-llm",
        )
        assert result["added"] > 0
        contents = {u.content for u in await store.list_active(UID, "test")}
        assert "A fact extracted by the LLM ingestion path" in contents


# ---------------------------------------------------------------------------
# render_for_prompt
# ---------------------------------------------------------------------------

class TestRenderForPrompt:
    async def test_empty_units_returns_empty_string(self, store, monkeypatch, fake_uuid):
        _patch_time(monkeypatch)
        mgr = _manager(store)
        assert await mgr.render_for_prompt([]) == ""

    async def test_renders_content(self, store, monkeypatch, fake_uuid):
        _patch_time(monkeypatch)
        mgr = _manager(store)
        u = _make_unit(
            memory_id="rend-001", namespace="test",
            memory_type=MemoryType.SEMANTIC,
            content="PostgreSQL is the database",
            updated_at="2025-01-01T00:00:01+00:00",
        )
        rendered = await mgr.render_for_prompt([u])
        assert "PostgreSQL is the database" in rendered

    async def test_groups_by_memory_type(self, store, monkeypatch, fake_uuid):
        _patch_time(monkeypatch)
        mgr = _manager(store)
        u_sem = _make_unit(
            memory_id="g-001", namespace="test",
            memory_type=MemoryType.SEMANTIC,
            content="semantic content",
            updated_at="2025-01-01T00:00:01+00:00",
        )
        u_ep = _make_unit(
            memory_id="g-002", namespace="test",
            memory_type=MemoryType.EPISODIC,
            content="episodic content",
            updated_at="2025-01-01T00:00:02+00:00",
        )
        rendered = await mgr.render_for_prompt([u_sem, u_ep])
        assert "### semantic" in rendered
        assert "### episodic" in rendered

    async def test_pinned_high_importance_appears_first(self, store, monkeypatch, fake_uuid):
        _patch_time(monkeypatch)
        mgr = _manager(store)
        unpinned = _make_unit(
            memory_id="p-001", namespace="test",
            memory_type=MemoryType.SEMANTIC,
            content="regular memory unit",
            importance=0.5,
            updated_at="2025-01-01T00:00:02+00:00",
        )
        pinned = _make_unit(
            memory_id="p-002", namespace="test",
            memory_type=MemoryType.EPISODIC,
            content="pinned memory unit",
            importance=0.99,
            updated_at="2025-01-01T00:00:01+00:00",
        )
        rendered = await mgr.render_for_prompt([unpinned, pinned])
        pos_pinned = rendered.find("pinned memory unit")
        pos_regular = rendered.find("regular memory unit")
        assert pos_pinned < pos_regular


class TestConflictResolution:
    """detect_conflicts / auto_resolve_conflicts — admin/manual path.

    These methods are no longer wired into ingest; they are tested here
    as standalone admin operations. Uses explicit user_id (a required arg)
    rather than the manager's own self.user_id.
    """

    def _conflicting_pair(self):
        older = _make_unit(
            memory_id="c-001", namespace="test",
            memory_type=MemoryType.PREFERENCE,
            content="Include the numerical mark in the feedback text",
            topics=["marks", "feedback"],
            created_at="2025-01-01T00:00:00+00:00",
            updated_at="2025-01-01T00:00:00+00:00",
        )
        newer = _make_unit(
            memory_id="c-002", namespace="test",
            memory_type=MemoryType.PREFERENCE,
            content="Never include the numerical mark in the feedback text",
            topics=["marks", "feedback"],
            created_at="2025-02-01T00:00:00+00:00",
            updated_at="2025-02-01T00:00:00+00:00",
        )
        return older, newer

    async def test_detect_flags_overlapping_different_content(self, store):
        older, newer = self._conflicting_pair()
        await store.add_memories([older, newer])
        conflicts = await _manager(store).detect_conflicts(UID, "test")
        assert len(conflicts) == 1
        assert conflicts[0]["overlap"] == 1.0
        assert conflicts[0]["type"] == MemoryType.PREFERENCE.value

    async def test_detect_ignores_different_types(self, store):
        older, newer = self._conflicting_pair()
        newer.memory_type = MemoryType.SEMANTIC
        await store.add_memories([older, newer])
        assert await _manager(store).detect_conflicts(UID, "test") == []

    async def test_resolve_supersedes_older_and_reports_dropped(self, store, monkeypatch):
        _patch_time(monkeypatch)
        older, newer = self._conflicting_pair()
        await store.add_memories([older, newer])
        result = await _manager(store).auto_resolve_conflicts(UID, "test")

        assert result["resolved"] == 1
        assert result["total_conflicts"] == 1
        # The older unit is superseded by the newer one.
        fetched = {u.memory_id: u for u in await store.get_by_ids(["c-001", "c-002"])}
        assert fetched["c-001"].status == MemoryStatus.SUPERSEDED
        assert fetched["c-001"].superseded_by == "c-002"
        assert fetched["c-002"].status == MemoryStatus.ACTIVE
        # Dropped payload mirrors the consolidate span's shape.
        drop = result["dropped"][0]
        assert drop["dropped_id"] == "c-001"
        assert drop["kept_id"] == "c-002"
        assert drop["reason"] == "conflict"
        assert drop["dropped_content"] == older.content
        assert drop["kept_content"] == newer.content

    async def test_resolve_skips_pinned(self, store, monkeypatch):
        _patch_time(monkeypatch)
        older, newer = self._conflicting_pair()
        newer.importance = 0.99  # pinned — must not be superseded away
        await store.add_memories([older, newer])
        result = await _manager(store).auto_resolve_conflicts(UID, "test")
        assert result["resolved"] == 0
        assert (await store.get_by_ids(["c-001"]))[0].status == MemoryStatus.ACTIVE

    async def test_resolve_no_conflicts_returns_zero(self, store):
        assert await _manager(store).auto_resolve_conflicts(UID, "test") == {"resolved": 0}


# ---------------------------------------------------------------------------
# clone_namespace
# ---------------------------------------------------------------------------

class TestCloneNamespace:
    async def _seed_source(self, store):
        await store.add_memories([
            _make_unit(memory_id="s-1", namespace="src", content="alpha fact one"),
            _make_unit(memory_id="s-2", namespace="src", content="beta fact two"),
        ])

    async def test_clones_all_active_units(self, store):
        await self._seed_source(store)
        result = await _manager(store).clone_namespace(UID, source_namespace="src", target_namespace="dst")
        assert result["cloned"] == 2
        assert len(await store.list_active(UID, "dst")) == 2

    async def test_creates_new_ids(self, store):
        await self._seed_source(store)
        await _manager(store).clone_namespace(UID, source_namespace="src", target_namespace="dst")
        source_ids = {u.memory_id for u in await store.list_active(UID, "src")}
        cloned_ids = {u.memory_id for u in await store.list_active(UID, "dst")}
        assert source_ids.isdisjoint(cloned_ids)

    async def test_source_unchanged(self, store):
        await self._seed_source(store)
        await _manager(store).clone_namespace(UID, source_namespace="src", target_namespace="dst")
        assert len(await store.list_active(UID, "src")) == 2

    async def test_empty_source_returns_zero(self, store):
        result = await _manager(store).clone_namespace(UID, source_namespace="nope", target_namespace="dst")
        assert result["cloned"] == 0
        assert await store.list_active(UID, "dst") == []

    async def test_does_not_clone_other_user_units(self, store):
        await store.add_memories([
            _make_unit(memory_id="mine-1", namespace="src"),
            _make_unit(memory_id="other-1", namespace="src", user_id=UID2),
        ])
        result = await _manager(store).clone_namespace(UID, source_namespace="src", target_namespace="dst")
        assert result["cloned"] == 1
        assert await store.list_active(UID2, "dst") == []


# ---------------------------------------------------------------------------
# archive_namespace
# ---------------------------------------------------------------------------

class TestArchiveNamespace:
    async def test_archives_non_pinned(self, store):
        await store.add_memories([
            _make_unit(memory_id="a-1", namespace="arch", importance=0.5),
            _make_unit(memory_id="a-2", namespace="arch", importance=0.6),
        ])
        result = await _manager(store).archive_namespace(UID, "arch")
        assert result["archived"] == 2
        assert result["pinned_kept"] == 0
        assert result["total_before"] == 2
        assert await store.list_active(UID, "arch") == []

    async def test_keeps_pinned(self, store):
        await store.add_memories([
            _make_unit(memory_id="a-1", namespace="arch", importance=0.5),
            _make_unit(memory_id="pin-1", namespace="arch", importance=0.99),
        ])
        result = await _manager(store).archive_namespace(UID, "arch")
        assert result["archived"] == 1
        assert result["pinned_kept"] == 1
        assert {u.memory_id for u in await store.list_active(UID, "arch")} == {"pin-1"}

    async def test_empty_scope_returns_zero(self, store):
        result = await _manager(store).archive_namespace(UID, "no-such-scope")
        assert result["archived"] == 0
        assert result["total_before"] == 0

    async def test_does_not_affect_other_scope_or_user(self, store):
        await store.add_memories([
            _make_unit(memory_id="t-1", namespace="arch"),
            _make_unit(memory_id="o-1", namespace="keep"),
            _make_unit(memory_id="u2-1", namespace="arch", user_id=UID2),
        ])
        await _manager(store).archive_namespace(UID, "arch")
        assert {u.memory_id for u in await store.list_active(UID, "keep")} == {"o-1"}
        assert {u.memory_id for u in await store.list_active(UID2, "arch")} == {"u2-1"}
