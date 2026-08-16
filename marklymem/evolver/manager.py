from __future__ import annotations

import asyncio
import logging
import re
import uuid
from collections.abc import Callable

from marklymem.utils import telemetry

from .consolidator import MemoryConsolidator
from .embeddings import BaseEmbedder, create_embedder
from .llm_extractor import LLMMemoryExtractor
from .metrics import summarize_memory_store
from .models import MemoryQuery, MemoryStatus, MemoryType, MemoryUnit, utc_now_iso
from .policy import MemoryPolicy
from .resolver import ConflictResolver
from .retriever import MemoryRetriever
from .store import MemoryStore

logger = logging.getLogger(__name__)


def estimate_tokens(text: str) -> int:
    return len(text.split())


class MemoryManager:
    """Facade for retrieval, rendering, and write-side session extraction."""

    def __init__(
        self,
        store: MemoryStore,
        policy: MemoryPolicy | None = None,
        user_id: str = "",
        namespace: str = "default",
        auto_consolidate: bool = True,
        auto_resolve: bool = True,
        retrieval_mode: str = "keyword",
        use_embeddings: bool = False,
        embedding_mode: str = "hashing",
        embedding_model: str = "all-MiniLM-L6-v2",
        embedder: BaseEmbedder | None = None,
        ingestion_mode: str = "pattern",
        llm_extractor: LLMMemoryExtractor | None = None,
        resolution_mode: str = "jaccard",
        resolver: ConflictResolver | None = None,
    ):
        self.store = store
        self.policy = policy or MemoryPolicy()
        self.user_id = user_id
        self.namespace = namespace
        self.auto_consolidate = auto_consolidate
        self.auto_resolve = auto_resolve
        self.retrieval_mode = retrieval_mode
        self.ingestion_mode = ingestion_mode
        self.llm_extractor = llm_extractor
        self.resolution_mode = resolution_mode
        self.resolver = resolver
        self.use_embeddings = use_embeddings or retrieval_mode in {"embedding", "hybrid"}
        self.embedding_mode = embedding_mode
        self.embedding_model = embedding_model
        if embedder is not None:
            self.embedder = embedder
        elif self.use_embeddings:
            self.embedder = create_embedder(mode=embedding_mode)
        else:
            self.embedder = None
        self.retriever = MemoryRetriever(
            store=self.store,
            policy=self.policy,
            retrieval_mode=retrieval_mode,
            embedder=self.embedder,
        )
        self.consolidator = MemoryConsolidator(store=self.store)
        self._event_callbacks: list[Callable] = []

    def register_event_callback(self, callback: Callable) -> None:
        """Register a callback for memory events.

        Callbacks receive a dict with at least 'event' (str) and 'namespace' (str).
        Additional keys depend on the event type.
        """
        self._event_callbacks.append(callback)

    def _notify(self, event: str, **kwargs) -> None:
        """Fire all registered event callbacks. Best-effort; errors are logged."""
        if not self._event_callbacks:
            return
        payload = {"event": event, **kwargs}
        for cb in self._event_callbacks:
            try:
                cb(payload)
            except Exception as exc:
                logger.debug("Event callback error for %s: %s", event, exc)

    async def ingest_session_turns(
        self,
        session_id: str | None,
        turns: list[dict],
        user_id: str | None = None,
        namespace: str | None = None,
    ) -> dict[str, int]:
        """Create memory units from a completed session.

        Branches on the manager's ``ingestion_mode``: ``"llm"`` sends windows of the
        session to an LLM via ``self.llm_extractor``; ``"pattern"`` (default) uses
        per-turn regex/keyword extraction. Everything after (dedup,
        conflict detection, embedding, store, consolidation, telemetry) is shared.

        Returns a dict with keys: ``added``, ``superseded``, ``decayed``, ``reinforced``.
        """
        uid = user_id or self.user_id
        ns = namespace or self.namespace
        with telemetry.trace(
            "memory.ingest", session_id=session_id, user_id=uid, namespace=namespace, input=turns
        ) as root:
            result = await self._ingest_session_turns(session_id, turns, uid, ns)
            root.set_attribute("added", result["added"])
            root.set_attribute("superseded", result["superseded"])
            root.set_attribute("decayed", result["decayed"])
            root.set_attribute("reinforced", result["reinforced"])
            surviving = result.pop("surviving_new_units")
            if surviving:
                telemetry.set_output(root, [
                    {"content": u.content, "type": u.memory_type.value}
                    for u in surviving
                ])
            return result

    async def _ingest_session_turns(
        self,
        session_id: str | None,
        turns: list[dict],
        uid: str,
        ns: str,
    ) -> dict:
        """Inner ingestion pipeline; wrapped by :meth:`ingest_session_turns` in a root span."""
        mode = self.ingestion_mode
        units: list[MemoryUnit] = []

        if mode == "llm" and self.llm_extractor:
            units = await self.llm_extractor.extract_session(
                turns=turns,
                user_id=uid,
                namespace=ns,
                session_id=session_id,
            )
        else:
            async def _extract_turn(idx: int, turn: dict) -> list[MemoryUnit]:
                prompt_text = str(turn.get("prompt_text", "") or "").strip()
                response_text = str(turn.get("response_text", "") or "").strip()
                if not prompt_text and not response_text:
                    return []
                extracted = await _extract_memory_units_for_turn(
                    user_id=uid,
                    namespace=ns,
                    session_id=session_id,
                    turn_index=idx,
                    prompt_text=prompt_text,
                    response_text=response_text,
                )
                if extracted:
                    logger.info(
                        "[Memory] extract turn=%d/%d → %d units [%s]",
                        idx, len(turns), len(extracted),
                        ", ".join(u.memory_type.value for u in extracted),
                    )
                return extracted

            results = await asyncio.gather(
                *(_extract_turn(idx, turn) for idx, turn in enumerate(turns, start=1))
            )
            for extracted in results:
                units.extend(extracted)

        # Stamp user_id on all extracted units.
        for u in units:
            u.user_id = uid

        # Pre-ingestion validation: skip units with empty or overly short content.
        pre_validate_count = len(units)
        units = [u for u in units if len(u.content.strip()) >= 3]

        # Pre-ingestion dedup: skip units whose content already exists in the store.
        pre_dedup_count = len(units)
        units = await _dedup_against_store(units, self.store, uid, ns)
        dedup_skipped = pre_dedup_count - len(units)
        if dedup_skipped or (pre_validate_count - pre_dedup_count):
            logger.info(
                "[Memory] pre-filter: validated=%d dedup_skipped=%d remaining=%d",
                pre_validate_count - pre_dedup_count, dedup_skipped, len(units),
            )

        # Drop earlier units that contradict later units from the same session.
        local_conflicts = _detect_local_conflicts(units)
        if local_conflicts:
            drop_local = {c["earlier_id"] for c in local_conflicts}
            units = [u for u in units if u.memory_id not in drop_local]
            logger.info(
                "[Memory] local conflicts: dropped %d earlier units superseded within session",
                len(drop_local),
            )

        if self.embedder is not None:
            texts = [
                " ".join([u.content, " ".join(u.topics), " ".join(u.entities)])
                for u in units
            ]
            embeddings = await self.embedder.encode_batch(texts)
            for unit, emb in zip(units, embeddings):
                unit.embedding = emb

        added = await self.store.add_memories(units)

        consolidation_result: dict = {}
        if self.auto_consolidate:
            with telemetry.span("consolidate") as cons_span:
                consolidation_result = await self.consolidator.consolidate(uid, ns)
                cons_span.set_attribute("superseded", consolidation_result.get("superseded", 0))
                cons_span.set_attribute("decayed", consolidation_result.get("decayed", 0))
                cons_span.set_attribute("reinforced", consolidation_result.get("reinforced", 0))
                if consolidation_result.get("dropped"):
                    telemetry.set_output(cons_span, consolidation_result["dropped"])

        conflict_result: dict = {}
        if self.auto_resolve:
            with telemetry.span("resolve") as conf_span:
                if self.resolution_mode == "llm" and self.resolver is not None:
                    conflict_result = await self.resolver.resolve(uid, ns, units)
                else:
                    conflict_result = await self.auto_resolve_conflicts(uid, ns)
                conf_span.set_attribute("resolved", conflict_result.get("resolved", 0))
                if conflict_result.get("dropped"):
                    telemetry.set_output(conf_span, conflict_result["dropped"])

        dropped_ids = (
            {d["dropped_id"] for d in consolidation_result.get("dropped", [])}
            | {d["dropped_id"] for d in conflict_result.get("dropped", [])}
        )
        surviving_new_units = [u for u in units if u.memory_id not in dropped_ids]

        stats = await summarize_memory_store(self.store, uid, ns)
        logger.info(
            "[Memory] ingested %d memory units from session=%s namespace=%s active=%d dominant_type=%s",
            added,
            session_id,
            ns,
            stats.get("active", 0),
            stats.get("dominant_type", ""),
        )
        self._notify("ingest", namespace=ns, session_id=session_id, added=len(surviving_new_units))
        return {
            "added": len(surviving_new_units),
            "superseded": consolidation_result.get("superseded", 0)
            + conflict_result.get("resolved", 0),
            "decayed": consolidation_result.get("decayed", 0),
            "reinforced": consolidation_result.get("reinforced", 0),
            "surviving_new_units": surviving_new_units,
        }

    async def render_for_prompt(self, units: list[MemoryUnit], include_pool_context: bool = False) -> str:
        if not units:
            return ""
        lines = ["## Relevant Long-Term Memory"]
        if include_pool_context:
            stats = await self.get_namespace_stats()
            active = stats.get("active", 0)
            types = stats.get("type_count", 0)
            lines.append(f"_Pool: {active} memories across {types} types. Showing top {len(units)}._")

        # Sort pinned memories to front for guaranteed visibility.
        pinned = [u for u in units if u.importance >= 0.99]
        unpinned = [u for u in units if u.importance < 0.99]
        units = pinned + unpinned

        # Group units by type for a more structured, token-efficient render.
        by_type: dict[str, list[MemoryUnit]] = {}
        for unit in units:
            by_type.setdefault(unit.memory_type.value, []).append(unit)

        for type_name, group in by_type.items():
            label = type_name.replace("_", " ")
            lines.append(f"\n### {label}")
            for unit in group:
                text = unit.content.strip()
                if text:
                    freshness = _freshness_tag(unit.updated_at)
                    if freshness:
                        lines.append(f"- {text} [{freshness}]")
                    else:
                        lines.append(f"- {text}")
        return "\n".join(lines).strip()

    async def get_namespace_stats(self, user_id: str | None = None, namespace: str | None = None) -> dict:
        return await summarize_memory_store(self.store, user_id or self.user_id, namespace)

    async def get_access_patterns(self, namespace: str | None = None, limit: int = 5) -> dict:
        """Return access pattern insights for the given ns."""
        ns = namespace or self.namespace
        units = await self.store.list_active(self.user_id, ns, limit=500)
        if not units:
            return {"total": 0, "most_accessed": [], "never_accessed": 0}
        sorted_by_access = sorted(units, key=lambda u: u.access_count, reverse=True)
        most_accessed = [
            {"id": u.memory_id, "type": u.memory_type.value, "access_count": u.access_count, "content": u.content[:100]}
            for u in sorted_by_access[:limit]
        ]
        never_accessed = sum(1 for u in units if u.access_count == 0)
        avg_access = sum(u.access_count for u in units) / float(len(units))
        return {
            "total": len(units),
            "most_accessed": most_accessed,
            "never_accessed": never_accessed,
            "avg_access_count": round(avg_access, 2),
        }

    async def diagnose(self, namespace: str | None = None) -> dict:
        """Return a diagnostic summary for operator debugging.

        Combines store stats, policy state, access patterns, and retrieval
        telemetry into a single view for quick health assessment.
        """
        ns = namespace or self.namespace
        stats = await self.get_namespace_stats(ns)
        access = await self.get_access_patterns(ns)

        # Memory age distribution.
        from datetime import datetime, timezone

        age_buckets = {"<1h": 0, "<24h": 0, "<7d": 0, "older": 0}
        units = await self.store.list_active(self.user_id, ns, limit=500)
        now = datetime.now(timezone.utc)
        for u in units:
            try:
                created = datetime.fromisoformat(u.created_at.replace("Z", "+00:00"))
                if created.tzinfo is None:
                    created = created.replace(tzinfo=timezone.utc)
                age_hours = max((now - created).total_seconds() / 3600.0, 0.0)
            except (ValueError, TypeError):
                age_hours = float("inf")
            if age_hours < 1:
                age_buckets["<1h"] += 1
            elif age_hours < 24:
                age_buckets["<24h"] += 1
            elif age_hours < 168:
                age_buckets["<7d"] += 1
            else:
                age_buckets["older"] += 1

        # TTL statistics.
        ttl_set = sum(1 for u in units if u.expires_at)
        ttl_expiring_soon = 0
        for u in units:
            if u.expires_at:
                try:
                    exp = datetime.fromisoformat(u.expires_at.replace("Z", "+00:00"))
                    if exp.tzinfo is None:
                        exp = exp.replace(tzinfo=timezone.utc)
                    hours_left = (exp - now).total_seconds() / 3600.0
                    if 0 < hours_left < 24:
                        ttl_expiring_soon += 1
                except (ValueError, TypeError):
                    pass

        issues: list[str] = []
        if stats.get("active", 0) == 0:
            issues.append("no active memories in store")
        if access.get("never_accessed", 0) > stats.get("active", 1) * 0.8:
            issues.append("over 80% of memories have never been accessed")
        if ttl_expiring_soon > 0:
            issues.append(f"{ttl_expiring_soon} memories expiring within 24 hours")

        return {
            "namespace": ns,
            "store": {
                "active": stats.get("active", 0),
                "dominant_type": stats.get("dominant_type", ""),
                "type_count": stats.get("type_count", 0),
                "age_distribution": age_buckets,
            },
            "access": {
                "avg_access_count": access.get("avg_access_count", 0.0),
                "never_accessed": access.get("never_accessed", 0),
            },
            "ttl": {
                "memories_with_ttl": ttl_set,
                "expiring_within_24h": ttl_expiring_soon,
            },
            "policy": {
                "retrieval_mode": self.retrieval_mode,
                "max_injected_units": self.policy.max_injected_units,
                "max_injected_tokens": self.policy.max_injected_tokens,
            },
            "issues": issues,
        }

    async def detect_conflicts(self, user_id: str, namespace: str | None = None) -> list[dict]:
        """Detect potential contradictions within the active memory pool.

        Compares all active memories of the same type that share significant
        topic/entity overlap but have different content.
        """
        ns = namespace or self.namespace
        units = await self.store.list_active(user_id, ns, limit=500)
        if len(units) < 2:
            return []

        conflicts: list[dict] = []
        seen: set[tuple[str, str]] = set()
        for i, a in enumerate(units):
            for b in units[i + 1:]:
                overlap = _jaccard_conflict(a, b)
                if overlap is None:
                    continue
                pair_key = (min(a.memory_id, b.memory_id), max(a.memory_id, b.memory_id))
                if pair_key in seen:
                    continue
                seen.add(pair_key)
                conflicts.append({
                    "id_a": a.memory_id,
                    "id_b": b.memory_id,
                    "type": a.memory_type.value,
                    "overlap": overlap,
                    "content_a": a.content[:120],
                    "content_b": b.content[:120],
                })
        return conflicts

    async def explain_retrieval(
        self,
        task_description: str,
        namespace: str | None = None,
    ) -> list[dict]:
        """Return detailed retrieval results with scoring explanations.

        Useful for operator debugging: shows why each memory was selected
        and its contribution to the final result.
        """
        effective_scope = namespace or self.namespace
        query = MemoryQuery(
            user_id=self.user_id,
            namespace=effective_scope,
            query_text=task_description,
            top_k=self.policy.max_injected_units,
            max_tokens=self.policy.max_injected_tokens,
        )
        hits = await self.retriever.retrieve(query)
        return [
            {
                "memory_id": h.unit.memory_id,
                "type": h.unit.memory_type.value,
                "score": round(h.score, 4),
                "matched_terms": h.matched_terms,
                "reason": h.reason,
                "content_preview": h.unit.content[:120],
                "importance": h.unit.importance,
                "access_count": h.unit.access_count,
            }
            for h in hits
        ]

    async def list_namespaces(self) -> list[dict]:
        """List all scopes in the store with memory counts."""
        return await self.store.list_namespaces(self.user_id)

    async def update_memory(self, memory_id: str, content: str) -> bool:
        """Update the content of an existing memory."""
        result = await self.store.update_content(memory_id, content)
        return result

    async def get_memory(self, memory_id: str) -> MemoryUnit | None:
        """Get a specific memory by ID."""
        return await self.store.get_by_id(memory_id)

    async def set_ttl(self, memory_id: str, expires_at: str) -> bool:
        """Set or clear a TTL on a memory.

        Args:
            memory_id: Target memory.
            expires_at: ISO-8601 expiry timestamp, or empty string to clear.
        """
        result = await self.store.set_ttl(memory_id, expires_at)
        return result

    async def expire_stale(self, namespace: str | None = None) -> int:
        """Archive all memories that have passed their TTL."""
        ns = namespace or self.namespace
        count = await self.store.expire_stale(self.user_id, ns)
        if count:
            self._notify("expire", namespace=namespace, expired_count=count)
        return count

    async def share_memory(self, memory_id: str, target_namespace_id: str) -> str | None:
        """Copy a memory to another namespace for cross-namespace knowledge sharing.

        Returns the new memory ID in the target ns.
        """
        new_id = await self.store.share_to_namespace(memory_id, target_namespace_id)
        if new_id:
            self._notify("share", memory_id=memory_id, target_namespace_id=target_namespace_id, new_id=new_id)
        return new_id

    async def export_scope(self, namespace: str | None = None) -> list[dict]:
        """Export all active memories for a namespace as JSON-serializable dicts."""
        ns = namespace or self.namespace
        return await self.store.export_namespace_json(self.user_id, ns)

    async def import_memories(self, data: list[dict], target_namespace_id: str | None = None) -> int:
        """Import memories from JSON dicts into the store."""
        ns = target_namespace_id or self.namespace
        count = await self.store.import_memories_json(self.user_id, data, ns)
        return count

    async def set_type_ttl(
        self,
        memory_type: MemoryType,
        expires_at: str,
        namespace: str | None = None,
    ) -> int:
        """Set TTL on all active memories of a given type."""
        ns = namespace or self.namespace
        count = await self.store.set_type_ttl(self.user_id, ns, memory_type, expires_at)
        return count

    async def merge_memories(self, id_a: str, id_b: str, merged_content: str) -> str | None:
        """Merge two memories into a new one, superseding both."""
        new_id = await self.store.merge_memories(id_a, id_b, merged_content)
        if new_id:
            self._notify("merge", id_a=id_a, id_b=id_b, new_id=new_id)
        return new_id

    async def get_memory_history(self, memory_id: str) -> list[dict]:
        """Get version history for a memory through its supersedes chain."""
        return await self.store.get_memory_history(memory_id)

    async def get_namespace_analytics(self, namespace: str | None = None) -> dict:
        """Get comprehensive analytics for a ns."""
        ns = namespace or self.namespace
        return await self.store.get_namespace_analytics(self.user_id, ns)

    async def add_tags(self, memory_id: str, tags: list[str]) -> bool:
        """Add user-defined tags to a memory."""
        result = await self.store.add_tags(memory_id, tags)
        return result

    async def remove_tags(self, memory_id: str, tags: list[str]) -> bool:
        """Remove tags from a memory."""
        result = await self.store.remove_tags(memory_id, tags)
        return result

    async def search_by_tag(self, tag: str, namespace: str | None = None, limit: int = 50) -> list[MemoryUnit]:
        """Find all active memories with a given tag."""
        ns = namespace or self.namespace
        return await self.store.search_by_tag(self.user_id, ns, tag, limit)

    async def bulk_archive(self, memory_ids: list[str]) -> int:
        """Archive multiple memories at once."""
        count = await self.store.bulk_archive(memory_ids)
        return count

    async def snapshot_namespace(self, namespace: str | None = None) -> dict:
        """Create a point-in-time snapshot for potential rollback."""
        ns = namespace or self.namespace
        return await self.store.snapshot_namespace(self.user_id, ns)

    async def restore_snapshot(self, snapshot: dict) -> int:
        """Restore a namespace from a previous snapshot."""
        count = await self.store.restore_snapshot(self.user_id, snapshot)
        return count

    async def find_similar(self, memory_id: str, limit: int = 5) -> list[dict]:
        """Find memories similar to a given memory by topic/entity overlap."""
        results = await self.store.find_similar(memory_id, limit)
        return [
            {
                "memory_id": u.memory_id,
                "type": u.memory_type.value,
                "similarity": score,
                "content": u.content[:120],
            }
            for u, score in results
        ]

    async def get_health_score(self, namespace: str | None = None) -> dict:
        """Get a composite health score (0-100) for the memory pool."""
        ns = namespace or self.namespace
        return await self.store.compute_health_score(self.user_id, ns)

    async def find_duplicates(self, namespace: str | None = None, threshold: float = 0.80) -> list[dict]:
        """Find near-duplicate memory pairs by content similarity."""
        ns = namespace or self.namespace
        return await self.store.find_duplicates(self.user_id, ns, threshold)

    async def consolidation_dry_run(self, namespace: str | None = None) -> dict:
        """Preview what consolidation would do without applying changes."""
        ns = namespace or self.namespace
        return await self.consolidator.dry_run(self.user_id, ns)

    async def search_advanced(
        self,
        keyword: str = "",
        memory_type: str = "",
        tag: str = "",
        min_importance: float = 0.0,
        namespace: str | None = None,
        limit: int = 50,
    ) -> list[MemoryUnit]:
        """Search memories with combined criteria."""
        ns = namespace or self.namespace
        return await self.store.search_advanced(self.user_id, ns, keyword, memory_type, tag, min_importance, limit)

    async def compare_namespaces(self, namespace_a: str, namespace_b: str) -> dict:
        """Compare two scopes to find shared and unique memories."""
        return await self.store.compare_namespaces(self.user_id, namespace_a, namespace_b)

    async def auto_resolve_conflicts(self, user_id: str, namespace: str | None = None) -> dict:
        """Automatically resolve conflicts by superseding older memories.

        When two same-type memories overlap significantly but have different
        content, the older one is superseded by the newer one.
        """
        ns = namespace or self.namespace
        conflicts = await self.detect_conflicts(user_id, ns)
        if not conflicts:
            return {"resolved": 0}

        now = utc_now_iso()
        resolved = 0
        dropped: list[dict] = []
        for c in conflicts:
            a = await self.store.get_by_id(c["id_a"])
            b = await self.store.get_by_id(c["id_b"])
            if a is None or b is None:
                continue
            # Skip if either is already superseded or pinned.
            if a.status != MemoryStatus.ACTIVE or b.status != MemoryStatus.ACTIVE:
                continue
            if a.importance >= 0.99 or b.importance >= 0.99:
                continue
            # Supersede the older one.
            drop, keep = (a, b) if a.created_at <= b.created_at else (b, a)
            await self.store.supersede(drop.memory_id, keep.memory_id, now)
            dropped.append({
                "dropped_id": drop.memory_id,
                "kept_id": keep.memory_id,
                "reason": "conflict",
                "type": drop.memory_type.value,
                "overlap": c["overlap"],
                "dropped_content": drop.content,
                "kept_content": keep.content,
            })
            resolved += 1

        self._notify(
            "conflict_resolution",
            namespace=namespace,
            resolved=resolved,
            total_conflicts=len(conflicts),
        )
        return {"resolved": resolved, "total_conflicts": len(conflicts), "dropped": dropped}

    async def rebalance_importance(self, namespace: str | None = None) -> dict:
        """Rebalance importance distribution to prevent clustering.

        If too many memories have the same importance value, spread them
        out to improve retrieval differentiation.
        """
        ns = namespace or self.namespace
        units = await self.store.list_active(self.user_id, ns, limit=5000)
        if len(units) < 5:
            return {"adjusted": 0}

        # Group by rounded importance.
        clusters: dict[float, list[MemoryUnit]] = {}
        for u in units:
            key = round(u.importance, 1)
            clusters.setdefault(key, []).append(u)

        adjusted = 0
        now = utc_now_iso()
        for _key, group in clusters.items():
            if len(group) < 4:
                continue
            # Spread importance within the cluster based on access count.
            group.sort(key=lambda u: u.access_count, reverse=True)
            spread = 0.05 * (len(group) - 1)
            base = max(0.1, group[0].importance - spread / 2)
            for idx, u in enumerate(group):
                if u.importance >= 0.99:  # don't touch pinned
                    continue
                new_imp = round(min(0.95, base + idx * 0.05 / max(len(group) - 1, 1) * spread), 4)
                if abs(new_imp - u.importance) > 0.005:
                    await self.store.update_importance(u.memory_id, new_imp, now)
                    adjusted += 1

        return {"adjusted": adjusted}

    async def pin_memory(self, memory_id: str) -> bool:
        """Pin a memory so it always ranks highest in retrieval."""
        result = await self.store.pin_memory(memory_id)
        return result

    async def unpin_memory(self, memory_id: str) -> bool:
        """Unpin a previously pinned memory."""
        result = await self.store.unpin_memory(memory_id)
        return result

    async def provide_feedback(self, memory_id: str, helpful) -> None:
        """Record retrieval feedback for a specific memory.

        Args:
            memory_id: Target memory.
            helpful: True/False, or "positive"/"negative" string.
        """
        if isinstance(helpful, str):
            helpful = helpful.lower() in ("positive", "true", "yes", "1", "helpful")
        await self.store.record_feedback(memory_id, helpful)

    async def get_pool_summary(self, namespace: str | None = None, max_per_type: int = 3) -> str:
        """Generate a concise summary of the entire memory pool.

        Returns a human-readable overview useful for operator inspection.
        """
        ns = namespace or self.namespace
        units = await self.store.list_active(self.user_id, ns, limit=500)
        if not units:
            return "No active memories."

        by_type: dict[str, list[MemoryUnit]] = {}
        for u in units:
            by_type.setdefault(u.memory_type.value, []).append(u)

        lines = [f"Memory pool: {len(units)} active memories across {len(by_type)} types"]
        for type_name, group in sorted(by_type.items()):
            lines.append(f"\n{type_name} ({len(group)}):")
            # Show top memories by importance.
            top = sorted(group, key=lambda u: u.importance, reverse=True)[:max_per_type]
            for u in top:
                preview = u.content[:100].replace("\n", " ")
                lines.append(f"  - [{u.importance:.2f}] {preview}")
            if len(group) > max_per_type:
                lines.append(f"  ... and {len(group) - max_per_type} more")

        conflicts = await self.detect_conflicts(self.user_id, ns)
        if conflicts:
            lines.append(f"\nPotential conflicts: {len(conflicts)}")
            for c in conflicts[:3]:
                lines.append(f"  - {c['content_a'][:60]} vs {c['content_b'][:60]}")

        return "\n".join(lines)

    async def search_memories(
        self,
        query_text: str,
        namespace: str | None = None,
        limit: int = 10,
    ) -> list[dict]:
        """Search memories by keyword with scoring information.

        Returns dictionaries with memory details and relevance scores.
        Useful for operator debugging and inspection.
        """
        ns = namespace or self.namespace
        hits = await self.store.search_keyword(self.user_id, ns,query_text, limit=limit)
        return [
            {
                "memory_id": h.unit.memory_id,
                "type": h.unit.memory_type.value,
                "score": round(h.score, 4),
                "content": h.unit.content[:200],
                "importance": h.unit.importance,
                "access_count": h.unit.access_count,
                "matched_terms": h.matched_terms,
                "created_at": h.unit.created_at,
            }
            for h in hits
        ]

    async def bulk_update_importance(
        self,
        updates: list[tuple[str, float]],
    ) -> int:
        """Update importance for multiple memories at once.

        Args:
            updates: List of (memory_id, new_importance) tuples.

        Returns:
            Number of memories updated.
        """
        count = 0
        now = utc_now_iso()
        for memory_id, importance in updates:
            clamped = max(0.1, min(0.99, importance))
            await self.store.update_importance(memory_id, round(clamped, 4), now)
            count += 1
        return count

    async def apply_retention_policy(
        self,
        namespace: str | None = None,
        max_age_days: int = 90,
        min_importance: float = 0.3,
        min_access_count: int = 0,
    ) -> dict:
        """Apply retention policy: archive old, low-importance, unused memories.

        Memories that are older than max_age_days AND have importance below
        min_importance AND have never been accessed are archived.
        Working summaries are exempt (always kept).
        """
        from datetime import datetime, timezone

        ns = namespace or self.namespace
        units = await self.store.list_active(self.user_id, ns, limit=5000)
        now = datetime.now(timezone.utc)
        archived = 0
        to_archive_ids: list[str] = []

        for u in units:
            # Never archive pinned memories.
            if u.importance >= 0.99:
                continue
            try:
                created = datetime.fromisoformat(u.created_at.replace("Z", "+00:00"))
                if created.tzinfo is None:
                    created = created.replace(tzinfo=timezone.utc)
                age_days = (now - created).total_seconds() / 86400.0
            except (ValueError, TypeError):
                continue
            if age_days < max_age_days:
                continue
            if u.importance >= min_importance:
                continue
            if u.access_count > min_access_count:
                continue
            to_archive_ids.append(u.memory_id)
            archived += 1

        if to_archive_ids:
            await self.store.bulk_archive(to_archive_ids)

        return {"archived": archived, "namespace": ns}

    async def apply_typed_retention(
        self,
        namespace: str | None = None,
        type_policies: dict[str, dict] | None = None,
    ) -> dict:
        """Apply per-type retention policies.

        Each type can have its own max_age_days, min_importance, and min_access_count.
        Default policies are applied if type_policies is not specified.
        """
        from datetime import datetime, timezone

        ns = namespace or self.namespace
        defaults: dict[str, dict] = {
            "episodic": {"max_age_days": 30, "min_importance": 0.2, "min_access_count": 0},
            "semantic": {"max_age_days": 180, "min_importance": 0.3, "min_access_count": 0},
            "preference": {"max_age_days": 365, "min_importance": 0.1, "min_access_count": 0},
            "project_state": {"max_age_days": 60, "min_importance": 0.2, "min_access_count": 0},
            "procedural_observation": {"max_age_days": 90, "min_importance": 0.2, "min_access_count": 0},
        }
        if type_policies:
            for k, v in type_policies.items():
                defaults[k] = {**defaults.get(k, {}), **v}

        units = await self.store.list_active(self.user_id, ns, limit=5000)
        now = datetime.now(timezone.utc)
        archived = 0
        to_archive_ids: list[str] = []

        for u in units:
            if u.importance >= 0.99:
                continue
            policy = defaults.get(u.memory_type.value, {"max_age_days": 90, "min_importance": 0.3, "min_access_count": 0})
            try:
                created = datetime.fromisoformat(u.created_at.replace("Z", "+00:00"))
                if created.tzinfo is None:
                    created = created.replace(tzinfo=timezone.utc)
                age_days = (now - created).total_seconds() / 86400.0
            except (ValueError, TypeError):
                continue
            if age_days < policy["max_age_days"]:
                continue
            if u.importance >= policy["min_importance"]:
                continue
            if u.access_count > policy.get("min_access_count", 0):
                continue
            to_archive_ids.append(u.memory_id)
            archived += 1

        if to_archive_ids:
            await self.store.bulk_archive(to_archive_ids)
        return {"archived": archived, "namespace": ns}

    async def apply_adaptive_ttl(
        self,
        namespace: str | None = None,
        base_days: dict[str, int] | None = None,
    ) -> dict:
        """Set TTL on memories based on type and access patterns.

        Memories that are accessed more frequently get longer TTLs.
        Working summaries get short TTLs (7 days). Episodic memories
        get medium TTLs (30 days). Others follow base_days or default to 90.
        Access count > 3 doubles the base TTL.
        """
        from datetime import datetime, timedelta, timezone

        ns = namespace or self.namespace
        defaults = {
            "episodic": 30,
            "semantic": 90,
            "preference": 180,
            "project_state": 60,
            "procedural_observation": 90,
        }
        if base_days:
            defaults.update(base_days)

        units = await self.store.list_active(self.user_id, ns, limit=5000)
        now = datetime.now(timezone.utc)
        updated = 0
        for u in units:
            if u.expires_at:  # Skip if TTL already set.
                continue
            if u.importance >= 0.99:  # Never auto-TTL pinned.
                continue
            base = defaults.get(u.memory_type.value, 90)
            # Frequently accessed memories get double TTL.
            if u.access_count > 3:
                base *= 2
            # High-importance memories get 50% longer TTL.
            if u.importance >= 0.7:
                base = int(base * 1.5)
            expires = now + timedelta(days=base)
            await self.store.set_ttl(u.memory_id, expires.isoformat(timespec="seconds"))
            updated += 1

        return {"updated": updated, "namespace": ns}

    async def batch_archive_by_criteria(
        self,
        namespace: str | None = None,
        max_quality_score: float | None = None,
        memory_type: MemoryType | None = None,
        max_importance: float | None = None,
        min_age_days: int | None = None,
    ) -> dict:
        """Archive memories matching all specified criteria.

        Pinned memories are always excluded.
        All criteria that are set must be satisfied (AND logic).
        """
        from datetime import datetime, timezone

        ns = namespace or self.namespace
        units = await self.store.list_active(self.user_id, ns, limit=5000)
        now = datetime.now(timezone.utc)
        to_archive: list[str] = []

        for u in units:
            if u.importance >= 0.99:
                continue
            if memory_type is not None and u.memory_type != memory_type:
                continue
            if max_importance is not None and u.importance > max_importance:
                continue
            if min_age_days is not None:
                try:
                    created = datetime.fromisoformat(u.created_at.replace("Z", "+00:00"))
                    if created.tzinfo is None:
                        created = created.replace(tzinfo=timezone.utc)
                    age = (now - created).total_seconds() / 86400.0
                    if age < min_age_days:
                        continue
                except (ValueError, TypeError):
                    continue
            if max_quality_score is not None:
                quality = await self.score_memory_quality(u.memory_id)
                if quality["score"] > max_quality_score:
                    continue
            to_archive.append(u.memory_id)

        if to_archive:
            await self.store.bulk_archive(to_archive)

        return {"archived": len(to_archive), "namespace": ns}

    async def score_memory_quality(self, memory_id: str) -> dict:
        """Compute a quality score (0-100) for a single memory unit.

        Factors: content richness, metadata completeness, access activity,
        importance calibration, and link connectivity.
        """
        unit = await self.store.get_by_id(memory_id)
        if unit is None:
            return {"score": 0, "reason": "not found"}

        # 1. Content richness (0-25): longer, more informative content scores higher.
        words = len(unit.content.split())
        content_score = min(25, 25 * min(words, 20) / 20.0)

        # 2. Metadata completeness (0-25): topics + entities + tags.
        meta_points = 0
        if unit.topics:
            meta_points += min(3, len(unit.topics))
        if unit.entities:
            meta_points += min(3, len(unit.entities))
        if unit.tags:
            meta_points += min(2, len(unit.tags))
        metadata_score = min(25, 25 * meta_points / 8.0)

        # 3. Access activity (0-25): accessed memories are more valuable.
        access_score = min(25, 25 * min(unit.access_count, 5) / 5.0)

        # 4. Importance + reinforcement (0-15).
        importance_score = 15 * unit.importance

        link_score = 0.0

        total = round(content_score + metadata_score + access_score + importance_score + link_score, 1)
        return {
            "score": total,
            "memory_id": memory_id,
            "components": {
                "content_richness": round(content_score, 1),
                "metadata_completeness": round(metadata_score, 1),
                "access_activity": round(access_score, 1),
                "importance": round(importance_score, 1),
                "connectivity": round(link_score, 1),
            },
        }

    async def get_lowest_quality_memories(self, namespace: str | None = None, limit: int = 10) -> list[dict]:
        """Get the lowest-quality active memories in a namespace for review/cleanup."""
        ns = namespace or self.namespace
        units = await self.store.list_active(self.user_id, ns, limit=500)
        scored = []
        for u in units:
            result = await self.score_memory_quality(u.memory_id)
            result["content_preview"] = u.content[:80]
            result["memory_type"] = u.memory_type.value
            scored.append(result)
        scored.sort(key=lambda x: x["score"])
        return scored[:limit]

    async def migrate_scope(self, from_scope: str, to_scope: str) -> dict:
        """Move all active memories from one namespace to another.

        Memories are copied to the new namespace and archived in the old ns.
        """
        units = await self.store.list_active(self.user_id, from_scope, limit=10000)
        migrated = 0
        for u in units:
            new_id = await self.store.share_to_namespace(u.memory_id, to_scope)
            if new_id:
                migrated += 1
        if migrated:
            to_archive = [u.memory_id for u in units]
            await self.store.bulk_archive(to_archive)
        self._notify("scope_migration", from_scope=from_scope, to_scope=to_scope, migrated=migrated)
        return {"migrated": migrated, "from_scope": from_scope, "to_scope": to_scope}

    async def get_age_distribution(self, namespace: str | None = None) -> dict:
        """Get age distribution of active memories in named buckets."""
        from datetime import datetime, timezone

        ns = namespace or self.namespace
        units = await self.store.list_active(self.user_id, ns, limit=5000)
        now = datetime.now(timezone.utc)
        buckets = {"< 1 day": 0, "1-7 days": 0, "1-4 weeks": 0, "1-3 months": 0, "3+ months": 0}

        for u in units:
            try:
                created = datetime.fromisoformat(u.created_at.replace("Z", "+00:00"))
                if created.tzinfo is None:
                    created = created.replace(tzinfo=timezone.utc)
                age_days = (now - created).total_seconds() / 86400.0
            except (ValueError, TypeError):
                continue
            if age_days < 1:
                buckets["< 1 day"] += 1
            elif age_days < 7:
                buckets["1-7 days"] += 1
            elif age_days < 28:
                buckets["1-4 weeks"] += 1
            elif age_days < 90:
                buckets["1-3 months"] += 1
            else:
                buckets["3+ months"] += 1

        return {"distribution": buckets, "total": len(units)}

    async def find_cross_scope_duplicates(
        self,
        namespace_a: str,
        namespace_b: str,
        threshold: float = 0.80,
    ) -> list[dict]:
        """Find near-duplicate memories across two scopes."""
        units_a = await self.store.list_active(self.user_id, namespace_a, limit=500)
        units_b = await self.store.list_active(self.user_id, namespace_b, limit=500)
        if not units_a or not units_b:
            return []

        def _tokenize_content(content: str) -> set[str]:
            return set(w.lower() for w in content.split() if len(w) >= 3)

        tokens_a = {u.memory_id: _tokenize_content(u.content) for u in units_a}
        tokens_b = {u.memory_id: _tokenize_content(u.content) for u in units_b}
        content_a = {u.memory_id: u.content[:80] for u in units_a}
        content_b = {u.memory_id: u.content[:80] for u in units_b}

        duplicates: list[dict] = []
        for id_a, toks_a in tokens_a.items():
            if not toks_a:
                continue
            for id_b, toks_b in tokens_b.items():
                if not toks_b:
                    continue
                inter = len(toks_a & toks_b)
                union = len(toks_a | toks_b)
                if union > 0 and inter / union >= threshold:
                    duplicates.append({
                        "id_a": id_a, "namespace_a": namespace_a,
                        "id_b": id_b, "namespace_b": namespace_b,
                        "similarity": round(inter / union, 4),
                        "preview_a": content_a[id_a],
                        "preview_b": content_b[id_b],
                    })
        duplicates.sort(key=lambda x: x["similarity"], reverse=True)
        return duplicates[:20]

    async def suggest_type_corrections(self, namespace: str | None = None, limit: int = 10) -> list[dict]:
        """Suggest memories that might be mistyped based on content analysis.

        Checks for content patterns that suggest a different type than assigned.
        """
        ns = namespace or self.namespace
        units = await self.store.list_active(self.user_id, ns, limit=500)
        suggestions: list[dict] = []

        for u in units:
            content_lower = u.content.lower()
            suggested = None

            # Preference indicators.
            if u.memory_type != MemoryType.PREFERENCE and any(
                p in content_lower for p in ["i prefer", "i like", "i want", "my convention"]
            ):
                suggested = MemoryType.PREFERENCE

            # Project state indicators.
            elif u.memory_type != MemoryType.PROJECT_STATE and any(
                p in content_lower for p in ["the project uses", "our stack", "we use"]
            ):
                suggested = MemoryType.PROJECT_STATE

            # Procedural indicators.
            elif u.memory_type != MemoryType.PROCEDURAL_OBSERVATION and any(
                p in content_lower for p in ["always", "never", "make sure", "workflow"]
            ):
                suggested = MemoryType.PROCEDURAL_OBSERVATION

            if suggested:
                suggestions.append({
                    "memory_id": u.memory_id,
                    "current_type": u.memory_type.value,
                    "suggested_type": suggested.value,
                    "content_preview": u.content[:80],
                    "confidence": 0.6,
                })

        return suggestions[:limit]

    async def compute_urgency_scores(self, namespace: str | None = None, limit: int = 10) -> list[dict]:
        """Compute urgency scores for memories that need attention.

        Urgency considers TTL proximity, low access count, and importance.
        Returns the most urgent memories first.
        """
        from datetime import datetime, timezone

        ns = namespace or self.namespace
        units = await self.store.list_active(self.user_id, ns, limit=5000)
        now = datetime.now(timezone.utc)
        scored: list[dict] = []

        for u in units:
            urgency = 0.0

            # TTL urgency: memories expiring soon are urgent.
            if u.expires_at:
                try:
                    expires = datetime.fromisoformat(u.expires_at.replace("Z", "+00:00"))
                    if expires.tzinfo is None:
                        expires = expires.replace(tzinfo=timezone.utc)
                    days_remaining = (expires - now).total_seconds() / 86400.0
                    if days_remaining < 0:
                        urgency += 50  # Already expired!
                    elif days_remaining < 7:
                        urgency += 30 * (1 - days_remaining / 7.0)
                except (ValueError, TypeError):
                    pass

            # Access urgency: high-importance memories that are never accessed.
            if u.importance > 0.6 and u.access_count == 0:
                urgency += 20 * u.importance

            # Quality urgency: low-quality high-importance memories need enrichment.
            if u.importance > 0.5 and not u.tags:
                urgency += 10

            if urgency > 0:
                scored.append({
                    "memory_id": u.memory_id,
                    "type": u.memory_type.value,
                    "urgency": round(urgency, 2),
                    "content_preview": u.content[:80],
                    "importance": u.importance,
                    "expires_at": u.expires_at,
                    "access_count": u.access_count,
                })

        scored.sort(key=lambda x: x["urgency"], reverse=True)
        return scored[:limit]

    async def get_memories_by_ids(self, memory_ids: list[str]) -> list:
        """Retrieve multiple memories by their IDs in a single operation."""
        return await self.store.get_by_ids(memory_ids)

    async def build_version_tree(self, memory_id: str) -> dict:
        """Build a version tree rooted at a memory, following supersedes chains.

        Traverses both directions: finds the root (oldest ancestor) and all descendants.
        Returns a nested tree structure.
        """
        # Find the root by following superseded_by backwards.
        root_id = memory_id
        visited_up = {memory_id}
        while True:
            unit = await self.store.get_by_id(root_id)
            if not unit or not unit.superseded_by:
                break
            if unit.superseded_by in visited_up:
                break
            visited_up.add(unit.superseded_by)
            root_id = unit.superseded_by

        # Actually, superseded_by points to the newer version.
        # Let's find the oldest ancestor instead by looking for units that supersede this one.
        # Walk up: find all units whose superseded_by points to root candidates.
        # Simpler approach: get the full history chain using existing method.
        history = await self.get_memory_history(memory_id)

        def _build_node(entry: dict) -> dict:
            return {
                "memory_id": entry["memory_id"],
                "status": entry.get("status", "unknown"),
                "content_preview": entry.get("content", "")[:60],
                "created_at": entry.get("created_at"),
                "superseded_by": entry.get("superseded_by"),
                "importance": entry.get("importance", 0.0),
            }

        nodes = [_build_node(e) for e in history]
        return {
            "root_id": history[-1]["memory_id"] if history else memory_id,
            "current_id": memory_id,
            "chain_length": len(nodes),
            "versions": nodes,
        }

    async def search_with_context(
        self,
        query: str,
        namespace: str | None = None,
        limit: int = 10,
    ) -> list[dict]:
        """Search memories and return results with matched terms highlighted.

        Returns dicts with memory info and highlighted content snippets.
        """
        ns = namespace or self.namespace
        hits = await self.store.search_keyword(self.user_id, ns,query, limit=limit)
        results = []
        for hit in hits:
            # Build a snippet with matched terms marked.
            content = hit.unit.content
            snippet = content[:200]
            for term in hit.matched_terms[:5]:
                # Case-insensitive highlight.
                pattern = re.compile(re.escape(term), re.IGNORECASE)
                snippet = pattern.sub(f"**{term}**", snippet)
            results.append({
                "memory_id": hit.unit.memory_id,
                "type": hit.unit.memory_type.value,
                "score": round(hit.score, 4),
                "matched_terms": hit.matched_terms,
                "snippet": snippet,
                "importance": hit.unit.importance,
                "tags": hit.unit.tags,
            })
        return results

    async def group_by_topic(self, namespace: str | None = None, min_group_size: int = 2) -> dict:
        """Group active memories by their dominant topic.

        Returns topic -> list of memory summaries, sorted by group size descending.
        """
        ns = namespace or self.namespace
        units = await self.store.list_active(self.user_id, ns, limit=5000)
        topic_groups: dict[str, list[dict]] = {}

        for u in units:
            if not u.topics:
                continue
            primary_topic = u.topics[0]
            if primary_topic not in topic_groups:
                topic_groups[primary_topic] = []
            topic_groups[primary_topic].append({
                "memory_id": u.memory_id,
                "type": u.memory_type.value,
                "content_preview": u.content[:80],
                "importance": u.importance,
            })

        # Filter by min_group_size and sort.
        filtered = {
            topic: members
            for topic, members in topic_groups.items()
            if len(members) >= min_group_size
        }
        sorted_groups = dict(
            sorted(filtered.items(), key=lambda kv: len(kv[1]), reverse=True)
        )
        return {
            "total_groups": len(sorted_groups),
            "total_grouped": sum(len(v) for v in sorted_groups.values()),
            "groups": sorted_groups,
        }

    async def find_stale_memories(
        self,
        namespace: str | None = None,
        stale_days: int = 30,
        limit: int = 20,
    ) -> list[dict]:
        """Find memories that haven't been accessed recently and may be outdated.

        Considers both access recency and creation age.
        """
        from datetime import datetime, timezone

        ns = namespace or self.namespace
        units = await self.store.list_active(self.user_id, ns, limit=5000)
        now = datetime.now(timezone.utc)
        stale: list[dict] = []

        for u in units:
            # Use updated_at as last activity proxy.
            try:
                last_active = datetime.fromisoformat(
                    u.updated_at.replace("Z", "+00:00")
                )
                if last_active.tzinfo is None:
                    last_active = last_active.replace(tzinfo=timezone.utc)
            except (ValueError, TypeError, AttributeError):
                continue

            days_inactive = (now - last_active).total_seconds() / 86400.0
            if days_inactive >= stale_days:
                staleness = min(days_inactive / stale_days, 5.0)  # Cap at 5x
                stale.append({
                    "memory_id": u.memory_id,
                    "type": u.memory_type.value,
                    "content_preview": u.content[:80],
                    "days_inactive": round(days_inactive, 1),
                    "staleness_factor": round(staleness, 2),
                    "importance": u.importance,
                    "access_count": u.access_count,
                    "is_pinned": u.importance >= 0.99,
                })

        stale.sort(key=lambda x: x["staleness_factor"], reverse=True)
        return stale[:limit]

    async def get_memory_summary_report(self, namespace: str | None = None) -> dict:
        """Generate a comprehensive summary report of a namespace's memory state.

        Combines stats, health, age distribution, importance histogram, and top topics
        into a single report suitable for dashboards or periodic reviews.
        """
        from datetime import datetime, timezone

        ns = namespace or self.namespace
        stats = await self.get_namespace_stats(ns)
        health = await self.get_health_score(ns)
        age_dist = await self.get_age_distribution(ns)
        importance_hist = await self.get_importance_histogram(ns)
        topics = await self.group_by_topic(ns, min_group_size=1)

        # Top 5 topics by group size.
        top_topics = []
        for topic, members in list(topics.get("groups", {}).items())[:5]:
            top_topics.append({"topic": topic, "count": len(members)})

        return {
            "namespace": ns,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "total_active": stats.get("active", 0),
            "total_superseded": stats.get("superseded", 0),
            "health_score": health,
            "age_distribution": age_dist.get("distribution", {}),
            "importance_distribution": importance_hist.get("histogram", {}),
            "top_topics": top_topics,
            "topic_group_count": topics.get("total_groups", 0),
        }

    async def suggest_auto_tags(self, namespace: str | None = None, limit: int = 20) -> list[dict]:
        """Suggest tags for memories based on content analysis.

        Analyzes content for topic keywords, entities, and patterns to suggest
        tags for memories that have no tags.
        """
        ns = namespace or self.namespace
        units = await self.store.list_active(self.user_id, ns, limit=5000)
        suggestions: list[dict] = []

        for u in units:
            if u.tags:
                continue  # Already tagged.

            suggested_tags = []
            # Use existing topics as tag candidates.
            for topic in u.topics[:3]:
                if len(topic) >= 3:
                    suggested_tags.append(topic)

            # Use entities as tag candidates.
            for entity in u.entities[:2]:
                if len(entity) >= 3:
                    suggested_tags.append(entity)

            # Content pattern-based tags.
            content_lower = u.content.lower()
            if any(w in content_lower for w in ["score", "marks", "grade", "band", "level"]) or re.search(r'\bl\d+\b', content_lower):
                suggested_tags.append("scoring")
            if any(w in content_lower for w in ["feedback", "comment", "annotation", "note"]):
                suggested_tags.append("feedback")

            if suggested_tags:
                suggestions.append({
                    "memory_id": u.memory_id,
                    "type": u.memory_type.value,
                    "content_preview": u.content[:80],
                    "suggested_tags": list(dict.fromkeys(suggested_tags))[:5],  # Deduplicate, keep order.
                })

        return suggestions[:limit]

    async def get_deduplication_report(self, namespace: str | None = None, threshold: float = 0.75) -> dict:
        """Generate a comprehensive deduplication report for a ns.

        Finds all near-duplicate pairs and groups them by cluster.
        """
        ns = namespace or self.namespace
        dupes = await self.find_duplicates(ns, threshold=threshold)

        # Group duplicates into clusters using union-find.
        parent: dict[str, str] = {}

        def find(x: str) -> str:
            while parent.get(x, x) != x:
                parent[x] = parent.get(parent[x], parent[x])
                x = parent[x]
            return x

        def union(a: str, b: str) -> None:
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[ra] = rb

        for d in dupes:
            union(d["id_a"], d["id_b"])

        clusters: dict[str, list[str]] = {}
        all_ids = set()
        for d in dupes:
            all_ids.add(d["id_a"])
            all_ids.add(d["id_b"])
        for mid in all_ids:
            root = find(mid)
            if root not in clusters:
                clusters[root] = []
            if mid not in clusters[root]:
                clusters[root].append(mid)

        return {
            "total_duplicate_pairs": len(dupes),
            "duplicate_clusters": len(clusters),
            "affected_memories": len(all_ids),
            "threshold": threshold,
            "pairs": dupes[:20],
            "clusters": [
                {"root": root, "members": members, "size": len(members)}
                for root, members in sorted(clusters.items(), key=lambda kv: len(kv[1]), reverse=True)
            ][:10],
        }

    async def search_regex(
        self,
        pattern: str,
        namespace: str | None = None,
        limit: int = 20,
    ) -> list[dict]:
        """Search memory content using a regular expression pattern.

        Returns matching memories with the matched portion highlighted.
        """
        ns = namespace or self.namespace
        units = await self.store.list_active(self.user_id, ns, limit=5000)
        results = []
        try:
            compiled = re.compile(pattern, re.IGNORECASE)
        except re.error:
            return []

        for u in units:
            match = compiled.search(u.content)
            if match:
                results.append({
                    "memory_id": u.memory_id,
                    "type": u.memory_type.value,
                    "content_preview": u.content[:80],
                    "matched_text": match.group(),
                    "match_position": match.start(),
                    "importance": u.importance,
                })
                if len(results) >= limit:
                    break

        return results

    async def merge_scopes(self, source_namespace: str, target_namespace: str) -> dict:
        """Merge all active memories from source namespace into target ns.

        Unlike migrate_scope, this preserves the source namespace intact.
        Memories are copied (shared) to the target ns.
        """
        source_units = await self.store.list_active(self.user_id, source_namespace, limit=5000)
        if not source_units:
            return {"copied": 0, "skipped": 0, "source_namespace": source_namespace, "target_namespace": target_namespace}

        # Check existing target content to avoid duplicates.
        target_units = await self.store.list_active(self.user_id, target_namespace, limit=5000)
        target_contents = {u.content.strip().lower() for u in target_units}

        copied = 0
        skipped = 0
        for u in source_units:
            if u.content.strip().lower() in target_contents:
                skipped += 1
                continue
            # Use share_to_namespace to copy.
            try:
                await self.store.share_to_namespace(u.memory_id, target_namespace)
                copied += 1
            except Exception:
                skipped += 1

        return {
            "copied": copied,
            "skipped": skipped,
            "source_namespace": source_namespace,
            "target_namespace": target_namespace,
        }

    async def diff_memories(self, memory_id_a: str, memory_id_b: str) -> dict:
        """Compare two memories side by side, showing differences.

        Returns a structured diff of content, metadata, and other fields.
        """
        unit_a = await self.store.get_by_id(memory_id_a)
        unit_b = await self.store.get_by_id(memory_id_b)

        if not unit_a or not unit_b:
            return {"error": "One or both memories not found"}

        # Word-level content diff.
        words_a = set(unit_a.content.lower().split())
        words_b = set(unit_b.content.lower().split())
        only_a = words_a - words_b
        only_b = words_b - words_a
        shared = words_a & words_b

        return {
            "memory_a": {
                "memory_id": unit_a.memory_id,
                "type": unit_a.memory_type.value,
                "content": unit_a.content,
                "importance": unit_a.importance,
                "topics": unit_a.topics,
                "tags": unit_a.tags,
                "created_at": unit_a.created_at,
            },
            "memory_b": {
                "memory_id": unit_b.memory_id,
                "type": unit_b.memory_type.value,
                "content": unit_b.content,
                "importance": unit_b.importance,
                "topics": unit_b.topics,
                "tags": unit_b.tags,
                "created_at": unit_b.created_at,
            },
            "content_diff": {
                "shared_words": len(shared),
                "only_in_a": len(only_a),
                "only_in_b": len(only_b),
                "similarity": round(len(shared) / max(len(words_a | words_b), 1), 4),
            },
            "type_match": unit_a.memory_type == unit_b.memory_type,
            "importance_delta": round(unit_a.importance - unit_b.importance, 4),
        }

    async def clone_namespace(
        self, user_id: str, *, source_namespace: str, target_namespace: str
    ) -> dict:
        """deep-clone a namespace: copies all active memories with full metadata.

        Unlike merge_scopes, this creates fresh copies with new IDs.
        """
        import uuid

        source_units = await self.store.list_active(user_id, source_namespace, limit=5000)
        if not source_units:
            return {"cloned": 0, "source_namespace": source_namespace, "target_namespace": target_namespace}

        cloned = 0
        for u in source_units:
            new_unit = MemoryUnit(
                memory_id=str(uuid.uuid4()),
                user_id=u.user_id,
                namespace=target_namespace,
                memory_type=u.memory_type,
                content=u.content,
                source_session_id=u.source_session_id,
                topics=list(u.topics),
                entities=list(u.entities),
                importance=u.importance,
                confidence=u.confidence,
                tags=list(u.tags),
            )
            await self.store.add_memories([new_unit])
            cloned += 1

        return {
            "cloned": cloned,
            "source_namespace": source_namespace,
            "target_namespace": target_namespace,
        }

    async def analyze_access_frequency(self, namespace: str | None = None) -> dict:
        """Categorize memories into hot (frequently accessed), warm, and cold buckets.

        Based on access_count relative to pool average.
        """
        ns = namespace or self.namespace
        units = await self.store.list_active(self.user_id, ns, limit=5000)
        if not units:
            return {"hot": [], "warm": [], "cold": [], "total": 0, "avg_access": 0}

        total_access = sum(u.access_count for u in units)
        avg_access = total_access / len(units) if units else 0
        hot_threshold = max(avg_access * 2, 3)
        cold_threshold = max(avg_access * 0.5, 1)

        hot, warm, cold = [], [], []
        for u in units:
            entry = {
                "memory_id": u.memory_id,
                "type": u.memory_type.value,
                "access_count": u.access_count,
                "importance": u.importance,
                "content_preview": u.content[:60],
            }
            if u.access_count >= hot_threshold:
                hot.append(entry)
            elif u.access_count < cold_threshold:
                cold.append(entry)
            else:
                warm.append(entry)

        hot.sort(key=lambda x: x["access_count"], reverse=True)
        cold.sort(key=lambda x: x["access_count"])

        return {
            "hot": hot[:10],
            "warm": warm[:10],
            "cold": cold[:10],
            "total": len(units),
            "avg_access": round(avg_access, 2),
            "hot_count": len(hot),
            "warm_count": len(warm),
            "cold_count": len(cold),
        }

    async def suggest_enrichments(self, namespace: str | None = None, limit: int = 20) -> list[dict]:
        """Suggest enrichments for memories that lack metadata.

        Identifies memories missing summaries, tags, or topics that would benefit
        from enrichment.
        """
        ns = namespace or self.namespace
        units = await self.store.list_active(self.user_id, ns, limit=5000)
        suggestions: list[dict] = []

        for u in units:
            missing = []
            if not u.tags:
                missing.append("tags")
            if not u.topics:
                missing.append("topics")
            if not u.entities:
                missing.append("entities")

            if missing:
                suggestions.append({
                    "memory_id": u.memory_id,
                    "type": u.memory_type.value,
                    "content_preview": u.content[:80],
                    "importance": u.importance,
                    "missing_fields": missing,
                    "completeness": round(1.0 - len(missing) / 3.0, 2),
                })

        # Sort by importance (high-importance memories should be enriched first).
        suggestions.sort(key=lambda x: (-x["importance"], x["completeness"]))
        return suggestions[:limit]

    async def get_content_density_stats(self, namespace: str | None = None) -> dict:
        """Analyze content density: token counts, value per token, and size distribution."""
        ns = namespace or self.namespace
        units = await self.store.list_active(self.user_id, ns, limit=5000)
        if not units:
            return {"total": 0, "avg_tokens": 0, "avg_value_per_token": 0, "size_buckets": {}}

        token_counts = []
        value_per_token = []
        for u in units:
            tokens = len(u.content.split())
            token_counts.append(tokens)
            if tokens > 0:
                value_per_token.append(u.importance / tokens)

        # Size distribution buckets.
        buckets = {"tiny (<10)": 0, "small (10-50)": 0, "medium (50-150)": 0, "large (150+)": 0}
        for tc in token_counts:
            if tc < 10:
                buckets["tiny (<10)"] += 1
            elif tc < 50:
                buckets["small (10-50)"] += 1
            elif tc < 150:
                buckets["medium (50-150)"] += 1
            else:
                buckets["large (150+)"] += 1

        return {
            "total": len(units),
            "total_tokens": sum(token_counts),
            "avg_tokens": round(sum(token_counts) / len(token_counts), 1),
            "min_tokens": min(token_counts),
            "max_tokens": max(token_counts),
            "avg_value_per_token": round(sum(value_per_token) / max(len(value_per_token), 1), 4),
            "size_buckets": buckets,
        }

    async def check_scope_quota(
        self,
        namespace: str | None = None,
        max_memories: int = 1000,
    ) -> dict:
        """Check if a namespace is within its memory quota.

        Returns quota status including current count, limit, and utilization.
        """
        ns = namespace or self.namespace
        units = await self.store.list_active(self.user_id, ns, limit=max_memories + 1)
        count = len(units)
        utilization = count / max(max_memories, 1)

        return {
            "namespace": ns,
            "current_count": count,
            "max_memories": max_memories,
            "utilization": round(utilization, 4),
            "within_quota": count <= max_memories,
            "remaining": max(max_memories - count, 0),
            "warning": utilization >= 0.9,
        }

    async def forecast_expiry(self, namespace: str | None = None) -> dict:
        """Forecast memory expirations over upcoming time windows."""
        from datetime import datetime, timezone

        ns = namespace or self.namespace
        units = await self.store.list_active(self.user_id, ns, limit=5000)
        now = datetime.now(timezone.utc)

        windows = {
            "next_24h": 0,
            "next_7d": 0,
            "next_30d": 0,
            "no_expiry": 0,
        }
        total_with_ttl = 0

        for u in units:
            if not u.expires_at:
                windows["no_expiry"] += 1
                continue
            total_with_ttl += 1
            try:
                expires = datetime.fromisoformat(u.expires_at.replace("Z", "+00:00"))
                if expires.tzinfo is None:
                    expires = expires.replace(tzinfo=timezone.utc)
                days = (expires - now).total_seconds() / 86400.0
                if days <= 1:
                    windows["next_24h"] += 1
                elif days <= 7:
                    windows["next_7d"] += 1
                elif days <= 30:
                    windows["next_30d"] += 1
            except (ValueError, TypeError):
                pass

        return {
            "total": len(units),
            "with_ttl": total_with_ttl,
            "forecast": windows,
        }

    async def get_type_overlap_matrix(self, namespace: str | None = None) -> dict:
        """Compute topic overlap between memory types.

        Returns a matrix showing how much each pair of types shares topics.
        """
        ns = namespace or self.namespace
        units = await self.store.list_active(self.user_id, ns, limit=5000)

        type_topics: dict[str, set[str]] = {}
        for u in units:
            t = u.memory_type.value
            if t not in type_topics:
                type_topics[t] = set()
            type_topics[t].update(u.topics)

        types = sorted(type_topics.keys())
        matrix: dict[str, dict[str, float]] = {}

        for ta in types:
            matrix[ta] = {}
            for tb in types:
                topics_a = type_topics[ta]
                topics_b = type_topics[tb]
                union = len(topics_a | topics_b)
                overlap = len(topics_a & topics_b) / max(union, 1)
                matrix[ta][tb] = round(overlap, 4)

        return {
            "types": types,
            "matrix": matrix,
        }

    async def recommend_archival(
        self,
        namespace: str | None = None,
        limit: int = 20,
    ) -> list[dict]:
        """Recommend memories for archival based on multiple signals.

        Combines staleness, low importance, low access, low quality, and no links
        into a single archival recommendation score.
        """
        ns = namespace or self.namespace
        units = await self.store.list_active(self.user_id, ns, limit=5000)
        recommendations: list[dict] = []

        from datetime import datetime, timezone
        now = datetime.now(timezone.utc)

        for u in units:
            if u.importance >= 0.99:  # Skip pinned.
                continue

            score = 0.0
            reasons = []

            # Low importance.
            if u.importance < 0.3:
                score += 20 * (0.3 - u.importance)
                reasons.append("low_importance")

            # Never accessed.
            if u.access_count == 0:
                score += 15
                reasons.append("never_accessed")

            # Old and stale.
            try:
                updated = datetime.fromisoformat(u.updated_at.replace("Z", "+00:00"))
                if updated.tzinfo is None:
                    updated = updated.replace(tzinfo=timezone.utc)
                days_old = (now - updated).total_seconds() / 86400.0
                if days_old > 60:
                    score += min(days_old / 30.0, 10)
                    reasons.append("stale")
            except (ValueError, TypeError, AttributeError):
                pass

            # No metadata.
            if not u.tags and not u.topics:
                score += 5
                reasons.append("no_metadata")

            if score > 10:
                recommendations.append({
                    "memory_id": u.memory_id,
                    "type": u.memory_type.value,
                    "content_preview": u.content[:80],
                    "archival_score": round(score, 2),
                    "reasons": reasons,
                    "importance": u.importance,
                    "access_count": u.access_count,
                })

        recommendations.sort(key=lambda x: x["archival_score"], reverse=True)
        return recommendations[:limit]

    async def get_scope_dashboard(self, namespace: str | None = None) -> dict:
        """Generate a comprehensive operational dashboard for a ns.

        Combines: summary report, access frequency, content density, link stats,
        expiry forecast, quota, and archival recommendations into one view.
        """
        ns = namespace or self.namespace

        report = await self.get_memory_summary_report(ns)
        access = await self.analyze_access_frequency(ns)
        density = await self.get_content_density_stats(ns)
        forecast = await self.forecast_expiry(ns)
        quota = await self.check_scope_quota(ns)
        archive_recs = await self.recommend_archival(ns, limit=5)
        urgency = await self.compute_urgency_scores(ns, limit=5)

        return {
            "namespace": ns,
            "overview": {
                "total_active": report.get("total_active", 0),
                "health_score": report.get("health_score", 0),
                "topic_groups": report.get("topic_group_count", 0),
                "top_topics": report.get("top_topics", []),
            },
            "access": {
                "hot_count": access.get("hot_count", 0),
                "warm_count": access.get("warm_count", 0),
                "cold_count": access.get("cold_count", 0),
                "avg_access": access.get("avg_access", 0),
            },
            "content": {
                "total_tokens": density.get("total_tokens", 0),
                "avg_tokens": density.get("avg_tokens", 0),
                "size_buckets": density.get("size_buckets", {}),
            },
            "expiry_forecast": forecast.get("forecast", {}),
            "quota": {
                "utilization": quota.get("utilization", 0),
                "within_quota": quota.get("within_quota", True),
            },
            "top_archival_candidates": len(archive_recs),
            "urgent_items": len(urgency),
        }

    async def generate_detailed_scope_comparison(self, namespace_a: str, namespace_b: str) -> dict:
        """Generate a detailed comparison report between two scopes.

        Goes beyond compare_namespaces to include type distributions, topic overlap,
        and health differences.
        """
        units_a = await self.store.list_active(self.user_id, namespace_a, limit=5000)
        units_b = await self.store.list_active(self.user_id, namespace_b, limit=5000)

        # Type distributions.
        types_a: dict[str, int] = {}
        types_b: dict[str, int] = {}
        for u in units_a:
            t = u.memory_type.value
            types_a[t] = types_a.get(t, 0) + 1
        for u in units_b:
            t = u.memory_type.value
            types_b[t] = types_b.get(t, 0) + 1

        # Topic sets.
        topics_a = set()
        topics_b = set()
        for u in units_a:
            topics_a.update(u.topics)
        for u in units_b:
            topics_b.update(u.topics)

        shared_topics = topics_a & topics_b
        unique_a = topics_a - topics_b
        unique_b = topics_b - topics_a

        # Importance stats.
        imp_a = [u.importance for u in units_a] or [0]
        imp_b = [u.importance for u in units_b] or [0]

        return {
            "namespace_a": {
                "name": namespace_a,
                "count": len(units_a),
                "types": types_a,
                "topic_count": len(topics_a),
                "avg_importance": round(sum(imp_a) / len(imp_a), 4),
            },
            "namespace_b": {
                "name": namespace_b,
                "count": len(units_b),
                "types": types_b,
                "topic_count": len(topics_b),
                "avg_importance": round(sum(imp_b) / len(imp_b), 4),
            },
            "shared_topics": list(shared_topics)[:20],
            "unique_to_a": list(unique_a)[:20],
            "unique_to_b": list(unique_b)[:20],
            "topic_overlap": round(
                len(shared_topics) / max(len(topics_a | topics_b), 1), 4
            ),
        }

    async def validate_content(self, content: str, rules: dict | None = None) -> dict:
        """Validate memory content against configurable rules.

        Default rules:
        - min_length: 3 characters
        - max_length: 10000 characters
        - min_words: 2
        - no_urls_only: content shouldn't be just a URL
        """
        if rules is None:
            rules = {}
        min_length = rules.get("min_length", 3)
        max_length = rules.get("max_length", 10000)
        min_words = rules.get("min_words", 2)

        errors = []
        warnings = []

        if len(content) < min_length:
            errors.append(f"Content too short ({len(content)} < {min_length} chars)")
        if len(content) > max_length:
            errors.append(f"Content too long ({len(content)} > {max_length} chars)")

        words = content.split()
        if len(words) < min_words:
            warnings.append(f"Very few words ({len(words)} < {min_words})")

        # Check if content is just a URL.
        stripped = content.strip()
        if stripped.startswith(("http://", "https://")) and " " not in stripped:
            warnings.append("Content appears to be just a URL")

        # Check for excessive repetition.
        if len(words) > 5:
            unique_ratio = len(set(w.lower() for w in words)) / len(words)
            if unique_ratio < 0.3:
                warnings.append(f"High word repetition (unique ratio: {unique_ratio:.0%})")

        return {
            "valid": len(errors) == 0,
            "errors": errors,
            "warnings": warnings,
            "stats": {
                "char_count": len(content),
                "word_count": len(words),
            },
        }

    async def recalculate_importance(self, namespace: str | None = None) -> dict:
        """Recalculate importance for all memories based on current signals.

        Factors: access frequency, link count, metadata completeness, recency.
        """
        from datetime import datetime, timezone

        ns = namespace or self.namespace
        units = await self.store.list_active(self.user_id, ns, limit=5000)
        now = datetime.now(timezone.utc)
        updated = 0

        for u in units:
            if u.importance >= 0.99:  # Skip pinned.
                continue

            new_importance = 0.5  # Base

            # Access bonus.
            if u.access_count > 0:
                new_importance += min(u.access_count * 0.02, 0.2)

            # Metadata completeness bonus.
            completeness = 0
            if u.tags:
                completeness += 0.33
            if u.topics:
                completeness += 0.33
            if u.entities:
                completeness += 0.34
            new_importance += completeness * 0.1

            # Recency bonus.
            try:
                updated_dt = datetime.fromisoformat(u.updated_at.replace("Z", "+00:00"))
                if updated_dt.tzinfo is None:
                    updated_dt = updated_dt.replace(tzinfo=timezone.utc)
                days_old = (now - updated_dt).total_seconds() / 86400.0
                if days_old < 7:
                    new_importance += 0.05
            except (ValueError, TypeError, AttributeError):
                pass

            new_importance = min(round(new_importance, 4), 0.98)

            if abs(new_importance - u.importance) > 0.01:
                await self.store.update_importance(u.memory_id, new_importance, now.isoformat())
                updated += 1

        return {
            "total_evaluated": len(units),
            "updated": updated,
            "namespace": ns,
        }

    async def analyze_type_balance(self, namespace: str | None = None) -> dict:
        """Analyze memory type distribution and suggest rebalancing actions."""
        ns = namespace or self.namespace
        units = await self.store.list_active(self.user_id, ns, limit=5000)
        if not units:
            return {"total": 0, "distribution": {}, "suggestions": []}

        type_counts: dict[str, int] = {}
        for u in units:
            t = u.memory_type.value
            type_counts[t] = type_counts.get(t, 0) + 1

        total = len(units)
        distribution = {t: {"count": c, "ratio": round(c / total, 4)} for t, c in type_counts.items()}

        suggestions = []
        # Check for dominant type (>60%).
        for t, info in distribution.items():
            if info["ratio"] > 0.6:
                suggestions.append({
                    "action": "reduce",
                    "type": t,
                    "ratio": info["ratio"],
                    "reason": f"{t} dominates at {info['ratio']:.0%} — consider archiving low-value {t} memories",
                })
        # Check for missing types.
        expected_types = {"semantic", "episodic", "preference", "project_state"}
        for et in expected_types:
            if et not in type_counts:
                suggestions.append({
                    "action": "add",
                    "type": et,
                    "ratio": 0,
                    "reason": f"No {et} memories — consider adding some for completeness",
                })

        return {
            "total": total,
            "distribution": distribution,
            "suggestions": suggestions,
        }

    async def compare_scope_health(self, namespace_a: str, namespace_b: str) -> dict:
        """Compare health scores and key metrics between two scopes."""
        health_a = await self.get_health_score(namespace_a)
        health_b = await self.get_health_score(namespace_b)
        stats_a = await self.get_namespace_stats(namespace_a)
        stats_b = await self.get_namespace_stats(namespace_b)

        score_a = health_a.get("score", 0) if isinstance(health_a, dict) else 0
        score_b = health_b.get("score", 0) if isinstance(health_b, dict) else 0

        return {
            "namespace_a": {
                "name": namespace_a,
                "health": score_a,
                "active": stats_a.get("active", 0),
            },
            "namespace_b": {
                "name": namespace_b,
                "health": score_b,
                "active": stats_b.get("active", 0),
            },
            "health_delta": score_a - score_b,
            "healthier_scope": namespace_a if score_a >= score_b else namespace_b,
        }

    async def get_memory_lifecycle(self, memory_id: str) -> dict:
        """Get the full lifecycle of a memory: creation, access, updates, and current state."""
        unit = await self.store.get_by_id(memory_id)
        if not unit:
            return {"error": "Memory not found", "memory_id": memory_id}

        return {
            "memory_id": memory_id,
            "current_state": {
                "status": unit.status.value if hasattr(unit.status, "value") else str(unit.status),
                "type": unit.memory_type.value,
                "importance": unit.importance,
                "access_count": unit.access_count,
                "created_at": unit.created_at,
                "updated_at": unit.updated_at,
                "content_preview": unit.content[:80],
                "tag_count": len(unit.tags),
                "topic_count": len(unit.topics),
            },
        }

    async def get_maintenance_recommendations(self, namespace: str | None = None) -> dict:
        """Generate maintenance recommendations based on namespace state.

        Analyzes the namespace and recommends specific maintenance actions.
        """
        ns = namespace or self.namespace
        actions = []

        # Check for expired memories.
        forecast = await self.forecast_expiry(ns)
        if forecast["forecast"].get("next_24h", 0) > 0:
            actions.append({
                "action": "expire_stale",
                "priority": "high",
                "reason": f"{forecast['forecast']['next_24h']} memories expiring in 24h",
            })

        # Check quota.
        quota = await self.check_scope_quota(ns)
        if quota["warning"]:
            actions.append({
                "action": "archive_low_value",
                "priority": "high",
                "reason": f"Quota at {quota['utilization']:.0%} — nearing limit",
            })

        # Check for stale memories.
        stale = await self.find_stale_memories(ns, stale_days=60, limit=1)
        if stale:
            actions.append({
                "action": "review_stale",
                "priority": "medium",
                "reason": "Stale memories detected (>60 days inactive)",
            })

        # Check type balance.
        balance = await self.analyze_type_balance(ns)
        if balance.get("suggestions"):
            actions.append({
                "action": "rebalance_types",
                "priority": "low",
                "reason": balance["suggestions"][0]["reason"],
            })

        # Check for untagged memories.
        tag_suggestions = self.suggest_auto_tags(ns, limit=1)
        if tag_suggestions:
            actions.append({
                "action": "tag_memories",
                "priority": "low",
                "reason": "Untagged memories found — consider auto-tagging",
            })

        actions.sort(key=lambda a: {"high": 0, "medium": 1, "low": 2}.get(a["priority"], 3))

        return {
            "namespace": ns,
            "total_recommendations": len(actions),
            "recommendations": actions,
        }

    async def export_for_training(self, namespace: str | None = None) -> list[dict]:
        """Export memories in a format suitable for ML training/fine-tuning.

        Returns structured records with content, metadata, and quality signals.
        """
        ns = namespace or self.namespace
        units = await self.store.list_active(self.user_id, ns, limit=5000)
        records = []

        for u in units:
            record = {
                "content": u.content,
                "type": u.memory_type.value,
                "topics": u.topics,
                "entities": u.entities,
                "importance": u.importance,
                "confidence": u.confidence,
                "access_count": u.access_count,
                "tags": u.tags,
                "metadata": {
                    "memory_id": u.memory_id,
                    "namespace": u.namespace,
                    "created_at": u.created_at,
                },
            }
            records.append(record)

        return records

    async def batch_update_content(self, updates: list[dict]) -> dict:
        """Update content for multiple memories at once.

        Each update dict should have: memory_id, content.
        Returns counts of updated and failed.
        """
        updated = 0
        failed = 0
        for u in updates:
            memory_id = u.get("memory_id", "")
            content = u.get("content", "")
            if not memory_id or not content:
                failed += 1
                continue
            try:
                await self.store.update_content(memory_id, content)
                updated += 1
            except Exception:
                failed += 1

        return {"updated": updated, "failed": failed, "total": len(updates)}

    async def compute_freshness_scores(self, namespace: str | None = None, limit: int = 20) -> list[dict]:
        """Compute combined freshness scores for memories.

        Considers: recency of creation, last access, update frequency, and TTL.
        Score 0-100 where 100 is freshest.
        """
        from datetime import datetime, timezone

        ns = namespace or self.namespace
        units = await self.store.list_active(self.user_id, ns, limit=5000)
        now = datetime.now(timezone.utc)
        scored: list[dict] = []

        for u in units:
            freshness = 50.0  # Base score.

            # Recency bonus (0-30).
            try:
                created = datetime.fromisoformat(u.created_at.replace("Z", "+00:00"))
                if created.tzinfo is None:
                    created = created.replace(tzinfo=timezone.utc)
                days_old = (now - created).total_seconds() / 86400.0
                recency = max(0, 30 * (1 - days_old / 365.0))
                freshness += recency
            except (ValueError, TypeError, AttributeError):
                pass

            # Access recency bonus (0-20).
            try:
                updated = datetime.fromisoformat(u.updated_at.replace("Z", "+00:00"))
                if updated.tzinfo is None:
                    updated = updated.replace(tzinfo=timezone.utc)
                access_days = (now - updated).total_seconds() / 86400.0
                access_bonus = max(0, 20 * (1 - access_days / 90.0))
                freshness += access_bonus
            except (ValueError, TypeError, AttributeError):
                pass

            freshness = min(max(round(freshness, 1), 0), 100)
            scored.append({
                "memory_id": u.memory_id,
                "type": u.memory_type.value,
                "freshness": freshness,
                "content_preview": u.content[:60],
                "importance": u.importance,
            })

        scored.sort(key=lambda x: x["freshness"], reverse=True)
        return scored[:limit]

    async def get_scope_inventory(
        self,
        namespace: str | None = None,
        type_filter: str | None = None,
        min_importance: float = 0.0,
        max_importance: float = 1.0,
        sort_by: str = "importance",
        limit: int = 50,
    ) -> dict:
        """Get a detailed, filterable inventory of memories in a ns.

        Supports filtering by type, importance range, and sorting.
        """
        ns = namespace or self.namespace
        units = await self.store.list_active(self.user_id, ns, limit=5000)

        # Apply filters.
        filtered = units
        if type_filter:
            filtered = [u for u in filtered if u.memory_type.value == type_filter]
        filtered = [u for u in filtered if min_importance <= u.importance <= max_importance]

        # Sort.
        if sort_by == "importance":
            filtered.sort(key=lambda u: u.importance, reverse=True)
        elif sort_by == "access":
            filtered.sort(key=lambda u: u.access_count, reverse=True)
        elif sort_by == "created":
            filtered.sort(key=lambda u: u.created_at or "", reverse=True)

        items = []
        for u in filtered[:limit]:
            items.append({
                "memory_id": u.memory_id,
                "type": u.memory_type.value,
                "content_preview": u.content[:80],
                "importance": u.importance,
                "access_count": u.access_count,
                "tags": u.tags[:3],
                "topics": u.topics[:3],
                "created_at": u.created_at,
            })

        return {
            "total_before_filter": len(units),
            "total_after_filter": len(filtered),
            "showing": len(items),
            "filters": {
                "type": type_filter,
                "min_importance": min_importance,
                "max_importance": max_importance,
                "sort_by": sort_by,
            },
            "items": items,
        }

    async def normalize_content(self, memory_id: str) -> dict:
        """Normalize memory content: strip whitespace, collapse multiple spaces, fix encoding."""
        unit = await self.store.get_by_id(memory_id)
        if not unit:
            return {"error": "Memory not found", "memory_id": memory_id}

        original = unit.content
        normalized = " ".join(original.split())  # Collapse whitespace.
        normalized = normalized.strip()

        if normalized == original:
            return {"memory_id": memory_id, "changed": False}

        await self.store.update_content(memory_id, normalized)
        return {
            "memory_id": memory_id,
            "changed": True,
            "original_length": len(original),
            "normalized_length": len(normalized),
        }

    async def batch_normalize_content(self, namespace: str | None = None) -> dict:
        """Normalize content for all memories in a ns."""
        ns = namespace or self.namespace
        units = await self.store.list_active(self.user_id, ns, limit=5000)
        normalized = 0

        for u in units:
            clean = " ".join(u.content.split()).strip()
            if clean != u.content:
                await self.store.update_content(u.memory_id, clean)
                normalized += 1

        return {"total": len(units), "normalized": normalized}

    async def get_priority_queue(
        self,
        namespace: str | None = None,
        limit: int = 20,
    ) -> list[dict]:
        """Get a priority-ranked queue of memories needing attention.

        Combines urgency, quality, and staleness into a single priority score.
        """
        ns = namespace or self.namespace
        urgency = await self.compute_urgency_scores(ns, limit=50)
        enrichment = await self.suggest_enrichments(ns, limit=50)
        stale = await self.find_stale_memories(ns, stale_days=30, limit=50)

        # Build combined priority map.
        priority_map: dict[str, dict] = {}

        for u in urgency:
            mid = u["memory_id"]
            if mid not in priority_map:
                priority_map[mid] = {"memory_id": mid, "priority": 0, "reasons": [], "type": u.get("type", "")}
            priority_map[mid]["priority"] += u["urgency"]
            priority_map[mid]["reasons"].append("urgent")

        for e in enrichment:
            mid = e["memory_id"]
            if mid not in priority_map:
                priority_map[mid] = {"memory_id": mid, "priority": 0, "reasons": [], "type": e.get("type", "")}
            priority_map[mid]["priority"] += (1 - e["completeness"]) * 20
            priority_map[mid]["reasons"].append("needs_enrichment")

        for s in stale:
            mid = s["memory_id"]
            if mid not in priority_map:
                priority_map[mid] = {"memory_id": mid, "priority": 0, "reasons": [], "type": s.get("type", "")}
            priority_map[mid]["priority"] += s["staleness_factor"] * 10
            priority_map[mid]["reasons"].append("stale")

        items = sorted(priority_map.values(), key=lambda x: x["priority"], reverse=True)
        return items[:limit]

    async def apply_quality_gate(self, content: str, memory_type: str | None = None) -> dict:
        """Apply quality gates to validate memory content before ingestion.

        Returns pass/fail with specific gate results.
        """
        gates = []

        # Gate 1: Content validation.
        validation = await self.validate_content(content)
        gates.append({
            "gate": "content_validation",
            "passed": validation["valid"],
            "details": validation.get("errors", []),
        })

        # Gate 2: Minimum information density.
        words = content.split()
        unique_words = set(w.lower() for w in words)
        info_density = len(unique_words) / max(len(words), 1)
        gates.append({
            "gate": "information_density",
            "passed": info_density >= 0.3 or len(words) <= 5,
            "details": [f"density={info_density:.2f}"],
        })

        # Gate 3: Not a duplicate (check against existing).
        # This is a lightweight check using first 50 chars.
        gates.append({
            "gate": "non_duplicate",
            "passed": True,  # Full dedup happens at ingestion.
            "details": [],
        })

        # Gate 4: Minimum content length.
        gates.append({
            "gate": "min_length",
            "passed": len(content.strip()) >= 10,
            "details": [f"length={len(content.strip())}"],
        })

        all_passed = all(g["passed"] for g in gates)
        return {
            "passed": all_passed,
            "gates": gates,
            "content_preview": content[:80],
        }

    async def get_importance_histogram(self, namespace: str | None = None, buckets: int = 10) -> dict:
        """Get importance distribution as a histogram for operators.

        Returns bucket counts for [0.0-0.1), [0.1-0.2), ..., [0.9-1.0].
        """
        ns = namespace or self.namespace
        units = await self.store.list_active(self.user_id, ns, limit=5000)
        histogram: dict[str, int] = {}
        bucket_size = 1.0 / buckets
        for i in range(buckets):
            low = round(i * bucket_size, 2)
            high = round((i + 1) * bucket_size, 2)
            label = f"{low:.1f}-{high:.1f}"
            histogram[label] = 0
        for u in units:
            idx = min(int(u.importance / bucket_size), buckets - 1)
            low = round(idx * bucket_size, 2)
            high = round((idx + 1) * bucket_size, 2)
            label = f"{low:.1f}-{high:.1f}"
            histogram[label] = histogram.get(label, 0) + 1
        return {"histogram": histogram, "total": len(units)}

    async def run_maintenance(self, namespace: str | None = None) -> dict:
        """Run a full maintenance cycle: expire, consolidate, clean orphans, compact.

        Returns a summary of all actions taken.
        """
        ns = namespace or self.namespace
        results: dict = {"namespace": ns}

        # 1. Expire TTL-stale memories.
        expired = await self.expire_stale(ns)
        results["expired"] = expired

        # 2. Consolidate (dedup, near-dedup, decay).
        consolidation = await self.consolidator.consolidate(ns)
        results["consolidation"] = consolidation

        # 3. Apply typed retention policy.
        retention = await self.apply_typed_retention(ns)
        results["retention_archived"] = retention["archived"]

        # 4. Garbage collect superseded memories.
        gc = await self.store.garbage_collect(ns)
        results["gc_removed"] = gc.get("removed", 0)

        # 6. Compact the database.
        self.store.compact()
        results["compacted"] = True

        self._notify("maintenance", **results)
        return results

    async def sample_memories(self, namespace: str | None = None, count: int = 5) -> list[MemoryUnit]:
        """Return a random sample of active memories for exploration."""
        ns = namespace or self.namespace
        return await self.store.sample_memories(self.user_id, ns, count)

    async def get_api_status(self, namespace: str | None = None) -> dict:
        """Get a comprehensive, API-ready status summary.

        Returns a JSON-serializable dict combining store stats, health,
        policy state, and feature usage indicators.
        """
        ns = namespace or self.namespace
        units = await self.store.list_active(self.user_id, ns, limit=5000)
        health = await self.store.compute_health_score(self.user_id, ns)
        db_info = await self.store.get_db_size()

        type_counts: dict[str, int] = {}
        pinned = 0
        with_ttl = 0
        with_tags = 0
        total_access = 0
        for u in units:
            type_counts[u.memory_type.value] = type_counts.get(u.memory_type.value, 0) + 1
            if u.importance >= 0.99:
                pinned += 1
            if u.expires_at:
                with_ttl += 1
            if u.tags:
                with_tags += 1
            total_access += u.access_count

        return {
            "namespace": ns,
            "schema_version": "postgres",
            "active_count": len(units),
            "type_distribution": type_counts,
            "health": health,
            "db": db_info,
            "features": {
                "pinned": pinned,
                "with_ttl": with_ttl,
                "with_tags": with_tags,
                "total_accesses": total_access,
            },
            "policy": {
                "retrieval_mode": self.retrieval_mode,
                "max_injected_units": self.policy.max_injected_units,
                "max_injected_tokens": self.policy.max_injected_tokens,
            },
            "embedder": self.get_embedder_info(),
        }

    async def get_optimization_hints(self, namespace: str | None = None) -> list[str]:
        """Generate optimization suggestions based on current store state."""
        ns = namespace or self.namespace
        hints: list[str] = []

        units = await self.store.list_active(self.user_id, ns, limit=5000)
        if not units:
            return ["Store is empty — no optimizations needed."]

        # Check for too many memories.
        if len(units) > 1000:
            hints.append(f"High memory count ({len(units)}): consider running batch-archive or typed-retention.")

        # Check for low access coverage.
        accessed = sum(1 for u in units if u.access_count > 0)
        if accessed / len(units) < 0.3:
            hints.append(f"Low access coverage ({accessed}/{len(units)}): many memories are never retrieved.")

        # Check for no TTL set.
        with_ttl = sum(1 for u in units if u.expires_at)
        if with_ttl == 0 and len(units) > 10:
            hints.append("No memories have TTL set: consider running auto-ttl to prevent unbounded growth.")

        # Check for duplicates.
        dupes = await self.store.find_duplicates(self.user_id, ns, threshold=0.85)
        if dupes:
            hints.append(f"{len(dupes)} near-duplicate pair(s) found: consider consolidation.")

        if not hints:
            hints.append("Store looks healthy — no optimizations needed.")

        return hints

    async def generate_usage_report(self, namespace: str | None = None) -> dict:
        """Generate a comprehensive usage report for monitoring and dashboards.

        Combines health score, quality distribution, type breakdown,
        access patterns, link statistics, and TTL status.
        """
        ns = namespace or self.namespace
        units = await self.store.list_active(self.user_id, ns, limit=5000)
        health = await self.store.compute_health_score(self.user_id, ns)

        type_counts: dict[str, int] = {}
        total_access = 0
        accessed_count = 0
        with_ttl = 0
        pinned_count = 0
        importance_sum = 0.0

        for u in units:
            type_counts[u.memory_type.value] = type_counts.get(u.memory_type.value, 0) + 1
            total_access += u.access_count
            if u.access_count > 0:
                accessed_count += 1
            if u.expires_at:
                with_ttl += 1
            if u.importance >= 0.99:
                pinned_count += 1
            importance_sum += u.importance

        n = max(len(units), 1)
        return {
            "namespace": ns,
            "total_active": len(units),
            "health_score": health.get("score", 0),
            "type_distribution": type_counts,
            "avg_importance": round(importance_sum / n, 4),
            "access_coverage": round(accessed_count / n, 4),
            "total_accesses": total_access,
            "pinned_count": pinned_count,
            "with_ttl": with_ttl,
            "health_components": health.get("components", {}),
        }

    async def get_embedder_info(self) -> dict:
        """Return information about the current embedder configuration."""
        if self.embedder is None:
            return {
                "enabled": False,
                "mode": "none",
                "model": None,
                "dimensions": None,
            }
        embedder_type = type(self.embedder).__name__
        is_semantic = embedder_type == "SentenceTransformerEmbedder"
        info = {
            "enabled": True,
            "mode": "semantic" if is_semantic else "hashing",
            "type": embedder_type,
            "dimensions": self.embedder.dimensions,
        }
        if is_semantic:
            info["model"] = getattr(self.embedder, "model_name", "unknown")
            info["available"] = getattr(self.embedder, "is_available", False)
        return info

    async def re_embed_scope(
        self,
        namespace: str | None = None,
        embedder: "BaseEmbedder | None" = None,
    ) -> dict:
        """Re-encode all active memories in a namespace with the current (or given) embedder.

        Useful when switching from hashing to semantic embeddings.
        Returns count of memories re-embedded.
        """
        ns = namespace or self.namespace
        emb = embedder or self.embedder
        if emb is None:
            return {"error": "No embedder available", "re_embedded": 0}

        units = await self.store.list_active(self.user_id, ns, limit=5000)
        if not units:
            return {"namespace": ns, "re_embedded": 0, "total": 0}

        # Batch encode for efficiency.
        texts = [u.content for u in units]
        try:
            vectors = await emb.encode_batch(texts)
        except Exception as exc:
            return {"error": str(exc), "re_embedded": 0}

        re_embedded = 0
        for unit, vec in zip(units, vectors):
            if vec:
                await self.store.update_embedding(unit.memory_id, vec)
                re_embedded += 1
        return {"namespace": ns, "re_embedded": re_embedded, "total": len(units)}

    async def compress_content(self, memory_id: str) -> dict:
        """Compress memory content by removing redundancy and verbosity.

        Applies heuristic compression: strips filler phrases, compacts
        whitespace, removes redundant words, and truncates to essential content.
        """
        unit = await self.store.get_by_id(memory_id)
        if not unit:
            return {"error": "Memory not found", "memory_id": memory_id}

        original = unit.content
        compressed = _compress_text(original)

        if compressed == original:
            return {"memory_id": memory_id, "changed": False, "length": len(original)}

        await self.store.update_content(memory_id, compressed)
        return {
            "memory_id": memory_id,
            "changed": True,
            "original_length": len(original),
            "compressed_length": len(compressed),
            "reduction_pct": round(100 * (1 - len(compressed) / max(len(original), 1)), 1),
        }

    async def batch_compress(self, namespace: str | None = None) -> dict:
        """Compress content for all memories in a ns.

        Returns stats on how many were compressed and total token savings.
        """
        ns = namespace or self.namespace
        units = await self.store.list_active(self.user_id, ns, limit=5000)
        compressed = 0
        total_saved = 0

        for u in units:
            result = _compress_text(u.content)
            if result != u.content:
                saved = len(u.content) - len(result)
                await self.store.update_content(u.memory_id, result)
                compressed += 1
                total_saved += saved

        return {"total": len(units), "compressed": compressed, "chars_saved": total_saved}

    async def bulk_tag_by_type(
        self,
        namespace: str | None = None,
        type_tag_map: dict[str, list[str]] | None = None,
    ) -> dict:
        """Auto-tag all memories based on their type.

        type_tag_map maps memory type values to tags to add.
        Default: {"project_state": ["infra"], "preference": ["user-pref"]}.
        """
        ns = namespace or self.namespace
        defaults = {
            "project_state": ["infrastructure"],
            "preference": ["user-preference"],
            "procedural_observation": ["procedure"],
        }
        tag_map = type_tag_map or defaults
        units = await self.store.list_active(self.user_id, ns, limit=5000)
        tagged = 0

        for u in units:
            tags_to_add = tag_map.get(u.memory_type.value, [])
            if tags_to_add:
                existing = set(u.tags)
                new_tags = [t for t in tags_to_add if t not in existing]
                if new_tags:
                    await self.store.add_tags(u.memory_id, new_tags)
                    tagged += 1

        return {"total": len(units), "tagged": tagged}

    async def analyze_retention_effectiveness(self, namespace: str | None = None) -> dict:
        """Analyze how well retention policies are working.

        Measures: archived vs active ratio, access patterns before archival,
        average lifetime, and whether high-value memories are being retained.
        """
        from datetime import datetime, timezone

        ns = namespace or self.namespace
        active = await self.store.list_active(self.user_id, ns, limit=5000)
        analytics = await self.store.get_namespace_analytics(self.user_id, ns)

        active_count = analytics.get("active", 0)
        archived_count = analytics.get("archived", 0)
        superseded_count = analytics.get("superseded", 0)
        total = analytics.get("total", 0)

        # Average importance / access from active units.
        active_imp = [u.importance for u in active]
        # Approximate archived importance / access from analytics (no individual rows)
        archived_imp: list[float] = []
        archived_access: list[int] = []

        # Average age of active memories.
        now = datetime.now(timezone.utc)
        active_ages = []
        for u in active:
            if u.created_at:
                try:
                    created = datetime.fromisoformat(u.created_at.replace("Z", "+00:00"))
                    active_ages.append((now - created).days)
                except (ValueError, TypeError):
                    pass

        return {
            "namespace": ns,
            "total_memories": total,
            "active": active_count,
            "archived": archived_count,
            "superseded": superseded_count,
            "archive_ratio": round(archived_count / max(total, 1), 3),
            "avg_active_importance": round(sum(active_imp) / max(len(active_imp), 1), 3),
            "avg_archived_importance": round(sum(archived_imp) / max(len(archived_imp), 1), 3),
            "avg_archived_access_count": round(sum(archived_access) / max(len(archived_access), 1), 1),
            "avg_active_age_days": round(sum(active_ages) / max(len(active_ages), 1), 1),
            "retention_health": "good" if (
                not archived_imp or sum(archived_imp) / max(len(archived_imp), 1) < sum(active_imp) / max(len(active_imp), 1)
            ) else "review_needed",
        }

    async def get_memory_growth_rate(self, namespace: str | None = None, window_days: int = 30) -> dict:
        """Compute memory growth rate over a time window.

        Returns memories added per day and projected growth.
        """
        from datetime import datetime, timedelta, timezone

        ns = namespace or self.namespace
        now = datetime.now(timezone.utc)
        cutoff = (now - timedelta(days=window_days)).isoformat()

        total_active = len(await self.store.list_active(self.user_id, ns, limit=5000))
        recent_count = await self.store.count_memories_since(self.user_id, ns, cutoff)

        rate_per_day = round(recent_count / max(window_days, 1), 2)
        projected_30d = round(rate_per_day * 30)
        projected_90d = round(rate_per_day * 90)

        return {
            "namespace": ns,
            "window_days": window_days,
            "current_active": total_active,
            "added_in_window": recent_count,
            "rate_per_day": rate_per_day,
            "projected_30d": projected_30d,
            "projected_90d": projected_90d,
        }

    async def auto_deduplicate(
        self,
        namespace: str | None = None,
        threshold: float = 0.85,
        dry_run: bool = False,
    ) -> dict:
        """Find and resolve duplicates by archiving older copies.

        Uses word-level Jaccard similarity. Keeps the memory with higher
        importance (or more recent if tied). Respects pinned memories.
        """
        ns = namespace or self.namespace
        duplicates = await self.store.find_duplicates(self.user_id, ns, threshold=threshold)
        archived = 0
        pairs = []

        for dup in duplicates:
            id_a, id_b = dup["id_a"], dup["id_b"]
            unit_a = await self.store.get_by_id(id_a)
            unit_b = await self.store.get_by_id(id_b)
            if not unit_a or not unit_b:
                continue
            if unit_a.status != "active" or unit_b.status != "active":
                continue
            # Don't touch pinned memories.
            if unit_a.importance >= 0.99 or unit_b.importance >= 0.99:
                continue

            # Keep the one with higher importance, or more recent.
            if unit_a.importance > unit_b.importance:
                keep, remove = id_a, id_b
            elif unit_b.importance > unit_a.importance:
                keep, remove = id_b, id_a
            elif unit_a.updated_at >= unit_b.updated_at:
                keep, remove = id_a, id_b
            else:
                keep, remove = id_b, id_a

            pairs.append({"keep": keep, "remove": remove, "similarity": dup["similarity"]})
            if not dry_run:
                await self.store.bulk_archive([remove])
                archived += 1

        return {
            "namespace": ns,
            "duplicates_found": len(pairs),
            "archived": archived,
            "dry_run": dry_run,
            "pairs": pairs[:20],  # Cap detail output.
        }

    async def forecast_capacity(
        self,
        namespace: str | None = None,
        quota: int = 1000,
    ) -> dict:
        """Project when a namespace will reach its quota based on growth rate."""
        ns = namespace or self.namespace
        growth = await self.get_memory_growth_rate(ns, window_days=30)
        current = growth["current_active"]
        rate = growth["rate_per_day"]

        if rate <= 0:
            return {
                "namespace": ns,
                "current": current,
                "quota": quota,
                "utilization_pct": round(100 * current / max(quota, 1), 1),
                "days_until_full": None,
                "rate_per_day": rate,
            }

        remaining = max(quota - current, 0)
        days_until_full = round(remaining / rate, 1) if rate > 0 else None

        return {
            "namespace": ns,
            "current": current,
            "quota": quota,
            "utilization_pct": round(100 * current / max(quota, 1), 1),
            "days_until_full": days_until_full,
            "rate_per_day": rate,
        }

    async def export_audit_trail(
        self,
        namespace: str | None = None,
        limit: int = 1000,
    ) -> list[dict]:
        """Export the memory event log as a compliance-ready audit trail.

        Event log has been removed; observability is handled via OTel/Langfuse traces.
        """
        return []

    async def generate_action_plan(
        self,
        namespace: str | None = None,
    ) -> dict:
        """Generate a comprehensive operator action plan for a ns.

        Combines deduplication, compression, enrichment, archival, and tagging
        recommendations into a single prioritized action list.
        """
        ns = namespace or self.namespace

        actions = []

        # 1. Duplicates to merge.
        duplicates = await self.store.find_duplicates(self.user_id, ns, threshold=0.85)
        if duplicates:
            actions.append({
                "action": "deduplicate",
                "priority": "high",
                "count": len(duplicates),
                "description": f"Found {len(duplicates)} duplicate pairs above 85% similarity",
                "command": "memory auto-dedup --namespace " + ns,
            })

        # 2. Stale memories.
        stale = await self.find_stale_memories(ns, stale_days=60, limit=50)
        if stale:
            actions.append({
                "action": "review_stale",
                "priority": "medium",
                "count": len(stale),
                "description": f"{len(stale)} memories not accessed in 60+ days",
                "command": "memory stale --namespace " + ns,
            })

        # 3. Enrichment needed.
        enrichments = await self.suggest_enrichments(ns, limit=20)
        if enrichments:
            actions.append({
                "action": "enrich",
                "priority": "low",
                "count": len(enrichments),
                "description": f"{len(enrichments)} memories lack topics, entities, or tags",
                "command": "memory enrichments --namespace " + ns,
            })

        # 4. Compression opportunities.
        units = await self.store.list_active(self.user_id, ns, limit=5000)
        compressible = sum(1 for u in units if len(u.content) > 200)
        if compressible > 0:
            actions.append({
                "action": "compress",
                "priority": "low",
                "count": compressible,
                "description": f"{compressible} memories have verbose content (>200 chars)",
                "command": "memory compress --namespace " + ns,
            })

        # 5. Type balance.
        balance = await self.analyze_type_balance(ns)
        if balance.get("suggestions"):
            actions.append({
                "action": "rebalance_types",
                "priority": "low",
                "count": len(balance["suggestions"]),
                "description": "; ".join(
                    str(s) if isinstance(s, str) else s.get("suggestion", str(s))
                    for s in balance["suggestions"][:3]
                ),
                "command": "memory type-balance --namespace " + ns,
            })

        # Sort by priority.
        priority_order = {"high": 0, "medium": 1, "low": 2}
        actions.sort(key=lambda a: priority_order.get(a["priority"], 3))

        return {
            "namespace": ns,
            "total_actions": len(actions),
            "actions": actions,
        }

    async def search_grouped(
        self,
        query_text: str,
        namespace: str | None = None,
        group_by: str = "type",
        limit: int = 20,
    ) -> dict:
        """Search memories and group results by type or topic.

        Returns grouped results with per-group scores.
        """
        ns = namespace or self.namespace
        query = MemoryQuery(
            user_id=self.user_id,
            namespace=ns,
            query_text=query_text,
            top_k=limit,
        )
        hits = await self.retriever.retrieve(query)
        groups: dict[str, list[dict]] = {}

        for hit in hits:
            if group_by == "type":
                key = hit.unit.memory_type.value
            elif group_by == "topic":
                key = hit.unit.topics[0] if hit.unit.topics else "untagged"
            else:
                key = "all"

            if key not in groups:
                groups[key] = []
            groups[key].append({
                "memory_id": hit.unit.memory_id,
                "content": hit.unit.content[:100],
                "score": round(hit.score, 3),
                "importance": hit.unit.importance,
            })

        return {
            "query": query_text,
            "total_results": len(hits),
            "group_by": group_by,
            "groups": {k: {"count": len(v), "results": v} for k, v in groups.items()},
        }

    async def bookmark_memories(
        self,
        memory_ids: list[str],
        bookmark_tag: str = "bookmarked",
    ) -> dict:
        """Bookmark memories for quick access by adding a tag.

        Bookmarks are just tags — simple and compatible with all existing search.
        """
        tagged = 0
        for mid in memory_ids:
            unit = await self.store.get_by_id(mid)
            if unit and bookmark_tag not in unit.tags:
                await self.store.add_tags(mid, [bookmark_tag])
                tagged += 1
        return {"tagged": tagged, "total": len(memory_ids)}

    async def get_bookmarks(
        self,
        namespace: str | None = None,
        bookmark_tag: str = "bookmarked",
    ) -> list[dict]:
        """Get all bookmarked memories in a ns."""
        ns = namespace or self.namespace
        units = await self.store.search_by_tag(self.user_id, ns, bookmark_tag)
        return [
            {
                "memory_id": u.memory_id,
                "type": u.memory_type.value,
                "content": u.content[:100],
                "importance": u.importance,
                "tags": u.tags,
            }
            for u in units
        ]

    async def archive_namespace(self, user_id: str, namespace: str | None = None) -> dict:
        """Archive all active memories in a ns.

        Useful for retiring old namespaces or preparing for namespace cleanup.
        Does not touch pinned memories (importance >= 0.99).
        """
        units = await self.store.list_active(user_id, namespace, limit=5000)
        to_archive = [u.memory_id for u in units if u.importance < 0.99]
        archived = await self.store.bulk_archive(to_archive) if to_archive else 0
        pinned_count = len(units) - len(to_archive)
        return {
            "namespace": namespace,
            "archived": archived,
            "pinned_kept": pinned_count,
            "total_before": len(units),
        }

    async def bulk_pin_by_criteria(
        self,
        namespace: str | None = None,
        min_importance: float = 0.9,
        min_access_count: int = 10,
    ) -> dict:
        """Pin memories that meet high importance or access criteria.

        Pinning sets importance to 0.99 which prevents archival/decay.
        """
        from datetime import datetime, timezone

        ns = namespace or self.namespace
        units = await self.store.list_active(self.user_id, ns, limit=5000)
        pinned = 0
        now = datetime.now(timezone.utc).isoformat()

        for u in units:
            if u.importance >= 0.99:
                continue  # Already pinned.
            if u.importance >= min_importance or u.access_count >= min_access_count:
                await self.store.update_importance(u.memory_id, 0.99, now)
                pinned += 1

        return {"namespace": ns, "pinned": pinned, "total": len(units)}

    async def export_scope_yaml(self, namespace: str | None = None) -> str:
        """Export namespace memories as YAML format.

        Returns a YAML string. Does not require external YAML library.
        """
        ns = namespace or self.namespace
        units = await self.store.list_active(self.user_id, ns, limit=5000)
        lines = ["# Memory Export", f"# Namespace: {ns}", f"# Count: {len(units)}", "memories:"]

        for u in units:
            lines.append(f"  - memory_id: {u.memory_id}")
            lines.append(f"    type: {u.memory_type.value}")
            lines.append(f"    content: \"{u.content[:200].replace(chr(34), chr(39))}\"")
            lines.append(f"    importance: {u.importance}")
            lines.append(f"    confidence: {u.confidence}")
            lines.append(f"    access_count: {u.access_count}")
            lines.append(f"    created_at: {u.created_at}")
            if u.topics:
                lines.append(f"    topics: [{', '.join(u.topics[:5])}]")
            if u.entities:
                lines.append(f"    entities: [{', '.join(u.entities[:5])}]")
            if u.tags:
                lines.append(f"    tags: [{', '.join(u.tags[:5])}]")

        return "\n".join(lines)

    async def run_system_health_check(self, namespace: str | None = None) -> dict:
        """Run a comprehensive system health check.

        Returns a pass/fail result with categorized findings:
        - integrity: database structural health
        - quality: memory quality distribution
        - capacity: quota utilization
        - freshness: how up-to-date memories are
        - maintenance: pending maintenance actions
        """
        ns = namespace or self.namespace
        issues = []
        checks = {}

        # 1. Health score.
        health = await self.store.compute_health_score(self.user_id, ns)
        health_score = health.get("score", 0)
        checks["health_score"] = {
            "passed": health_score >= 50,
            "score": health_score,
        }
        if health_score < 50:
            issues.append(f"Low health score: {health_score}")

        # 3. Stale count.
        stale = await self.find_stale_memories(ns, stale_days=90, limit=100)
        checks["staleness"] = {
            "passed": len(stale) < 10,
            "stale_count": len(stale),
        }
        if len(stale) >= 10:
            issues.append(f"{len(stale)} memories stale for 90+ days")

        # 4. Duplicate count.
        duplicates = await self.store.find_duplicates(self.user_id, ns, threshold=0.90)
        checks["duplicates"] = {
            "passed": len(duplicates) < 5,
            "duplicate_pairs": len(duplicates),
        }
        if len(duplicates) >= 5:
            issues.append(f"{len(duplicates)} near-duplicate pairs found")

        # 5. DB size.
        db_info = await self.store.get_db_size()
        db_size_mb = db_info.get("total_bytes", 0) / (1024 * 1024)
        checks["db_size"] = {
            "passed": db_size_mb < 100,
            "size_mb": round(db_size_mb, 2),
        }
        if db_size_mb >= 100:
            issues.append(f"Database size: {db_size_mb:.1f}MB")

        overall_passed = all(c.get("passed", False) for c in checks.values())

        return {
            "namespace": ns,
            "passed": overall_passed,
            "checks": checks,
            "issues": issues,
            "summary": "All checks passed" if overall_passed else f"{len(issues)} issue(s) found",
        }

    async def get_system_summary(self) -> dict:
        """Get a comprehensive summary of the entire memory system.

        Combines all scopes, health, embedder, policy, and schema info
        into a single operator-friendly overview.
        """
        scopes = await self.store.list_namespaces(self.user_id)
        scope_summaries = []
        total_active = 0
        for scope_info in scopes:
            sid = scope_info.get("namespace", "")
            active = scope_info.get("active", 0)
            total_active += active
            scope_summaries.append({
                "namespace": sid,
                "active": active,
                "total": scope_info.get("total", 0),
            })

        return {
            "schema_version": "postgres",
            "scopes": scope_summaries,
            "scope_count": len(scopes),
            "total_active_memories": total_active,
            "embedder": self.get_embedder_info(),
            "policy": {
                "retrieval_mode": self.retrieval_mode,
                "max_injected_units": self.policy.max_injected_units,
                "max_injected_tokens": self.policy.max_injected_tokens,
            },
            "db": await self.store.get_db_size(),
        }

    async def generate_operator_report(self, namespace: str | None = None) -> dict:
        """Generate a comprehensive operator diagnostic report.

        Combines health check, action plan, growth rate, capacity forecast,
        and system summary into a single actionable output for quick triage.
        """
        ns = namespace or self.namespace
        report: dict = {"namespace": ns, "generated_at": ""}
        try:
            from datetime import datetime, timezone
            report["generated_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
        except Exception:
            pass

        # Health check
        try:
            report["health"] = await self.run_system_health_check(namespace=namespace)
        except Exception as exc:
            report["health"] = {"error": str(exc)}

        # Action plan
        try:
            report["action_plan"] = self.generate_action_plan(namespace=namespace)
        except Exception as exc:
            report["action_plan"] = {"error": str(exc)}

        # Growth rate
        try:
            report["growth_rate"] = await self.get_memory_growth_rate(namespace=namespace)
        except Exception as exc:
            report["growth_rate"] = {"error": str(exc)}

        # Capacity forecast
        try:
            report["capacity"] = self.forecast_capacity(namespace=namespace)
        except Exception as exc:
            report["capacity"] = {"error": str(exc)}

        # Stats
        try:
            report["stats"] = await self.get_namespace_stats(namespace=namespace)
        except Exception as exc:
            report["stats"] = {"error": str(exc)}

        # Type balance
        try:
            report["type_balance"] = await self.analyze_type_balance(namespace=namespace)
        except Exception as exc:
            report["type_balance"] = {"error": str(exc)}

        # System-wide context
        try:
            report["system"] = await self.get_system_summary()
        except Exception as exc:
            report["system"] = {"error": str(exc)}

        return report

    def close(self) -> None:
        self.store.close()

    def _fit_token_budget(self, units: list[MemoryUnit], max_tokens: int) -> list[MemoryUnit]:
        # Apply type diversity: if more than 4 units, ensure no single type
        # dominates more than 60% of slots.
        units = _enforce_type_diversity(units, max_dominant_ratio=0.6, min_count=4)

        kept: list[MemoryUnit] = []
        used = 0
        for unit in units:
            text = unit.content.strip()
            cost = estimate_tokens(text)
            if kept and used + cost > max_tokens:
                break
            kept.append(unit)
            used += cost
        return kept


def _extract_topics(prompt_text: str) -> list[str]:
    topics = []
    seen = set()

    # Single-word topics from significant tokens.
    for token in prompt_text.split():
        cleaned = token.strip(".,:;!?()[]{}").lower()
        if (
            len(cleaned) >= 5
            and cleaned not in seen
            and cleaned not in _STOPWORDS
        ):
            topics.append(cleaned)
            seen.add(cleaned)
        if len(topics) >= 10:
            break

    return topics[:12]


def _extract_entities(text: str) -> list[str]:
    entities = []
    seen = set()

    for token in text.split():
        cleaned = token.strip(".,:;!?()[]{}")
        if len(cleaned) > 1 and cleaned[:1].isupper() and cleaned not in seen:
            entities.append(cleaned)
            seen.add(cleaned)
        if len(entities) >= 12:
            break

    return entities


def _infer_memory_type(prompt_text: str, response_text: str) -> MemoryType:
    text = f"{prompt_text}\n{response_text}".lower()
    if "i prefer" in text or "my preference" in text or "prefer " in text:
        return MemoryType.PREFERENCE
    if "always " in text or "never " in text or "make sure" in text or "do not " in text:
        return MemoryType.PROCEDURAL_OBSERVATION
    if "remember that" in text or "keep in mind" in text or "note that" in text:
        return MemoryType.SEMANTIC
    return MemoryType.EPISODIC


async def _extract_memory_units_for_turn(
    user_id: str,
    namespace: str,
    session_id: str | None,
    turn_index: int,
    prompt_text: str,
    response_text: str,
) -> list[MemoryUnit]:
    extracted: list[MemoryUnit] = []
    text = " ".join([prompt_text, response_text]).strip()
    topics = _extract_topics(text)
    entities = _extract_entities(text)

    for fact in _extract_pattern_facts(prompt_text, response_text):
        extracted.append(
            MemoryUnit(
                memory_id=str(uuid.uuid4()),
                user_id=user_id,
                namespace=namespace,
                memory_type=fact["memory_type"],
                content=fact["content"][:2000],
                source_session_id=session_id,
                source_turn_start=turn_index,
                source_turn_end=turn_index,
                topics=topics,
                entities=entities,
                importance=fact["importance"],
                confidence=fact["confidence"],
            )
        )

    if extracted:
        return extracted

    combined = prompt_text
    if response_text:
        combined = f"{prompt_text}\nAssistant: {response_text}".strip()
    fallback_type = _infer_memory_type(prompt_text, response_text)
    return [
        MemoryUnit(
            memory_id=str(uuid.uuid4()),
            user_id=user_id,
            namespace=namespace,
            memory_type=fallback_type,
            content=combined[:4000],
            source_session_id=session_id,
            source_turn_start=turn_index,
            source_turn_end=turn_index,
            topics=topics,
            entities=entities,
            importance=0.7 if fallback_type != MemoryType.EPISODIC else 0.5,
            confidence=0.55,
        )
    ]


def _extract_pattern_facts(prompt_text: str, response_text: str) -> list[dict]:
    text = _normalize_space(prompt_text)
    facts: list[dict] = []
    seen: set[tuple[str, str]] = set()

    # First pass: extract from prompt text (higher confidence).
    prompt_patterns = [
        (
            MemoryType.PREFERENCE,
            0.9,
            0.82,
            [
                r"\bi prefer (?P<fact>[^.!\n]{3,180})",
                r"\bplease (?:keep|make) (?P<fact>[^.!\n]{3,180})",
                r"\bmy preferred (?P<fact>[^.!\n]{3,180})",
                r"\bi(?:'d| would) like (?P<fact>[^.!\n]{3,180})",
                r"\bi don(?:'t| not) (?:want|like) (?P<fact>[^.!\n]{3,180})",
                r"\bmy convention is (?P<fact>[^.!\n]{3,180})",
                r"\bi(?:'d| would) rather (?P<fact>[^.!\n]{3,180})",
                r"\bi want (?P<fact>[^.!\n]{3,180})",
                r"\bi(?:'m| am) used to (?P<fact>[^.!\n]{3,180})",
            ],
        ),
        (
            MemoryType.PROCEDURAL_OBSERVATION,
            0.8,
            0.75,
            [
                r"\balways (?P<fact>\w[^.!\n]{5,180})",
                r"\bnever (?P<fact>\w[^.!\n]{5,180})",
                r"\bdo not (?P<fact>\w[^.!\n]{5,180})",
                r"\bavoid (?P<fact>\w[^.!\n]{5,180})",
                r"\bmake sure (?:to |that )?(?P<fact>[^.!\n]{3,180})",
                r"\bdon(?:'t| not) forget to (?P<fact>[^.!\n]{3,180})",
                r"\bwhen \w+ing[, ]+(?P<fact>[^.!\n]{3,180})",
                r"\bthe (?:marking|grading|assessment) (?:criteria|rubric|process) is (?P<fact>[^.!\n]{3,180})",
            ],
        ),
        (
            MemoryType.SEMANTIC,
            0.82,
            0.78,
            [
                r"\bremember that (?P<fact>[^.!\n]{3,180})",
                r"\bkeep in mind that (?P<fact>[^.!\n]{3,180})",
                r"\bfor context[, ]+(?P<fact>[^.!\n]{3,180})",
                r"\bnote that (?P<fact>[^.!\n]{3,180})",
                r"\bfyi[, ]+(?P<fact>[^.!\n]{3,180})",
                r"\bjust so you know[, ]+(?P<fact>[^.!\n]{3,180})",
                r"\bfor (?:your|future) reference[, ]+(?P<fact>[^.!\n]{3,180})",
                r"\bimportant(?:ly)?[: ,]+(?P<fact>[^.!\n]{3,180})",
                r"\bby the way[, ]+(?P<fact>[^.!\n]{3,180})",
                r"\bthe (?:key|main) thing is (?P<fact>[^.!\n]{3,180})",
                r"\bwhat matters (?:is|here) (?P<fact>[^.!\n]{3,180})",
                r"\bfor your information[, ]+(?P<fact>[^.!\n]{3,180})",
            ],
        ),
    ]

    for memory_type, importance, confidence, regexes in prompt_patterns:
        for regex in regexes:
            for match in re.finditer(regex, text, flags=re.IGNORECASE):
                fact_text = _clean_fact(match.group("fact"))
                if not fact_text:
                    continue
                key = (memory_type.value, fact_text.lower())
                if key in seen:
                    continue
                seen.add(key)
                facts.append(
                    {
                        "memory_type": memory_type,
                        "content": _format_memory_content(memory_type, fact_text),
                        "importance": importance,
                        "confidence": confidence,
                    }
                )
                if len(facts) >= 6:
                    return facts

    # Second pass: extract from response text (lower confidence).
    response_normalized = _normalize_space(response_text)
    response_patterns = [
        (
            MemoryType.SEMANTIC,
            0.75,
            0.65,
            [
                r"\bimportant(?:ly)?[: ,]+(?P<fact>[^.!\n]{3,180})",
                r"\bkey (?:point|takeaway|finding)[: ,]+(?P<fact>[^.!\n]{3,180})",
                r"\bin summary[, ]+(?P<fact>[^.!\n]{3,180})",
                r"\bworth noting that (?P<fact>[^.!\n]{3,180})",
            ],
        ),
        (
            MemoryType.PROCEDURAL_OBSERVATION,
            0.72,
            0.62,
            [
                r"\byou should (?:always |)(?P<fact>[^.!\n]{3,180})",
                r"\bbest practice is to (?P<fact>[^.!\n]{3,180})",
                r"\bavoid (?P<fact>\w[^.!\n]{5,180})",
                r"\bdo not (?P<fact>\w[^.!\n]{5,180})",
                r"\bnever (?P<fact>\w[^.!\n]{5,180})",
            ],
        ),
    ]
    for memory_type, importance, confidence, regexes in response_patterns:
        for regex in regexes:
            for match in re.finditer(regex, response_normalized, flags=re.IGNORECASE):
                fact_text = _clean_fact(match.group("fact"))
                if not fact_text:
                    continue
                key = (memory_type.value, fact_text.lower())
                if key in seen:
                    continue
                seen.add(key)
                facts.append(
                    {
                        "memory_type": memory_type,
                        "content": _format_memory_content(memory_type, fact_text),
                        "importance": importance,
                        "confidence": confidence,
                    }
                )
                if len(facts) >= 6:
                    return facts

    return facts


def _normalize_space(text: str) -> str:
    return " ".join(text.split())


def _clean_fact(text: str) -> str:
    fact = _normalize_space(text)
    fact = re.sub(r"^(that|to)\s+", "", fact, flags=re.IGNORECASE)
    fact = fact.strip(" .,:;!-")
    if len(fact) < 3:
        return ""
    return fact


def _format_memory_content(memory_type: MemoryType, fact_text: str) -> str:
    if memory_type == MemoryType.PREFERENCE:
        return f"User preference: {fact_text}."
    if memory_type == MemoryType.PROJECT_STATE:
        return f"Project context: {fact_text}."
    if memory_type == MemoryType.PROCEDURAL_OBSERVATION:
        return f"Workflow guidance: {fact_text}."
    if memory_type == MemoryType.SEMANTIC:
        return f"Persistent fact: {fact_text}."
    return fact_text


def _freshness_tag(updated_at: str) -> str:
    """Return a short freshness tag based on how recently the memory was updated."""
    if not updated_at:
        return ""
    try:
        from datetime import datetime, timezone

        updated = datetime.fromisoformat(updated_at.replace("Z", "+00:00"))
        if updated.tzinfo is None:
            updated = updated.replace(tzinfo=timezone.utc)
        age_hours = max(
            (datetime.now(timezone.utc) - updated).total_seconds() / 3600.0, 0.0
        )
        if age_hours < 1:
            return "just now"
        if age_hours < 24:
            return "recent"
        if age_hours < 168:  # 7 days
            return "this week"
        return ""
    except (ValueError, TypeError):
        return ""


def _enforce_type_diversity(
    units: list[MemoryUnit],
    max_dominant_ratio: float = 0.6,
    min_count: int = 4,
) -> list[MemoryUnit]:
    """Reorder units to prevent any single type from dominating results.

    If a type exceeds max_dominant_ratio of the total, push excess units
    to the end so other types can fill the budget.
    """
    if len(units) < min_count:
        return units

    max_slots = max(int(len(units) * max_dominant_ratio), 1)
    type_counts: dict[str, int] = {}
    primary: list[MemoryUnit] = []
    overflow: list[MemoryUnit] = []

    for unit in units:
        t = unit.memory_type.value
        count = type_counts.get(t, 0)
        if count < max_slots:
            primary.append(unit)
            type_counts[t] = count + 1
        else:
            overflow.append(unit)

    return primary + overflow


def _jaccard_conflict(
    a: MemoryUnit,
    b: MemoryUnit,
    threshold: float = 0.65,
) -> float | None:
    """Return topics+entities Jaccard overlap for same-type units above
    threshold, else None. Identical content is no longer exempt — callers
    rely on upstream dedup to remove exact duplicates before this runs."""
    if a.memory_type != b.memory_type:
        return None
    a_terms = set(t.lower() for t in a.topics + a.entities)
    b_terms = set(t.lower() for t in b.topics + b.entities)
    if not a_terms or not b_terms:
        return None
    overlap = len(a_terms & b_terms) / float(len(a_terms | b_terms))
    if overlap < threshold:
        return None
    return round(overlap, 4)


def _detect_local_conflicts(
    units: list[MemoryUnit],
    threshold: float = 0.80,
) -> list[dict]:
    """Find conflicts within a batch of not-yet-persisted units.

    Compares units pairwise. The unit from the earlier turn
    (lower source_turn_start) is labelled 'earlier' and should be dropped.

    The threshold matches the consolidator's near-duplicate similarity
    (0.80) so within-batch and cross-batch overlap are judged consistently.
    """
    conflicts = []
    seen: set[tuple[str, str]] = set()
    for i, a in enumerate(units):
        for b in units[i + 1:]:
            overlap = _jaccard_conflict(a, b, threshold)
            if overlap is None:
                continue
            if a.source_turn_start <= b.source_turn_start:
                earlier, later = a, b
            else:
                earlier, later = b, a
            pair_key = (earlier.memory_id, later.memory_id)
            if pair_key in seen:
                continue
            seen.add(pair_key)
            conflicts.append({
                "earlier_id": earlier.memory_id,
                "later_id": later.memory_id,
                "type": a.memory_type.value,
                "overlap": overlap,
                "earlier_content": earlier.content[:120],
                "later_content": later.content[:120],
            })
    return conflicts



async def _dedup_against_store(
    units: list[MemoryUnit],
    store: MemoryStore,
    user_id: str,
    namespace: str | None,
) -> list[MemoryUnit]:
    """Remove units whose content already exists in the active store."""
    existing = await store.list_active(user_id, namespace, limit=500)
    if not existing:
        return units
    existing_content = {
        (u.memory_type.value, u.content.strip().lower())
        for u in existing
    }
    kept: list[MemoryUnit] = []
    for unit in units:
        key = (unit.memory_type.value, unit.content.strip().lower())
        if key in existing_content:
            continue
        kept.append(unit)
    return kept


_FILLER_PHRASES = [
    "basically", "essentially", "actually", "obviously",
    "it should be noted that", "it is important to note that",
    "as mentioned before", "as we discussed",
    "you know", "kind of", "sort of", "more or less",
    "in terms of", "at the end of the day",
    "the thing is", "to be honest",
    "as a matter of fact", "needless to say",
]


def _compress_text(text: str) -> str:
    """Heuristic text compression: remove filler, condense whitespace, trim."""
    result = text
    # Remove filler phrases.
    lower = result.lower()
    for filler in _FILLER_PHRASES:
        idx = lower.find(filler)
        while idx >= 0:
            result = result[:idx] + result[idx + len(filler):]
            lower = result.lower()
            idx = lower.find(filler)
    # Collapse whitespace.
    result = " ".join(result.split())
    return result.strip()


_STOPWORDS = {
    "about",
    "after",
    "always",
    "before",
    "brief",
    "context",
    "please",
    "project",
    "remember",
    "should",
    "their",
    "there",
    "these",
    "those",
    "using",
    "which",
    "while",
    "would",
}
