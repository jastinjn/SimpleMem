from __future__ import annotations

import json
import logging
import math
from typing import Iterable

from sqlalchemy import delete, func, select, text, update
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from .models import MemorySearchHit, MemoryStatus, MemoryType, MemoryUnit
from .models import utc_now_iso as _utc_now_iso
from .schema import (
    Memory,
    MemoryAnnotation,
    MemoryEvent,
    MemoryLink,
    MemoryWatch,
    ScopeAccess,
    StatsSnapshot,
)

logger = logging.getLogger(__name__)


class MemoryStore:
    """PostgreSQL-backed async store for long-term memory units."""

    def __init__(self, sm: async_sessionmaker[AsyncSession], enable_event_log: bool = True):
        self._sm = sm
        self._enable_event_log = enable_event_log

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _log_event(self, s: AsyncSession, event_type: str, memory_id: str, scope_id: str = "", detail: str = "") -> None:
        if not self._enable_event_log:
            return
        try:
            s.add(MemoryEvent(
                timestamp=_utc_now_iso(),
                event_type=event_type,
                memory_id=memory_id,
                scope_id=scope_id,
                detail=detail[:500],
            ))
        except Exception:
            pass

    def _where_user_scope(self, user_id: str, scope_id: str | None):
        if not user_id:
            raise ValueError("user_id is required")
        if scope_id is not None:
            return (Memory.user_id == user_id) & (Memory.scope_id == scope_id)
        return Memory.user_id == user_id

    # ------------------------------------------------------------------
    # Core CRUD
    # ------------------------------------------------------------------

    async def add_memories(self, units: Iterable[MemoryUnit]) -> int:
        count = 0
        async with self._sm() as s:
            for unit in units:
                vals = dict(
                    memory_id=unit.memory_id,
                    user_id=unit.user_id,
                    scope_id=unit.scope_id,
                    memory_type=unit.memory_type.value,
                    content=unit.content,
                    source_session_id=unit.source_session_id,
                    source_turn_start=unit.source_turn_start,
                    source_turn_end=unit.source_turn_end,
                    entities=unit.entities,
                    topics=unit.topics,
                    importance=unit.importance,
                    confidence=unit.confidence,
                    access_count=unit.access_count,
                    reinforcement_score=unit.reinforcement_score,
                    status=unit.status.value,
                    supersedes=unit.supersedes,
                    superseded_by=unit.superseded_by,
                    embedding=unit.embedding if unit.embedding else None,
                    created_at=unit.created_at,
                    updated_at=unit.updated_at,
                    last_accessed_at=unit.last_accessed_at,
                    expires_at=unit.expires_at,
                    tags=unit.tags,
                )
                stmt = pg_insert(Memory).values(**vals).on_conflict_do_update(
                    index_elements=["memory_id"],
                    set_={k: vals[k] for k in vals if k != "memory_id"},
                )
                await s.execute(stmt)
                await self._log_event(s, "create", unit.memory_id, unit.scope_id or "", f"type={unit.memory_type.value}")
                count += 1
            await s.commit()
        return count

    async def get_by_id(self, memory_id: str) -> MemoryUnit | None:
        async with self._sm() as s:
            row = await s.get(Memory, memory_id)
            return row.to_unit() if row else None

    async def get_by_ids(self, memory_ids: list[str]) -> list[MemoryUnit]:
        if not memory_ids:
            return []
        async with self._sm() as s:
            result = await s.execute(select(Memory).where(Memory.memory_id.in_(memory_ids)))
            return [r.to_unit() for r in result.scalars()]

    async def list_active(self, user_id: str, scope_id: str | None = None, limit: int = 100) -> list[MemoryUnit]:
        async with self._sm() as s:
            cond = self._where_user_scope(user_id, scope_id) & (Memory.status == MemoryStatus.ACTIVE.value)
            result = await s.execute(
                select(Memory).where(cond).order_by(Memory.updated_at.desc()).limit(limit)
            )
            units = [r.to_unit() for r in result.scalars()]
        now_iso = _utc_now_iso()
        return [u for u in units if not u.expires_at or u.expires_at > now_iso]

    async def update_content(self, memory_id: str, content: str) -> bool:
        async with self._sm() as s:
            row = await s.get(Memory, memory_id)
            if row is None:
                return False
            row.content = content
            row.updated_at = _utc_now_iso()
            await s.commit()
        return True

    async def update_importance(self, memory_id: str, importance: float, updated_at: str) -> None:
        async with self._sm() as s:
            await s.execute(
                update(Memory).where(Memory.memory_id == memory_id)
                .values(importance=importance, updated_at=updated_at)
            )
            await s.commit()

    async def update_reinforcement(self, memory_id: str, reinforcement_score: float, updated_at: str) -> None:
        async with self._sm() as s:
            await s.execute(
                update(Memory).where(Memory.memory_id == memory_id)
                .values(reinforcement_score=reinforcement_score, updated_at=updated_at)
            )
            await s.commit()

    async def supersede(self, memory_id: str, superseded_by: str, updated_at: str) -> None:
        async with self._sm() as s:
            await s.execute(
                update(Memory).where(Memory.memory_id == memory_id)
                .values(status=MemoryStatus.SUPERSEDED.value, superseded_by=superseded_by, updated_at=updated_at)
            )
            await self._log_event(s, "supersede", memory_id, detail=f"by={superseded_by[:36]}")
            await s.commit()

    async def mark_accessed(self, memory_ids: Iterable[str], accessed_at: str) -> None:
        ids = list(memory_ids)
        if not ids:
            return
        async with self._sm() as s:
            await s.execute(
                update(Memory).where(Memory.memory_id.in_(ids))
                .values(
                    access_count=Memory.access_count + 1,
                    last_accessed_at=accessed_at,
                    updated_at=accessed_at,
                )
            )
            await s.commit()

    async def bulk_archive(self, memory_ids: list[str]) -> int:
        if not memory_ids:
            return 0
        now = _utc_now_iso()
        async with self._sm() as s:
            result = await s.execute(
                select(Memory.memory_id, Memory.scope_id)
                .where(Memory.memory_id.in_(memory_ids))
                .where(Memory.status == MemoryStatus.ACTIVE.value)
            )
            rows = result.all()
            if not rows:
                return 0
            active_ids = [r[0] for r in rows]
            await s.execute(
                update(Memory).where(Memory.memory_id.in_(active_ids))
                .values(status=MemoryStatus.ARCHIVED.value, updated_at=now)
            )
            for mid, scope in rows:
                await self._log_event(s, "archive", mid, scope or "")
            await s.commit()
        return len(active_ids)

    async def set_ttl(self, memory_id: str, expires_at: str) -> bool:
        async with self._sm() as s:
            row = await s.get(Memory, memory_id)
            if row is None:
                return False
            row.expires_at = expires_at
            row.updated_at = _utc_now_iso()
            await s.commit()
        return True

    async def expire_stale(self, user_id: str, scope_id: str) -> int:
        now_iso = _utc_now_iso()
        async with self._sm() as s:
            result = await s.execute(
                select(Memory.memory_id).where(
                    (Memory.user_id == user_id)
                    & (Memory.scope_id == scope_id)
                    & (Memory.status == MemoryStatus.ACTIVE.value)
                    & (Memory.expires_at != "")
                    & (Memory.expires_at <= now_iso)
                )
            )
            ids = [r[0] for r in result.all()]
            if not ids:
                return 0
            await s.execute(
                update(Memory).where(Memory.memory_id.in_(ids))
                .values(status=MemoryStatus.ARCHIVED.value, updated_at=now_iso)
            )
            await s.commit()
        return len(ids)

    async def pin_memory(self, memory_id: str) -> bool:
        async with self._sm() as s:
            row = await s.get(Memory, memory_id)
            if row is None:
                return False
            row.importance = 0.99
            row.updated_at = _utc_now_iso()
            await self._log_event(s, "pin", memory_id)
            await s.commit()
        return True

    async def unpin_memory(self, memory_id: str, restore_importance: float = 0.7) -> bool:
        async with self._sm() as s:
            row = await s.get(Memory, memory_id)
            if row is None:
                return False
            if float(row.importance) >= 0.99:
                row.importance = restore_importance
                row.updated_at = _utc_now_iso()
                await s.commit()
        return True

    async def record_feedback(self, memory_id: str, helpful: bool) -> None:
        async with self._sm() as s:
            row = await s.get(Memory, memory_id)
            if row is None:
                return
            current = float(row.importance)
            scope = row.scope_id or ""
            if helpful:
                new_importance = min(0.95, current + 0.03)
            else:
                new_importance = max(0.1, current - 0.05)
            row.importance = round(new_importance, 4)
            row.updated_at = _utc_now_iso()
            direction = "positive" if helpful else "negative"
            await self._log_event(s, "feedback", memory_id, scope_id=scope, detail=f"{direction}: {current:.4f} -> {new_importance:.4f}")
            await s.commit()

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    async def search_keyword(self, user_id: str, scope_id: str | None, query_text: str, limit: int = 6) -> list[MemorySearchHit]:
        terms = [t.lower() for t in _tokenize(query_text) if t]
        if not terms:
            return []
        # Build OR query for websearch_to_tsquery
        ts_query = " OR ".join(terms[:12])
        async with self._sm() as s:
            cond = (
                self._where_user_scope(user_id, scope_id)
                & (Memory.status == MemoryStatus.ACTIVE.value)
                & text("content_tsv @@ websearch_to_tsquery('english', :q)").bindparams(q=ts_query)
            )
            result = await s.execute(
                select(Memory).where(cond).limit(limit * 5)
            )
            units = [r.to_unit() for r in result.scalars()]
        if not units:
            return self._search_keyword_manual_sync(
                await self._list_active_sync(user_id, scope_id, limit=500), terms, limit
            )
        return self._rank_with_idf(units, terms, limit)

    async def _list_active_sync(self, user_id: str, scope_id: str | None, limit: int) -> list[MemoryUnit]:
        return await self.list_active(user_id, scope_id, limit=limit)

    def _search_keyword_manual_sync(self, units: list[MemoryUnit], terms: list[str], limit: int) -> list[MemorySearchHit]:
        if not units:
            return []
        doc_freq: dict[str, int] = {}
        haystacks: list[str] = []
        for unit in units:
            haystack = " ".join([
                unit.content.lower(),
                " ".join(x.lower() for x in unit.entities),
                " ".join(x.lower() for x in unit.topics),
            ])
            haystacks.append(haystack)
            for term in set(terms):
                if term in haystack:
                    doc_freq[term] = doc_freq.get(term, 0) + 1
        num_docs = float(len(units))
        hits: list[MemorySearchHit] = []
        for idx, unit in enumerate(units):
            haystack = haystacks[idx]
            matched = [t for t in terms if t in haystack]
            if not matched:
                continue
            idf_score = sum(_log2(num_docs / float(doc_freq.get(t, 1))) for t in matched)
            score = idf_score + unit.importance + unit.reinforcement_score
            hits.append(MemorySearchHit(unit=unit, score=score, matched_terms=matched))
        hits.sort(key=lambda h: (h.score, h.unit.updated_at), reverse=True)
        return hits[:limit]

    def _rank_with_idf(self, units: list[MemoryUnit], terms: list[str], limit: int) -> list[MemorySearchHit]:
        return self._search_keyword_manual_sync(units, terms, limit)

    async def search_vector(self, user_id: str, scope_id: str | None, query_embedding: list[float], limit: int = 10) -> list[MemoryUnit]:
        if not query_embedding:
            return []
        vec_str = "[" + ",".join(str(v) for v in query_embedding) + "]"
        async with self._sm() as s:
            cond = (
                self._where_user_scope(user_id, scope_id)
                & (Memory.status == MemoryStatus.ACTIVE.value)
                & Memory.embedding.isnot(None)
            )
            result = await s.execute(
                select(Memory).where(cond)
                .order_by(text(f"embedding <=> '{vec_str}'::vector"))
                .limit(limit)
            )
            return [r.to_unit() for r in result.scalars()]

    async def search_by_tag(self, user_id: str, scope_id: str | None = None, tag: str = "", limit: int = 50) -> list[MemoryUnit]:
        units = await self.list_active(user_id, scope_id, limit=limit)
        tag_lower = tag.lower().strip()
        return [u for u in units if tag_lower in {t.lower() for t in u.tags}]

    async def search_advanced(
        self,
        user_id: str,
        scope_id: str | None = None,
        keyword: str = "",
        memory_type: str = "",
        tag: str = "",
        min_importance: float = 0.0,
        limit: int = 50,
    ) -> list[MemoryUnit]:
        units = await self.list_active(user_id, scope_id, limit=limit * 5)
        results = []
        keyword_lower = keyword.lower() if keyword else ""
        tag_lower = tag.lower().strip() if tag else ""
        for u in units:
            if keyword_lower and keyword_lower not in u.content.lower():
                continue
            if memory_type and u.memory_type.value != memory_type:
                continue
            if tag_lower and tag_lower not in {t.lower() for t in u.tags}:
                continue
            if u.importance < min_importance:
                continue
            results.append(u)
            if len(results) >= limit:
                break
        return results

    # ------------------------------------------------------------------
    # Stats / analytics
    # ------------------------------------------------------------------

    async def get_stats(self, user_id: str, scope_id: str | None = None) -> dict:
        async with self._sm() as s:
            cond = self._where_user_scope(user_id, scope_id)
            total_row = (await s.execute(select(func.count()).where(cond))).scalar() or 0
            active_row = (await s.execute(
                select(func.count()).where(cond & (Memory.status == MemoryStatus.ACTIVE.value))
            )).scalar() or 0
            type_rows = (await s.execute(
                select(Memory.memory_type, func.count().label("cnt"))
                .where(cond & (Memory.status == MemoryStatus.ACTIVE.value))
                .group_by(Memory.memory_type)
                .order_by(func.count().desc(), Memory.memory_type.asc())
            )).all()
        return {
            "total": int(total_row),
            "active": int(active_row),
            "active_by_type": {str(r[0]): int(r[1]) for r in type_rows},
        }

    async def list_scopes(self, user_id: str) -> list[dict]:
        async with self._sm() as s:
            rows2 = (await s.execute(
                select(Memory.scope_id, func.count().label("total"))
                .where(Memory.user_id == user_id)
                .group_by(Memory.scope_id).order_by(func.count().desc())
            )).all()
            active_rows = (await s.execute(
                select(Memory.scope_id, func.count().label("active"))
                .where((Memory.user_id == user_id) & (Memory.status == MemoryStatus.ACTIVE.value))
                .group_by(Memory.scope_id)
            )).all()
        active_map = {r[0]: int(r[1]) for r in active_rows}
        return [{"scope_id": r[0], "total": int(r[1]), "active": active_map.get(r[0], 0)} for r in rows2]

    async def compute_health_score(self, user_id: str, scope_id: str | None = None) -> dict:
        from datetime import datetime as _dt
        from datetime import timezone as _tz
        units = await self.list_active(user_id, scope_id, limit=5000)
        if not units:
            return {"score": 0, "components": {}, "active_count": 0}
        now = _dt.now(_tz.utc)
        n = float(len(units))
        accessed = sum(1 for u in units if u.access_count > 0)
        access_score = min(25, 25 * (accessed / n))
        avg_imp = sum(u.importance for u in units) / n
        imp_center_dist = abs(avg_imp - 0.55)
        importance_score = max(0, 25 * (1 - imp_center_dist / 0.45))
        types_present = len({u.memory_type.value for u in units})
        type_diversity_score = min(25, 25 * types_present / 4.0)
        fresh_count = 0
        for u in units:
            try:
                updated = _dt.fromisoformat(u.updated_at.replace("Z", "+00:00"))
                if updated.tzinfo is None:
                    updated = updated.replace(tzinfo=_tz.utc)
                if (now - updated).total_seconds() / 86400.0 < 30:
                    fresh_count += 1
            except (ValueError, TypeError):
                pass
        freshness_score = min(25, 25 * (fresh_count / n))
        total = round(access_score + importance_score + type_diversity_score + freshness_score, 1)
        return {
            "score": total,
            "components": {
                "access_coverage": round(access_score, 1),
                "importance_health": round(importance_score, 1),
                "type_diversity": round(type_diversity_score, 1),
                "freshness": round(freshness_score, 1),
            },
            "active_count": len(units),
        }

    async def find_duplicates(self, user_id: str, scope_id: str | None = None, threshold: float = 0.80) -> list[dict]:
        units = await self.list_active(user_id, scope_id, limit=500)
        if len(units) < 2:
            return []
        word_sets = [set(u.content.lower().split()) for u in units]
        duplicates: list[dict] = []
        for i in range(len(units)):
            if not word_sets[i]:
                continue
            for j in range(i + 1, len(units)):
                if not word_sets[j]:
                    continue
                intersection = len(word_sets[i] & word_sets[j])
                union = len(word_sets[i] | word_sets[j])
                if union == 0:
                    continue
                sim = intersection / float(union)
                if sim >= threshold:
                    duplicates.append({
                        "id_a": units[i].memory_id, "id_b": units[j].memory_id,
                        "similarity": round(sim, 4), "type": units[i].memory_type.value,
                        "content_a": units[i].content[:100], "content_b": units[j].content[:100],
                    })
        duplicates.sort(key=lambda d: d["similarity"], reverse=True)
        return duplicates

    async def count_memories_since(self, user_id: str, scope_id: str | None, since_iso: str) -> int:
        async with self._sm() as s:
            cond = self._where_user_scope(user_id, scope_id) & (Memory.created_at >= since_iso)
            return (await s.execute(select(func.count()).where(cond))).scalar() or 0

    async def get_db_size(self) -> dict:
        async with self._sm() as s:
            try:
                row = (await s.execute(text(
                    "SELECT pg_database_size(current_database()) AS db_bytes"
                ))).one()
                size_bytes = int(row[0])
            except Exception:
                size_bytes = 0
        return {
            "size_bytes": size_bytes,
            "size_mb": round(size_bytes / (1024 * 1024), 2),
        }

    def compact(self) -> None:
        pass  # Aurora/PG autovacuums; no-op here.

    # ------------------------------------------------------------------
    # Event log
    # ------------------------------------------------------------------

    async def get_event_log(self, scope_id: str = "", limit: int = 50) -> list[dict]:
        async with self._sm() as s:
            q = select(MemoryEvent).order_by(MemoryEvent.event_id.desc()).limit(limit)
            if scope_id:
                q = q.where(MemoryEvent.scope_id == scope_id)
            rows = (await s.execute(q)).scalars().all()
        return [
            {"event_id": r.event_id, "timestamp": r.timestamp, "event_type": r.event_type,
             "memory_id": r.memory_id, "scope_id": r.scope_id, "detail": r.detail}
            for r in rows
        ]

    # ------------------------------------------------------------------
    # Stats snapshots
    # ------------------------------------------------------------------

    async def save_stats_snapshot(self, user_id: str, scope_id: str | None = None) -> dict:
        stats = await self.get_stats(user_id, scope_id)
        health = await self.compute_health_score(user_id, scope_id)
        snapshot = {
            "timestamp": _utc_now_iso(),
            "scope_id": scope_id,
            "active": stats.get("active", 0),
            "total": stats.get("total", 0),
            "active_by_type": stats.get("active_by_type", {}),
            "health_score": health.get("score", 0),
        }
        async with self._sm() as s:
            s.add(StatsSnapshot(
                timestamp=snapshot["timestamp"],
                user_id=user_id,
                scope_id=scope_id or "",
                data_json=json.dumps(snapshot, ensure_ascii=False),
            ))
            await s.commit()
        return snapshot

    async def get_stats_trend(self, user_id: str, scope_id: str, limit: int = 20) -> list[dict]:
        async with self._sm() as s:
            rows = (await s.execute(
                select(StatsSnapshot.data_json)
                .where((StatsSnapshot.user_id == user_id) & (StatsSnapshot.scope_id == scope_id))
                .order_by(StatsSnapshot.snapshot_id.desc())
                .limit(limit)
            )).scalars().all()
        snapshots = []
        for raw in rows:
            try:
                snapshots.append(json.loads(raw))
            except Exception:
                pass
        snapshots.reverse()
        return snapshots

    # ------------------------------------------------------------------
    # Tags
    # ------------------------------------------------------------------

    async def add_tags(self, memory_id: str, tags: list[str]) -> bool:
        async with self._sm() as s:
            row = await s.get(Memory, memory_id)
            if row is None:
                return False
            existing = set(row.tags or [])
            for tag in tags:
                existing.add(tag.lower().strip())
            row.tags = sorted(existing)
            row.updated_at = _utc_now_iso()
            await s.commit()
        return True

    async def remove_tags(self, memory_id: str, tags: list[str]) -> bool:
        async with self._sm() as s:
            row = await s.get(Memory, memory_id)
            if row is None:
                return False
            to_remove = {t.lower().strip() for t in tags}
            row.tags = [t for t in (row.tags or []) if t not in to_remove]
            row.updated_at = _utc_now_iso()
            await s.commit()
        return True

    async def bulk_add_tags(self, memory_ids: list[str], tags: list[str]) -> int:
        count = 0
        for mid in memory_ids:
            if await self.add_tags(mid, tags):
                count += 1
        return count

    # ------------------------------------------------------------------
    # Scope operations
    # ------------------------------------------------------------------

    async def export_scope_json(self, user_id: str, scope_id: str | None = None) -> list[dict]:
        units = await self.list_active(user_id, scope_id, limit=10000)
        result = []
        for u in units:
            result.append({
                "memory_id": u.memory_id, "scope_id": u.scope_id,
                "memory_type": u.memory_type.value, "content": u.content,
                "source_session_id": u.source_session_id,
                "source_turn_start": u.source_turn_start, "source_turn_end": u.source_turn_end,
                "entities": u.entities, "topics": u.topics, "importance": u.importance,
                "confidence": u.confidence, "access_count": u.access_count,
                "reinforcement_score": u.reinforcement_score, "status": u.status.value,
                "created_at": u.created_at, "updated_at": u.updated_at,
                "last_accessed_at": u.last_accessed_at, "expires_at": u.expires_at, "tags": u.tags,
            })
        return result

    async def export_csv(self, user_id: str, scope_id: str | None = None) -> str:
        units = await self.list_active(user_id, scope_id, limit=10000)
        lines = ["memory_id,type,content,importance,confidence,access_count,created_at,tags"]
        for u in units:
            content = u.content.replace('"', '""')
            tags = ";".join(u.tags)
            lines.append(f'"{u.memory_id}","{u.memory_type.value}","{content}",{u.importance},{u.confidence},{u.access_count},"{u.created_at}","{tags}"')
        return "\n".join(lines)

    async def import_memories_json(self, user_id: str, data: list[dict], target_scope_id: str | None = None) -> int:
        import uuid as _uuid
        units = []
        for item in data:
            try:
                mt = MemoryType(item.get("memory_type", "episodic"))
            except ValueError:
                mt = MemoryType.EPISODIC
            scope = target_scope_id or item.get("scope_id", "default")
            units.append(MemoryUnit(
                memory_id=str(_uuid.uuid4()),
                user_id=user_id,
                scope_id=scope,
                memory_type=mt,
                content=item.get("content", ""),
                source_session_id=item.get("source_session_id", ""),
                source_turn_start=int(item.get("source_turn_start", 0)),
                source_turn_end=int(item.get("source_turn_end", 0)),
                entities=item.get("entities", []),
                topics=item.get("topics", []),
                importance=float(item.get("importance", 0.5)),
                confidence=float(item.get("confidence", 0.7)),
                access_count=0,
                reinforcement_score=0.0,
                expires_at=item.get("expires_at", ""),
                tags=item.get("tags", []),
            ))
        return await self.add_memories(units)

    async def snapshot_scope(self, user_id: str, scope_id: str | None = None) -> dict:
        units = await self.export_scope_json(user_id, scope_id)
        stats = await self.get_stats(user_id, scope_id)
        return {"snapshot_at": _utc_now_iso(), "scope_id": scope_id, "stats": stats, "memories": units}

    async def restore_snapshot(self, user_id: str, snapshot: dict) -> int:
        scope_id = snapshot.get("scope_id")
        memories = snapshot.get("memories", [])
        if not memories:
            return 0
        current = await self.list_active(user_id, scope_id, limit=10000)
        if current:
            await self.bulk_archive([u.memory_id for u in current])
        return await self.import_memories_json(user_id, memories, target_scope_id=scope_id)

    async def set_type_ttl(self, user_id: str, scope_id: str, memory_type: MemoryType, expires_at: str) -> int:
        now = _utc_now_iso()
        async with self._sm() as s:
            result = await s.execute(
                update(Memory)
                .where(
                    (Memory.user_id == user_id)
                    & (Memory.scope_id == scope_id)
                    & (Memory.memory_type == memory_type.value)
                    & (Memory.status == MemoryStatus.ACTIVE.value)
                )
                .values(expires_at=expires_at, updated_at=now)
            )
            await s.commit()
        return result.rowcount or 0  # type: ignore[union-attr]

    async def share_to_scope(self, memory_id: str, target_scope_id: str) -> str | None:
        import uuid as _uuid
        source = await self.get_by_id(memory_id)
        if source is None:
            return None
        new_id = str(_uuid.uuid4())
        shared = MemoryUnit(
            memory_id=new_id, user_id=source.user_id, scope_id=target_scope_id,
            memory_type=source.memory_type, content=source.content,
            source_session_id=source.source_session_id,
            source_turn_start=source.source_turn_start, source_turn_end=source.source_turn_end,
            entities=list(source.entities), topics=list(source.topics),
            importance=source.importance, confidence=max(source.confidence - 0.05, 0.5),
            embedding=list(source.embedding),
        )
        await self.add_memories([shared])
        async with self._sm() as s:
            await self._log_event(s, "share", memory_id, target_scope_id, f"new_id={new_id[:36]}")
            await s.commit()
        return new_id

    # ------------------------------------------------------------------
    # Merge / similarity
    # ------------------------------------------------------------------

    async def merge_memories(self, id_a: str, id_b: str, merged_content: str) -> str | None:
        import uuid as _uuid
        a = await self.get_by_id(id_a)
        b = await self.get_by_id(id_b)
        if a is None or b is None:
            return None
        now = _utc_now_iso()
        new_id = str(_uuid.uuid4())
        entities = list(dict.fromkeys(a.entities + b.entities))[:12]
        topics = list(dict.fromkeys(a.topics + b.topics))[:12]
        merged = MemoryUnit(
            memory_id=new_id, user_id=a.user_id, scope_id=a.scope_id, memory_type=a.memory_type,
            content=merged_content,
            source_session_id=a.source_session_id,
            source_turn_start=min(a.source_turn_start, b.source_turn_start),
            source_turn_end=max(a.source_turn_end, b.source_turn_end),
            entities=entities, topics=topics,
            importance=max(a.importance, b.importance),
            confidence=max(a.confidence, b.confidence),
            supersedes=[id_a, id_b],
            embedding=list(a.embedding) if a.embedding else list(b.embedding),
        )
        await self.add_memories([merged])
        await self.supersede(id_a, new_id, now)
        await self.supersede(id_b, new_id, now)
        async with self._sm() as s:
            await self._log_event(s, "merge", new_id, a.scope_id or "", f"from={id_a[:18]}+{id_b[:18]}")
            await s.commit()
        return new_id

    async def find_similar(self, memory_id: str, limit: int = 5) -> list[tuple[MemoryUnit, float]]:
        source = await self.get_by_id(memory_id)
        if source is None:
            return []
        source_terms = set(t.lower() for t in source.topics + source.entities)
        if not source_terms:
            return []
        units = await self.list_active(source.user_id, source.scope_id, limit=500)
        scored: list[tuple[MemoryUnit, float]] = []
        for u in units:
            if u.memory_id == memory_id:
                continue
            u_terms = set(t.lower() for t in u.topics + u.entities)
            if not u_terms:
                continue
            overlap = len(source_terms & u_terms) / float(len(source_terms | u_terms))
            if overlap > 0.1:
                scored.append((u, round(overlap, 4)))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:limit]

    async def get_memory_history(self, memory_id: str) -> list[dict]:
        history: list[dict] = []
        visited: set[str] = set()
        queue = [memory_id]
        while queue:
            mid = queue.pop(0)
            if mid in visited:
                continue
            visited.add(mid)
            unit = await self.get_by_id(mid)
            if unit is None:
                continue
            history.append({
                "memory_id": unit.memory_id, "content": unit.content[:200],
                "status": unit.status.value, "created_at": unit.created_at,
                "updated_at": unit.updated_at, "importance": unit.importance,
                "supersedes": unit.supersedes, "superseded_by": unit.superseded_by,
            })
            for sid in unit.supersedes:
                if sid not in visited:
                    queue.append(sid)
        return sorted(history, key=lambda h: h["created_at"])

    async def get_scope_analytics(self, user_id: str, scope_id: str) -> dict:
        async with self._sm() as s:
            result = await s.execute(select(Memory).where(Memory.user_id == user_id).where(Memory.scope_id == scope_id))
            all_rows = result.scalars().all()
        if not all_rows:
            return {"total": 0}
        units = [r.to_unit() for r in all_rows]
        active = [u for u in units if u.status == MemoryStatus.ACTIVE]
        superseded = [u for u in units if u.status == MemoryStatus.SUPERSEDED]
        archived = [u for u in units if u.status == MemoryStatus.ARCHIVED]
        type_dist: dict[str, int] = {}
        for u in active:
            type_dist[u.memory_type.value] = type_dist.get(u.memory_type.value, 0) + 1
        total_accesses = sum(u.access_count for u in active)
        avg_access = total_accesses / max(len(active), 1)
        avg_importance = sum(u.importance for u in active) / max(len(active), 1)
        return {
            "total": len(units), "active": len(active), "superseded": len(superseded),
            "archived": len(archived), "type_distribution": type_dist,
            "access": {"total_accesses": total_accesses, "avg_access_count": round(avg_access, 2),
                       "never_accessed": sum(1 for u in active if u.access_count == 0),
                       "highly_accessed": sum(1 for u in active if u.access_count >= 5)},
            "importance": {"average": round(avg_importance, 4),
                           "high_count": sum(1 for u in active if u.importance >= 0.8),
                           "low_count": sum(1 for u in active if u.importance < 0.3)},
            "features": {
                "with_ttl": sum(1 for u in active if u.expires_at),
                "with_tags": sum(1 for u in active if u.tags),
                "pinned": sum(1 for u in active if u.importance >= 0.99),
            },
        }

    async def compare_scopes(self, user_id: str, scope_a: str | None, scope_b: str | None) -> dict:
        units_a = await self.list_active(user_id, scope_a, limit=1000)
        units_b = await self.list_active(user_id, scope_b, limit=1000)
        content_a = {u.content.strip().lower(): u for u in units_a}
        content_b = {u.content.strip().lower(): u for u in units_b}
        shared_keys = set(content_a.keys()) & set(content_b.keys())
        unique_a = set(content_a.keys()) - shared_keys
        unique_b = set(content_b.keys()) - shared_keys
        return {
            "scope_a": scope_a, "scope_b": scope_b,
            "scope_a_count": len(units_a), "scope_b_count": len(units_b),
            "shared_count": len(shared_keys), "unique_to_a": len(unique_a), "unique_to_b": len(unique_b),
            "shared_content": [content_a[k].content[:100] for k in list(shared_keys)[:5]],
        }

    async def garbage_collect(self, user_id: str, scope_id: str | None = None) -> dict:
        async with self._sm() as s:
            cond = self._where_user_scope(user_id, scope_id) & (Memory.status == MemoryStatus.SUPERSEDED.value)
            result = await s.execute(select(Memory.memory_id).where(cond))
            superseded_ids = {r[0] for r in result.all()}
        active_units = await self.list_active(user_id, scope_id, limit=10000)
        referenced = {sid for u in active_units for sid in u.supersedes}
        orphans = superseded_ids - referenced
        if orphans:
            async with self._sm() as s:
                await s.execute(delete(Memory).where(Memory.memory_id.in_(orphans)))
                await s.commit()
        return {"removed": len(orphans), "kept_superseded": len(superseded_ids) - len(orphans)}

    async def validate_integrity(self) -> dict:
        async with self._sm() as s:
            orphaned_links = (await s.execute(text(
                "SELECT COUNT(*) FROM memory_links l WHERE NOT EXISTS "
                "(SELECT 1 FROM memories m WHERE m.memory_id = l.source_id) "
                "OR NOT EXISTS (SELECT 1 FROM memories m WHERE m.memory_id = l.target_id)"
            ))).scalar() or 0
            orphaned_watches = (await s.execute(text(
                "SELECT COUNT(*) FROM memory_watches w WHERE NOT EXISTS "
                "(SELECT 1 FROM memories m WHERE m.memory_id = w.memory_id)"
            ))).scalar() or 0
            orphaned_annotations = (await s.execute(text(
                "SELECT COUNT(*) FROM memory_annotations a WHERE NOT EXISTS "
                "(SELECT 1 FROM memories m WHERE m.memory_id = a.memory_id)"
            ))).scalar() or 0
            dangling_superseded = (await s.execute(text(
                "SELECT COUNT(*) FROM memories m WHERE m.superseded_by != '' "
                "AND NOT EXISTS (SELECT 1 FROM memories m2 WHERE m2.memory_id = m.superseded_by)"
            ))).scalar() or 0
        issues = []
        if orphaned_links:
            issues.append(f"{orphaned_links} orphaned link(s)")
        if orphaned_watches:
            issues.append(f"{orphaned_watches} orphaned watch(es)")
        if orphaned_annotations:
            issues.append(f"{orphaned_annotations} orphaned annotation(s)")
        if dangling_superseded:
            issues.append(f"{dangling_superseded} dangling superseded_by")
        return {
            "valid": len(issues) == 0, "issues": issues,
            "orphaned_links": orphaned_links, "orphaned_watches": orphaned_watches,
            "orphaned_annotations": orphaned_annotations, "dangling_superseded": dangling_superseded,
        }

    async def cleanup_orphans(self) -> dict:
        async with self._sm() as s:
            r_links = await s.execute(text(
                "DELETE FROM memory_links WHERE NOT EXISTS (SELECT 1 FROM memories m WHERE m.memory_id = memory_links.source_id) "
                "OR NOT EXISTS (SELECT 1 FROM memories m WHERE m.memory_id = memory_links.target_id)"
            ))
            r_watches = await s.execute(text(
                "DELETE FROM memory_watches WHERE NOT EXISTS (SELECT 1 FROM memories m WHERE m.memory_id = memory_watches.memory_id)"
            ))
            r_annotations = await s.execute(text(
                "DELETE FROM memory_annotations WHERE NOT EXISTS (SELECT 1 FROM memories m WHERE m.memory_id = memory_annotations.memory_id)"
            ))
            await s.commit()
        return {
            "removed_links": r_links.rowcount,  # type: ignore[union-attr]
            "removed_watches": r_watches.rowcount,  # type: ignore[union-attr]
            "removed_annotations": r_annotations.rowcount,  # type: ignore[union-attr]
            "total_removed": (r_links.rowcount or 0) + (r_watches.rowcount or 0) + (r_annotations.rowcount or 0),  # type: ignore[union-attr]
        }

    async def sample_memories(self, user_id: str, scope_id: str | None = None, count: int = 5) -> list[MemoryUnit]:
        units = await self.list_active(user_id, scope_id, limit=500)
        if len(units) <= count:
            return units
        import random
        return random.sample(units, count)

    async def update_embedding(self, memory_id: str, embedding: list[float]) -> None:
        async with self._sm() as s:
            await s.execute(
                update(Memory).where(Memory.memory_id == memory_id)
                .values(embedding=embedding if embedding else None)
            )
            await s.commit()

    # ------------------------------------------------------------------
    # Scope access control
    # ------------------------------------------------------------------

    async def grant_access(self, scope_id: str, principal: str, permission: str = "read") -> bool:
        if permission not in ("read", "write", "admin"):
            return False
        async with self._sm() as s:
            try:
                s.add(ScopeAccess(scope_id=scope_id, principal=principal, permission=permission, created_at=_utc_now_iso()))
                await s.commit()
                return True
            except IntegrityError:
                await s.rollback()
                return False

    async def revoke_access(self, scope_id: str, principal: str, permission: str | None = None) -> int:
        async with self._sm() as s:
            cond = (ScopeAccess.scope_id == scope_id) & (ScopeAccess.principal == principal)
            if permission:
                cond = cond & (ScopeAccess.permission == permission)
            result = await s.execute(delete(ScopeAccess).where(cond))
            await s.commit()
        return result.rowcount or 0  # type: ignore[union-attr]

    async def check_access(self, scope_id: str, principal: str, permission: str = "read") -> bool:
        async with self._sm() as s:
            count = (await s.execute(
                select(func.count()).where(
                    (ScopeAccess.scope_id == scope_id) & (ScopeAccess.principal == principal)
                    & ((ScopeAccess.permission == permission) | (ScopeAccess.permission == "admin"))
                )
            )).scalar() or 0
        return count > 0

    async def list_scope_grants(self, scope_id: str) -> list[dict]:
        async with self._sm() as s:
            rows = (await s.execute(
                select(ScopeAccess).where(ScopeAccess.scope_id == scope_id)
                .order_by(ScopeAccess.principal, ScopeAccess.permission)
            )).scalars().all()
        return [{"scope_id": r.scope_id, "principal": r.principal, "permission": r.permission, "created_at": r.created_at} for r in rows]

    async def list_principal_scopes(self, principal: str) -> list[dict]:
        async with self._sm() as s:
            rows = (await s.execute(
                select(ScopeAccess).where(ScopeAccess.principal == principal)
                .order_by(ScopeAccess.scope_id, ScopeAccess.permission)
            )).scalars().all()
        return [{"scope_id": r.scope_id, "principal": r.principal, "permission": r.permission, "created_at": r.created_at} for r in rows]

    # ------------------------------------------------------------------
    # Memory links
    # ------------------------------------------------------------------

    async def add_link(self, source_id: str, target_id: str, link_type: str = "related") -> bool:
        async with self._sm() as s:
            try:
                s.add(MemoryLink(source_id=source_id, target_id=target_id, link_type=link_type, created_at=_utc_now_iso()))
                await s.commit()
                return True
            except IntegrityError:
                await s.rollback()
                return False

    async def remove_link(self, source_id: str, target_id: str, link_type: str | None = None) -> int:
        async with self._sm() as s:
            cond = (MemoryLink.source_id == source_id) & (MemoryLink.target_id == target_id)
            if link_type:
                cond = cond & (MemoryLink.link_type == link_type)
            result = await s.execute(delete(MemoryLink).where(cond))
            await s.commit()
        return result.rowcount or 0  # type: ignore[union-attr]

    async def get_links(self, memory_id: str, direction: str = "both") -> list[dict]:
        results = []
        async with self._sm() as s:
            if direction in ("outgoing", "both"):
                rows = (await s.execute(select(MemoryLink).where(MemoryLink.source_id == memory_id))).scalars().all()
                results.extend({"source_id": r.source_id, "target_id": r.target_id, "link_type": r.link_type, "direction": "outgoing"} for r in rows)
            if direction in ("incoming", "both"):
                rows = (await s.execute(select(MemoryLink).where(MemoryLink.target_id == memory_id))).scalars().all()
                results.extend({"source_id": r.source_id, "target_id": r.target_id, "link_type": r.link_type, "direction": "incoming"} for r in rows)
        return results

    async def get_linked_memories(self, memory_id: str, link_type: str | None = None) -> list[MemoryUnit]:
        links = await self.get_links(memory_id, direction="both")
        linked_ids = set()
        for lnk in links:
            if link_type is None or lnk["link_type"] == link_type:
                linked_ids.add(lnk["target_id"] if lnk["direction"] == "outgoing" else lnk["source_id"])
        if not linked_ids:
            return []
        units = await self.get_by_ids(list(linked_ids))
        return [u for u in units if u.status == MemoryStatus.ACTIVE]

    # ------------------------------------------------------------------
    # Watches
    # ------------------------------------------------------------------

    async def add_watch(self, memory_id: str, watcher: str) -> bool:
        async with self._sm() as s:
            try:
                s.add(MemoryWatch(memory_id=memory_id, watcher=watcher, created_at=_utc_now_iso()))
                await s.commit()
                return True
            except IntegrityError:
                await s.rollback()
                return False

    async def remove_watch(self, memory_id: str, watcher: str) -> bool:
        async with self._sm() as s:
            result = await s.execute(
                delete(MemoryWatch).where((MemoryWatch.memory_id == memory_id) & (MemoryWatch.watcher == watcher))
            )
            await s.commit()
        return (result.rowcount or 0) > 0  # type: ignore[union-attr]

    async def get_watchers(self, memory_id: str) -> list[str]:
        async with self._sm() as s:
            rows = (await s.execute(select(MemoryWatch.watcher).where(MemoryWatch.memory_id == memory_id).order_by(MemoryWatch.watcher))).scalars().all()
        return list(rows)

    async def get_watched_memories(self, watcher: str) -> list[str]:
        async with self._sm() as s:
            rows = (await s.execute(select(MemoryWatch.memory_id).where(MemoryWatch.watcher == watcher).order_by(MemoryWatch.memory_id))).scalars().all()
        return list(rows)

    # ------------------------------------------------------------------
    # Annotations
    # ------------------------------------------------------------------

    async def add_annotation(self, memory_id: str, content: str, author: str = "") -> int:
        async with self._sm() as s:
            ann = MemoryAnnotation(memory_id=memory_id, author=author, content=content, created_at=_utc_now_iso())
            s.add(ann)
            await s.flush()
            ann_id = ann.annotation_id
            await s.commit()
        return ann_id

    async def get_annotations(self, memory_id: str) -> list[dict]:
        async with self._sm() as s:
            rows = (await s.execute(
                select(MemoryAnnotation).where(MemoryAnnotation.memory_id == memory_id).order_by(MemoryAnnotation.created_at)
            )).scalars().all()
        return [{"annotation_id": r.annotation_id, "memory_id": r.memory_id, "author": r.author, "content": r.content, "created_at": r.created_at} for r in rows]

    async def delete_annotation(self, annotation_id: int) -> bool:
        async with self._sm() as s:
            result = await s.execute(delete(MemoryAnnotation).where(MemoryAnnotation.annotation_id == annotation_id))
            await s.commit()
        return (result.rowcount or 0) > 0  # type: ignore[union-attr]

    def close(self) -> None:
        pass  # Engine disposal handled in app lifespan.


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _log2(x: float) -> float:
    return math.log2(max(x, 1.0))


def _tokenize(text: str) -> list[str]:
    token: list[str] = []
    out: list[str] = []
    for ch in text.lower():
        if ch.isalnum() or ch in {"_", "-"}:
            token.append(ch)
            continue
        if token:
            out.append("".join(token))
            token = []
    if token:
        out.append("".join(token))
    return out
