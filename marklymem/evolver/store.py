from __future__ import annotations

import logging
import math
from typing import Iterable

from sqlalchemy import func, select, text, update
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from .models import MemorySearchHit, MemoryStatus, MemoryType, MemoryUnit
from .models import utc_now_iso as _utc_now_iso
from .schema import Memory

logger = logging.getLogger(__name__)


class MemoryStore:
    """PostgreSQL-backed async store for long-term memory units."""

    def __init__(self, sm: async_sessionmaker[AsyncSession]):
        self._sm = sm

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _namespace_cond(self, namespace: str):
        escaped = namespace.replace("_", r"\_")
        return (Memory.namespace == namespace) | Memory.namespace.like(escaped + "/%", escape="\\")

    def _where_user_namespace(self, user_id: str, namespace: str | None):
        if not user_id:
            raise ValueError("user_id is required")
        if namespace is not None:
            return (Memory.user_id == user_id) & self._namespace_cond(namespace)
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
                    namespace=unit.namespace,
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

    async def _list_active_with_cond(self, namespace_cond, limit: int) -> list[MemoryUnit]:
        async with self._sm() as s:
            result = await s.execute(
                select(Memory)
                .where(namespace_cond & (Memory.status == MemoryStatus.ACTIVE.value))
                .order_by(Memory.updated_at.desc())
                .limit(limit)
            )
            units = [r.to_unit() for r in result.scalars()]
        now_iso = _utc_now_iso()
        return [u for u in units if not u.expires_at or u.expires_at > now_iso]

    async def list_active_exact(self, user_id: str, namespace: str | None = None, limit: int = 100) -> list[MemoryUnit]:
        """Same as list_active but matches namespace exactly — no subtree expansion."""
        if namespace is not None:
            cond = (Memory.user_id == user_id) & (Memory.namespace == namespace)
        else:
            cond = Memory.user_id == user_id
        return await self._list_active_with_cond(cond, limit)

    async def list_active(self, user_id: str, namespace: str | None = None, limit: int = 100) -> list[MemoryUnit]:
        return await self._list_active_with_cond(self._where_user_namespace(user_id, namespace), limit)

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
                select(Memory.memory_id, Memory.namespace)
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
            await s.commit()
        return len(active_ids)

    async def expire_stale(self, user_id: str, namespace: str) -> int:
        now_iso = _utc_now_iso()
        async with self._sm() as s:
            result = await s.execute(
                select(Memory.memory_id).where(
                    (Memory.user_id == user_id)
                    & self._namespace_cond(namespace)
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

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    async def search_keyword(self, user_id: str, namespace: str | None, query_text: str, limit: int = 6) -> list[MemorySearchHit]:
        terms = [t.lower() for t in _tokenize(query_text) if t]
        if not terms:
            return []
        # Build OR query for websearch_to_tsquery
        ts_query = " OR ".join(terms[:12])
        async with self._sm() as s:
            cond = (
                self._where_user_namespace(user_id, namespace)
                & (Memory.status == MemoryStatus.ACTIVE.value)
                & text("content_tsv @@ websearch_to_tsquery('english', :q)").bindparams(q=ts_query)
            )
            result = await s.execute(
                select(Memory).where(cond).limit(limit * 5)
            )
            units = [r.to_unit() for r in result.scalars()]
        if not units:
            return self._search_keyword_manual_sync(
                await self._list_active_sync(user_id, namespace, limit=500), terms, limit
            )
        return self._rank_with_idf(units, terms, limit)

    async def _list_active_sync(self, user_id: str, namespace: str | None, limit: int) -> list[MemoryUnit]:
        return await self.list_active(user_id, namespace, limit=limit)

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

    async def search_vector(self, user_id: str, namespace: str | None, query_embedding: list[float], limit: int = 10) -> list[MemoryUnit]:
        if not query_embedding:
            return []
        vec_str = "[" + ",".join(str(v) for v in query_embedding) + "]"
        async with self._sm() as s:
            cond = (
                self._where_user_namespace(user_id, namespace)
                & (Memory.status == MemoryStatus.ACTIVE.value)
                & Memory.embedding.isnot(None)
            )
            result = await s.execute(
                select(Memory).where(cond)
                .order_by(text(f"embedding <=> '{vec_str}'::vector"))
                .limit(limit)
            )
            return [r.to_unit() for r in result.scalars()]

    # ------------------------------------------------------------------
    # Stats / analytics
    # ------------------------------------------------------------------

    async def get_stats(self, user_id: str, namespace: str | None = None) -> dict:
        async with self._sm() as s:
            cond = self._where_user_namespace(user_id, namespace)
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

    async def get_namespace_analytics(self, user_id: str, namespace: str) -> dict:
        async with self._sm() as s:
            result = await s.execute(
                select(Memory).where(self._where_user_namespace(user_id, namespace))
            )
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

    # ------------------------------------------------------------------
    # Scope operations
    # ------------------------------------------------------------------

    async def set_type_ttl(self, user_id: str, namespace: str, memory_type: MemoryType, expires_at: str) -> int:
        now = _utc_now_iso()
        async with self._sm() as s:
            result = await s.execute(
                update(Memory)
                .where(
                    (Memory.user_id == user_id)
                    & self._namespace_cond(namespace)
                    & (Memory.memory_type == memory_type.value)
                    & (Memory.status == MemoryStatus.ACTIVE.value)
                )
                .values(expires_at=expires_at, updated_at=now)
            )
            await s.commit()
        return result.rowcount or 0  # type: ignore[union-attr]

    async def update_embedding(self, memory_id: str, embedding: list[float]) -> None:
        async with self._sm() as s:
            await s.execute(
                update(Memory).where(Memory.memory_id == memory_id)
                .values(embedding=embedding if embedding else None)
            )
            await s.commit()

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
