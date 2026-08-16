from __future__ import annotations

import logging
import math
from datetime import datetime, timezone

from marklymem.utils import telemetry

from .embeddings import cosine_similarity
from .models import MemoryQuery, MemorySearchHit
from .policy import MemoryPolicy
from .store import MemoryStore

logger = logging.getLogger(__name__)


class MemoryRetriever:
    """Retrieves memory hits from store using the current policy."""

    def __init__(
        self,
        store: MemoryStore,
        policy: MemoryPolicy | None = None,
        retrieval_mode: str = "keyword",
        embedder=None,
    ):
        self.store = store
        self.policy = policy or MemoryPolicy()
        self.retrieval_mode = retrieval_mode
        self.embedder = embedder

    async def retrieve(self, query: MemoryQuery) -> list[MemorySearchHit]:
        with telemetry.trace(
            "memory.retrieve",
            session_id=query.session_id,
            user_id=query.user_id,
            namespace=query.namespace,
            input=query.query_text,
        ) as root:
            mode = self.retrieval_mode
            if mode == "auto":
                mode = self._auto_select_mode(query)
                logger.info("[Retriever] auto-selected mode=%s for query(%d chars)", mode, len(query.query_text))
            root.set_attribute("retrieval.mode", mode)
            root.set_attribute("top_k", query.top_k)

            if mode == "embedding":
                hits = await self._retrieve_embedding(query)
                logger.info("[Retriever] mode=embedding namespace=%s hits=%d", query.namespace, len(hits))
            elif mode == "hybrid":
                hits = await self._retrieve_hybrid(query)
                logger.info("[Retriever] mode=hybrid namespace=%s hits=%d", query.namespace, len(hits))
            else:
                limit = min(query.top_k, self.policy.max_injected_units)
                hits = await self.store.search_keyword(
                    query.user_id,
                    query.namespace,
                    query_text=query.query_text,
                    limit=limit,
                )
                # Apply tag-based boosting if context tags are provided.
                if query.context_tags:
                    hits = _apply_tag_boost(hits, query.context_tags)
                logger.info(
                    "[Retriever] mode=keyword namespace=%s hits=%d",
                    query.namespace, len(hits),
                )

            root.set_attribute("result_count", len(hits))
            telemetry.set_output(
                root,
                [
                    {
                        "memory_id": h.unit.memory_id,
                        "score": round(float(h.score), 4),
                        "matched_terms": h.matched_terms,
                        "content": h.unit.content,
                    }
                    for h in hits
                ],
            )
            return hits

    def _auto_select_mode(self, query: MemoryQuery) -> str:
        """Auto-select retrieval mode based on query characteristics.

        - Short queries (< 4 words): keyword is most reliable
        - Medium queries with embedder available: hybrid for best recall
        - Long queries without embedder: keyword
        """
        terms = _tokenize(query.query_text)
        if self.embedder is not None and len(terms) >= 4:
            return "hybrid"
        return "keyword"

    async def _retrieve_hybrid(self, query: MemoryQuery) -> list[MemorySearchHit]:
        query_terms = _tokenize(query.query_text)
        if not query_terms:
            return []

        limit = min(query.top_k * 4, 200)
        if self.embedder:
            query_embedding = self.embedder.encode(query.query_text)
        else:
            query_embedding = []

        # Source candidates from both keyword and (if available) vector search.
        kw_hits = await self.store.search_keyword(
            query.user_id, query.namespace, query_text=query.query_text, limit=limit
        )
        kw_units = {h.unit.memory_id: h.unit for h in kw_hits}

        if query_embedding:
            vec_units = await self.store.search_vector(
                query.user_id, query.namespace, query_embedding=query_embedding, limit=limit
            )
            for u in vec_units:
                kw_units.setdefault(u.memory_id, u)

        units = list(kw_units.values())
        if not units:
            return []

        # Build IDF weights across the active corpus.
        doc_freq: dict[str, int] = {}
        unit_content_terms: list[set[str]] = []
        unit_metadata_terms: list[set[str]] = []
        for unit in units:
            content = set(_tokenize(unit.content))
            metadata = set(_tokenize(" ".join(unit.topics + unit.entities)))
            unit_content_terms.append(content)
            unit_metadata_terms.append(metadata)
            all_terms = content | metadata
            for term in set(query_terms):
                if term in all_terms:
                    doc_freq[term] = doc_freq.get(term, 0) + 1

        num_docs = float(len(units))
        hits: list[MemorySearchHit] = []
        for idx, unit in enumerate(units):
            if query.include_types and unit.memory_type not in query.include_types:
                continue

            content_terms = unit_content_terms[idx]
            metadata_terms = unit_metadata_terms[idx]
            matched = sorted(
                term for term in query_terms
                if term in content_terms or term in metadata_terms
            )

            # IDF-weighted keyword and metadata overlap.
            keyword_idf = sum(
                _log2(num_docs / float(doc_freq.get(term, 1)))
                for term in query_terms if term in content_terms
            )
            metadata_idf = sum(
                _log2(num_docs / float(doc_freq.get(term, 1)))
                for term in query_terms if term in metadata_terms
            )
            embedding_score = (
                cosine_similarity(query_embedding, unit.embedding)
                if query_embedding and unit.embedding
                else 0.0
            )
            recency_bonus = _estimate_recency_bonus(unit.updated_at, self.policy.recent_bonus_hours)
            type_boost = self.policy.type_boosts.get(unit.memory_type.value, 1.0)
            # Confidence factor: memories with higher confidence score slightly better.
            confidence_factor = 0.8 + 0.2 * unit.confidence
            score = (
                self.policy.keyword_weight * keyword_idf
                + self.policy.metadata_weight * metadata_idf
                + embedding_score
                + self.policy.importance_weight * unit.importance
                + self.policy.recency_weight * recency_bonus
                + unit.reinforcement_score
            ) * type_boost * confidence_factor
            reason_parts = [f"matched: {', '.join(matched[:5])}"] if matched else ["vector match"]
            if recency_bonus > 0.3:
                reason_parts.append("recent")
            if unit.importance >= 0.8:
                reason_parts.append("high importance")
            if unit.reinforcement_score > 0.1:
                reason_parts.append("reinforced")
            hits.append(MemorySearchHit(
                unit=unit, score=score, matched_terms=matched,
                reason="; ".join(reason_parts),
            ))

        hits.sort(key=lambda hit: (hit.score, hit.unit.updated_at), reverse=True)
        hits = hits[: min(query.top_k, self.policy.max_injected_units)]
        if query.context_tags:
            hits = _apply_tag_boost(hits, query.context_tags)
        return hits

    async def _retrieve_embedding(self, query: MemoryQuery) -> list[MemorySearchHit]:
        if self.embedder is None:
            return []
        query_embedding = self.embedder.encode(query.query_text)
        if not query_embedding:
            return []

        limit = min(query.top_k * 3, 100)
        hits: list[MemorySearchHit] = []
        for unit in await self.store.search_vector(
            query.user_id, query.namespace, query_embedding=query_embedding, limit=limit
        ):
            if query.include_types and unit.memory_type not in query.include_types:
                continue
            similarity = cosine_similarity(query_embedding, unit.embedding)
            if similarity <= 0.0:
                continue
            type_boost = self.policy.type_boosts.get(unit.memory_type.value, 1.0)
            confidence_factor = 0.8 + 0.2 * unit.confidence
            score = (
                similarity
                + self.policy.importance_weight * unit.importance
                + unit.reinforcement_score
            ) * type_boost * confidence_factor
            hits.append(MemorySearchHit(unit=unit, score=score, matched_terms=[]))

        hits.sort(key=lambda hit: (hit.score, hit.unit.updated_at), reverse=True)
        hits = hits[: min(query.top_k, self.policy.max_injected_units)]
        if query.context_tags:
            hits = _apply_tag_boost(hits, query.context_tags)
        return hits


def _apply_tag_boost(hits: list[MemorySearchHit], context_tags: list[str]) -> list[MemorySearchHit]:
    """Boost scores for memories whose tags overlap with the query's context tags.

    Each matching tag adds a 15% boost, capped at 50% total. Re-sorts after boosting.
    """
    if not context_tags:
        return hits
    tag_set = set(t.lower() for t in context_tags)
    for hit in hits:
        unit_tags = set(t.lower() for t in hit.unit.tags)
        overlap = len(tag_set & unit_tags)
        if overlap:
            boost = min(0.15 * overlap, 0.5)
            hit.score *= 1.0 + boost
    hits.sort(key=lambda h: (h.score, h.unit.updated_at), reverse=True)
    return hits


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



def _estimate_recency_bonus(updated_at: str, recent_bonus_hours: int) -> float:
    if not updated_at:
        return 0.0
    try:
        updated = datetime.fromisoformat(updated_at.replace("Z", "+00:00"))
    except ValueError:
        return 0.0
    age_seconds = max((datetime.now(timezone.utc) - updated).total_seconds(), 0.0)
    age_hours = age_seconds / 3600.0
    if recent_bonus_hours <= 0:
        return 0.0
    return max(0.0, 1.0 - (age_hours / float(recent_bonus_hours)))
