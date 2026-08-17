from __future__ import annotations

import logging
import re
import uuid

from marklymem.utils import telemetry

from .consolidator import MemoryConsolidator
from .embeddings import BaseEmbedder, create_embedder
from .llm_extractor import LLMMemoryExtractor
from .metrics import summarize_memory_store
from .models import MemoryStatus, MemoryType, MemoryUnit, utc_now_iso
from .policy import MemoryPolicy
from .resolver import ConflictResolver
from .retriever import MemoryRetriever
from .store import MemoryStore

logger = logging.getLogger(__name__)


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
            for idx, turn in enumerate(turns, start=1):
                prompt_text = str(turn.get("prompt_text", "") or "").strip()
                response_text = str(turn.get("response_text", "") or "").strip()
                if not prompt_text and not response_text:
                    continue
                extracted = _extract_memory_units_for_turn(
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

        return {"resolved": resolved, "total_conflicts": len(conflicts), "dropped": dropped}

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

    def close(self) -> None:
        self.store.close()


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


def _extract_memory_units_for_turn(
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
