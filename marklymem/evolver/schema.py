from __future__ import annotations

import os

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    BigInteger,
    Float,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from .db import Base
from .models import MemoryStatus, MemoryType, MemoryUnit

EMBEDDING_DIM: int = int(os.getenv("EVOLVER_EMBEDDING_DIM", "1024"))

_TSVEC_EXPR = "to_tsvector('english', coalesce(content,''))"


class Memory(Base):
    __tablename__ = "memories"
    __table_args__ = (
        Index("idx_memories_user", "user_id"),
        Index("idx_memories_user_scope", "user_id", "scope_id"),
        Index("idx_memories_scope_status", "scope_id", "status"),
        Index("idx_memories_scope_type", "scope_id", "memory_type"),
    )

    memory_id: Mapped[str] = mapped_column(String, primary_key=True)
    user_id: Mapped[str] = mapped_column(String, nullable=False)
    scope_id: Mapped[str | None] = mapped_column(String, nullable=True)
    memory_type: Mapped[str] = mapped_column(String, nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    source_session_id: Mapped[str | None] = mapped_column(String, nullable=True, default=None)
    source_turn_start: Mapped[int] = mapped_column(Integer, default=0)
    source_turn_end: Mapped[int] = mapped_column(Integer, default=0)
    entities: Mapped[list] = mapped_column(JSONB, default=list)
    topics: Mapped[list] = mapped_column(JSONB, default=list)
    importance: Mapped[float] = mapped_column(Float, default=0.5)
    confidence: Mapped[float] = mapped_column(Float, default=0.7)
    access_count: Mapped[int] = mapped_column(Integer, default=0)
    reinforcement_score: Mapped[float] = mapped_column(Float, default=0.0)
    status: Mapped[str] = mapped_column(String, default="active")
    supersedes: Mapped[list] = mapped_column(JSONB, default=list)
    superseded_by: Mapped[str] = mapped_column(String, default="")
    embedding: Mapped[list | None] = mapped_column(Vector(EMBEDDING_DIM), nullable=True)
    created_at: Mapped[str] = mapped_column(String, nullable=False)
    updated_at: Mapped[str] = mapped_column(String, nullable=False)
    last_accessed_at: Mapped[str] = mapped_column(String, default="")
    expires_at: Mapped[str] = mapped_column(String, default="")
    tags: Mapped[list] = mapped_column(JSONB, default=list)
    content_tsv: Mapped[str | None] = mapped_column(
        "content_tsv",
        nullable=True,
        # The Computed column is defined in the Alembic migration via raw DDL.
        # This declaration is only informational for the ORM.
        deferred=True,
    )

    def to_unit(self) -> MemoryUnit:
        return MemoryUnit(
            memory_id=self.memory_id,
            user_id=self.user_id,
            scope_id=self.scope_id,
            memory_type=MemoryType(self.memory_type),
            content=self.content,
            source_session_id=self.source_session_id or None,
            source_turn_start=int(self.source_turn_start or 0),
            source_turn_end=int(self.source_turn_end or 0),
            entities=list(self.entities or []),
            topics=list(self.topics or []),
            importance=float(self.importance or 0.0),
            confidence=float(self.confidence or 0.0),
            access_count=int(self.access_count or 0),
            reinforcement_score=float(self.reinforcement_score or 0.0),
            status=MemoryStatus(self.status),
            supersedes=list(self.supersedes or []),
            superseded_by=self.superseded_by or "",
            embedding=list(self.embedding) if self.embedding is not None else [],
            created_at=self.created_at,
            updated_at=self.updated_at,
            last_accessed_at=self.last_accessed_at or "",
            expires_at=self.expires_at or "",
            tags=list(self.tags or []),
        )

    @staticmethod
    def from_unit(unit: MemoryUnit) -> "Memory":
        return Memory(
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


class MemoryEvent(Base):
    __tablename__ = "memory_events"

    event_id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    timestamp: Mapped[str] = mapped_column(String, nullable=False)
    event_type: Mapped[str] = mapped_column(String, nullable=False)
    memory_id: Mapped[str] = mapped_column(String, nullable=False)
    scope_id: Mapped[str] = mapped_column(String, default="")
    detail: Mapped[str] = mapped_column(Text, default="")


class MemoryLink(Base):
    __tablename__ = "memory_links"
    __table_args__ = (
        UniqueConstraint("source_id", "target_id", "link_type", name="uq_memory_links"),
    )

    source_id: Mapped[str] = mapped_column(String, nullable=False, primary_key=True)
    target_id: Mapped[str] = mapped_column(String, nullable=False, primary_key=True)
    link_type: Mapped[str] = mapped_column(String, nullable=False, primary_key=True, default="related")
    created_at: Mapped[str] = mapped_column(String, nullable=False)


class MemoryWatch(Base):
    __tablename__ = "memory_watches"
    __table_args__ = (
        UniqueConstraint("memory_id", "watcher", name="uq_memory_watches"),
    )

    watch_id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    memory_id: Mapped[str] = mapped_column(String, nullable=False)
    watcher: Mapped[str] = mapped_column(String, nullable=False)
    created_at: Mapped[str] = mapped_column(String, nullable=False)


class MemoryAnnotation(Base):
    __tablename__ = "memory_annotations"

    annotation_id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    memory_id: Mapped[str] = mapped_column(String, nullable=False)
    author: Mapped[str] = mapped_column(String, default="")
    content: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[str] = mapped_column(String, nullable=False)


