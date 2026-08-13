"""Initial schema — PostgreSQL + pgvector

Revision ID: 0001
Revises:
Create Date: 2026-08-13
"""

from __future__ import annotations

import os

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB

# revision identifiers, used by Alembic.
revision = "0001"
down_revision = None
branch_labels = None
depends_on = None

EMBEDDING_DIM: int = int(os.getenv("EVOLVER_EMBEDDING_DIM", "1024"))

_TSVEC_EXPR = (
    "to_tsvector('english', coalesce(content,'') || ' ' || coalesce(summary,''))"
)


def upgrade() -> None:
    # pgvector extension — installs the vector type into public schema (DB-scoped).
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")

    # -----------------------------------------------------------------------
    # memories
    # -----------------------------------------------------------------------
    op.create_table(
        "memories",
        sa.Column("memory_id", sa.String(), primary_key=True),
        sa.Column("user_id", sa.String(), nullable=False),
        sa.Column("scope_id", sa.String(), nullable=True),
        sa.Column("memory_type", sa.String(), nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("summary", sa.Text(), server_default=""),
        sa.Column("source_session_id", sa.String(), server_default=""),
        sa.Column("source_turn_start", sa.Integer(), server_default="0"),
        sa.Column("source_turn_end", sa.Integer(), server_default="0"),
        sa.Column("entities", JSONB(), server_default=sa.text("'[]'")),
        sa.Column("topics", JSONB(), server_default=sa.text("'[]'")),
        sa.Column("importance", sa.Float(), server_default="0.5"),
        sa.Column("confidence", sa.Float(), server_default="0.7"),
        sa.Column("access_count", sa.Integer(), server_default="0"),
        sa.Column("reinforcement_score", sa.Float(), server_default="0.0"),
        sa.Column("status", sa.String(), server_default="active"),
        sa.Column("supersedes", JSONB(), server_default=sa.text("'[]'")),
        sa.Column("superseded_by", sa.String(), server_default=""),
        sa.Column(
            "embedding",
            sa.Text(),  # placeholder; altered to vector below
            nullable=True,
        ),
        sa.Column("created_at", sa.String(), nullable=False),
        sa.Column("updated_at", sa.String(), nullable=False),
        sa.Column("last_accessed_at", sa.String(), server_default=""),
        sa.Column("expires_at", sa.String(), server_default=""),
        sa.Column("tags", JSONB(), server_default=sa.text("'[]'")),
    )

    # Alter embedding column to the real vector type (pgvector DDL).
    op.execute(f"ALTER TABLE memories ALTER COLUMN embedding TYPE vector({EMBEDDING_DIM}) USING NULL")

    # Generated tsvector column for full-text search.
    op.execute(
        f"ALTER TABLE memories ADD COLUMN content_tsv tsvector "
        f"GENERATED ALWAYS AS ({_TSVEC_EXPR}) STORED"
    )

    # Indexes on memories.
    op.create_index("idx_memories_user", "memories", ["user_id"])
    op.create_index("idx_memories_user_scope", "memories", ["user_id", "scope_id"])
    op.create_index("idx_memories_scope_status", "memories", ["scope_id", "status"])
    op.create_index("idx_memories_scope_type", "memories", ["scope_id", "memory_type"])
    op.execute("CREATE INDEX idx_memories_content_tsv ON memories USING GIN (content_tsv)")
    op.execute(
        "CREATE INDEX idx_memories_embedding ON memories "
        "USING hnsw (embedding vector_cosine_ops)"
    )

    # -----------------------------------------------------------------------
    # memory_events
    # -----------------------------------------------------------------------
    op.execute(
        """
        CREATE TABLE memory_events (
            event_id  BIGSERIAL PRIMARY KEY,
            timestamp VARCHAR NOT NULL,
            event_type VARCHAR NOT NULL,
            memory_id VARCHAR NOT NULL,
            scope_id  VARCHAR NOT NULL DEFAULT '',
            detail    TEXT    NOT NULL DEFAULT ''
        )
        """
    )

    op.create_index("idx_memory_events_memory_id", "memory_events", ["memory_id"])
    op.create_index("idx_memory_events_scope", "memory_events", ["scope_id"])

    # -----------------------------------------------------------------------
    # memory_links
    # -----------------------------------------------------------------------
    op.create_table(
        "memory_links",
        sa.Column("source_id", sa.String(), nullable=False, primary_key=True),
        sa.Column("target_id", sa.String(), nullable=False, primary_key=True),
        sa.Column("link_type", sa.String(), nullable=False, primary_key=True, server_default="related"),
        sa.Column("created_at", sa.String(), nullable=False),
        sa.UniqueConstraint("source_id", "target_id", "link_type", name="uq_memory_links"),
    )

    # -----------------------------------------------------------------------
    # memory_watches
    # -----------------------------------------------------------------------
    op.create_table(
        "memory_watches",
        sa.Column("watch_id", sa.BigInteger(), primary_key=True, autoincrement=True),
        sa.Column("memory_id", sa.String(), nullable=False),
        sa.Column("watcher", sa.String(), nullable=False),
        sa.Column("created_at", sa.String(), nullable=False),
        sa.UniqueConstraint("memory_id", "watcher", name="uq_memory_watches"),
    )

    # -----------------------------------------------------------------------
    # memory_annotations
    # -----------------------------------------------------------------------
    op.create_table(
        "memory_annotations",
        sa.Column("annotation_id", sa.BigInteger(), primary_key=True, autoincrement=True),
        sa.Column("memory_id", sa.String(), nullable=False),
        sa.Column("author", sa.String(), server_default=""),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("created_at", sa.String(), nullable=False),
    )
    op.create_index("idx_memory_annotations_memory_id", "memory_annotations", ["memory_id"])

    # -----------------------------------------------------------------------
    # scope_access
    # -----------------------------------------------------------------------
    op.create_table(
        "scope_access",
        sa.Column("scope_id", sa.String(), nullable=False, primary_key=True),
        sa.Column("principal", sa.String(), nullable=False, primary_key=True),
        sa.Column("permission", sa.String(), nullable=False, primary_key=True, server_default="read"),
        sa.Column("created_at", sa.String(), nullable=False),
        sa.UniqueConstraint("scope_id", "principal", "permission", name="uq_scope_access"),
    )

    # -----------------------------------------------------------------------
    # stats_snapshots
    # -----------------------------------------------------------------------
    op.create_table(
        "stats_snapshots",
        sa.Column("snapshot_id", sa.BigInteger(), primary_key=True, autoincrement=True),
        sa.Column("timestamp", sa.String(), nullable=False),
        sa.Column("scope_id", sa.String(), nullable=False),
        sa.Column("data_json", sa.Text(), nullable=False),
    )
    op.create_index("idx_stats_snapshots_scope", "stats_snapshots", ["scope_id"])


def downgrade() -> None:
    op.drop_table("stats_snapshots")
    op.drop_table("scope_access")
    op.drop_table("memory_annotations")
    op.drop_table("memory_watches")
    op.drop_table("memory_links")
    op.drop_table("memory_events")
    op.drop_table("memories")
    op.execute("DROP EXTENSION IF EXISTS vector")
