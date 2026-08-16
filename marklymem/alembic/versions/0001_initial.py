"""Initial schema — PostgreSQL + pgvector

Revision ID: 0001
Revises:
Create Date: 2026-08-16
"""

from __future__ import annotations

import os

import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB

from alembic import op

revision = "0001"
down_revision = None
branch_labels = None
depends_on = None

EMBEDDING_DIM: int = int(os.getenv("EVOLVER_EMBEDDING_DIM", "1024"))


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")

    op.create_table(
        "memories",
        sa.Column("memory_id", sa.String(), primary_key=True),
        sa.Column("user_id", sa.String(), nullable=False),
        sa.Column("namespace", sa.String(), nullable=True),
        sa.Column("memory_type", sa.String(), nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("source_session_id", sa.String(), nullable=True),
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
        sa.Column("embedding", sa.Text(), nullable=True),  # altered to vector below
        sa.Column("created_at", sa.String(), nullable=False),
        sa.Column("updated_at", sa.String(), nullable=False),
        sa.Column("last_accessed_at", sa.String(), server_default=""),
        sa.Column("expires_at", sa.String(), server_default=""),
        sa.Column("tags", JSONB(), server_default=sa.text("'[]'")),
    )

    op.execute(
        f"ALTER TABLE memories ALTER COLUMN embedding TYPE vector({EMBEDDING_DIM}) USING NULL"
    )
    op.execute(
        "ALTER TABLE memories ADD COLUMN content_tsv tsvector "
        "GENERATED ALWAYS AS (to_tsvector('english', coalesce(content,''))) STORED"
    )

    op.create_index("idx_memories_user", "memories", ["user_id"])
    op.create_index("idx_memories_user_namespace", "memories", ["user_id", "namespace"])
    op.create_index("idx_memories_namespace_status", "memories", ["namespace", "status"])
    op.create_index("idx_memories_namespace_type", "memories", ["namespace", "memory_type"])
    op.execute("CREATE INDEX idx_memories_content_tsv ON memories USING GIN (content_tsv)")
    op.execute(
        "CREATE INDEX idx_memories_embedding ON memories "
        "USING hnsw (embedding vector_cosine_ops)"
    )
    op.execute(
        "CREATE INDEX ix_memories_namespace_pattern ON memories (namespace text_pattern_ops)"
    )


def downgrade() -> None:
    op.drop_table("memories")
    op.execute("DROP EXTENSION IF EXISTS vector")
