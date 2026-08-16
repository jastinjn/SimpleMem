"""Add text_pattern_ops index on memories.scope_id for hierarchical prefix queries

Revision ID: 0007
Revises: 0006
Create Date: 2026-08-16
"""

from __future__ import annotations

from alembic import op

revision = "0007"
down_revision = "0006"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        "CREATE INDEX ix_memories_scope_id_pattern "
        "ON memories (scope_id text_pattern_ops)"
    )


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS ix_memories_scope_id_pattern")
