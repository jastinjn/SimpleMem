"""Drop memory_events, memory_links, memory_watches, memory_annotations

Revision ID: 0006
Revises: 0005
Create Date: 2026-08-14
"""

from __future__ import annotations

import sqlalchemy as sa

from alembic import op

# revision identifiers, used by Alembic.
revision = "0006"
down_revision = "0005"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("DROP TABLE IF EXISTS memory_annotations")
    op.execute("DROP TABLE IF EXISTS memory_watches")
    op.execute("DROP TABLE IF EXISTS memory_links")
    op.execute("DROP TABLE IF EXISTS memory_events")


def downgrade() -> None:
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

    op.create_table(
        "memory_links",
        sa.Column("source_id", sa.String(), nullable=False, primary_key=True),
        sa.Column("target_id", sa.String(), nullable=False, primary_key=True),
        sa.Column("link_type", sa.String(), nullable=False, primary_key=True, server_default="related"),
        sa.Column("created_at", sa.String(), nullable=False),
        sa.UniqueConstraint("source_id", "target_id", "link_type", name="uq_memory_links"),
    )

    op.create_table(
        "memory_watches",
        sa.Column("watch_id", sa.BigInteger(), primary_key=True, autoincrement=True),
        sa.Column("memory_id", sa.String(), nullable=False),
        sa.Column("watcher", sa.String(), nullable=False),
        sa.Column("created_at", sa.String(), nullable=False),
        sa.UniqueConstraint("memory_id", "watcher", name="uq_memory_watches"),
    )

    op.create_table(
        "memory_annotations",
        sa.Column("annotation_id", sa.BigInteger(), primary_key=True, autoincrement=True),
        sa.Column("memory_id", sa.String(), nullable=False),
        sa.Column("author", sa.String(), server_default=""),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("created_at", sa.String(), nullable=False),
    )
    op.create_index("idx_memory_annotations_memory_id", "memory_annotations", ["memory_id"])
