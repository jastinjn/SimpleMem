"""Drop scope_access and stats_snapshots tables

Revision ID: 0005
Revises: 0004
Create Date: 2026-08-14
"""

from __future__ import annotations

import sqlalchemy as sa

from alembic import op

revision = "0005"
down_revision = "0004"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.drop_table("scope_access")
    op.drop_table("stats_snapshots")


def downgrade() -> None:
    op.create_table(
        "stats_snapshots",
        sa.Column("snapshot_id", sa.BigInteger(), primary_key=True, autoincrement=True),
        sa.Column("timestamp", sa.String(), nullable=False),
        sa.Column("user_id", sa.String(), nullable=False),
        sa.Column("scope_id", sa.String(), nullable=False),
        sa.Column("data_json", sa.Text(), nullable=False),
    )
    op.create_index("idx_stats_snapshots_scope", "stats_snapshots", ["scope_id"])

    op.create_table(
        "scope_access",
        sa.Column("scope_id", sa.String(), nullable=False, primary_key=True),
        sa.Column("principal", sa.String(), nullable=False, primary_key=True),
        sa.Column("permission", sa.String(), nullable=False, primary_key=True, server_default="read"),
        sa.Column("created_at", sa.String(), nullable=False),
        sa.UniqueConstraint("scope_id", "principal", "permission", name="uq_scope_access"),
    )
