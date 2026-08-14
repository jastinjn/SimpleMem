"""make source_session_id nullable

Revision ID: 0004
Revises: 0003
Create Date: 2026-08-14
"""

from typing import Sequence, Union

import sqlalchemy as sa

from alembic import op

revision: str = "0004"
down_revision: Union[str, Sequence[str], None] = "0003"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.alter_column("memories", "source_session_id", existing_type=sa.String(), nullable=True)
    op.execute("UPDATE memories SET source_session_id = NULL WHERE source_session_id = ''")


def downgrade() -> None:
    op.execute("UPDATE memories SET source_session_id = '' WHERE source_session_id IS NULL")
    op.alter_column("memories", "source_session_id", existing_type=sa.String(), nullable=False)
