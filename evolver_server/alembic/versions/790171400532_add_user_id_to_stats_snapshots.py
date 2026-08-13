"""add_user_id_to_stats_snapshots

Revision ID: 790171400532
Revises: 0001
Create Date: 2026-08-13 22:34:43.532334

"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = '790171400532'
down_revision: Union[str, Sequence[str], None] = '0001'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column('stats_snapshots', sa.Column('user_id', sa.String(), nullable=False, server_default=''))
    op.alter_column('stats_snapshots', 'user_id', server_default=None)


def downgrade() -> None:
    op.drop_column('stats_snapshots', 'user_id')
