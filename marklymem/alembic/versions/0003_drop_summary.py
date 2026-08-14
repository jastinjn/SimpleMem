"""drop summary column and update content_tsv generated expression

Revision ID: 0003
Revises: 790171400532
Create Date: 2026-08-14

"""
from typing import Sequence, Union

from alembic import op

revision: str = "0003"
down_revision: Union[str, Sequence[str], None] = "790171400532"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Drop the generated column first — it references summary in its expression.
    op.execute("ALTER TABLE memories DROP COLUMN content_tsv")
    op.execute("ALTER TABLE memories DROP COLUMN summary")
    op.execute(
        "ALTER TABLE memories ADD COLUMN content_tsv tsvector "
        "GENERATED ALWAYS AS (to_tsvector('english', coalesce(content,''))) STORED"
    )
    op.execute(
        "CREATE INDEX IF NOT EXISTS idx_memories_content_tsv "
        "ON memories USING gin(content_tsv)"
    )


def downgrade() -> None:
    op.execute("ALTER TABLE memories DROP COLUMN content_tsv")
    op.execute("ALTER TABLE memories ADD COLUMN summary TEXT NOT NULL DEFAULT ''")
    op.execute(
        "ALTER TABLE memories ADD COLUMN content_tsv tsvector "
        "GENERATED ALWAYS AS "
        "(to_tsvector('english', coalesce(content,'') || ' ' || coalesce(summary,''))) STORED"
    )
    op.execute(
        "CREATE INDEX IF NOT EXISTS idx_memories_content_tsv "
        "ON memories USING gin(content_tsv)"
    )
