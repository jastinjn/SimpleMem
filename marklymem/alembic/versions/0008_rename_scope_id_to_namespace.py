"""rename scope_id to namespace in memories table

Revision ID: 0008
Revises: 0007
"""
from alembic import op

revision = "0008"
down_revision = "0007"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Drop old indexes before renaming the column
    op.drop_index("idx_memories_user_scope", table_name="memories")
    op.drop_index("idx_memories_scope_status", table_name="memories")
    op.drop_index("idx_memories_scope_type", table_name="memories")
    op.drop_index("ix_memories_scope_id_pattern", table_name="memories")

    op.alter_column("memories", "scope_id", new_column_name="namespace")

    # Recreate indexes with the new column name
    op.create_index("idx_memories_user_namespace", "memories", ["user_id", "namespace"])
    op.create_index("idx_memories_namespace_status", "memories", ["namespace", "status"])
    op.create_index("idx_memories_namespace_type", "memories", ["namespace", "memory_type"])
    op.execute(
        "CREATE INDEX ix_memories_namespace_pattern ON memories (namespace text_pattern_ops)"
    )


def downgrade() -> None:
    op.drop_index("idx_memories_user_namespace", table_name="memories")
    op.drop_index("idx_memories_namespace_status", table_name="memories")
    op.drop_index("idx_memories_namespace_type", table_name="memories")
    op.execute("DROP INDEX IF EXISTS ix_memories_namespace_pattern")

    op.alter_column("memories", "namespace", new_column_name="scope_id")

    op.create_index("idx_memories_user_scope", "memories", ["user_id", "scope_id"])
    op.create_index("idx_memories_scope_status", "memories", ["scope_id", "status"])
    op.create_index("idx_memories_scope_type", "memories", ["scope_id", "memory_type"])
    op.execute(
        "CREATE INDEX ix_memories_scope_id_pattern ON memories (scope_id text_pattern_ops)"
    )
