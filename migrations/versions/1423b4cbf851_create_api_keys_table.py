"""create_api_keys_table

Revision ID: 1423b4cbf851
Revises: 7d0614baf146
Create Date: 2026-11-03 23:02:00.252900

"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "1423b4cbf851"
down_revision: Union[str, Sequence[str], None] = "7d0614baf146"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema: create api_keys table with lock_timeout guard."""
    bind = op.get_bind()
    if bind.dialect.name == "postgresql":
        op.execute(sa.text("SET lock_timeout = '30s'"))

    op.create_table(
        "api_keys",
        sa.Column("name", sa.String(length=64), nullable=False),
        sa.Column("key_hash", sa.String(length=64), nullable=False),
        sa.Column("role", sa.String(length=32), nullable=False),
        sa.Column("created_at", sa.String(length=64), nullable=False),
        sa.Column("revoked_at", sa.String(length=64), nullable=True),
        sa.Column("last_used_at", sa.String(length=64), nullable=True),
        sa.PrimaryKeyConstraint("name"),
        sa.UniqueConstraint("key_hash", name="uq_api_keys_key_hash"),
    )
    op.create_index("idx_api_keys_key_hash", "api_keys", ["key_hash"])


def downgrade() -> None:
    """Downgrade schema: drop api_keys table with lock_timeout guard."""
    bind = op.get_bind()
    if bind.dialect.name == "postgresql":
        op.execute(sa.text("SET lock_timeout = '30s'"))

    op.drop_index("idx_api_keys_key_hash", table_name="api_keys")
    op.drop_table("api_keys")
