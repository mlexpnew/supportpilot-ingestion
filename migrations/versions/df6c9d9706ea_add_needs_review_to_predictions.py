"""add_needs_review_to_predictions

Revision ID: df6c9d9706ea
Revises: 488d32cf6509
Create Date: 2026-10-29 17:59:29.575821

"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "df6c9d9706ea"
down_revision: Union[str, Sequence[str], None] = "488d32cf6509"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema: add needs_review boolean column to predictions table."""
    with op.batch_alter_table("predictions", schema=None) as batch_op:
        batch_op.add_column(
            sa.Column(
                "needs_review",
                sa.Boolean(),
                nullable=True,
            )
        )


def downgrade() -> None:
    """Downgrade schema: drop needs_review column from predictions table."""
    with op.batch_alter_table("predictions", schema=None) as batch_op:
        batch_op.drop_column("needs_review")
