"""create_kb_chunks (SP-122 ReplyPilot knowledge base)

Revision ID: 9c4e2a7b1f30
Revises: 1423b4cbf851
Create Date: 2026-11-10 14:00:00

PostgreSQL only: enables pgvector and creates kb_chunks with an HNSW cosine
index. On SQLite (local dev/tests) this migration is a no-op, because vector
search is not supported there.
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision: str = "9c4e2a7b1f30"
down_revision: Union[str, Sequence[str], None] = "1423b4cbf851"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Enable pgvector and create kb_chunks (Postgres only)."""
    bind = op.get_bind()
    if bind.dialect.name != "postgresql":
        return

    op.execute(sa.text("SET lock_timeout = '30s'"))
    op.execute(sa.text("CREATE EXTENSION IF NOT EXISTS vector"))
    op.execute(
        sa.text(
            """
            CREATE TABLE kb_chunks (
                id BIGSERIAL PRIMARY KEY,
                article_id VARCHAR(128) NOT NULL,
                title TEXT NOT NULL,
                category VARCHAR(32) NOT NULL,
                chunk_index INTEGER NOT NULL,
                text TEXT NOT NULL,
                content_hash CHAR(64) NOT NULL,
                model_name VARCHAR(128) NOT NULL,
                embedding vector(384) NOT NULL,
                created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                CONSTRAINT uq_kb_chunks_content_hash UNIQUE (content_hash)
            )
            """
        )
    )
    op.execute(
        sa.text(
            "CREATE INDEX idx_kb_chunks_embedding ON kb_chunks "
            "USING hnsw (embedding vector_cosine_ops)"
        )
    )
    op.execute(
        sa.text("CREATE INDEX idx_kb_chunks_article_id ON kb_chunks (article_id)")
    )


def downgrade() -> None:
    """Drop kb_chunks (keeps the vector extension, other objects may use it)."""
    bind = op.get_bind()
    if bind.dialect.name != "postgresql":
        return

    op.execute(sa.text("SET lock_timeout = '30s'"))
    op.execute(sa.text("DROP TABLE IF EXISTS kb_chunks"))
