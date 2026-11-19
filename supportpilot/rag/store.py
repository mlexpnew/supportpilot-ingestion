"""pgvector storage and search for knowledge-base chunks (SP-122).

PostgreSQL only. The table is created by the Alembic migration
``9c4e2a7b1f30_create_kb_chunks``; :func:`create_kb_schema` mirrors it for tests.

Ingest is idempotent: chunks are keyed by ``content_hash``. Re-running ingest
adds nothing new, and chunks whose text changed (or were embedded with a
different model) are replaced.
"""

from __future__ import annotations

import os
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import numpy as np
from sqlalchemy import bindparam, create_engine, text
from sqlalchemy.engine import Connection, Engine

from supportpilot.feedback.store import normalize_database_url

from .chunker import Chunk


class KBStoreError(RuntimeError):
    """Raised for configuration problems (e.g. not a Postgres database)."""


@dataclass(frozen=True)
class SyncResult:
    added: int
    unchanged: int
    removed: int

    @property
    def total(self) -> int:
        return self.added + self.unchanged


@dataclass(frozen=True)
class SearchResult:
    article_id: str
    title: str
    category: str
    chunk_index: int
    text: str
    score: float


KB_SCHEMA_SQL: tuple[str, ...] = (
    "CREATE EXTENSION IF NOT EXISTS vector",
    """
    CREATE TABLE IF NOT EXISTS kb_chunks (
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
    """,
    (
        "CREATE INDEX IF NOT EXISTS idx_kb_chunks_embedding ON kb_chunks "
        "USING hnsw (embedding vector_cosine_ops)"
    ),
    "CREATE INDEX IF NOT EXISTS idx_kb_chunks_article_id ON kb_chunks (article_id)",
)


def create_kb_schema(conn: Connection) -> None:
    """Create the kb_chunks table if missing (tests / local setup only)."""
    for statement in KB_SCHEMA_SQL:
        conn.execute(text(statement))


def get_kb_engine(url: str | None = None) -> Engine:
    """Engine for the knowledge base. Requires a PostgreSQL DATABASE_URL.

    There is deliberately no SQLite fallback: vector search needs pgvector.
    """
    raw = url or os.getenv("DATABASE_URL", "")
    if not raw.strip():
        raise KBStoreError("DATABASE_URL is not set (PostgreSQL with pgvector needed)")
    engine = create_engine(normalize_database_url(raw), pool_pre_ping=True)
    if engine.dialect.name != "postgresql":
        raise KBStoreError(
            f"knowledge base needs PostgreSQL, got {engine.dialect.name!r}"
        )
    return engine


def to_vector_literal(vector: np.ndarray | Sequence[float]) -> str:
    """Format a vector as a pgvector text literal, e.g. '[0.1,0.2]'."""
    return "[" + ",".join(f"{float(x):.7f}" for x in vector) + "]"


_INSERT_SQL = text(
    """
    INSERT INTO kb_chunks
        (article_id, title, category, chunk_index, text,
         content_hash, model_name, embedding)
    VALUES
        (:article_id, :title, :category, :chunk_index, :text,
         :content_hash, :model_name, CAST(:embedding AS vector))
    ON CONFLICT (content_hash) DO NOTHING
    """
)

_DELETE_STALE_SQL = text(
    "DELETE FROM kb_chunks "
    "WHERE model_name <> :model_name OR content_hash NOT IN :keep"
).bindparams(bindparam("keep", expanding=True))

_SEARCH_SQL = text(
    """
    SELECT article_id, title, category, chunk_index, text,
           1 - (embedding <=> CAST(:query AS vector)) AS score
    FROM kb_chunks
    WHERE model_name = :model_name
    ORDER BY embedding <=> CAST(:query AS vector)
    LIMIT :k
    """
)


GROUP_OVERFETCH: int = 5


def best_chunk_per_article(
    chunks: Sequence[SearchResult], k: int
) -> list[SearchResult]:
    """Keep the highest-scoring chunk of each article, then the top k articles.

    ``chunks`` may arrive in any order; the result is sorted best first.
    """
    best: dict[str, SearchResult] = {}
    for chunk in chunks:
        current = best.get(chunk.article_id)
        if current is None or chunk.score > current.score:
            best[chunk.article_id] = chunk
    return sorted(best.values(), key=lambda r: r.score, reverse=True)[:k]


class KBStore:
    """Read/write access to the kb_chunks table."""

    def __init__(self, engine: Engine) -> None:
        if engine.dialect.name != "postgresql":
            raise KBStoreError("KBStore requires PostgreSQL with pgvector")
        self.engine = engine

    def existing_hashes(self, model_name: str) -> set[str]:
        """Hashes already stored for this model (no need to re-embed them)."""
        with self.engine.connect() as conn:
            rows = conn.execute(
                text("SELECT content_hash FROM kb_chunks WHERE model_name = :m"),
                {"m": model_name},
            )
            return {str(h).strip() for h in rows.scalars()}

    def apply(
        self,
        new_chunks: Sequence[Chunk],
        new_vectors: np.ndarray,
        keep_hashes: Iterable[str],
        model_name: str,
    ) -> SyncResult:
        """Insert new chunks and delete stale ones, in one transaction."""
        if len(new_chunks) != len(new_vectors):
            raise ValueError("new_chunks and new_vectors must have the same length")
        keep = sorted(set(keep_hashes))
        if not keep:
            raise ValueError("refusing to sync an empty knowledge base")

        rows = [
            {
                "article_id": c.article_id,
                "title": c.title,
                "category": c.category,
                "chunk_index": c.chunk_index,
                "text": c.text,
                "content_hash": c.content_hash,
                "model_name": model_name,
                "embedding": to_vector_literal(v),
            }
            for c, v in zip(new_chunks, new_vectors, strict=True)
        ]

        with self.engine.begin() as conn:
            removed = conn.execute(
                _DELETE_STALE_SQL, {"model_name": model_name, "keep": keep}
            ).rowcount
            added = 0
            for row in rows:
                added += conn.execute(_INSERT_SQL, row).rowcount
            total = conn.execute(text("SELECT count(*) FROM kb_chunks")).scalar_one()

        return SyncResult(added=added, unchanged=int(total) - added, removed=removed)

    def article_ids(self, model_name: str) -> set[str]:
        """Distinct article ids indexed for this model."""
        with self.engine.connect() as conn:
            rows = conn.execute(
                text("SELECT DISTINCT article_id FROM kb_chunks WHERE model_name = :m"),
                {"m": model_name},
            )
            return {str(a) for a in rows.scalars()}

    def count(self) -> int:
        with self.engine.connect() as conn:
            return int(
                conn.execute(text("SELECT count(*) FROM kb_chunks")).scalar_one()
            )

    def search(
        self,
        query_vector: np.ndarray,
        model_name: str,
        k: int = 5,
        group_by_article: bool = True,
    ) -> list[SearchResult]:
        """Top-k results by cosine similarity (score 1.0 = identical meaning).

        With ``group_by_article`` (the default, SP-123) we over-fetch
        ``k * GROUP_OVERFETCH`` chunks and keep only the best-scoring chunk per
        article, so one long article cannot fill every slot and push a second
        relevant article out of the top k. With it off, the raw top-k chunks
        are returned (the pre-SP-123 behaviour).
        """
        if k < 1:
            raise ValueError("k must be at least 1")
        fetch_k = k * GROUP_OVERFETCH if group_by_article else k
        chunks = self._search_chunks(query_vector, model_name, fetch_k)
        if not group_by_article:
            return chunks
        return best_chunk_per_article(chunks, k)

    def _search_chunks(
        self, query_vector: np.ndarray, model_name: str, k: int
    ) -> list[SearchResult]:
        """Raw top-k chunks, best first."""
        with self.engine.connect() as conn:
            rows = conn.execute(
                _SEARCH_SQL,
                {
                    "query": to_vector_literal(query_vector),
                    "model_name": model_name,
                    "k": k,
                },
            ).mappings()
            return [
                SearchResult(
                    article_id=r["article_id"],
                    title=r["title"],
                    category=r["category"],
                    chunk_index=int(r["chunk_index"]),
                    text=r["text"],
                    score=float(r["score"]),
                )
                for r in rows
            ]
