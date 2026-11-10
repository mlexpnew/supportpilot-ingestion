"""pgvector store, ingest and search tests (SP-122).

Needs a PostgreSQL DATABASE_URL with the pgvector extension available
(CI uses the pgvector/pgvector:pg18 image). Skipped otherwise.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from supportpilot.rag.embedder import EMBEDDING_DIM, Embedder

pytestmark = pytest.mark.skipif(
    not os.getenv("DATABASE_URL", "").startswith("postgres"),
    reason="needs a PostgreSQL DATABASE_URL with pgvector",
)

HELP_CENTER = Path(__file__).resolve().parents[1] / "data" / "help_center"


class FakeModel:
    def encode(self, sentences: list[str], **kwargs: Any) -> np.ndarray:
        rows = []
        for s in sentences:
            seed = int(hashlib.sha256(s.encode()).hexdigest()[:8], 16)
            rows.append(np.random.default_rng(seed).normal(size=EMBEDDING_DIM))
        return np.array(rows)


@pytest.fixture()
def store() -> Any:
    from sqlalchemy import text

    from supportpilot.rag.store import KBStore, create_kb_schema, get_kb_engine

    engine = get_kb_engine()
    with engine.begin() as conn:
        create_kb_schema(conn)
        conn.execute(text("TRUNCATE kb_chunks"))
    yield KBStore(engine)
    with engine.begin() as conn:
        conn.execute(text("TRUNCATE kb_chunks"))


@pytest.fixture()
def fake_embedder() -> Embedder:
    return Embedder(model=FakeModel(), model_name="fake-model")


def _copy_articles(dst: Path) -> None:
    for src in HELP_CENTER.glob("*.md"):
        (dst / src.name).write_text(src.read_text(encoding="utf-8"), encoding="utf-8")


def test_ingest_twice_adds_nothing_second_time(
    store: Any, fake_embedder: Embedder
) -> None:
    from supportpilot.rag.ingest import ingest_folder

    first = ingest_folder(HELP_CENTER, store, fake_embedder)
    assert first.added > 0 and first.removed == 0
    rows_after_first = store.count()

    second = ingest_folder(HELP_CENTER, store, fake_embedder)
    assert second.added == 0
    assert second.removed == 0
    assert second.unchanged == rows_after_first
    assert store.count() == rows_after_first


def test_edited_article_replaces_only_changed_chunk(
    store: Any, fake_embedder: Embedder, tmp_path: Path
) -> None:
    from supportpilot.rag.ingest import ingest_folder

    _copy_articles(tmp_path)
    ingest_folder(tmp_path, store, fake_embedder)
    before = store.count()

    path = tmp_path / "track-order.md"
    path.write_text(
        path.read_text(encoding="utf-8").replace("24 hours", "48 hours"),
        encoding="utf-8",
    )
    result = ingest_folder(tmp_path, store, fake_embedder)
    assert result.added == 1
    assert result.removed == 1
    assert store.count() == before


def test_model_change_replaces_everything(store: Any, tmp_path: Path) -> None:
    from supportpilot.rag.ingest import ingest_folder

    _copy_articles(tmp_path)
    ingest_folder(tmp_path, store, Embedder(model=FakeModel(), model_name="m1"))
    n = store.count()

    result = ingest_folder(
        tmp_path, store, Embedder(model=FakeModel(), model_name="m2")
    )
    assert result.removed == n
    assert result.added == n
    assert store.count() == n


def test_search_ranks_exact_match_first(store: Any, fake_embedder: Embedder) -> None:
    from supportpilot.rag.chunker import chunk_folder
    from supportpilot.rag.ingest import ingest_folder

    ingest_folder(HELP_CENTER, store, fake_embedder)
    target = chunk_folder(HELP_CENTER)[5]

    results = store.search(
        fake_embedder.embed_query(target.text), fake_embedder.model_name, k=3
    )
    assert len(results) == 3
    assert results[0].text == target.text
    assert results[0].score == pytest.approx(1.0, abs=1e-4)
    scores = [r.score for r in results]
    assert scores == sorted(scores, reverse=True)


def test_search_ignores_other_models(store: Any, fake_embedder: Embedder) -> None:
    from supportpilot.rag.ingest import ingest_folder

    ingest_folder(HELP_CENTER, store, fake_embedder)
    query = fake_embedder.embed_query("anything")
    assert store.search(query, "some-other-model", k=5) == []


SANITY_QUESTIONS: list[tuple[str, str]] = [
    ("where is my parcel", "kb-track-order"),
    ("I was charged two times for the same order", "kb-double-charge"),
    ("how many days until I get my money back", "kb-refund-timeline"),
    ("I forgot my password and cannot log in", "kb-reset-password"),
    ("I want to stop my monthly subscription", "kb-cancel-subscription"),
]


def test_sanity_questions_find_right_article_in_top_3(store: Any) -> None:
    pytest.importorskip("sentence_transformers")
    from supportpilot.rag.ingest import ingest_folder
    from supportpilot.rag.search import search

    embedder = Embedder()  # real pinned model
    ingest_folder(HELP_CENTER, store, embedder)

    misses = []
    for question, expected in SANITY_QUESTIONS:
        top3 = [r.article_id for r in search(question, store, embedder, k=3)]
        if expected not in top3:
            misses.append((question, expected, top3))
    assert not misses, f"retrieval misses: {misses}"
