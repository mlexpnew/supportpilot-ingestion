"""Retrieval evaluation tests and regression gate (SP-123).

The scoring and golden-set tests run everywhere. The regression gate needs a
PostgreSQL DATABASE_URL with pgvector and the real embedding model; it runs in
the ``test-postgres`` CI job, which sets ``REQUIRE_RETRIEVAL_GATE=1`` so the
gate can never be silently skipped there.
"""

from __future__ import annotations

import json
import os
from collections import Counter
from pathlib import Path
from typing import Any

import pytest

from supportpilot.rag.chunker import load_articles
from supportpilot.rag.evaluate import (
    GoldenQuestion,
    GoldenSetError,
    evaluate_questions,
    first_correct_rank,
    load_golden,
)

ROOT = Path(__file__).resolve().parents[1]
HELP_CENTER = ROOT / "data" / "help_center"
DEV = ROOT / "data" / "eval" / "retrieval_dev.jsonl"
HELDOUT = ROOT / "data" / "eval" / "retrieval_test.jsonl"
HELDOUT_BASELINE = ROOT / "data" / "eval" / "retrieval_test_baseline.json"
GATE_MARGIN = 0.05
CATEGORIES = {"delivery", "refund", "billing", "cancellation", "account", "other"}
STYLES = {"paraphrase", "typo", "hinglish", "vague", "short"}


def _kb_ids() -> set[str]:
    return {a.article_id for a in load_articles(HELP_CENTER)}


# --------------------------------------------------------------------------
# Scoring
# --------------------------------------------------------------------------


def test_first_correct_rank() -> None:
    assert first_correct_rank(["a", "b", "c"], ["a"]) == 1
    assert first_correct_rank(["a", "b", "c"], ["c", "b"]) == 2
    assert first_correct_rank(["a", "b"], ["z"]) == 0
    assert first_correct_rank([], ["a"]) == 0


def test_metrics_on_known_ranks() -> None:
    questions = [
        GoldenQuestion("q1", ("a",), "delivery", "short"),  # rank 1
        GoldenQuestion("q2", ("b",), "delivery", "typo"),  # rank 2
        GoldenQuestion("q3", ("c",), "refund", "typo"),  # rank 4 (miss @3)
        GoldenQuestion("q4", ("z",), "refund", "vague"),  # not found
    ]
    ranked = {
        "q1": ["a", "x", "y"],
        "q2": ["x", "b", "y"],
        "q3": ["x", "y", "w", "c"],
        "q4": ["x", "y", "w"],
    }
    report = evaluate_questions(
        questions, ranked.__getitem__, group_by_article=True, model_name="fake"
    )
    assert report.recall_at_1 == 0.25
    assert report.recall_at_3 == 0.5
    assert report.mrr == pytest.approx((1 + 1 / 2 + 1 / 4 + 0) / 4, abs=1e-4)
    assert report.recall_at_3_by_category == {"delivery": 1.0, "refund": 0.0}
    assert [m["question"] for m in report.misses] == ["q3", "q4"]
    assert report.misses[0]["rank"] == 4


def test_raw_chunks_rank_by_slot_position() -> None:
    """Without grouping, one article filling 3 slots pushes the answer to 4."""
    questions = [GoldenQuestion("q", ("b",), "delivery", "short")]
    report = evaluate_questions(
        questions,
        lambda _: ["a", "a", "a", "b"],
        group_by_article=False,
        model_name="fake",
    )
    assert report.results[0]["rank"] == 4
    assert report.recall_at_3 == 0.0


def test_best_chunk_per_article_keeps_highest_score() -> None:
    from supportpilot.rag.store import SearchResult, best_chunk_per_article

    def r(article: str, idx: int, s: float) -> SearchResult:
        return SearchResult(article, article, "delivery", idx, "t", s)

    chunks = [r("a", 0, 0.9), r("a", 1, 0.95), r("a", 2, 0.8), r("b", 0, 0.7)]
    top = best_chunk_per_article(chunks, k=2)
    assert [(x.article_id, x.chunk_index) for x in top] == [("a", 1), ("b", 0)]
    assert len(best_chunk_per_article(chunks, k=1)) == 1


# --------------------------------------------------------------------------
# Golden set quality
# --------------------------------------------------------------------------


def test_dev_and_heldout_split_is_clean() -> None:
    dev = load_golden(DEV, _kb_ids())
    test = load_golden(HELDOUT, _kb_ids())
    assert (len(dev), len(test)) == (25, 15)

    dev_q = {q.question.lower() for q in dev}
    test_q = {q.question.lower() for q in test}
    assert len(dev_q) == 25 and len(test_q) == 15, "duplicate question"
    assert not dev_q & test_q, "held-out question leaked into dev"

    for name, qs in (("dev", dev), ("held-out", test)):
        assert {q.category for q in qs} == CATEGORIES, name
        assert {q.style for q in qs} == STYLES, name
    assert min(Counter(q.category for q in test).values()) >= 2
    assert set(Counter(q.style for q in test).values()) == {3}
    assert min(Counter(q.category for q in [*dev, *test]).values()) >= 5


def test_golden_category_matches_an_expected_article() -> None:
    category_of = {a.article_id: a.category for a in load_articles(HELP_CENTER)}
    for path in (DEV, HELDOUT):
        for q in load_golden(path, category_of):
            cats = {category_of[a] for a in q.expected_article_ids}
            assert q.category in cats, (q.question, q.category, cats)


def test_golden_set_fails_loudly_on_unknown_id(tmp_path: Path) -> None:
    bad = tmp_path / "golden.jsonl"
    bad.write_text(
        json.dumps(
            {
                "question": "where is it",
                "expected_article_ids": ["kb-does-not-exist"],
                "category": "delivery",
                "style": "vague",
            }
        )
        + "\n"
    )
    with pytest.raises(GoldenSetError, match="kb-does-not-exist"):
        load_golden(bad, _kb_ids())


# --------------------------------------------------------------------------
# Regression gate (real model + pgvector)
# --------------------------------------------------------------------------

_HAS_POSTGRES = os.getenv("DATABASE_URL", "").startswith("postgres")
_GATE_REQUIRED = os.getenv("REQUIRE_RETRIEVAL_GATE") == "1"


@pytest.mark.skipif(
    not (_HAS_POSTGRES or _GATE_REQUIRED),
    reason="needs a PostgreSQL DATABASE_URL with pgvector",
)
def test_heldout_retrieval_does_not_regress() -> None:
    """CI gate: held-out set only, recall@3 AND MRR within 0.05 of baseline.

    With 15 questions one extra miss costs 0.067 recall@3, so in practice the
    gate allows no new held-out misses.
    """
    from sqlalchemy import text

    from supportpilot.rag.embedder import Embedder
    from supportpilot.rag.evaluate import format_report, run_evaluation
    from supportpilot.rag.ingest import ingest_folder
    from supportpilot.rag.store import KBStore, create_kb_schema, get_kb_engine

    assert HELDOUT_BASELINE.exists(), (
        f"{HELDOUT_BASELINE} missing: run `python -m supportpilot.rag.evaluate "
        "--golden data/eval/retrieval_test.jsonl "
        "--json data/eval/retrieval_test_baseline.json`"
    )
    baseline: dict[str, Any] = json.loads(HELDOUT_BASELINE.read_text())
    recall_floor = round(baseline["recall_at_3"] - GATE_MARGIN, 4)
    mrr_floor = round(baseline["mrr"] - GATE_MARGIN, 4)

    engine = get_kb_engine()
    with engine.begin() as conn:
        create_kb_schema(conn)
        conn.execute(text("TRUNCATE kb_chunks"))
    store = KBStore(engine)
    embedder = Embedder()  # real pinned model, no fakes
    assert embedder.model_name == baseline["model_name"], "model changed: re-baseline"

    try:
        ingest_folder(HELP_CENTER, store, embedder)
        report = run_evaluation(HELDOUT, HELP_CENTER, store=store, embedder=embedder)
    finally:
        with engine.begin() as conn:
            conn.execute(text("TRUNCATE kb_chunks"))

    details = format_report(report)
    assert report.recall_at_3 >= recall_floor, (
        f"held-out recall@3 {report.recall_at_3:.3f} < floor {recall_floor:.3f}\n"
        + details
    )
    assert report.mrr >= mrr_floor, (
        f"held-out MRR {report.mrr:.3f} < floor {mrr_floor:.3f}\n" + details
    )
