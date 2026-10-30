"""Unit and integration tests for FeedbackStore.get_stats and needs_review persistence (SP-114 Part A)."""

from __future__ import annotations

from datetime import date, datetime, timezone
from pathlib import Path

from starlette.testclient import TestClient

from supportpilot.feedback.models import DateRangeStats
from supportpilot.feedback.store import FeedbackStore, init_db
from supportpilot.triage import LABELS


def test_needs_review_column_saved_and_retrieved(tmp_path: Path) -> None:
    """Verify needs_review column is correctly persisted and retrieved as a boolean."""
    db_file = tmp_path / "test_nr.db"
    init_db(db_file)
    store = FeedbackStore(db_file)

    try:
        store.save_prediction(
            ticket_id="T-NR-1",
            predicted_label="delivery",
            confidence=0.92,
            model_version="triage-v1.0.0-20261022",
            created_at="2026-10-26T10:00:00+00:00",
            needs_review=False,
        )
        store.save_prediction(
            ticket_id="T-NR-2",
            predicted_label="billing",
            confidence=0.31,
            model_version="triage-v1.0.0-20261022",
            created_at="2026-10-26T11:00:00+00:00",
            needs_review=True,
        )

        p1 = store.get_prediction("T-NR-1")
        assert p1 is not None
        assert p1["needs_review"] is False

        p2 = store.get_prediction("T-NR-2")
        assert p2 is not None
        assert p2["needs_review"] is True
    finally:
        store.close()


def test_get_stats_empty_range(tmp_path: Path) -> None:
    """Verify get_stats on an empty database returns zero counts and None for accuracy/recalls."""
    db_file = tmp_path / "test_empty.db"
    init_db(db_file)
    store = FeedbackStore(db_file)

    try:
        stats = store.get_stats("2026-10-26", "2026-11-01")
        assert isinstance(stats, DateRangeStats)
        assert stats.start == "2026-10-26"
        assert stats.end == "2026-11-01"
        assert stats.total_triaged == 0
        assert stats.tickets_triaged == 0
        assert stats.needs_review_count == 0
        assert stats.needs_review_rate == 0.0
        assert stats.feedback_total == 0
        assert stats.feedback_count == 0
        assert stats.feedback_agreed == 0
        assert stats.live_accuracy is None
        assert stats.model_versions == []
        assert stats.daily_triage_counts == {}

        # All labels present in per_label_counts with 0
        assert set(stats.predictions_by_label.keys()) == set(LABELS)
        assert all(cnt == 0 for cnt in stats.predictions_by_label.values())

        # All labels present in per_label_recall with None
        assert set(stats.per_label_recall.keys()) == set(LABELS)
        assert all(rec is None for rec in stats.per_label_recall.values())
    finally:
        store.close()


def test_get_stats_aggregates_and_label_metrics(tmp_path: Path) -> None:
    """Verify get_stats computes exact triage counts, review rate, live accuracy, and per-label recall."""
    db_file = tmp_path / "test_aggregates.db"
    init_db(db_file)
    store = FeedbackStore(db_file)

    try:
        # Day 1: 2026-10-26 (2 delivery: 1 high conf, 1 low conf needs_review)
        store.save_prediction(
            "T-1",
            "delivery",
            0.95,
            "triage-v1.0.0-20261022",
            created_at="2026-10-26T09:00:00+00:00",
            needs_review=False,
        )
        store.save_prediction(
            "T-2",
            "delivery",
            0.30,
            "triage-v1.0.0-20261022",
            created_at="2026-10-26T14:30:00+00:00",
            needs_review=True,
        )

        # Day 2: 2026-10-27 (2 billing: both needs_review=False, model bumped to v1.1.0)
        store.save_prediction(
            "T-3",
            "billing",
            0.89,
            "triage-v1.1.0-20261027",
            created_at="2026-10-27T10:00:00+00:00",
            needs_review=False,
        )
        store.save_prediction(
            "T-4",
            "billing",
            0.85,
            "triage-v1.1.0-20261027",
            created_at="2026-10-27T16:00:00+00:00",
            needs_review=False,
        )

        # Day 3: 2026-10-28 (1 refund: needs_review=True)
        store.save_prediction(
            "T-5",
            "refund",
            0.25,
            "triage-v1.1.0-20261027",
            created_at="2026-10-28T11:00:00+00:00",
            needs_review=True,
        )

        # Out-of-range ticket: Before range (2026-10-25)
        store.save_prediction(
            "T-PAST",
            "account",
            0.90,
            "triage-v1.0.0-20261022",
            created_at="2026-10-25T23:59:59+00:00",
            needs_review=False,
        )

        # Out-of-range ticket: After range (2026-11-02)
        store.save_prediction(
            "T-FUTURE",
            "account",
            0.90,
            "triage-v1.1.0-20261027",
            created_at="2026-11-02T08:00:00+00:00",
            needs_review=False,
        )

        # Feedback in range:
        # T-1: Agent agrees delivery (correct_label='delivery') -> Agreed!
        store.save_feedback(
            "T-1",
            "delivery",
            "AG-1",
            created_at="2026-10-26T12:00:00+00:00",
        )
        # T-2: Predicted delivery, but Agent says refund -> Disagreed!
        store.save_feedback(
            "T-2",
            "refund",
            "AG-2",
            created_at="2026-10-27T09:00:00+00:00",
        )
        # T-3: Predicted billing, Agent says billing -> Agreed!
        store.save_feedback(
            "T-3",
            "billing",
            "AG-1",
            created_at="2026-10-28T14:00:00+00:00",
        )

        # Feedback out-of-range: T-4 feedback on 2026-11-02
        store.save_feedback(
            "T-4",
            "billing",
            "AG-3",
            created_at="2026-11-02T10:00:00+00:00",
        )

        # Query stats for pilot week
        stats = store.get_stats("2026-10-26", "2026-11-01")

        # 1. Total triaged: 5 tickets in range
        assert stats.total_triaged == 5
        assert stats.predictions_by_label["delivery"] == 2
        assert stats.predictions_by_label["billing"] == 2
        assert stats.predictions_by_label["refund"] == 1
        assert stats.predictions_by_label["account"] == 0
        assert stats.predictions_by_label["cancellation"] == 0
        assert stats.predictions_by_label["other"] == 0

        # 2. Needs review: T-2 and T-5 (2 of 5 = 40.0%)
        assert stats.needs_review_count == 2
        assert stats.needs_review_rate == 0.4

        # 3. Model versions used in range
        assert sorted(stats.model_versions) == [
            "triage-v1.0.0-20261022",
            "triage-v1.1.0-20261027",
        ]

        # 4. Daily triage counts
        assert stats.daily_triage_counts == {
            "2026-10-26": 2,
            "2026-10-27": 2,
            "2026-10-28": 1,
        }

        # 5. Feedback in range: T-1, T-2, T-3 (3 total, 2 agreed)
        assert stats.feedback_total == 3
        assert stats.feedback_agreed == 2
        assert stats.live_accuracy == 0.6667  # 2 / 3

        # 6. Per-label recall:
        # - Ground truth 'delivery': 1 ticket (T-1), agreed 1 -> 1.0
        # - Ground truth 'billing': 1 ticket (T-3), agreed 1 -> 1.0
        # - Ground truth 'refund': 1 ticket (T-2), agreed 0 (model predicted delivery) -> 0.0
        # - Others: 0 feedback -> None
        assert stats.per_label_recall["delivery"] == 1.0
        assert stats.per_label_recall["billing"] == 1.0
        assert stats.per_label_recall["refund"] == 0.0
        assert stats.per_label_recall["cancellation"] is None
        assert stats.per_label_recall["account"] is None
        assert stats.per_label_recall["other"] is None

        assert stats.per_label_feedback["delivery"] == {
            "total": 1,
            "agreed": 1,
            "recall": 1.0,
        }
        assert stats.per_label_feedback["refund"] == {
            "total": 1,
            "agreed": 0,
            "recall": 0.0,
        }
    finally:
        store.close()


def test_get_stats_date_boundaries_inclusive_and_exclusive(tmp_path: Path) -> None:
    """Verify date boundaries include the full end day (23:59:59) and exclude the next day."""
    db_file = tmp_path / "test_bounds.db"
    init_db(db_file)
    store = FeedbackStore(db_file)

    try:
        # Start edge: exact midnight
        store.save_prediction(
            "T-START",
            "delivery",
            0.8,
            "v1",
            created_at="2026-10-26T00:00:00+00:00",
        )
        # End edge: late night on end day
        store.save_prediction(
            "T-END-NIGHT",
            "billing",
            0.8,
            "v1",
            created_at="2026-11-01T23:59:59.999999+00:00",
        )
        # Next day midnight: excluded
        store.save_prediction(
            "T-NEXT-DAY",
            "refund",
            0.8,
            "v1",
            created_at="2026-11-02T00:00:00+00:00",
        )

        # Test string YYYY-MM-DD
        stats_str = store.get_stats("2026-10-26", "2026-11-01")
        assert stats_str.total_triaged == 2
        assert stats_str.predictions_by_label["delivery"] == 1
        assert stats_str.predictions_by_label["billing"] == 1
        assert stats_str.predictions_by_label["refund"] == 0

        # Test date objects
        stats_date = store.get_stats(date(2026, 10, 26), date(2026, 11, 1))
        assert stats_date.total_triaged == 2

        # Test datetime objects (exact range)
        stats_dt = store.get_stats(
            datetime(2026, 10, 26, 0, 0, 0, tzinfo=timezone.utc),
            datetime(2026, 11, 1, 12, 0, 0, tzinfo=timezone.utc),
        )
        # T-END-NIGHT was at 23:59:59, so it should be excluded when end datetime is 12:00:00
        assert stats_dt.total_triaged == 1
        assert stats_dt.predictions_by_label["delivery"] == 1
    finally:
        store.close()


def test_get_stats_start_inclusive_end_exclusive(tmp_path: Path) -> None:
    """Verify get_stats strictly includes start timestamp (>=) and excludes end timestamp (<)."""
    db_file = tmp_path / "test_start_inc_end_exc.db"
    init_db(db_file)
    store = FeedbackStore(db_file)

    try:
        # Row 1: Exactly at start boundary (2026-10-26T00:00:00+00:00) -> INCLUDED
        store.save_prediction(
            "T-START-EXACT",
            "delivery",
            0.9,
            "v1",
            created_at="2026-10-26T00:00:00+00:00",
        )
        # Row 2: Mid-range ticket -> INCLUDED
        store.save_prediction(
            "T-MID",
            "billing",
            0.9,
            "v1",
            created_at="2026-10-28T12:00:00+00:00",
        )
        # Row 3: Exactly at end boundary (2026-11-01T00:00:00+00:00) -> EXCLUDED
        store.save_prediction(
            "T-END-EXACT",
            "refund",
            0.9,
            "v1",
            created_at="2026-11-01T00:00:00+00:00",
        )

        # Test with timestamp strings: start inclusive, end exclusive
        stats = store.get_stats(
            "2026-10-26T00:00:00+00:00",
            "2026-11-01T00:00:00+00:00",
        )

        assert stats.total_triaged == 2
        assert stats.predictions_by_label["delivery"] == 1
        assert stats.predictions_by_label["billing"] == 1
        assert stats.predictions_by_label["refund"] == 0

        # Test with datetime objects: start inclusive, end exclusive
        stats_dt = store.get_stats(
            datetime(2026, 10, 26, 0, 0, 0, tzinfo=timezone.utc),
            datetime(2026, 11, 1, 0, 0, 0, tzinfo=timezone.utc),
        )
        assert stats_dt.total_triaged == 2
        assert stats_dt.predictions_by_label["delivery"] == 1
        assert stats_dt.predictions_by_label["billing"] == 1
        assert stats_dt.predictions_by_label["refund"] == 0
    finally:
        store.close()


def test_triage_api_persists_needs_review(auth_headers: dict[str, str]) -> None:
    """Verify POST /triage persists needs_review value into predictions table."""
    from supportpilot.api.main import app

    with TestClient(app) as client:
        # 1. High-confidence ticket (needs_review: False)
        resp1 = client.post(
            "/triage",
            json={
                "ticket_id": "T-API-NR-1",
                "body": "Where is my parcel tracking delivery?",
            },
            headers=auth_headers,
        )
        assert resp1.status_code == 200
        data1 = resp1.json()

        saved1 = app.state.feedback_store.get_prediction("T-API-NR-1")
        assert saved1 is not None
        assert saved1["needs_review"] == data1["needs_review"]
        assert saved1["needs_review"] is False

        # 2. Ambiguous noise ticket (needs_review: True)
        resp2 = client.post(
            "/triage",
            json={"ticket_id": "T-API-NR-2", "body": "hello ?"},
            headers=auth_headers,
        )
        assert resp2.status_code == 200
        data2 = resp2.json()

        saved2 = app.state.feedback_store.get_prediction("T-API-NR-2")
        assert saved2 is not None
        assert saved2["needs_review"] == data2["needs_review"]
        assert saved2["needs_review"] is True


def test_get_stats_null_needs_review_excluded_from_rate(tmp_path: Path) -> None:
    """Verify historical rows with NULL needs_review are excluded from review rate."""
    db_file = tmp_path / "test_null_nr.db"
    init_db(db_file)
    store = FeedbackStore(db_file)

    try:
        # 1. Old historical row (pre-migration / untracked): needs_review is None
        store.save_prediction(
            ticket_id="T-OLD-1",
            predicted_label="delivery",
            confidence=0.9,
            model_version="v1",
            created_at="2026-10-26T10:00:00+00:00",
            needs_review=None,
        )
        store.save_prediction(
            ticket_id="T-OLD-2",
            predicted_label="billing",
            confidence=0.8,
            model_version="v1",
            created_at="2026-10-27T10:00:00+00:00",
            needs_review=None,
        )

        # 2. New rows where needs_review was tracked (starting 2026-10-29)
        # 2 tracked rows: 1 flagged (needs_review=True), 1 not flagged (needs_review=False)
        store.save_prediction(
            ticket_id="T-NEW-1",
            predicted_label="delivery",
            confidence=0.3,
            model_version="v1",
            created_at="2026-10-29T11:00:00+00:00",
            needs_review=True,
        )
        store.save_prediction(
            ticket_id="T-NEW-2",
            predicted_label="refund",
            confidence=0.95,
            model_version="v1",
            created_at="2026-10-30T12:00:00+00:00",
            needs_review=False,
        )

        stats = store.get_stats("2026-10-26", "2026-11-01")

        # Total tickets triaged = 4 (2 old + 2 new)
        assert stats.total_triaged == 4

        # Tracked tickets = 2 (only the new ones)
        assert stats.needs_review_tracked == 2
        assert stats.needs_review_count == 1

        # Review rate must be 1 / 2 = 50.0% (NOT 1 / 4 = 25.0%)
        assert stats.needs_review_rate == 0.50

        # First tracked date is 2026-10-29
        assert stats.first_tracked_date == "2026-10-29"
    finally:
        store.close()
