"""Tests for Model Quality Monitoring (SP-120).

Covers:
  - Quality data store method joining predictions and feedback on ticket_id for a date range
  - Returns strictly predicted_label and correct_label, never ticket text
  - Counts predicted labels from the last 7 days
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from supportpilot.feedback import FeedbackStore
from supportpilot.feedback.store import init_db
from supportpilot.triage import LABELS


@pytest.fixture
def store(tmp_path: Path) -> FeedbackStore:
    db_file = tmp_path / "quality_test.db"
    init_db(db_file)
    return FeedbackStore(db_path=db_file)


def test_quality_data_joins_predictions_and_feedback(store: FeedbackStore) -> None:
    """Verify quality data method joins on ticket_id, filters by date range, and returns only labels."""
    # Seed predictions
    store.save_prediction(
        ticket_id="TICKET-1",
        predicted_label="billing",
        confidence=0.9,
        model_version="triage-v1.0.0-20261103",
        created_at="2026-11-01T10:00:00Z",
    )
    store.save_prediction(
        ticket_id="TICKET-2",
        predicted_label="technical",
        confidence=0.8,
        model_version="triage-v1.0.0-20261103",
        created_at="2026-11-02T10:00:00Z",
    )
    store.save_prediction(
        ticket_id="TICKET-3",
        predicted_label="delivery",
        confidence=0.85,
        model_version="triage-v1.0.0-20261103",
        created_at="2026-11-03T10:00:00Z",
    )
    # TICKET-4 has prediction but NO feedback
    store.save_prediction(
        ticket_id="TICKET-4",
        predicted_label="other",
        confidence=0.7,
        model_version="triage-v1.0.0-20261103",
        created_at="2026-11-04T10:00:00Z",
    )

    # Seed feedback for TICKET-1, TICKET-2, TICKET-3
    store.save_feedback(
        ticket_id="TICKET-1",
        correct_label="billing",
        agent_id="agent-1",
        created_at="2026-11-01T11:00:00Z",
    )
    store.save_feedback(
        ticket_id="TICKET-2",
        correct_label="refund",
        agent_id="agent-2",
        created_at="2026-11-02T11:00:00Z",
    )
    store.save_feedback(
        ticket_id="TICKET-3",
        correct_label="delivery",
        agent_id="agent-3",
        created_at="2026-11-03T11:00:00Z",
    )

    # 1. Query all pairs
    all_pairs = store.get_prediction_feedback_pairs()
    assert len(all_pairs) == 3
    # Check strict fields: only predicted_label and correct_label
    for pair in all_pairs:
        assert set(pair.keys()) == {"predicted_label", "correct_label"}
        assert "ticket_id" not in pair
        assert "body" not in pair
        assert "text" not in pair

    assert all_pairs[0] == {"predicted_label": "billing", "correct_label": "billing"}
    assert all_pairs[1] == {"predicted_label": "technical", "correct_label": "refund"}
    assert all_pairs[2] == {"predicted_label": "delivery", "correct_label": "delivery"}

    # 2. Query with date range
    filtered = store.get_prediction_feedback_pairs(
        start="2026-11-02T00:00:00Z",
        end="2026-11-03T00:00:00Z",
    )
    assert len(filtered) == 1
    assert filtered[0] == {"predicted_label": "technical", "correct_label": "refund"}

    # 3. Test alias get_quality_data
    alias_pairs = store.get_quality_data()
    assert alias_pairs == all_pairs


def test_counts_predicted_labels_last_7_days(store: FeedbackStore) -> None:
    """Verify second method counts predictions from last 7 days and excludes older ones."""
    ref_time = datetime(2026, 11, 10, 12, 0, 0, tzinfo=UTC)

    # 2 days old (within 7 days)
    store.save_prediction(
        ticket_id="T1",
        predicted_label="billing",
        confidence=0.9,
        model_version="v1",
        created_at="2026-11-08T12:00:00Z",
    )
    store.save_prediction(
        ticket_id="T2",
        predicted_label="billing",
        confidence=0.9,
        model_version="v1",
        created_at="2026-11-07T12:00:00Z",
    )
    store.save_prediction(
        ticket_id="T3",
        predicted_label="technical",
        confidence=0.8,
        model_version="v1",
        created_at="2026-11-05T12:00:00Z",
    )

    # 10 days old (outside 7 days: before 2026-11-03 12:00:00)
    store.save_prediction(
        ticket_id="T-OLD",
        predicted_label="delivery",
        confidence=0.9,
        model_version="v1",
        created_at="2026-10-30T12:00:00Z",
    )

    counts = store.get_predicted_label_counts_last_7_days(now=ref_time)
    assert counts["billing"] == 2
    assert counts["technical"] == 1
    assert counts["delivery"] == 0  # older than 7 days
    # All known labels are present in the dictionary
    for lbl in LABELS:
        assert lbl in counts

    # Test alias
    alias_counts = store.get_last_7_days_predicted_counts(now=ref_time)
    assert alias_counts == counts
