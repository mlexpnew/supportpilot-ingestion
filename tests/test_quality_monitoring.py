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


def test_confusion_matrix_and_accuracy_on_known_data() -> None:
    """Verify confusion matrix counts (predicted, actual) pairs and computes accuracy."""
    from supportpilot.monitoring.quality import (
        compute_accuracy,
        compute_confusion_matrix,
        find_most_confused_pair,
    )

    # 10 test pairs: 7 correct, 3 confused (2 billing->refund, 1 delivery->other)
    pairs = [
        {"predicted_label": "billing", "correct_label": "billing"},
        {"predicted_label": "billing", "correct_label": "billing"},
        {"predicted_label": "billing", "correct_label": "billing"},
        {"predicted_label": "billing", "correct_label": "refund"},
        {"predicted_label": "billing", "correct_label": "refund"},
        {"predicted_label": "delivery", "correct_label": "delivery"},
        {"predicted_label": "delivery", "correct_label": "delivery"},
        {"predicted_label": "delivery", "correct_label": "other"},
        {"predicted_label": "account", "correct_label": "account"},
        {"predicted_label": "cancellation", "correct_label": "cancellation"},
    ]

    cm = compute_confusion_matrix(pairs)

    # Check known counts
    assert cm["billing"]["billing"] == 3
    assert cm["billing"]["refund"] == 2
    assert cm["delivery"]["delivery"] == 2
    assert cm["delivery"]["other"] == 1
    assert cm["account"]["account"] == 1
    assert cm["cancellation"]["cancellation"] == 1
    # Check 2-tuple indexing
    assert cm[("billing", "refund")] == 2
    # Check unobserved pairs are 0
    assert cm["billing"]["delivery"] == 0
    assert cm["refund"]["refund"] == 0

    # Accuracy: 7 / 10 = 0.7
    acc = compute_accuracy(cm)
    assert acc == 0.7000

    # Accuracy from pairs directly
    acc_pairs = compute_accuracy(pairs)
    assert acc_pairs == 0.7000

    # Accuracy on empty data returns None
    assert compute_accuracy([]) is None
    assert compute_accuracy(compute_confusion_matrix([])) is None

    # Top confused pair: billing -> refund (count: 2)
    top_confused = find_most_confused_pair(cm)
    assert top_confused == ("billing", "refund", 2)


def test_precision_recall_derived_from_matrix() -> None:
    """Verify precision and recall per label derived from confusion matrix, returning None on zero division."""
    from supportpilot.monitoring.quality import (
        compute_confusion_matrix,
        compute_precision,
        compute_precision_recall,
        compute_recall,
    )

    pairs = [
        # billing: predicted 5 times (3 actual billing, 2 actual refund)
        {"predicted_label": "billing", "correct_label": "billing"},
        {"predicted_label": "billing", "correct_label": "billing"},
        {"predicted_label": "billing", "correct_label": "billing"},
        {"predicted_label": "billing", "correct_label": "refund"},
        {"predicted_label": "billing", "correct_label": "refund"},
        # delivery: predicted 2 times (2 actual delivery)
        {"predicted_label": "delivery", "correct_label": "delivery"},
        {"predicted_label": "delivery", "correct_label": "delivery"},
        # actual refund: 2 from billing, 1 from refund
        {"predicted_label": "refund", "correct_label": "refund"},
    ]

    cm = compute_confusion_matrix(pairs)
    precision, recall = compute_precision_recall(cm)

    # Precision(billing) = TP(3) / Total Pred(5) = 0.6
    assert precision["billing"] == 0.6000
    # Recall(billing) = TP(3) / Total Actual(3) = 1.0
    assert recall["billing"] == 1.0000

    # Precision(refund) = TP(1) / Total Pred(1) = 1.0
    assert precision["refund"] == 1.0000
    # Recall(refund) = TP(1) / Total Actual(3: 2 from billing, 1 from refund) = 1/3 = 0.3333
    assert recall["refund"] == 0.3333

    # Precision(delivery) = 2/2 = 1.0
    assert precision["delivery"] == 1.0000
    # Recall(delivery) = 2/2 = 1.0
    assert recall["delivery"] == 1.0000

    # Zero count labels: cancellation, account, other -> None (null)
    assert precision["cancellation"] is None
    assert recall["cancellation"] is None
    assert precision["account"] is None
    assert recall["account"] is None

    # Standalone helpers
    assert compute_precision(cm)["billing"] == 0.6000
    assert compute_recall(cm)["refund"] == 0.3333


def test_psi_matches_hand_calculated_value() -> None:
    """Verify PSI calculation matches a hand-calculated value and handles zero counts."""
    import math
    from supportpilot.monitoring.quality import compute_psi

    test_labels = ("delivery", "billing")
    exp = {"delivery": 50, "billing": 50}  # share 0.5, 0.5
    act = {"delivery": 60, "billing": 40}  # share 0.6, 0.4

    # Hand calculation:
    # term1 = (0.6 - 0.5) * ln(0.6 / 0.5) = 0.1 * ln(1.2) = 0.018232155679
    # term2 = (0.4 - 0.5) * ln(0.4 / 0.5) = -0.1 * ln(0.8) = 0.022314355131
    # total = 0.0405465... -> 0.0405
    expected_hand = round(
        (0.6 - 0.5) * math.log(0.6 / 0.5) + (0.4 - 0.5) * math.log(0.4 / 0.5), 4
    )
    assert expected_hand == 0.0405

    psi = compute_psi(actual_counts=act, expected_counts=exp, labels=test_labels)
    assert psi == 0.0405

    # Zero drift (identical distribution) -> 0.0
    zero_drift = compute_psi(actual_counts=exp, expected_counts=exp, labels=test_labels)
    assert zero_drift == 0.0000

    # Zero count handles epsilon 0.0001 avoiding log(0)
    act_with_zero = {"delivery": 100, "billing": 0}
    psi_zero = compute_psi(
        actual_counts=act_with_zero,
        expected_counts=exp,
        eps=0.0001,
        labels=test_labels,
    )
    eps = 0.0001
    hand_zero = round(
        (1.0 - 0.5) * math.log(1.0 / 0.5) + (eps - 0.5) * math.log(eps / 0.5), 4
    )
    assert psi_zero == hand_zero
    assert psi_zero == 4.6043

    # Empty actual counts returns None
    assert compute_psi(actual_counts={}, labels=test_labels) is None

