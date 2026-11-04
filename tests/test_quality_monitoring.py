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


def test_metrics_quality_endpoint_auth_and_insufficient_data() -> None:
    """Verify GET /metrics/quality enforces admin auth and returns insufficient_data when < 30 feedback rows."""
    from fastapi.testclient import TestClient
    from supportpilot.api.main import app

    with TestClient(app) as client:
        # 1. Unauthenticated -> 401
        res_no_auth = client.get("/metrics/quality")
        assert res_no_auth.status_code == 401

        # 2. Client key -> 401 (only admin allowed)
        res_client = client.get("/metrics/quality", headers={"X-API-Key": "test-key-1"})
        assert res_client.status_code == 401

        # 3. Admin key with 0 feedback rows -> insufficient_data
        res_admin = client.get(
            "/metrics/quality", headers={"X-API-Key": "test-admin-key-1"}
        )
        assert res_admin.status_code == 200
        data = res_admin.json()
        assert data["status"] == "insufficient_data"
        assert "not enough data" in data["message"].lower()
        assert data["sample_size"] == 0
        assert data["accuracy"] is None
        assert data["precision"] is None
        assert data["recall"] is None
        assert data["confusion_matrix"] is None
        assert data["matrix"] is None


def test_metrics_quality_endpoint_sufficient_data_and_days_filter() -> None:
    """Verify GET /metrics/quality returns full metrics when >= 30 rows and respects days parameter."""
    from datetime import UTC, datetime
    from fastapi.testclient import TestClient
    from supportpilot.api.main import app

    now = datetime.now(UTC)
    now_iso = now.isoformat()
    old_iso = (now - timedelta(days=5)).isoformat()

    with TestClient(app) as client:
        store: FeedbackStore = client.app.state.feedback_store

        # Seed 32 recent feedback rows (within 2 days): 28 billing agreed, 4 billing->refund
        for i in range(28):
            tid = f"T-RECENT-{i}"
            store.save_prediction(
                ticket_id=tid,
                predicted_label="billing",
                confidence=0.9,
                model_version="triage-v1.0.0-20261103",
                created_at=now_iso,
            )
            store.save_feedback(
                ticket_id=tid,
                correct_label="billing",
                agent_id="agent-1",
                created_at=now_iso,
            )

        for i in range(28, 32):
            tid = f"T-RECENT-{i}"
            store.save_prediction(
                ticket_id=tid,
                predicted_label="billing",
                confidence=0.85,
                model_version="triage-v1.0.0-20261103",
                created_at=now_iso,
            )
            store.save_feedback(
                ticket_id=tid,
                correct_label="refund",
                agent_id="agent-2",
                created_at=now_iso,
            )

        # Seed 5 older feedback rows (5 days old): 5 delivery agreed
        for i in range(5):
            tid = f"T-OLD-{i}"
            store.save_prediction(
                ticket_id=tid,
                predicted_label="delivery",
                confidence=0.9,
                model_version="triage-v1.0.0-20261103",
                created_at=old_iso,
            )
            store.save_feedback(
                ticket_id=tid,
                correct_label="delivery",
                agent_id="agent-3",
                created_at=old_iso,
            )

        # 1. Query with default days=7 (total 37 rows >= 30)
        res_7d = client.get(
            "/metrics/quality", headers={"X-API-Key": "test-admin-key-1"}
        )
        assert res_7d.status_code == 200
        data_7d = res_7d.json()
        assert data_7d["status"] == "ok"
        assert data_7d["sample_size"] == 37
        assert data_7d["samples"] == 37
        assert data_7d["days"] == 7
        # Agreed: 28 billing + 5 delivery = 33 / 37 = 0.8919
        assert data_7d["accuracy"] == round(33 / 37, 4)
        assert data_7d["precision"]["billing"] == round(28 / 32, 4)
        assert data_7d["recall"]["billing"] == 1.0
        assert data_7d["confusion_matrix"]["billing"]["refund"] == 4
        assert data_7d["matrix"]["billing"]["refund"] == 4

        # 2. Query with days=2 (only the 32 recent rows: 32 >= 30)
        res_2d = client.get(
            "/metrics/quality?days=2", headers={"X-API-Key": "test-admin-key-1"}
        )
        assert res_2d.status_code == 200
        data_2d = res_2d.json()
        assert data_2d["status"] == "ok"
        assert data_2d["sample_size"] == 32
        assert data_2d["days"] == 2
        # Delivery was 5 days ago, so not present in 2-day window
        assert data_2d["confusion_matrix"]["delivery"]["delivery"] == 0
        assert data_2d["accuracy"] == round(28 / 32, 4)


def test_label_drift_alert_guards_and_firing() -> None:
    """Verify label_drift alert only fires when predictions >= 50 and PSI > 0.2."""
    from supportpilot.api.alerts import (
        ALERT_NAME_LABEL_DRIFT,
        SEVERITY_CRITICAL,
        SEVERITY_WARNING,
        AlertThresholds,
        evaluate_alerts,
    )
    from supportpilot.api.models import MetricsResponse

    thresholds = AlertThresholds(max_psi=0.20, min_drift_predictions=50)
    dummy_metrics = MetricsResponse(
        requests_total=0,
        triage_total=0,
        errors_total=0,
        avg_latency_ms=0.0,
        needs_review_rate=0.0,
        rate_limited_total=0,
    )

    # 1. Under 50 predictions (49) even with high drift -> NO alert
    alerts_low_samples = evaluate_alerts(
        dummy_metrics,
        thresholds,
        psi=0.50,
        total_predictions=49,
    )
    assert not any(a.name == ALERT_NAME_LABEL_DRIFT for a in alerts_low_samples)

    # 2. >= 50 predictions (50) with PSI <= 0.20 (0.15) -> NO alert
    alerts_no_drift = evaluate_alerts(
        dummy_metrics,
        thresholds,
        psi=0.15,
        total_predictions=50,
    )
    assert not any(a.name == ALERT_NAME_LABEL_DRIFT for a in alerts_no_drift)

    # 3. >= 50 predictions (50) with PSI > 0.20 (0.25) -> Warning alert
    alerts_warn = evaluate_alerts(
        dummy_metrics,
        thresholds,
        psi=0.25,
        total_predictions=50,
    )
    drift_warn = [a for a in alerts_warn if a.name == ALERT_NAME_LABEL_DRIFT]
    assert len(drift_warn) == 1
    assert drift_warn[0].value == 0.25
    assert drift_warn[0].threshold == 0.20
    assert drift_warn[0].severity == SEVERITY_WARNING

    # 4. Critical severity (PSI >= 0.40)
    alerts_crit = evaluate_alerts(
        dummy_metrics,
        thresholds,
        psi=0.45,
        total_predictions=60,
    )
    drift_crit = [a for a in alerts_crit if a.name == ALERT_NAME_LABEL_DRIFT]
    assert len(drift_crit) == 1
    assert drift_crit[0].value == 0.45
    assert drift_crit[0].severity == SEVERITY_CRITICAL


def test_low_live_accuracy_alert_guards_and_firing() -> None:
    """Verify low_live_accuracy alert only fires when feedback >= 30 and accuracy < 0.75."""
    from supportpilot.api.alerts import (
        ALERT_NAME_LOW_LIVE_ACCURACY,
        SEVERITY_CRITICAL,
        SEVERITY_WARNING,
        AlertThresholds,
        evaluate_alerts,
    )
    from supportpilot.api.models import MetricsResponse

    thresholds = AlertThresholds(min_live_accuracy=0.75, min_quality_feedback=30)
    dummy_metrics = MetricsResponse(
        requests_total=0,
        triage_total=0,
        errors_total=0,
        avg_latency_ms=0.0,
        needs_review_rate=0.0,
        rate_limited_total=0,
    )

    # 1. Under 30 feedback rows (29) with low accuracy (0.50) -> NO alert
    alerts_low_samples = evaluate_alerts(
        dummy_metrics,
        thresholds,
        feedback_rows=29,
        live_accuracy=0.50,
    )
    assert not any(a.name == ALERT_NAME_LOW_LIVE_ACCURACY for a in alerts_low_samples)

    # 2. >= 30 feedback rows (30) with accuracy >= 0.75 (0.76) -> NO alert
    alerts_good_acc = evaluate_alerts(
        dummy_metrics,
        thresholds,
        feedback_rows=30,
        live_accuracy=0.76,
    )
    assert not any(a.name == ALERT_NAME_LOW_LIVE_ACCURACY for a in alerts_good_acc)

    # 3. >= 30 feedback rows (30) with accuracy < 0.75 (0.72) -> Warning alert
    alerts_warn = evaluate_alerts(
        dummy_metrics,
        thresholds,
        feedback_rows=30,
        live_accuracy=0.72,
    )
    acc_warn = [a for a in alerts_warn if a.name == ALERT_NAME_LOW_LIVE_ACCURACY]
    assert len(acc_warn) == 1
    assert acc_warn[0].value == 0.72
    assert acc_warn[0].threshold == 0.75
    assert acc_warn[0].severity == SEVERITY_WARNING

    # 4. Critical severity (< 0.65, e.g. 0.60)
    alerts_crit = evaluate_alerts(
        dummy_metrics,
        thresholds,
        feedback_rows=35,
        live_accuracy=0.60,
    )
    acc_crit = [a for a in alerts_crit if a.name == ALERT_NAME_LOW_LIVE_ACCURACY]
    assert len(acc_crit) == 1
    assert acc_crit[0].value == 0.60
    assert acc_crit[0].severity == SEVERITY_CRITICAL


def test_alerts_endpoint_picks_up_drift_and_low_accuracy() -> None:
    """Verify GET /alerts automatically evaluates and includes both new alerts."""
    from datetime import UTC, datetime
    from fastapi.testclient import TestClient
    from supportpilot.api.alerts import (
        ALERT_NAME_LABEL_DRIFT,
        ALERT_NAME_LOW_LIVE_ACCURACY,
    )
    from supportpilot.api.main import app

    now = datetime.now(UTC).isoformat()

    with TestClient(app) as client:
        store: FeedbackStore = client.app.state.feedback_store

        # 1. Seed 60 predictions skewed entirely into 'billing' -> high drift vs uniform frozen eval set
        for i in range(60):
            tid = f"T-DRIFT-{i}"
            store.save_prediction(
                ticket_id=tid,
                predicted_label="billing",
                confidence=0.9,
                model_version="triage-v1.0.0-20261103",
                created_at=now,
            )

        # 2. Seed 35 feedback rows with only 20 correct -> accuracy = 20/35 = 0.5714 < 0.75
        for i in range(20):
            store.save_feedback(
                ticket_id=f"T-DRIFT-{i}",
                correct_label="billing",
                agent_id="agent-1",
                created_at=now,
            )
        for i in range(20, 35):
            store.save_feedback(
                ticket_id=f"T-DRIFT-{i}",
                correct_label="refund",
                agent_id="agent-2",
                created_at=now,
            )

        res = client.get("/alerts", headers={"X-API-Key": "test-key-1"})
        assert res.status_code == 200
        data = res.json()

        alert_names = [a["name"] for a in data["alerts"]]
        assert ALERT_NAME_LABEL_DRIFT in alert_names
        assert ALERT_NAME_LOW_LIVE_ACCURACY in alert_names
