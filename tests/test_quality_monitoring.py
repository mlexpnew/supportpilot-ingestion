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


import logging
import math
import os
from collections.abc import Generator


@pytest.fixture
def store(tmp_path: Path) -> Generator[FeedbackStore, None, None]:
    db_url = os.getenv("DATABASE_URL")
    if db_url is not None and db_url.strip():
        s = FeedbackStore()
        try:
            yield s
        finally:
            s.close()
    else:
        db_file = tmp_path / "quality_test.db"
        init_db(db_file)
        s = FeedbackStore(db_path=db_file)
        try:
            yield s
        finally:
            s.close()


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


def test_alerts_endpoint_picks_up_drift_and_low_accuracy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verify GET /alerts automatically evaluates and includes both new alerts when baseline exists."""
    import json
    from datetime import UTC, datetime
    from fastapi.testclient import TestClient
    from supportpilot.api.alerts import (
        ALERT_NAME_LABEL_DRIFT,
        ALERT_NAME_LOW_LIVE_ACCURACY,
    )
    from supportpilot.api.main import app

    baseline_path = tmp_path / "drift_baseline.json"
    with open(baseline_path, "w", encoding="utf-8") as f:
        json.dump({lbl: 10 for lbl in LABELS}, f)
    monkeypatch.setenv("DRIFT_BASELINE_PATH", str(baseline_path))

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


def test_known_prediction_feedback_list_exact_matrix_accuracy_precision_recall() -> (
    None
):
    """Verify known prediction and feedback list yields exact matrix, accuracy, precision, and recall (Step 6)."""
    from supportpilot.monitoring.quality import (
        compute_accuracy,
        compute_confusion_matrix,
        compute_precision_recall,
    )

    # 10 known pairs:
    # billing: 3 predictions (2 actual billing, 1 actual refund)
    # refund: 2 predictions (2 actual refund)
    # delivery: 2 predictions (1 actual delivery, 1 actual other)
    # other: 1 prediction (1 actual delivery)
    # account: 1 prediction (1 actual account)
    # cancellation: 1 prediction (1 actual cancellation)
    known_pairs = [
        {"predicted_label": "billing", "correct_label": "billing"},
        {"predicted_label": "billing", "correct_label": "billing"},
        {"predicted_label": "billing", "correct_label": "refund"},
        {"predicted_label": "refund", "correct_label": "refund"},
        {"predicted_label": "refund", "correct_label": "refund"},
        {"predicted_label": "delivery", "correct_label": "delivery"},
        {"predicted_label": "delivery", "correct_label": "other"},
        {"predicted_label": "other", "correct_label": "delivery"},
        {"predicted_label": "account", "correct_label": "account"},
        {"predicted_label": "cancellation", "correct_label": "cancellation"},
    ]

    # 1. Exact Confusion Matrix
    cm = compute_confusion_matrix(known_pairs)
    assert cm["billing"]["billing"] == 2
    assert cm["billing"]["refund"] == 1
    assert cm["refund"]["refund"] == 2
    assert cm["delivery"]["delivery"] == 1
    assert cm["delivery"]["other"] == 1
    assert cm["other"]["delivery"] == 1
    assert cm["account"]["account"] == 1
    assert cm["cancellation"]["cancellation"] == 1

    # Verify zero for unobserved combinations
    assert cm["billing"]["delivery"] == 0
    assert cm["refund"]["billing"] == 0
    assert cm["delivery"]["refund"] == 0
    assert cm["other"]["other"] == 0

    # 2. Exact Accuracy: 7 correct out of 10 = 0.7000
    acc = compute_accuracy(cm)
    assert acc == 0.7000
    assert compute_accuracy(known_pairs) == 0.7000

    # 3. Exact Precision and Recall per label
    precision, recall = compute_precision_recall(cm)

    # Precision = TP / Total Predicted
    assert precision["billing"] == round(2 / 3, 4)  # 0.6667
    assert precision["refund"] == 1.0000  # 2 / 2
    assert precision["delivery"] == 0.5000  # 1 / 2
    assert precision["account"] == 1.0000  # 1 / 1
    assert precision["cancellation"] == 1.0000  # 1 / 1
    assert precision["other"] == 0.0000  # 0 / 1

    # Recall = TP / Total Actual
    # Actual counts: billing=2, refund=3 (1 from billing, 2 from refund), delivery=2 (1 from delivery, 1 from other), other=1, account=1, cancellation=1
    assert recall["billing"] == 1.0000  # 2 / 2
    assert recall["refund"] == round(2 / 3, 4)  # 0.6667
    assert recall["delivery"] == 0.5000  # 1 / 2
    assert recall["account"] == 1.0000  # 1 / 1
    assert recall["cancellation"] == 1.0000  # 1 / 1
    assert recall["other"] == 0.0000  # 0 / 1


def test_psi_hand_calculated_and_identical_distributions_give_zero() -> None:
    """Verify PSI equals hand-calculated value and identical distributions give PSI = 0 (Step 6)."""
    from supportpilot.monitoring.quality import compute_psi

    test_labels = ("delivery", "billing")
    exp = {"delivery": 50, "billing": 50}
    act = {"delivery": 60, "billing": 40}

    # Hand calculation:
    # A_delivery = 0.6, E_delivery = 0.5 -> (0.6 - 0.5) * ln(0.6 / 0.5) = 0.1 * ln(1.2) = 0.018232155679
    # A_billing  = 0.4, E_billing  = 0.5 -> (0.4 - 0.5) * ln(0.4 / 0.5) = -0.1 * ln(0.8) = 0.022314355131
    # Total sum = 0.0405465... -> 0.0405
    hand_calc = round(
        (0.6 - 0.5) * math.log(0.6 / 0.5) + (0.4 - 0.5) * math.log(0.4 / 0.5), 4
    )
    assert hand_calc == 0.0405

    psi = compute_psi(actual_counts=act, expected_counts=exp, labels=test_labels)
    assert psi == hand_calc
    assert psi == 0.0405

    # Identical distributions give PSI = 0
    zero_psi_binary = compute_psi(
        actual_counts=exp, expected_counts=exp, labels=test_labels
    )
    assert zero_psi_binary == 0.0

    # Identical across all 6 model labels give PSI = 0
    uniform_all = {lbl: 20 for lbl in LABELS}
    zero_psi_all = compute_psi(actual_counts=uniform_all, expected_counts=uniform_all)
    assert zero_psi_all == 0.0


def test_under_30_feedback_rows_insufficient_data_and_alerts_do_not_fire() -> None:
    """Verify under 30 feedback rows gives status: insufficient_data and alerts don't fire (Step 6)."""
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

        # Seed exactly 29 feedback rows (under 30) where 100% are wrong (0% accuracy)
        for i in range(29):
            tid = f"T-GUARD-{i}"
            store.save_prediction(
                ticket_id=tid,
                predicted_label="billing",
                confidence=0.9,
                model_version="triage-v1.0.0-20261103",
                created_at=now,
            )
            store.save_feedback(
                ticket_id=tid,
                correct_label="refund",  # 0% accuracy
                agent_id="agent-qa",
                created_at=now,
            )

        # 1. GET /metrics/quality must return status: "insufficient_data" and null metrics
        res_quality = client.get(
            "/metrics/quality", headers={"X-API-Key": "test-admin-key-1"}
        )
        assert res_quality.status_code == 200
        data_quality = res_quality.json()

        assert data_quality["status"] == "insufficient_data"
        assert "not enough data" in data_quality["message"].lower()
        assert data_quality["sample_size"] == 29
        assert data_quality["accuracy"] is None
        assert data_quality["precision"] is None
        assert data_quality["recall"] is None
        assert data_quality["confusion_matrix"] is None
        assert data_quality["matrix"] is None

        # 2. GET /alerts must NOT fire low_live_accuracy (guarded by 30-feedback minimum)
        #    and must NOT fire label_drift (29 predictions < 50-prediction minimum)
        res_alerts = client.get("/alerts", headers={"X-API-Key": "test-key-1"})
        assert res_alerts.status_code == 200
        data_alerts = res_alerts.json()

        active_alert_names = [a["name"] for a in data_alerts["alerts"]]
        assert ALERT_NAME_LOW_LIVE_ACCURACY not in active_alert_names
        assert ALERT_NAME_LABEL_DRIFT not in active_alert_names


def test_no_ticket_text_appears_in_quality_response_or_logs(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Verify no customer ticket text appears in /metrics/quality response, logs, or store pairs (Step 6)."""
    from fastapi.testclient import TestClient
    from supportpilot.api.main import app

    secret_pii = "CONFIDENTIAL_CARD_4111_SECRET_USER_MESSAGE_DO_NOT_LEAK"

    with TestClient(app) as client:
        store: FeedbackStore = client.app.state.feedback_store

        # 1. Triage a ticket containing sensitive PII text
        res_triage = client.post(
            "/triage",
            headers={"X-API-Key": "test-key-1"},
            json={"ticket_id": "TICKET-PII-LEAK-CHECK", "body": secret_pii},
        )
        assert res_triage.status_code == 200

        # 2. Submit feedback for the ticket
        res_fb = client.post(
            "/feedback",
            headers={"X-API-Key": "test-key-1"},
            json={
                "ticket_id": "TICKET-PII-LEAK-CHECK",
                "correct_label": "billing",
                "agent_id": "agent-qa-1",
            },
        )
        assert res_fb.status_code == 200

        # 3. Call GET /metrics/quality while capturing logs
        caplog.clear()
        with caplog.at_level(logging.DEBUG):
            res_quality = client.get(
                "/metrics/quality",
                headers={"X-API-Key": "test-admin-key-1"},
            )
            assert res_quality.status_code == 200

            # Verify no ticket text in response JSON
            assert secret_pii not in res_quality.text

            # Verify no ticket text in captured logs
            assert secret_pii not in caplog.text

        # 4. Verify quality data store method returns only predicted_label and correct_label
        pairs = store.get_prediction_feedback_pairs()
        assert len(pairs) >= 1
        for pair in pairs:
            assert secret_pii not in str(pair)
            assert set(pair.keys()) == {"predicted_label", "correct_label"}
            assert "text" not in pair
            assert "body" not in pair
            assert "ticket_id" not in pair


def test_no_baseline_means_no_alert(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Verify that when drift baseline file is absent, label_drift reports baseline_missing and doesn't fire."""
    import logging
    from datetime import UTC, datetime
    from fastapi.testclient import TestClient
    from supportpilot.api.alerts import (
        ALERT_NAME_DRIFT_BASELINE_MISSING,
        ALERT_NAME_LABEL_DRIFT,
        AlertThresholds,
        evaluate_alerts,
    )
    from supportpilot.api.main import app
    from supportpilot.api.models import MetricsResponse
    from supportpilot.monitoring.quality import compute_psi, load_drift_baseline

    # Point to a guaranteed non-existent file
    missing_baseline = tmp_path / "non_existent_drift_baseline.json"
    monkeypatch.setenv("DRIFT_BASELINE_PATH", str(missing_baseline))

    # 1. load_drift_baseline returns None and logs warning
    caplog.clear()
    with caplog.at_level(logging.WARNING):
        baseline = load_drift_baseline()
        assert baseline is None
        assert "baseline_missing" in caplog.text
        assert "Never falling back silently" in caplog.text

    # 2. compute_psi returns None and logs warning (never falling back silently)
    counts = {"delivery": 30, "billing": 30}
    caplog.clear()
    with caplog.at_level(logging.WARNING):
        psi = compute_psi(counts)
        assert psi is None
        assert "baseline_missing" in caplog.text
        assert "Never falling back silently" in caplog.text

    # 3. evaluate_alerts: 60 predictions (>= 50 min) 100% skewed in billing
    thresholds = AlertThresholds()
    dummy_metrics = MetricsResponse(
        requests_total=0,
        triage_total=0,
        errors_total=0,
        avg_latency_ms=0.0,
        needs_review_rate=0.0,
        rate_limited_total=0,
    )
    caplog.clear()
    with caplog.at_level(logging.WARNING):
        alerts = evaluate_alerts(
            dummy_metrics,
            thresholds,
            recent_predictions_counts={"billing": 60},
        )
        # Visible drift_baseline_missing warning alert is shown on /alerts when baseline is missing
        missing_alerts = [
            a for a in alerts if a.name == ALERT_NAME_DRIFT_BASELINE_MISSING
        ]
        assert len(missing_alerts) == 1
        assert "baseline_missing" in missing_alerts[0].message
        assert missing_alerts[0].severity == "warning"
        # label_drift itself does not fire
        assert not any(a.name == ALERT_NAME_LABEL_DRIFT for a in alerts)
        assert "baseline_missing" in caplog.text

    # 4. Live GET /alerts endpoint: displays visible drift_baseline_missing warning alert
    now = datetime.now(UTC).isoformat()
    with TestClient(app) as client:
        feedback_store: FeedbackStore = client.app.state.feedback_store
        for i in range(60):
            feedback_store.save_prediction(
                ticket_id=f"T-NO-BASELINE-{i}",
                predicted_label="billing",
                confidence=0.9,
                model_version="triage-v1.0.0-20261103",
                created_at=now,
            )

        res = client.get("/alerts", headers={"X-API-Key": "test-key-1"})
        assert res.status_code == 200
        active_alerts = res.json()["alerts"]
        missing_alerts = [
            a for a in active_alerts if a["name"] == ALERT_NAME_DRIFT_BASELINE_MISSING
        ]
        assert len(missing_alerts) == 1
        assert "baseline_missing" in missing_alerts[0]["message"]
        assert ALERT_NAME_LABEL_DRIFT not in [a["name"] for a in active_alerts]


def test_baseline_present_gives_correct_psi(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verify that when drift baseline is present, it gives the exact hand-calculated PSI and fires when breached."""
    import json
    from supportpilot.api.alerts import (
        ALERT_NAME_LABEL_DRIFT,
        AlertThresholds,
        evaluate_alerts,
    )
    from supportpilot.api.models import MetricsResponse
    from supportpilot.monitoring.quality import compute_psi, load_drift_baseline

    baseline_file = tmp_path / "drift_baseline.json"
    baseline_data = {
        "counts": {
            "delivery": 50,
            "billing": 50,
            "refund": 0,
            "cancellation": 0,
            "account": 0,
            "other": 0,
        }
    }
    with open(baseline_file, "w", encoding="utf-8") as f:
        json.dump(baseline_data, f)
    monkeypatch.setenv("DRIFT_BASELINE_PATH", str(baseline_file))

    # Verify load_drift_baseline reads it correctly
    loaded = load_drift_baseline()
    assert loaded is not None
    assert loaded["delivery"] == 50
    assert loaded["billing"] == 50

    # Hand calculation check:
    # delivery: 60/100 = 0.6 vs 50/100 = 0.5 -> (0.6 - 0.5) * ln(0.6/0.5) = 0.018232
    # billing:  40/100 = 0.4 vs 50/100 = 0.5 -> (0.4 - 0.5) * ln(0.4/0.5) = 0.022314
    # sum = 0.0405
    act = {"delivery": 60, "billing": 40}
    psi = compute_psi(act, labels=("delivery", "billing"))
    assert psi == 0.0405

    # High drift test (> 0.20): 90 delivery, 10 billing vs 50/50 baseline
    # (0.9 - 0.5) * ln(0.9/0.5) + (0.1 - 0.5) * ln(0.1/0.5) = 0.4 * ln(1.8) - 0.4 * ln(0.2) = 0.2351 + 0.6437 = 0.8789 > 0.20
    skewed_mix = {"delivery": 90, "billing": 10}
    skewed_psi = compute_psi(skewed_mix, labels=("delivery", "billing"))
    assert skewed_psi is not None
    assert skewed_psi > 0.20

    thresholds = AlertThresholds(max_psi=0.20, min_drift_predictions=50)
    dummy_metrics = MetricsResponse(
        requests_total=0,
        triage_total=0,
        errors_total=0,
        avg_latency_ms=0.0,
        needs_review_rate=0.0,
        rate_limited_total=0,
    )
    alerts = evaluate_alerts(
        dummy_metrics,
        thresholds,
        recent_predictions_counts=skewed_mix,
    )
    drift_alerts = [a for a in alerts if a.name == ALERT_NAME_LABEL_DRIFT]
    assert len(drift_alerts) == 1
    assert drift_alerts[0].value == skewed_psi


def test_snapshot_baseline_under_500_predictions_fails_fast(
    tmp_path: Path,
) -> None:
    """Verify snapshot script fails fast when there are fewer than 500 predictions."""
    import pytest
    from supportpilot.monitoring.snapshot_baseline import main, snapshot_baseline

    db_file = tmp_path / "snapshot_test_small.db"
    init_db(db_file)
    store = FeedbackStore(db_path=db_file)
    out_file = tmp_path / "drift_baseline.json"

    try:
        # Seed 120 predictions (under 500)
        for i in range(120):
            store.save_prediction(
                ticket_id=f"T-SMALL-{i}",
                predicted_label="billing",
                confidence=0.9,
                model_version="v1",
                created_at="2026-11-05T12:00:00Z",
            )

        # 1. snapshot_baseline raises ValueError
        with pytest.raises(
            ValueError,
            match="Cannot snapshot baseline: found 120 real predictions, but at least 500 are required.",
        ):
            snapshot_baseline(store=store, out_path=out_file)

        assert not out_file.exists()

        # 2. CLI main returns code 1
        rc = main(["--db-path", str(db_file), "--output", str(out_file)])
        assert rc == 1
        assert not out_file.exists()
    finally:
        store.close()


def test_snapshot_baseline_over_500_predictions_saves_valid_baseline(
    tmp_path: Path,
) -> None:
    """Verify snapshot script saves a valid baseline inside package once >= 500 predictions exist."""
    from supportpilot.monitoring.quality import load_drift_baseline
    from supportpilot.monitoring.snapshot_baseline import main, snapshot_baseline

    db_file = tmp_path / "snapshot_test_500.db"
    init_db(db_file)
    store = FeedbackStore(db_path=db_file)
    out_file = tmp_path / "drift_baseline.json"

    try:
        # Seed 520 predictions across labels
        for i in range(200):
            store.save_prediction(
                ticket_id=f"T-BILL-{i}",
                predicted_label="billing",
                confidence=0.9,
                model_version="v1",
                created_at="2026-11-05T12:00:00Z",
            )
        for i in range(120):
            store.save_prediction(
                ticket_id=f"T-DELIV-{i}",
                predicted_label="delivery",
                confidence=0.9,
                model_version="v1",
                created_at="2026-11-05T12:00:00Z",
            )
        for i in range(100):
            store.save_prediction(
                ticket_id=f"T-REF-{i}",
                predicted_label="refund",
                confidence=0.9,
                model_version="v1",
                created_at="2026-11-05T12:00:00Z",
            )
        for i in range(100):
            store.save_prediction(
                ticket_id=f"T-ACC-{i}",
                predicted_label="account",
                confidence=0.9,
                model_version="v1",
                created_at="2026-11-05T12:00:00Z",
            )

        # 1. snapshot_baseline successfully executes
        payload = snapshot_baseline(store=store, out_path=out_file)
        assert payload["total_predictions"] == 520
        assert payload["counts"]["billing"] == 200
        assert payload["counts"]["delivery"] == 120
        assert payload["counts"]["refund"] == 100
        assert payload["counts"]["account"] == 100

        assert out_file.exists()

        # 2. load_drift_baseline can read the generated file
        loaded = load_drift_baseline(out_file)
        assert loaded is not None
        assert loaded["billing"] == 200
        assert loaded["delivery"] == 120
        assert loaded["refund"] == 100
        assert loaded["account"] == 100

        # 3. CLI main returns code 0
        cli_out_file = tmp_path / "drift_baseline_cli.json"
        rc = main(["--db-path", str(db_file), "--output", str(cli_out_file)])
        assert rc == 0
        assert cli_out_file.exists()
    finally:
        store.close()


def test_metrics_quality_days_upper_limit_90() -> None:
    """Verify GET /metrics/quality rejects days > 90 with 422 to prevent full table scans."""
    from fastapi.testclient import TestClient
    from supportpilot.api.main import app

    with TestClient(app) as client:
        # days=91 exceeds le=90 -> 422
        res_exceeded = client.get(
            "/metrics/quality?days=91", headers={"X-API-Key": "test-admin-key-1"}
        )
        assert res_exceeded.status_code == 422

        # days=90 is allowed -> 200
        res_valid = client.get(
            "/metrics/quality?days=90", headers={"X-API-Key": "test-admin-key-1"}
        )
        assert res_valid.status_code == 200


def test_metrics_quality_uses_min_quality_feedback_setting(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify /metrics/quality reads ALERT_MIN_QUALITY_FEEDBACK setting rather than hardcoding 30."""
    from fastapi.testclient import TestClient
    from supportpilot.api.main import app

    # Configure min_quality_feedback to 45
    monkeypatch.setenv("ALERT_MIN_QUALITY_FEEDBACK", "45")

    with TestClient(app) as client:
        res = client.get("/metrics/quality", headers={"X-API-Key": "test-admin-key-1"})
        assert res.status_code == 200
        data = res.json()
        assert data["status"] == "insufficient_data"
        assert "45 feedback rows required" in data["message"]


def test_low_live_accuracy_alert_uses_7_day_window() -> None:
    """Verify low_live_accuracy evaluates only feedback from the last 7 days, ignoring older data."""
    from datetime import UTC, datetime, timedelta
    from fastapi.testclient import TestClient
    from supportpilot.api.alerts import ALERT_NAME_LOW_LIVE_ACCURACY
    from supportpilot.api.main import app

    now = datetime.now(UTC)
    old_time = (now - timedelta(days=10)).isoformat()
    recent_time = now.isoformat()

    with TestClient(app) as client:
        store: FeedbackStore = client.app.state.feedback_store

        # 35 feedback rows that are 10 days old (all wrong, 0% accuracy)
        for i in range(35):
            tid = f"T-OLD-ACC-{i}"
            store.save_prediction(
                ticket_id=tid,
                predicted_label="billing",
                confidence=0.9,
                model_version="v1",
                created_at=old_time,
            )
            store.save_feedback(
                ticket_id=tid,
                correct_label="refund",
                agent_id="agent-1",
                created_at=old_time,
            )

        # 5 recent feedback rows (within 7 days) with 100% accuracy
        for i in range(5):
            tid = f"T-RECENT-ACC-{i}"
            store.save_prediction(
                ticket_id=tid,
                predicted_label="billing",
                confidence=0.9,
                model_version="v1",
                created_at=recent_time,
            )
            store.save_feedback(
                ticket_id=tid,
                correct_label="billing",
                agent_id="agent-1",
                created_at=recent_time,
            )

        # In a 7-day window, there are only 5 feedback rows (< 30 min required),
        # so low_live_accuracy must NOT fire despite the 35 old 0%-accuracy rows.
        res = client.get("/alerts", headers={"X-API-Key": "test-key-1"})
        assert res.status_code == 200
        alert_names = [a["name"] for a in res.json()["alerts"]]
        assert ALERT_NAME_LOW_LIVE_ACCURACY not in alert_names
