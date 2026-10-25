"""Tests for SP-112 Part C: Model health alerts and GET /alerts endpoint."""

from __future__ import annotations

from datetime import datetime

import pytest
from fastapi.testclient import TestClient

from supportpilot.api.alerts import (
    ALERT_NAME_ACCURACY,
    ALERT_NAME_ERROR_RATE,
    ALERT_NAME_REVIEW_RATE,
    SEVERITY_CRITICAL,
    SEVERITY_WARNING,
    AlertThresholds,
    evaluate_alerts,
    get_alert_thresholds,
)
from supportpilot.api.main import MetricsResponse, app


# =====================================================================
# 1. Healthy State Tests
# =====================================================================


def test_evaluate_alerts_healthy_returns_empty_list() -> None:
    """Verify healthy operational metrics return an empty list of alerts."""
    thresholds = AlertThresholds(
        min_accuracy=0.80,
        max_review_rate=0.30,
        max_error_rate=0.05,
        min_feedback=20,
    )
    healthy_metrics = {
        "requests_total": 100,
        "triage_total": 50,
        "errors_total": 2,  # error_rate = 2/100 = 0.02 <= 0.05
        "needs_review_rate": 0.15,  # <= 0.30
        "feedback_total": 30,  # >= 20
        "live_accuracy": 0.90,  # >= 0.80
    }

    alerts = evaluate_alerts(healthy_metrics, thresholds)
    assert alerts == []


def test_evaluate_alerts_supports_metrics_response_object() -> None:
    """Verify evaluate_alerts accepts a Pydantic MetricsResponse object."""
    thresholds = AlertThresholds()
    metrics_obj = MetricsResponse(
        requests_total=100,
        triage_total=80,
        errors_total=1,
        avg_latency_ms=12.5,
        needs_review_rate=0.20,
        rate_limited_total=0,
        feedback_total=25,
        live_accuracy=0.88,
    )

    alerts = evaluate_alerts(metrics_obj, thresholds)
    assert alerts == []


# =====================================================================
# 2. Minimum Data Invariant Tests (No alert below threshold minimums)
# =====================================================================


def test_accuracy_alert_does_not_fire_when_feedback_below_minimum() -> None:
    """Verify accuracy alert NEVER fires if feedback_total < min_feedback (20)."""
    thresholds = AlertThresholds(min_accuracy=0.80, min_feedback=20)

    # 19 feedback rows with terrible accuracy (0.10) must NOT alert
    metrics_low_data = {
        "requests_total": 50,
        "triage_total": 30,
        "errors_total": 0,
        "needs_review_rate": 0.10,
        "feedback_total": 19,
        "live_accuracy": 0.10,
    }
    alerts = evaluate_alerts(metrics_low_data, thresholds)
    assert not any(a.name == ALERT_NAME_ACCURACY for a in alerts)


def test_accuracy_alert_does_not_fire_when_live_accuracy_is_none() -> None:
    """Verify accuracy alert does not fire if live_accuracy is None."""
    thresholds = AlertThresholds(min_accuracy=0.80, min_feedback=20)
    metrics_none_accuracy = {
        "requests_total": 50,
        "triage_total": 30,
        "errors_total": 0,
        "needs_review_rate": 0.10,
        "feedback_total": 25,
        "live_accuracy": None,
    }
    alerts = evaluate_alerts(metrics_none_accuracy, thresholds)
    assert not any(a.name == ALERT_NAME_ACCURACY for a in alerts)


def test_review_rate_alert_does_not_fire_when_triage_total_zero() -> None:
    """Verify review_rate alert does not fire if triage_total is 0."""
    thresholds = AlertThresholds(max_review_rate=0.30)
    metrics_no_triage = {
        "requests_total": 10,
        "triage_total": 0,
        "errors_total": 0,
        "needs_review_rate": 0.99,
        "feedback_total": 0,
        "live_accuracy": None,
    }
    alerts = evaluate_alerts(metrics_no_triage, thresholds)
    assert not any(a.name == ALERT_NAME_REVIEW_RATE for a in alerts)


def test_error_rate_alert_does_not_fire_when_requests_total_zero() -> None:
    """Verify error_rate alert does not fire if requests_total is 0."""
    thresholds = AlertThresholds(max_error_rate=0.05)
    metrics_no_requests = {
        "requests_total": 0,
        "triage_total": 0,
        "errors_total": 0,
        "needs_review_rate": 0.0,
        "feedback_total": 0,
        "live_accuracy": None,
    }
    alerts = evaluate_alerts(metrics_no_requests, thresholds)
    assert not any(a.name == ALERT_NAME_ERROR_RATE for a in alerts)


# =====================================================================
# 3. Known Numbers and Severity Tests (Warning vs Critical)
# =====================================================================


def test_accuracy_alert_known_numbers_warning_and_critical() -> None:
    """Verify accuracy alert triggers warning vs critical with known numbers."""
    thresholds = AlertThresholds(min_accuracy=0.80, min_feedback=20)

    # 1. Warning: slightly past threshold (0.75 < 0.80, but >= 0.70)
    metrics_warning = {
        "requests_total": 100,
        "triage_total": 50,
        "errors_total": 0,
        "needs_review_rate": 0.10,
        "feedback_total": 20,
        "live_accuracy": 0.75,
    }
    alerts_warn = evaluate_alerts(metrics_warning, thresholds)
    assert len(alerts_warn) == 1
    alert_w = alerts_warn[0]
    assert alert_w.name == ALERT_NAME_ACCURACY
    assert alert_w.value == 0.75
    assert alert_w.threshold == 0.80
    assert alert_w.severity == SEVERITY_WARNING

    # 2. Critical: far past threshold (< 0.70)
    metrics_critical = {
        "requests_total": 100,
        "triage_total": 50,
        "errors_total": 0,
        "needs_review_rate": 0.10,
        "feedback_total": 20,
        "live_accuracy": 0.65,
    }
    alerts_crit = evaluate_alerts(metrics_critical, thresholds)
    assert len(alerts_crit) == 1
    alert_c = alerts_crit[0]
    assert alert_c.name == ALERT_NAME_ACCURACY
    assert alert_c.value == 0.65
    assert alert_c.threshold == 0.80
    assert alert_c.severity == SEVERITY_CRITICAL


def test_review_rate_alert_known_numbers_warning_and_critical() -> None:
    """Verify review_rate alert triggers warning vs critical with known numbers."""
    thresholds = AlertThresholds(max_review_rate=0.30)

    # 1. Warning: slightly past threshold (0.35 > 0.30, but < 0.40)
    metrics_warning = {
        "requests_total": 100,
        "triage_total": 50,
        "errors_total": 0,
        "needs_review_rate": 0.35,
        "feedback_total": 0,
        "live_accuracy": None,
    }
    alerts_warn = evaluate_alerts(metrics_warning, thresholds)
    assert len(alerts_warn) == 1
    alert_w = alerts_warn[0]
    assert alert_w.name == ALERT_NAME_REVIEW_RATE
    assert alert_w.value == 0.35
    assert alert_w.threshold == 0.30
    assert alert_w.severity == SEVERITY_WARNING

    # 2. Critical: far past threshold (0.45 >= 0.40)
    metrics_critical = {
        "requests_total": 100,
        "triage_total": 50,
        "errors_total": 0,
        "needs_review_rate": 0.45,
        "feedback_total": 0,
        "live_accuracy": None,
    }
    alerts_crit = evaluate_alerts(metrics_critical, thresholds)
    assert len(alerts_crit) == 1
    alert_c = alerts_crit[0]
    assert alert_c.name == ALERT_NAME_REVIEW_RATE
    assert alert_c.value == 0.45
    assert alert_c.threshold == 0.30
    assert alert_c.severity == SEVERITY_CRITICAL


def test_error_rate_alert_known_numbers_warning_and_critical() -> None:
    """Verify error_rate alert triggers warning vs critical with known numbers."""
    thresholds = AlertThresholds(max_error_rate=0.05)

    # 1. Warning: slightly past threshold (7/100 = 0.07 > 0.05, but < 0.10)
    metrics_warning = {
        "requests_total": 100,
        "triage_total": 50,
        "errors_total": 7,
        "needs_review_rate": 0.10,
        "feedback_total": 0,
        "live_accuracy": None,
    }
    alerts_warn = evaluate_alerts(metrics_warning, thresholds)
    assert len(alerts_warn) == 1
    alert_w = alerts_warn[0]
    assert alert_w.name == ALERT_NAME_ERROR_RATE
    assert alert_w.value == 0.07
    assert alert_w.threshold == 0.05
    assert alert_w.severity == SEVERITY_WARNING

    # 2. Critical: far past threshold (12/100 = 0.12 >= 0.10)
    metrics_critical = {
        "requests_total": 100,
        "triage_total": 50,
        "errors_total": 12,
        "needs_review_rate": 0.10,
        "feedback_total": 0,
        "live_accuracy": None,
    }
    alerts_crit = evaluate_alerts(metrics_critical, thresholds)
    assert len(alerts_crit) == 1
    alert_c = alerts_crit[0]
    assert alert_c.name == ALERT_NAME_ERROR_RATE
    assert alert_c.value == 0.12
    assert alert_c.threshold == 0.05
    assert alert_c.severity == SEVERITY_CRITICAL


def test_all_alerts_fire_simultaneously() -> None:
    """Verify all three alerts trigger simultaneously when every threshold is breached."""
    thresholds = AlertThresholds(
        min_accuracy=0.80,
        max_review_rate=0.30,
        max_error_rate=0.05,
        min_feedback=20,
    )
    unhealthy_metrics = {
        "requests_total": 200,
        "triage_total": 100,
        "errors_total": 25,  # 25/200 = 0.125 >= 0.10 -> critical
        "needs_review_rate": 0.35,  # > 0.30, < 0.40 -> warning
        "feedback_total": 50,
        "live_accuracy": 0.60,  # < 0.70 -> critical
    }

    alerts = evaluate_alerts(unhealthy_metrics, thresholds)
    assert len(alerts) == 3

    names = {a.name for a in alerts}
    assert names == {ALERT_NAME_ACCURACY, ALERT_NAME_REVIEW_RATE, ALERT_NAME_ERROR_RATE}

    acc_alert = next(a for a in alerts if a.name == ALERT_NAME_ACCURACY)
    assert acc_alert.severity == SEVERITY_CRITICAL
    assert acc_alert.value == 0.60

    review_alert = next(a for a in alerts if a.name == ALERT_NAME_REVIEW_RATE)
    assert review_alert.severity == SEVERITY_WARNING
    assert review_alert.value == 0.35

    err_alert = next(a for a in alerts if a.name == ALERT_NAME_ERROR_RATE)
    assert err_alert.severity == SEVERITY_CRITICAL
    assert err_alert.value == 0.125


# =====================================================================
# 4. Environment Variable Thresholds & Fail-Fast Tests
# =====================================================================


def test_get_alert_thresholds_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify default threshold values when env vars are unset."""
    monkeypatch.delenv("ALERT_MIN_ACCURACY", raising=False)
    monkeypatch.delenv("ALERT_MAX_REVIEW_RATE", raising=False)
    monkeypatch.delenv("ALERT_MAX_ERROR_RATE", raising=False)
    monkeypatch.delenv("ALERT_MIN_FEEDBACK", raising=False)

    thresholds = get_alert_thresholds()
    assert thresholds.min_accuracy == 0.80
    assert thresholds.max_review_rate == 0.30
    assert thresholds.max_error_rate == 0.05
    assert thresholds.min_feedback == 20


def test_get_alert_thresholds_custom_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify custom environment variables parse correctly."""
    monkeypatch.setenv("ALERT_MIN_ACCURACY", "0.85")
    monkeypatch.setenv("ALERT_MAX_REVIEW_RATE", "0.25")
    monkeypatch.setenv("ALERT_MAX_ERROR_RATE", "0.02")
    monkeypatch.setenv("ALERT_MIN_FEEDBACK", "50")

    thresholds = get_alert_thresholds()
    assert thresholds.min_accuracy == 0.85
    assert thresholds.max_review_rate == 0.25
    assert thresholds.max_error_rate == 0.02
    assert thresholds.min_feedback == 50


@pytest.mark.parametrize(
    "var_name, bad_value",
    [
        ("ALERT_MIN_ACCURACY", "1.5"),
        ("ALERT_MIN_ACCURACY", "-0.1"),
        ("ALERT_MIN_ACCURACY", "not_a_number"),
        ("ALERT_MAX_REVIEW_RATE", "2.0"),
        ("ALERT_MAX_REVIEW_RATE", "-0.01"),
        ("ALERT_MAX_REVIEW_RATE", "high"),
        ("ALERT_MAX_ERROR_RATE", "1.1"),
        ("ALERT_MAX_ERROR_RATE", "-0.5"),
        ("ALERT_MAX_ERROR_RATE", "invalid"),
        ("ALERT_MIN_FEEDBACK", "-1"),
        ("ALERT_MIN_FEEDBACK", "12.5"),
        ("ALERT_MIN_FEEDBACK", "none"),
    ],
)
def test_get_alert_thresholds_bad_env_fails_fast(
    monkeypatch: pytest.MonkeyPatch, var_name: str, bad_value: str
) -> None:
    """Verify invalid threshold env vars raise ValueError fail-fast."""
    monkeypatch.setenv(var_name, bad_value)
    with pytest.raises(ValueError):
        get_alert_thresholds()


def test_lifespan_fails_fast_on_bad_alert_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify application startup in lifespan fails fast if an alert env var is invalid."""
    monkeypatch.setenv("ALERT_MIN_ACCURACY", "invalid_threshold")
    with pytest.raises(ValueError):
        with TestClient(app):
            pass


# =====================================================================
# 5. GET /alerts Endpoint Integration Tests (Auth, Rate Limit, Contract)
# =====================================================================


def test_get_alerts_unauthenticated_returns_401() -> None:
    """Verify GET /alerts without API key returns 401 Unauthorized."""
    with TestClient(app) as client:
        res = client.get("/alerts")
        assert res.status_code == 401


def test_get_alerts_invalid_api_key_returns_401() -> None:
    """Verify GET /alerts with invalid API key returns 401 Unauthorized."""
    with TestClient(app) as client:
        res = client.get("/alerts", headers={"X-API-Key": "unauthorized-key"})
        assert res.status_code == 401


def test_get_alerts_contract_and_empty_list_when_healthy(
    auth_headers: dict[str, str],
) -> None:
    """Verify GET /alerts returns expected contract schema and empty list when healthy."""
    with TestClient(app) as client:
        res = client.get("/alerts", headers=auth_headers)
        assert res.status_code == 200

        data = res.json()
        assert "alerts" in data
        assert "checked_at" in data
        assert isinstance(data["alerts"], list)
        assert data["alerts"] == []

        # Verify checked_at is valid UTC ISO timestamp
        checked_dt = datetime.fromisoformat(data["checked_at"])
        assert checked_dt.tzinfo is not None


def test_get_alerts_fires_alert_when_threshold_breached(
    auth_headers: dict[str, str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify GET /alerts returns active alerts when metric thresholds are breached."""
    # Set a very strict review rate threshold to trigger review alert
    monkeypatch.setenv("ALERT_MAX_REVIEW_RATE", "0.01")
    monkeypatch.setenv("ALERT_MIN_FEEDBACK", "20")

    with TestClient(app) as client:
        # Submit a ticket that triggers needs_review (confidence fallback or low confidence)
        client.post(
            "/triage",
            headers=auth_headers,
            json={"ticket_id": "T-ALERT-1", "body": "random unclassifiable query"},
        )

        res = client.get("/alerts", headers=auth_headers)
        assert res.status_code == 200
        data = res.json()

        assert len(data["alerts"]) >= 1
        review_alerts = [
            a for a in data["alerts"] if a["name"] == ALERT_NAME_REVIEW_RATE
        ]
        assert len(review_alerts) == 1
        alert = review_alerts[0]
        assert alert["name"] == "review_rate"
        assert alert["threshold"] == 0.01
        assert alert["value"] > 0.01
        assert alert["severity"] in (SEVERITY_WARNING, SEVERITY_CRITICAL)
