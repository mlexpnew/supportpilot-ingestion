"""Tests for latency tracking and latency SLO breach alerts (SP-119 Part B)."""

from __future__ import annotations

from fastapi.testclient import TestClient

from supportpilot.api.alerts import (
    ALERT_NAME_LATENCY_SLO_BREACH,
    AlertThresholds,
    evaluate_alerts,
)
from supportpilot.api.main import app, calculate_percentiles
from supportpilot.api.models import EndpointLatency, MetricsResponse


def test_calculate_percentiles_accuracy() -> None:
    """Verify p50 and p95 percentiles calculation from sample lists."""
    assert calculate_percentiles([]) == (0.0, 0.0)
    assert calculate_percentiles([100.0]) == (100.0, 100.0)

    # 100 sequential values: 1 to 100
    samples = [float(i) for i in range(1, 101)]
    p50, p95 = calculate_percentiles(samples)
    assert p50 == 50.5
    assert p95 == 95.05


def test_endpoint_latencies_max_length_500() -> None:
    """Verify rolling memory buffer retains strictly the last 500 latencies per endpoint."""
    with TestClient(app) as client:
        latencies = client.app.state.endpoint_latencies
        assert "/triage" in latencies
        assert "/triage/batch" in latencies

        # Feed 550 entries into /triage
        latencies["/triage"].clear()
        for i in range(550):
            latencies["/triage"].append(float(i))

        assert len(latencies["/triage"]) == 500
        # The oldest 50 (0..49) should have rolled off; first element is 50
        assert latencies["/triage"][0] == 50.0
        assert latencies["/triage"][-1] == 549.0


def test_successful_requests_only_recorded() -> None:
    """Verify durations are recorded after request finishes for successful requests only."""
    with TestClient(app) as client:
        latencies = client.app.state.endpoint_latencies
        latencies["/triage"].clear()
        latencies["/triage/batch"].clear()

        # 1. Successful /triage (200 OK)
        res_triage_ok = client.post(
            "/triage",
            headers={"X-API-Key": "test-key-1"},
            json={"ticket_id": "T-OK-1", "body": "Where is my parcel?"},
        )
        assert res_triage_ok.status_code == 200
        assert len(latencies["/triage"]) == 1
        assert latencies["/triage"][0] > 0.0

        # 2. Failed /triage auth (401 Unauthorized) -> NOT recorded
        res_triage_401 = client.post(
            "/triage",
            headers={"X-API-Key": "wrong-key"},
            json={"ticket_id": "T-FAIL-1", "body": "Where is my parcel?"},
        )
        assert res_triage_401.status_code == 401
        assert len(latencies["/triage"]) == 1

        # 3. Failed /triage validation (422 Unprocessable Entity) -> NOT recorded
        res_triage_422 = client.post(
            "/triage",
            headers={"X-API-Key": "test-key-1"},
            json={"ticket_id": 12345},  # invalid ticket_id type
        )
        assert res_triage_422.status_code == 422
        assert len(latencies["/triage"]) == 1

        # 4. Successful /triage/batch (200 OK)
        res_batch_ok = client.post(
            "/triage/batch",
            headers={"X-API-Key": "test-key-1"},
            json={"tickets": [{"ticket_id": "T-B-1", "body": "Refund inquiry"}]},
        )
        assert res_batch_ok.status_code == 200
        assert len(latencies["/triage/batch"]) == 1
        assert latencies["/triage/batch"][0] > 0.0


def test_metrics_endpoint_reports_latencies_per_endpoint() -> None:
    """Verify /metrics returns p50, p95, and samples for /triage and /triage/batch."""
    with TestClient(app) as client:
        latencies = client.app.state.endpoint_latencies
        latencies["/triage"].clear()
        latencies["/triage/batch"].clear()

        # Populate known values: 100, 200, 300 ms for /triage
        latencies["/triage"].extend([100.0, 200.0, 300.0])
        # Populate 50, 150 ms for /triage/batch
        latencies["/triage/batch"].extend([50.0, 150.0])

        res = client.get("/metrics", headers={"X-API-Key": "test-key-1"})
        assert res.status_code == 200
        data = res.json()
        assert "latencies" in data

        triage_lat = data["latencies"]["/triage"]
        assert triage_lat["samples"] == 3
        assert triage_lat["p50_ms"] == 200.0
        assert triage_lat["p95_ms"] == 290.0

        batch_lat = data["latencies"]["/triage/batch"]
        assert batch_lat["samples"] == 2
        assert batch_lat["p50_ms"] == 100.0
        assert batch_lat["p95_ms"] == 145.0


def test_latency_slo_breach_alert_conditions() -> None:
    """Verify latency_slo_breach alert: fires when p95 > 500 ms and samples >= 20."""
    thresholds = AlertThresholds()

    # Base healthy metrics fixture
    def _make(lat_map: dict[str, EndpointLatency]) -> MetricsResponse:
        return MetricsResponse(
            requests_total=100,
            triage_total=50,
            errors_total=0,
            avg_latency_ms=10.0,
            needs_review_rate=0.10,
            rate_limited_total=0,
            feedback_total=25,
            live_accuracy=0.90,
            latencies=lat_map,
        )

    # 1. 19 samples with p95 = 600 ms -> NO alert (< 20 samples)
    metrics_few_samples = _make(
        {
            "/triage": EndpointLatency(p50_ms=300.0, p95_ms=600.0, samples=19),
            "/triage/batch": EndpointLatency(p50_ms=100.0, p95_ms=200.0, samples=10),
        }
    )
    alerts = evaluate_alerts(metrics_few_samples, thresholds)
    assert not any(a.name == ALERT_NAME_LATENCY_SLO_BREACH for a in alerts)

    # 2. 20 samples with p95 = 500.0 ms -> NO alert (not over 500 ms)
    metrics_at_boundary = _make(
        {
            "/triage": EndpointLatency(p50_ms=300.0, p95_ms=500.0, samples=20),
            "/triage/batch": EndpointLatency(p50_ms=100.0, p95_ms=200.0, samples=20),
        }
    )
    alerts = evaluate_alerts(metrics_at_boundary, thresholds)
    assert not any(a.name == ALERT_NAME_LATENCY_SLO_BREACH for a in alerts)

    # 3. 20 samples with p95 = 520.5 ms -> FIRES latency_slo_breach warning
    metrics_breach = _make(
        {
            "/triage": EndpointLatency(p50_ms=300.0, p95_ms=520.5, samples=20),
            "/triage/batch": EndpointLatency(p50_ms=100.0, p95_ms=200.0, samples=20),
        }
    )
    alerts = evaluate_alerts(metrics_breach, thresholds)
    slo_alerts = [a for a in alerts if a.name == ALERT_NAME_LATENCY_SLO_BREACH]
    assert len(slo_alerts) == 1
    assert slo_alerts[0].value == 520.5
    assert slo_alerts[0].threshold == 500.0
    assert slo_alerts[0].severity == "warning"
    assert "/triage" in (slo_alerts[0].message or "")

    # 4. Critical severity if p95 >= 1000 ms (2x threshold)
    metrics_critical = _make(
        {
            "/triage": EndpointLatency(p50_ms=600.0, p95_ms=1200.0, samples=25),
        }
    )
    alerts = evaluate_alerts(metrics_critical, thresholds)
    slo_alerts = [a for a in alerts if a.name == ALERT_NAME_LATENCY_SLO_BREACH]
    assert len(slo_alerts) == 1
    assert slo_alerts[0].severity == "critical"

    # 5. Batch triage breach fires alert as well
    metrics_batch_breach = _make(
        {
            "/triage": EndpointLatency(p50_ms=100.0, p95_ms=200.0, samples=30),
            "/triage/batch": EndpointLatency(p50_ms=400.0, p95_ms=550.0, samples=22),
        }
    )
    alerts = evaluate_alerts(metrics_batch_breach, thresholds)
    slo_alerts = [a for a in alerts if a.name == ALERT_NAME_LATENCY_SLO_BREACH]
    assert len(slo_alerts) == 1
    assert slo_alerts[0].value == 550.0
    assert "/triage/batch" in (slo_alerts[0].message or "")


def test_live_alerts_endpoint_fires_latency_slo_breach() -> None:
    """Verify live GET /alerts returns latency_slo_breach when p95 breaches SLO."""
    with TestClient(app) as client:
        latencies = client.app.state.endpoint_latencies
        latencies["/triage"].clear()

        # Seed 20 requests with p95 = 550ms
        samples = [100.0] * 18 + [550.0, 560.0]
        latencies["/triage"].extend(samples)

        res = client.get("/alerts", headers={"X-API-Key": "test-key-1"})
        assert res.status_code == 200
        data = res.json()
        alert_names = [a["name"] for a in data["alerts"]]
        assert ALERT_NAME_LATENCY_SLO_BREACH in alert_names

        # Clearing: add 30 fast requests so p95 drops below 500ms
        latencies["/triage"].extend([50.0] * 30)
        res_cleared = client.get("/alerts", headers={"X-API-Key": "test-key-1"})
        assert res_cleared.status_code == 200
        cleared_names = [a["name"] for a in res_cleared.json()["alerts"]]
        assert ALERT_NAME_LATENCY_SLO_BREACH not in cleared_names
