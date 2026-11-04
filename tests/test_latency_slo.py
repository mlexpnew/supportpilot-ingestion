"""Tests for latency tracking and latency SLO breach alerts (SP-119 Part B & Step 5).

Covers:
1. p95 is calculated correctly on a known list of latencies.
2. The SLO alert doesn't fire with under 20 samples.
3. The SLO alert fires when p95 is over 500 ms.
4. Errors aren't counted in the latency.
5. In-memory buffer rolling window (last 500 latencies per endpoint).
6. Metrics endpoint reporting and live alert clearance.
"""

from __future__ import annotations

from fastapi.testclient import TestClient

from supportpilot.api.alerts import (
    ALERT_NAME_LATENCY_SLO_BREACH,
    AlertThresholds,
    evaluate_alerts,
)
from supportpilot.api.main import (
    MAX_BODY_BYTES,
    app,
    calculate_percentiles,
)
from supportpilot.api.models import EndpointLatency, MetricsResponse


# =====================================================================
# 1. p95 is calculated correctly on a known list of latencies
# =====================================================================


def test_p95_calculated_correctly_on_known_list_of_latencies() -> None:
    """Verify p95 and p50 percentiles are calculated accurately on known lists."""
    # Empty and single-element edge cases
    assert calculate_percentiles([]) == (0.0, 0.0)
    assert calculate_percentiles([100.0]) == (100.0, 100.0)

    # 10 sequential elements: [100.0, 200.0, ..., 1000.0]
    # N = 10:
    #   p50 index = 9 * 0.50 = 4.5 -> 500.0 + 0.5 * (600.0 - 500.0) = 550.0
    #   p95 index = 9 * 0.95 = 8.55 -> 900.0 + 0.55 * (1000.0 - 900.0) = 955.0
    ten_samples = [float(i * 100) for i in range(1, 11)]
    p50, p95 = calculate_percentiles(ten_samples)
    assert p50 == 550.0
    assert p95 == 955.0

    # 20 sequential elements: [10.0, 20.0, ..., 200.0]
    # N = 20:
    #   p50 index = 19 * 0.50 = 9.5 -> 100.0 + 0.5 * 10.0 = 105.0
    #   p95 index = 19 * 0.95 = 18.05 -> 190.0 + 0.05 * 10.0 = 190.5
    twenty_samples = [float(i * 10) for i in range(1, 21)]
    p50_20, p95_20 = calculate_percentiles(twenty_samples)
    assert p50_20 == 105.0
    assert p95_20 == 190.5

    # 100 sequential elements: 1.0 to 100.0
    # N = 100: p50 = 50.5, p95 = 95.05
    hundred_samples = [float(i) for i in range(1, 101)]
    p50_100, p95_100 = calculate_percentiles(hundred_samples)
    assert p50_100 == 50.5
    assert p95_100 == 95.05


# =====================================================================
# 2. The SLO alert doesn't fire with under 20 samples
# =====================================================================


def test_slo_alert_does_not_fire_under_20_samples() -> None:
    """Verify latency_slo_breach alert does NOT fire when sample count is under 20."""
    thresholds = AlertThresholds()

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

    # 0 samples
    metrics_0 = _make({"/triage": EndpointLatency(p50_ms=0.0, p95_ms=0.0, samples=0)})
    alerts_0 = evaluate_alerts(metrics_0, thresholds)
    assert not any(a.name == ALERT_NAME_LATENCY_SLO_BREACH for a in alerts_0)

    # 1 sample with extreme latency (2000 ms)
    metrics_1 = _make(
        {"/triage": EndpointLatency(p50_ms=2000.0, p95_ms=2000.0, samples=1)}
    )
    alerts_1 = evaluate_alerts(metrics_1, thresholds)
    assert not any(a.name == ALERT_NAME_LATENCY_SLO_BREACH for a in alerts_1)

    # 19 samples with p95 = 850.0 ms (> 500 ms, but samples < 20)
    metrics_19 = _make(
        {
            "/triage": EndpointLatency(p50_ms=300.0, p95_ms=850.0, samples=19),
            "/triage/batch": EndpointLatency(p50_ms=400.0, p95_ms=900.0, samples=15),
        }
    )
    alerts_19 = evaluate_alerts(metrics_19, thresholds)
    assert not any(a.name == ALERT_NAME_LATENCY_SLO_BREACH for a in alerts_19)

    # Live endpoint check: 19 high-latency samples in memory do not fire alert
    with TestClient(app) as client:
        latencies = client.app.state.endpoint_latencies
        latencies["/triage"].clear()
        latencies["/triage/batch"].clear()
        latencies["/triage"].extend([600.0] * 19)

        res = client.get("/alerts", headers={"X-API-Key": "test-key-1"})
        assert res.status_code == 200
        alert_names = [a["name"] for a in res.json()["alerts"]]
        assert ALERT_NAME_LATENCY_SLO_BREACH not in alert_names


# =====================================================================
# 3. The SLO alert fires when p95 is over 500 ms
# =====================================================================


def test_slo_alert_fires_when_p95_is_over_500ms() -> None:
    """Verify latency_slo_breach alert fires when p95 > 500 ms with at least 20 samples."""
    thresholds = AlertThresholds()

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

    # Boundary case: exactly 500.0 ms -> does NOT fire (> 500 required)
    metrics_at_boundary = _make(
        {
            "/triage": EndpointLatency(p50_ms=250.0, p95_ms=500.0, samples=20),
        }
    )
    alerts_boundary = evaluate_alerts(metrics_at_boundary, thresholds)
    assert not any(a.name == ALERT_NAME_LATENCY_SLO_BREACH for a in alerts_boundary)

    # Breach case on /triage: 20 samples with p95 = 525.0 ms -> fires warning
    metrics_breach = _make(
        {
            "/triage": EndpointLatency(p50_ms=300.0, p95_ms=525.0, samples=20),
        }
    )
    alerts_breach = evaluate_alerts(metrics_breach, thresholds)
    slo_alerts = [a for a in alerts_breach if a.name == ALERT_NAME_LATENCY_SLO_BREACH]
    assert len(slo_alerts) == 1
    assert slo_alerts[0].value == 525.0
    assert slo_alerts[0].threshold == 500.0
    assert slo_alerts[0].severity == "warning"
    assert "/triage" in (slo_alerts[0].message or "")

    # Breach case on /triage/batch: 25 samples with p95 = 580.0 ms -> fires warning
    metrics_batch = _make(
        {
            "/triage": EndpointLatency(p50_ms=100.0, p95_ms=200.0, samples=20),
            "/triage/batch": EndpointLatency(p50_ms=350.0, p95_ms=580.0, samples=25),
        }
    )
    alerts_batch = evaluate_alerts(metrics_batch, thresholds)
    slo_batch = [a for a in alerts_batch if a.name == ALERT_NAME_LATENCY_SLO_BREACH]
    assert len(slo_batch) == 1
    assert slo_batch[0].value == 580.0
    assert slo_batch[0].severity == "warning"
    assert "/triage/batch" in (slo_batch[0].message or "")

    # Critical severity when p95 >= 1000 ms (2x threshold)
    metrics_critical = _make(
        {
            "/triage": EndpointLatency(p50_ms=800.0, p95_ms=1250.0, samples=30),
        }
    )
    alerts_critical = evaluate_alerts(metrics_critical, thresholds)
    slo_crit = [a for a in alerts_critical if a.name == ALERT_NAME_LATENCY_SLO_BREACH]
    assert len(slo_crit) == 1
    assert slo_crit[0].severity == "critical"

    # Live endpoint check: 20 samples exceeding 500ms trigger alert on GET /alerts
    with TestClient(app) as client:
        latencies = client.app.state.endpoint_latencies
        latencies["/triage"].clear()
        latencies["/triage"].extend([100.0] * 18 + [550.0, 600.0])

        res = client.get("/alerts", headers={"X-API-Key": "test-key-1"})
        assert res.status_code == 200
        alert_names = [a["name"] for a in res.json()["alerts"]]
        assert ALERT_NAME_LATENCY_SLO_BREACH in alert_names


# =====================================================================
# 4. Errors aren't counted in the latency
# =====================================================================


def test_errors_are_not_counted_in_the_latency() -> None:
    """Verify failed requests (401, 413, 422) are NOT counted in the latency list."""
    with TestClient(app) as client:
        latencies = client.app.state.endpoint_latencies
        latencies["/triage"].clear()
        latencies["/triage/batch"].clear()

        # 1. Failed authentication (401 Unauthorized) -> NOT recorded
        res_401 = client.post(
            "/triage",
            headers={"X-API-Key": "invalid-nonexistent-key"},
            json={"ticket_id": "T-ERR-1", "body": "Where is my parcel?"},
        )
        assert res_401.status_code == 401
        assert len(latencies["/triage"]) == 0

        # 2. Validation error: missing required body (422 Unprocessable Entity) -> NOT recorded
        res_422 = client.post(
            "/triage",
            headers={"X-API-Key": "test-key-1"},
            json={"ticket_id": "T-ERR-2"},
        )
        assert res_422.status_code == 422
        assert len(latencies["/triage"]) == 0

        # 3. Validation error: whitespace-only body (422) -> NOT recorded
        res_422_ws = client.post(
            "/triage",
            headers={"X-API-Key": "test-key-1"},
            json={"ticket_id": "T-ERR-3", "body": "   \n\t  "},
        )
        assert res_422_ws.status_code == 422
        assert len(latencies["/triage"]) == 0

        # 4. Payload too large: > 10 KB (413 Payload Too Large) -> NOT recorded
        big_body = "x" * (MAX_BODY_BYTES + 50)
        res_413 = client.post(
            "/triage",
            headers={"X-API-Key": "test-key-1"},
            json={"ticket_id": "T-ERR-4", "body": big_body},
        )
        assert res_413.status_code == 413
        assert len(latencies["/triage"]) == 0

        # 5. Batch triage validation error: tickets is not a list (422) -> NOT recorded
        res_batch_422 = client.post(
            "/triage/batch",
            headers={"X-API-Key": "test-key-1"},
            json={"tickets": "invalid_not_a_list"},
        )
        assert res_batch_422.status_code == 422
        assert len(latencies["/triage/batch"]) == 0

        # 6. Now make 1 SUCCESSFUL single request (200 OK)
        res_ok_single = client.post(
            "/triage",
            headers={"X-API-Key": "test-key-1"},
            json={"ticket_id": "T-OK-1", "body": "Need refund for late item"},
        )
        assert res_ok_single.status_code == 200
        # Exactly 1 latency recorded (the successful one)
        assert len(latencies["/triage"]) == 1
        assert latencies["/triage"][0] > 0.0

        # 7. Make 1 SUCCESSFUL batch request (200 OK)
        res_ok_batch = client.post(
            "/triage/batch",
            headers={"X-API-Key": "test-key-1"},
            json={"tickets": [{"ticket_id": "TB-OK-1", "body": "Order status"}]},
        )
        assert res_ok_batch.status_code == 200
        # Exactly 1 batch latency recorded
        assert len(latencies["/triage/batch"]) == 1
        assert latencies["/triage/batch"][0] > 0.0


# =====================================================================
# 5. In-Memory Rolling Buffer (last 500 latencies per endpoint)
# =====================================================================


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
        # The oldest 50 (0..49) rolled off; first element is 50.0, last is 549.0
        assert latencies["/triage"][0] == 50.0
        assert latencies["/triage"][-1] == 549.0


# =====================================================================
# 6. Metrics Endpoint Reporting and Alert Clearance
# =====================================================================


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


def test_alert_clears_when_p95_drops_below_500ms() -> None:
    """Verify live GET /alerts clears latency_slo_breach when faster requests lower p95."""
    with TestClient(app) as client:
        latencies = client.app.state.endpoint_latencies
        latencies["/triage"].clear()

        # 20 requests with p95 = 550ms -> triggers alert
        samples = [100.0] * 18 + [550.0, 560.0]
        latencies["/triage"].extend(samples)

        res = client.get("/alerts", headers={"X-API-Key": "test-key-1"})
        assert res.status_code == 200
        alert_names = [a["name"] for a in res.json()["alerts"]]
        assert ALERT_NAME_LATENCY_SLO_BREACH in alert_names

        # Clear alert by adding 30 fast requests (50ms) so p95 drops under 500ms
        latencies["/triage"].extend([50.0] * 30)
        res_cleared = client.get("/alerts", headers={"X-API-Key": "test-key-1"})
        assert res_cleared.status_code == 200
        cleared_names = [a["name"] for a in res_cleared.json()["alerts"]]
        assert ALERT_NAME_LATENCY_SLO_BREACH not in cleared_names
