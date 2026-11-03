"""Tests for batch triage endpoint (SP-117)."""

from __future__ import annotations

import time
import pytest
from fastapi.testclient import TestClient

from supportpilot.api.main import app

AUTH_HEADER = {"X-API-Key": "test-key-1"}


@pytest.fixture(autouse=True)
def reset_rate_limits_state():
    """Ensure rate limit state is clean before and after every test."""
    app.state.ticket_rate_limit_per_minute = 1000
    if hasattr(app.state, "ticket_rate_limits"):
        app.state.ticket_rate_limits.clear()
    if hasattr(app.state, "rate_limits"):
        app.state.rate_limits.clear()
    yield
    app.state.ticket_rate_limit_per_minute = 1000
    if hasattr(app.state, "ticket_rate_limits"):
        app.state.ticket_rate_limits.clear()
    if hasattr(app.state, "rate_limits"):
        app.state.rate_limits.clear()


def test_batch_triage_auth_required() -> None:
    """Verify POST /triage/batch enforces API key authentication."""
    with TestClient(app) as client:
        # 1. Missing auth header -> 401
        res = client.post("/triage/batch", json={"tickets": []})
        assert res.status_code == 401

        # 2. Invalid auth header -> 401
        res_bad = client.post(
            "/triage/batch",
            headers={"X-API-Key": "invalid-secret-key"},
            json={"tickets": []},
        )
        assert res_bad.status_code == 401

        # 3. Valid auth header -> 200
        res_ok = client.post("/triage/batch", headers=AUTH_HEADER, json={"tickets": []})
        assert res_ok.status_code == 200
        data = res_ok.json()
        assert data["results"] == []
        assert data["summary"] == {"total": 0, "ok": 0, "failed": 0}


def test_batch_triage_max_100_tickets_limit() -> None:
    """Verify batch limit: 100 tickets work, 101 tickets return 413 Payload Too Large."""
    with TestClient(app) as client:
        # Exactly 100 tickets -> 200 OK
        tickets_100 = [
            {"ticket_id": f"T-100-{i}", "body": "Where is my parcel delivery?"}
            for i in range(100)
        ]
        res_100 = client.post(
            "/triage/batch",
            headers=AUTH_HEADER,
            json={"tickets": tickets_100},
        )
        assert res_100.status_code == 200
        assert res_100.json()["summary"]["total"] == 100

        # 101 tickets -> 413 Payload Too Large
        tickets_101 = [
            {"ticket_id": f"T-101-{i}", "body": "Where is my parcel delivery?"}
            for i in range(101)
        ]
        res_101 = client.post(
            "/triage/batch",
            headers=AUTH_HEADER,
            json={"tickets": tickets_101},
        )
        assert res_101.status_code == 413
        assert "exceeds maximum limit of 100" in res_101.json()["detail"]


def test_batch_triage_accepts_loose_dict_without_422() -> None:
    """Verify one bad ticket doesn't fail the batch with 422 Unprocessable Entity."""
    with TestClient(app) as client:
        batch = {
            "tickets": [
                {"ticket_id": "T-VALID-01", "body": "Where is my delivery tracking?"},
                {"ticket_id": "", "body": "Missing id"},  # bad id
                {"ticket_id": "T-BAD-BODY", "body": ""},  # empty body
                "not-a-dict",  # non-dict item
            ]
        }
        res = client.post("/triage/batch", headers=AUTH_HEADER, json=batch)
        assert res.status_code == 200  # NOT 422!
        data = res.json()
        assert data["summary"]["total"] == 4
        assert data["summary"]["ok"] == 1
        assert data["summary"]["failed"] == 3


def test_batch_triage_ticket_rate_limit_sliding_window() -> None:
    """Verify second limiter: tickets per minute per key (sliding window, default 1,000)."""
    with TestClient(app) as client:
        # Override ticket limit to 10 tickets/min for this test
        try:
            client.app.state.ticket_rate_limit_per_minute = 10
            client.app.state.ticket_rate_limits.clear()
            initial_rate_limited = client.app.state.counters.rate_limited_total

            # Send batch of 8 tickets -> OK (8 <= 10)
            batch_8 = {
                "tickets": [
                    {"ticket_id": f"T-RL-{i}", "body": "Where is my order?"}
                    for i in range(8)
                ]
            }
            res_8 = client.post("/triage/batch", headers=AUTH_HEADER, json=batch_8)
            assert res_8.status_code == 200

            # Send batch of 3 tickets -> Exceeds 10 (8 + 3 = 11 > 10) -> 429
            batch_3 = {
                "tickets": [
                    {"ticket_id": f"T-RL-OVER-{i}", "body": "Order status?"}
                    for i in range(3)
                ]
            }
            res_429 = client.post("/triage/batch", headers=AUTH_HEADER, json=batch_3)
            assert res_429.status_code == 429
            assert "Retry-After" in res_429.headers
            assert int(res_429.headers["Retry-After"]) >= 1
            assert (
                client.app.state.counters.rate_limited_total == initial_rate_limited + 1
            )
        finally:
            client.app.state.ticket_rate_limit_per_minute = 1000
            client.app.state.ticket_rate_limits.clear()


def test_batch_triage_partial_failures_per_ticket_and_summary() -> None:
    """SP-117 Part B: Verify per-ticket validation, partial failure results, and summary counts."""
    with TestClient(app) as client:
        secret_bad_text = "CONFIDENTIAL_PAYMENT_FAILURE_DETAILS_SECRET_XYZ"
        oversized_body = "x" * 10241  # > 10 KB limit

        batch = {
            "tickets": [
                {
                    "ticket_id": "T-OK-1",
                    "subject": "Delivery inquiry",
                    "body": "Where is my package parcel tracking number?",
                },
                {
                    "ticket_id": "T-EMPTY-BODY",
                    "subject": "Billing issue",
                    "body": "   ",  # empty body
                },
                {
                    "ticket_id": "",  # empty ticket_id
                    "body": "Missing ticket id query",
                },
                {
                    "ticket_id": "T-OVERSIZED",
                    "body": oversized_body,  # exceeds 10 KB limit
                },
                {"ticket_id": "T-OK-2", "body": "I would like to cancel my order"},
            ]
        }
        res = client.post("/triage/batch", headers=AUTH_HEADER, json=batch)
        assert res.status_code == 200  # Partial failures must never sink the batch

        data = res.json()
        assert data["summary"] == {"total": 5, "ok": 2, "failed": 3}
        results = data["results"]
        assert len(results) == 5

        # Item 0: OK
        assert results[0]["ticket_id"] == "T-OK-1"
        assert results[0]["status"] == "ok"
        assert results[0]["label"] == "delivery"
        assert isinstance(results[0]["confidence"], float)

        # Item 1: Empty body error
        assert results[1]["ticket_id"] == "T-EMPTY-BODY"
        assert results[1]["status"] == "error"
        assert "body" in results[1]["error"]

        # Item 2: Missing ticket id
        assert results[2]["ticket_id"] == "unknown"
        assert results[2]["status"] == "error"
        assert "ticket_id" in results[2]["error"]

        # Item 3: Oversized body error
        assert results[3]["ticket_id"] == "T-OVERSIZED"
        assert results[3]["status"] == "error"
        assert "10240 bytes" in results[3]["error"]

        # Item 4: OK
        assert results[4]["ticket_id"] == "T-OK-2"
        assert results[4]["status"] == "ok"
        assert results[4]["label"] == "cancellation"

        # Verify zero body text in any error
        for r in results:
            if r["status"] == "error":
                assert secret_bad_text not in str(r)
                assert oversized_body not in str(r)


def test_batch_triage_duplicate_ticket_id_in_same_batch() -> None:
    """SP-117 Part C: Track seen_ids within batch -> repeat gets error: 'duplicate_in_batch'."""
    with TestClient(app) as client:
        batch = {
            "tickets": [
                {"ticket_id": "T-DUP-999", "body": "Where is my parcel?"},
                {
                    "ticket_id": "T-DUP-999",
                    "body": "Repeated request inside same batch",
                },
                {"ticket_id": "T-OTHER-111", "body": "Cancel my subscription"},
            ]
        }
        res = client.post("/triage/batch", headers=AUTH_HEADER, json=batch)
        assert res.status_code == 200

        data = res.json()
        assert data["summary"] == {"total": 3, "ok": 2, "failed": 1}
        results = data["results"]

        # First instance succeeds
        assert results[0]["ticket_id"] == "T-DUP-999"
        assert results[0]["status"] == "ok"
        assert results[0]["label"] == "delivery"
        assert results[0]["cached"] is False

        # Second instance gets duplicate_in_batch error
        assert results[1]["ticket_id"] == "T-DUP-999"
        assert results[1]["status"] == "error"
        assert results[1]["error"] == "duplicate_in_batch"

        # Third is distinct and succeeds
        assert results[2]["ticket_id"] == "T-OTHER-111"
        assert results[2]["status"] == "ok"
        assert results[2]["label"] == "cancellation"


def test_batch_triage_idempotent_re_send_returns_cached_prediction() -> None:
    """SP-117 Part C: A ticket already triaged before returns cached: True (no re-prediction, no overwrite)."""
    with TestClient(app) as client:
        batch = {
            "tickets": [
                {"ticket_id": "T-IDEM-01", "body": "Where is my parcel tracking?"},
                {"ticket_id": "T-IDEM-02", "body": "Please refund my wallet payment"},
            ]
        }

        # First call: newly predicted, cached=False
        res1 = client.post("/triage/batch", headers=AUTH_HEADER, json=batch)
        assert res1.status_code == 200
        data1 = res1.json()
        assert data1["summary"] == {"total": 2, "ok": 2, "failed": 0}
        assert data1["results"][0]["cached"] is False
        assert data1["results"][1]["cached"] is False
        label_1 = data1["results"][0]["label"]
        conf_1 = data1["results"][0]["confidence"]

        # Second call: identical request -> returned from DB with cached=True
        res2 = client.post("/triage/batch", headers=AUTH_HEADER, json=batch)
        assert res2.status_code == 200
        data2 = res2.json()
        assert data2["summary"] == {"total": 2, "ok": 2, "failed": 0}
        assert data2["results"][0]["cached"] is True
        assert data2["results"][1]["cached"] is True
        assert data2["results"][0]["label"] == label_1
        assert data2["results"][0]["confidence"] == conf_1


def test_batch_predictions_match_single_predictions_exactly() -> None:
    """SP-117 Part D (Rohan): Batch predictions must match single predictions exactly."""
    test_tickets = [
        ("T-EQ-1", "Where is my parcel? Delivery status says delayed."),
        ("T-EQ-2", "Charged twice on my credit card statement for same order."),
        ("T-EQ-3", "Please cancel my subscription immediately."),
        ("T-EQ-4", "I returned the product yesterday, when will refund arrive?"),
        ("T-EQ-5", "Cannot reset my account password otp not coming."),
    ]

    with TestClient(app) as client:
        single_results = []
        for tid, body in test_tickets:
            res = client.post(
                "/triage",
                headers=AUTH_HEADER,
                json={"ticket_id": f"SINGLE-{tid}", "body": body},
            )
            assert res.status_code == 200
            single_results.append(res.json())

        # Now send batch of exact same ticket bodies
        batch_payload = {
            "tickets": [
                {"ticket_id": f"BATCH-{tid}", "body": body}
                for tid, body in test_tickets
            ]
        }
        res_batch = client.post(
            "/triage/batch",
            headers=AUTH_HEADER,
            json=batch_payload,
        )
        assert res_batch.status_code == 200
        batch_results = res_batch.json()["results"]

        for single, batch in zip(single_results, batch_results):
            assert batch["label"] == single["label"]
            assert batch["confidence"] == single["confidence"]
            assert batch["needs_review"] == single["needs_review"]


def test_batch_100_tickets_finishes_under_two_seconds() -> None:
    """SP-117 Part D: 100 tickets finish in < 2 s locally (vectorized performance test)."""
    with TestClient(app) as client:
        batch_100 = {
            "tickets": [
                {
                    "ticket_id": f"T-PERF-{i}",
                    "subject": "Status check",
                    "body": f"Where is package order tracking shipment parcel #{i}?",
                }
                for i in range(100)
            ]
        }
        t0 = time.perf_counter()
        res = client.post("/triage/batch", headers=AUTH_HEADER, json=batch_100)
        elapsed = time.perf_counter() - t0

        assert res.status_code == 200
        data = res.json()
        assert data["summary"]["total"] == 100
        assert data["summary"]["ok"] == 100
        assert elapsed < 2.0, f"100 tickets took {elapsed:.2f}s, expected < 2.0s"


def test_batch_counters_increment_triage_total_and_needs_review() -> None:
    """SP-117 Part D (Meera): Count batch tickets in triage_total and needs_review_total."""
    with TestClient(app) as client:
        initial_triage_total = client.app.state.counters.triage_total

        batch = {
            "tickets": [
                {"ticket_id": "T-CTR-01", "body": "Where is my parcel tracking?"},
                {"ticket_id": "T-CTR-02", "body": "Cancel my order"},
            ]
        }
        res = client.post("/triage/batch", headers=AUTH_HEADER, json=batch)
        assert res.status_code == 200

        # Incremented by 2 newly predicted tickets
        assert client.app.state.counters.triage_total == initial_triage_total + 2

        # Re-sending the same cached tickets must NOT increment triage_total
        res_cached = client.post("/triage/batch", headers=AUTH_HEADER, json=batch)
        assert res_cached.status_code == 200
        assert client.app.state.counters.triage_total == initial_triage_total + 2


def test_batch_log_one_line_telemetry_without_ticket_text(caplog) -> None:
    """SP-117 Part D: Log one line: request_id, total, ok, failed, cached, latency_ms (no text)."""
    import logging

    with TestClient(app) as client:
        batch = {
            "tickets": [
                {
                    "ticket_id": "T-LOG-01",
                    "body": "SECRET_BODY_TEXT_CONFIDENTIAL_12345",
                }
            ]
        }
        with caplog.at_level(logging.INFO):
            res = client.post("/triage/batch", headers=AUTH_HEADER, json=batch)
            assert res.status_code == 200

        log_text = caplog.text
        assert "SECRET_BODY_TEXT_CONFIDENTIAL_12345" not in log_text
        assert "total=1" in log_text
        assert "ok=1" in log_text
        assert "failed=0" in log_text
        assert "cached=0" in log_text
        assert "latency_ms=" in log_text
