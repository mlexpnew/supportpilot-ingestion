"""Tests for per-key usage tracking (SP-115 Part A & Part D)."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any
from fastapi.testclient import TestClient
import pytest

from supportpilot.api.alerts import (
    ALERT_NAME_NEVER_SEEN,
    evaluate_alerts,
    get_alert_thresholds,
)
from supportpilot.api.main import (
    app,
    get_admin_api_keys,
    hash_api_key,
    validate_key_separation,
)
from supportpilot.api.models import MetricsResponse
from supportpilot.feedback.store import FeedbackStore, init_db


def test_record_key_usage_upsert(tmp_path: Any) -> None:
    """Verify record_key_usage correctly initializes and increments ok_count / auth_fail_count."""
    db_file = tmp_path / "test_usage.db"
    init_db(db_file)
    store = FeedbackStore(db_file)

    try:
        key_hash = "abc12345"

        # 1. Initial success
        store.record_key_usage(
            key_hash=key_hash, ok=True, timestamp="2026-11-01T10:00:00+00:00"
        )
        rows = store.get_key_usage(key_hash=key_hash)
        assert len(rows) == 1
        record = rows[0]
        assert record["key_hash"] == key_hash
        assert record["first_seen"] == "2026-11-01T10:00:00+00:00"
        assert record["last_seen"] == "2026-11-01T10:00:00+00:00"
        assert record["ok_count"] == 1
        assert record["auth_fail_count"] == 0

        # 2. Second success: updates last_seen, keeps first_seen, ok_count becomes 2
        store.record_key_usage(
            key_hash=key_hash, ok=True, timestamp="2026-11-01T11:00:00+00:00"
        )
        rows = store.get_key_usage(key_hash=key_hash)
        assert len(rows) == 1
        record = rows[0]
        assert record["first_seen"] == "2026-11-01T10:00:00+00:00"
        assert record["last_seen"] == "2026-11-01T11:00:00+00:00"
        assert record["ok_count"] == 2
        assert record["auth_fail_count"] == 0

        # 3. Failed auth: auth_fail_count becomes 1
        store.record_key_usage(
            key_hash=key_hash, ok=False, timestamp="2026-11-01T12:00:00+00:00"
        )
        rows = store.get_key_usage(key_hash=key_hash)
        assert len(rows) == 1
        record = rows[0]
        assert record["first_seen"] == "2026-11-01T10:00:00+00:00"
        assert record["last_seen"] == "2026-11-01T12:00:00+00:00"
        assert record["ok_count"] == 2
        assert record["auth_fail_count"] == 1
    finally:
        store.close()


def test_api_tracks_key_usage_on_success_and_failure() -> None:
    """Verify /triage tracks ok_count on success, and tracks auth_fail_count for invalid / missing keys."""
    with TestClient(app) as client:
        store: FeedbackStore = client.app.state.feedback_store

        # 1. Success with valid key 'test-key-1'
        valid_key = "test-key-1"
        valid_hash = hash_api_key(valid_key)

        res = client.post(
            "/triage",
            headers={"X-API-Key": valid_key},
            json={"ticket_id": "T-USAGE-001", "body": "Where is my delivery?"},
        )
        assert res.status_code == 200

        records = store.get_key_usage(valid_hash)
        assert len(records) == 1
        assert records[0]["ok_count"] >= 1
        assert records[0]["auth_fail_count"] == 0

        # 2. Failed auth with invalid key -> collapsed into shared "unknown" row (table limit protection)
        bad_key = "wrong-invalid-secret-key"
        bad_hash = hash_api_key(bad_key)

        res_bad = client.post(
            "/triage",
            headers={"X-API-Key": bad_key},
            json={"ticket_id": "T-USAGE-002", "body": "Where is my delivery?"},
        )
        assert res_bad.status_code == 401

        unknown_records = store.get_key_usage("unknown")
        assert len(unknown_records) == 1
        assert unknown_records[0]["auth_fail_count"] == 1
        assert unknown_records[0]["ok_count"] == 0

        # Verify no dedicated row was created for the bad hash (DoS protection)
        assert len(store.get_key_usage(bad_hash)) == 0

        # Verify plaintext key is NOT stored in DB (Zero-PII / security check)
        all_records = store.get_key_usage()
        for r in all_records:
            assert r["key_hash"] != bad_key
            assert r["key_hash"] != valid_key
            assert len(r["key_hash"]) <= 64

        # 3. Failed auth with missing key -> also increments "unknown"
        res_none = client.post(
            "/triage",
            json={"ticket_id": "T-USAGE-003", "body": "Where is my delivery?"},
        )
        assert res_none.status_code == 401
        unknown_records_after = store.get_key_usage("unknown")
        assert len(unknown_records_after) == 1
        assert unknown_records_after[0]["auth_fail_count"] == 2


def test_usage_tracking_failure_never_breaks_auth(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify that if the store raises an error during record_key_usage, authentication still succeeds."""
    with TestClient(app) as client:
        store: FeedbackStore = client.app.state.feedback_store

        def broken_record_key_usage(*args: Any, **kwargs: Any) -> None:
            raise RuntimeError("Database connection died!")

        monkeypatch.setattr(store, "record_key_usage", broken_record_key_usage)

        # Valid request should still succeed with 200 even though usage tracking failed
        res = client.post(
            "/triage",
            headers={"X-API-Key": "test-key-1"},
            json={"ticket_id": "T-USAGE-004", "body": "I need a refund for my order"},
        )
        assert res.status_code == 200
        assert res.json()["label"] == "refund"


def test_admin_api_keys_fail_fast(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify get_admin_api_keys raises ValueError when ADMIN_API_KEYS is missing or empty."""
    monkeypatch.delenv("ADMIN_API_KEYS", raising=False)
    with pytest.raises(
        ValueError, match="ADMIN_API_KEYS environment variable is required"
    ):
        get_admin_api_keys()

    monkeypatch.setenv("ADMIN_API_KEYS", "   ")
    with pytest.raises(
        ValueError, match="ADMIN_API_KEYS environment variable is required"
    ):
        get_admin_api_keys()

    monkeypatch.setenv("ADMIN_API_KEYS", ",,,")
    with pytest.raises(ValueError, match="must contain at least one valid key"):
        get_admin_api_keys()


def test_usage_endpoint_requires_admin_key() -> None:
    """Verify /usage rejects missing keys and client keys, and succeeds with admin key."""
    with TestClient(app) as client:
        # 1. Missing auth header -> 401
        res_no_auth = client.get("/usage")
        assert res_no_auth.status_code == 401

        # 2. Client key (least-privilege check: client key must NOT access /usage) -> 401
        res_client_auth = client.get("/usage", headers={"X-API-Key": "test-key-1"})
        assert res_client_auth.status_code == 401

        # 3. Admin key -> 200
        res_admin = client.get("/usage", headers={"X-API-Key": "test-admin-key-1"})
        assert res_admin.status_code == 200
        data = res_admin.json()
        assert isinstance(data, list)
        for item in data:
            assert "key_hash" in item
            assert "first_seen" in item
            assert "last_seen" in item
            assert "ok_count" in item
            assert "auth_fail_count" in item


def test_no_client_traffic_alert() -> None:
    """Verify that a known client key with last_seen older than 24h triggers warning: no_client_traffic in /alerts."""
    with TestClient(app) as client:
        store: FeedbackStore = client.app.state.feedback_store

        # Insert traffic for test-key-2 from 25 hours ago
        key_2_hash = hash_api_key("test-key-2")
        old_time = (datetime.now(timezone.utc) - timedelta(hours=25)).isoformat()
        store.record_key_usage(key_hash=key_2_hash, ok=True, timestamp=old_time)

        # Query /alerts with test-key-1
        res = client.get("/alerts", headers={"X-API-Key": "test-key-1"})
        assert res.status_code == 200
        body = res.json()
        alerts = body["alerts"]

        traffic_alerts = [a for a in alerts if a["name"] == "no_client_traffic"]
        assert len(traffic_alerts) >= 1
        alert = traffic_alerts[0]
        assert alert["severity"] == "warning"
        assert alert["threshold"] == 24.0
        assert alert["value"] >= 24.9

        # Now update test-key-2 traffic to current time (< 24h)
        now_time = datetime.now(timezone.utc).isoformat()
        store.record_key_usage(key_hash=key_2_hash, ok=True, timestamp=now_time)

        res_recent = client.get("/alerts", headers={"X-API-Key": "test-key-1"})
        assert res_recent.status_code == 200
        active_names = [a["name"] for a in res_recent.json()["alerts"]]
        assert "no_client_traffic" not in active_names


def test_stale_key_alert_fires_after_24h_faked_time() -> None:
    """Verify evaluate_alerts fires no_client_traffic alert when last_seen > 24 hours using faked reference time."""
    fake_now = datetime(2026, 11, 2, 12, 0, 0, tzinfo=timezone.utc)
    stale_time = fake_now - timedelta(hours=25)
    recent_time = fake_now - timedelta(hours=2)

    thresholds = get_alert_thresholds()
    metrics = MetricsResponse(
        requests_total=0,
        triage_total=0,
        errors_total=0,
        avg_latency_ms=0.0,
        needs_review_rate=0.0,
        rate_limited_total=0,
    )

    client_key_hashes = {"client_hash_1"}

    # 1. Stale key (25h ago) -> triggers alert
    stale_usage = [
        {
            "key_hash": "client_hash_1",
            "first_seen": stale_time.isoformat(),
            "last_seen": stale_time.isoformat(),
            "ok_count": 5,
            "auth_fail_count": 0,
        }
    ]
    alerts_stale = evaluate_alerts(
        metrics=metrics,
        thresholds=thresholds,
        key_usage=stale_usage,
        client_key_hashes=client_key_hashes,
        now=fake_now,
    )
    stale_alerts = [a for a in alerts_stale if a.name == "no_client_traffic"]
    assert len(stale_alerts) == 1
    assert stale_alerts[0].severity == "warning"
    assert stale_alerts[0].threshold == 24.0
    assert stale_alerts[0].value == 25.0

    # 2. Active key (2h ago) -> does not trigger alert
    recent_usage = [
        {
            "key_hash": "client_hash_1",
            "first_seen": recent_time.isoformat(),
            "last_seen": recent_time.isoformat(),
            "ok_count": 5,
            "auth_fail_count": 0,
        }
    ]
    alerts_recent = evaluate_alerts(
        metrics=metrics,
        thresholds=thresholds,
        key_usage=recent_usage,
        client_key_hashes=client_key_hashes,
        now=fake_now,
    )
    assert not any(a.name == "no_client_traffic" for a in alerts_recent)


def test_unknown_keys_shared_row_prevents_table_growth() -> None:
    """Verify failed attempts from multiple distinct random keys collapse into key_hash='unknown' (Sana DoS guard)."""
    with TestClient(app) as client:
        store: FeedbackStore = client.app.state.feedback_store

        # Send 10 different random keys
        for i in range(10):
            res = client.post(
                "/triage",
                headers={"X-API-Key": f"random-attack-key-{i}"},
                json={"ticket_id": f"T-ATTACK-{i}", "body": "Exploit attempt"},
            )
            assert res.status_code == 401

        # Must have only ONE row for all failures with key_hash = "unknown"
        unknown_records = store.get_key_usage("unknown")
        assert len(unknown_records) == 1
        assert unknown_records[0]["auth_fail_count"] == 10

        # Verify no individual rows were created for any of the attack keys
        all_records = store.get_key_usage()
        for r in all_records:
            assert not r["key_hash"].startswith("random-attack")


def test_admin_and_client_keys_overlap_fails_fast() -> None:
    """Verify startup fails fast if any key appears in both ADMIN_API_KEYS and TRIAGE_API_KEYS (Sana least privilege check)."""
    client_keys = {"client-key-1", "shared-compromised-key"}
    admin_keys = {"admin-key-1", "shared-compromised-key"}

    with pytest.raises(
        ValueError, match="ADMIN_API_KEYS and TRIAGE_API_KEYS must not overlap"
    ):
        validate_key_separation(client_keys, admin_keys)

    # Disjoint keys must pass without error
    validate_key_separation({"client-key-1"}, {"admin-key-1"})


def test_never_seen_alert_fires_once_service_up_24h() -> None:
    """Verify known client key with no usage row triggers alert 'never seen' once service has been up 24 h (pilot bug fix)."""
    fake_now = datetime(2026, 11, 2, 12, 0, 0, tzinfo=timezone.utc)
    service_started_stale = fake_now - timedelta(hours=26)
    service_started_recent = fake_now - timedelta(hours=5)

    thresholds = get_alert_thresholds()
    metrics = MetricsResponse(
        requests_total=0,
        triage_total=0,
        errors_total=0,
        avg_latency_ms=0.0,
        needs_review_rate=0.0,
        rate_limited_total=0,
    )
    client_key_hashes = {"client_hash_never_called"}

    # 1. Empty usage list, service up for 26h -> triggers 'never seen'
    alerts_26h = evaluate_alerts(
        metrics=metrics,
        thresholds=thresholds,
        key_usage=[],
        client_key_hashes=client_key_hashes,
        now=fake_now,
        started_at=service_started_stale,
    )
    never_seen_alerts = [a for a in alerts_26h if a.name == ALERT_NAME_NEVER_SEEN]
    assert len(never_seen_alerts) == 1
    assert never_seen_alerts[0].severity == "warning"
    assert never_seen_alerts[0].threshold == 24.0
    assert never_seen_alerts[0].value == 26.0

    # 2. Empty usage list, service up for only 5h -> does NOT trigger 'never seen'
    alerts_5h = evaluate_alerts(
        metrics=metrics,
        thresholds=thresholds,
        key_usage=[],
        client_key_hashes=client_key_hashes,
        now=fake_now,
        started_at=service_started_recent,
    )
    assert not any(a.name == ALERT_NAME_NEVER_SEEN for a in alerts_5h)
