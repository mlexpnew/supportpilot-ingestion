"""Tests for per-key usage tracking (SP-115 Part A & Part D)."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any
from fastapi.testclient import TestClient
import pytest

from supportpilot.api.main import app, get_admin_api_keys, hash_api_key
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

        # 2. Failed auth with invalid key
        bad_key = "wrong-invalid-secret-key"
        bad_hash = hash_api_key(bad_key)

        res_bad = client.post(
            "/triage",
            headers={"X-API-Key": bad_key},
            json={"ticket_id": "T-USAGE-002", "body": "Where is my delivery?"},
        )
        assert res_bad.status_code == 401

        bad_records = store.get_key_usage(bad_hash)
        assert len(bad_records) == 1
        assert bad_records[0]["auth_fail_count"] == 1
        assert bad_records[0]["ok_count"] == 0

        # Verify plaintext key is NOT stored in DB (Zero-PII / security check)
        all_records = store.get_key_usage()
        for r in all_records:
            assert r["key_hash"] != bad_key
            assert r["key_hash"] != valid_key
            assert len(r["key_hash"]) <= 64

        # 3. Failed auth with missing key -> tracked as "none"
        res_none = client.post(
            "/triage",
            json={"ticket_id": "T-USAGE-003", "body": "Where is my delivery?"},
        )
        assert res_none.status_code == 401
        none_records = store.get_key_usage("none")
        assert len(none_records) == 1
        assert none_records[0]["auth_fail_count"] >= 1


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
