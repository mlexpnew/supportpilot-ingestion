"""Tests for per-key usage tracking (SP-115 Part A & Part D)."""

from __future__ import annotations

from typing import Any
from fastapi.testclient import TestClient
import pytest

from supportpilot.api.main import app, hash_api_key
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
