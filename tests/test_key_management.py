"""Tests for dynamic API key management and authentication (SP-118 Part C & D)."""

from __future__ import annotations

import hashlib
from fastapi.testclient import TestClient

from supportpilot.api.main import app
from supportpilot.feedback.store import get_feedback_store


def test_auth_db_key_and_fallback_to_env() -> None:
    """SP-118 Part C: Test DB key lookup, revoked key 401, and fallback to env TRIAGE_API_KEYS."""
    store = get_feedback_store()

    # 1. Create active client key in DB
    raw_db_key = "dynamic-client-secret-12345"
    db_key_hash = hashlib.sha256(raw_db_key.encode("utf-8")).hexdigest()
    store.create_key(
        name="client-acme",
        key_hash=db_key_hash,
        role="client",
    )

    with TestClient(app) as client:
        # A. Key from DB works on /triage
        res_db = client.post(
            "/triage",
            headers={"X-API-Key": raw_db_key},
            json={"ticket_id": "T-DB-1", "body": "Where is my parcel?"},
        )
        assert res_db.status_code == 200

        # Verify last_used_at was updated in DB
        db_record = store.get_key_by_hash(db_key_hash)
        assert db_record is not None
        assert db_record["last_used_at"] is not None

        # B. Key from env (test-key-1) works as fallback
        res_env = client.post(
            "/triage",
            headers={"X-API-Key": "test-key-1"},
            json={"ticket_id": "T-ENV-1", "body": "Where is my parcel?"},
        )
        assert res_env.status_code == 200

        # C. Revoke the DB key -> immediately receives 401
        assert store.revoke_key("client-acme") is True
        res_revoked = client.post(
            "/triage",
            headers={"X-API-Key": raw_db_key},
            json={"ticket_id": "T-REVOKED-1", "body": "Where is my parcel?"},
        )
        assert res_revoked.status_code == 401

        # D. Invalid key -> 401
        res_bad = client.post(
            "/triage",
            headers={"X-API-Key": "completely-invalid-key"},
            json={"ticket_id": "T-BAD-1", "body": "Where is my parcel?"},
        )
        assert res_bad.status_code == 401
