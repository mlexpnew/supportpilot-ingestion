"""Tests for dynamic API key management and authentication (SP-118 Part C & D)."""

from __future__ import annotations

import hashlib
import logging
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import text

from supportpilot.api.main import app


def test_auth_db_key_and_fallback_to_env() -> None:
    """SP-118 Part C: Test DB key lookup, revoked key 401, and fallback to env TRIAGE_API_KEYS."""
    raw_db_key = "dynamic-client-secret-12345"
    db_key_hash = hashlib.sha256(raw_db_key.encode("utf-8")).hexdigest()

    with TestClient(app) as client:
        store = client.app.state.feedback_store
        # 1. Create active client key in DB
        store.create_key(
            name="client-acme",
            key_hash=db_key_hash,
            role="client",
        )

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


def test_admin_keys_endpoints_full_lifecycle(caplog: pytest.LogCaptureFixture) -> None:
    """SP-118 Part B & D: Verify /admin/keys CRUD, 403 for client keys, zero raw key leakage, and instant effect."""
    import logging

    ADMIN_HEADERS = {"X-API-Key": "test-admin-key-1"}
    CLIENT_HEADERS = {"X-API-Key": "test-key-1"}

    with TestClient(app) as client:
        # 1. POST /admin/keys auth enforcement
        # Missing auth -> 401
        res_no_auth = client.post("/admin/keys", json={"name": "key-foo"})
        assert res_no_auth.status_code == 401

        # Client key -> 403 Forbidden ("A client key can't call /admin/* (403)")
        res_client_forbidden = client.post(
            "/admin/keys",
            headers=CLIENT_HEADERS,
            json={"name": "key-foo"},
        )
        assert res_client_forbidden.status_code == 403
        assert "Forbidden" in res_client_forbidden.json()["detail"]

        # 2. POST /admin/keys with admin key -> 201 Created
        with caplog.at_level(logging.INFO):
            res_create = client.post(
                "/admin/keys",
                headers=ADMIN_HEADERS,
                json={"name": "key-test-client-alpha", "role": "client"},
            )
        assert res_create.status_code == 201
        created_data = res_create.json()
        raw_key = created_data["key"]
        assert raw_key.startswith("") and len(raw_key) > 20
        assert created_data["name"] == "key-test-client-alpha"
        assert created_data["role"] == "client"
        hash_prefix = created_data["key_hash"]
        assert len(hash_prefix) == 8

        # Verify raw key was NEVER logged in application logs
        log_text = caplog.text
        assert raw_key not in log_text
        assert "key-test-client-alpha" in log_text
        assert hash_prefix in log_text

        # Verify DB stores only the hash, NEVER raw key
        store = client.app.state.feedback_store
        db_rec = store.get_key_by_name("key-test-client-alpha")
        assert db_rec is not None
        assert db_rec["key_hash"] != raw_key
        assert db_rec["key_hash"].startswith(hash_prefix)

        # 3. GET /admin/keys
        # Client key -> 403
        res_get_client = client.get("/admin/keys", headers=CLIENT_HEADERS)
        assert res_get_client.status_code == 403

        # Admin key -> 200 OK
        res_get = client.get("/admin/keys", headers=ADMIN_HEADERS)
        assert res_get.status_code == 200
        keys_list = res_get.json()
        assert isinstance(keys_list, list)
        matching = [k for k in keys_list if k["name"] == "key-test-client-alpha"]
        assert len(matching) == 1
        item = matching[0]
        assert item["name"] == "key-test-client-alpha"
        assert item["role"] == "client"
        assert item["hash_prefix"] == hash_prefix
        assert item["active"] is True
        assert item["revoked_at"] is None
        # Raw key is NEVER returned in list
        assert raw_key not in str(keys_list)

        # 4. Newly created key works immediately on /triage without restart
        res_triage_ok = client.post(
            "/triage",
            headers={"X-API-Key": raw_key},
            json={"ticket_id": "T-NEW-1", "body": "Where is my parcel?"},
        )
        assert res_triage_ok.status_code == 200

        # 5. POST /admin/keys/{name}/revoke
        # Client key -> 403
        res_revoke_client = client.post(
            "/admin/keys/key-test-client-alpha/revoke",
            headers=CLIENT_HEADERS,
        )
        assert res_revoke_client.status_code == 403

        # Non-existent key -> 404 Not Found
        res_revoke_404 = client.post(
            "/admin/keys/non-existent-key-xyz/revoke",
            headers=ADMIN_HEADERS,
        )
        assert res_revoke_404.status_code == 404

        # Admin key revokes existing key -> 200 OK
        caplog.clear()
        with caplog.at_level(logging.INFO):
            res_revoke = client.post(
                "/admin/keys/key-test-client-alpha/revoke",
                headers=ADMIN_HEADERS,
            )
        assert res_revoke.status_code == 200
        assert res_revoke.json()["status"] == "revoked"

        # Log only name and hash prefix
        log_text_revoke = caplog.text
        assert raw_key not in log_text_revoke
        assert "key-test-client-alpha" in log_text_revoke
        assert hash_prefix in log_text_revoke

        # 6. Revoked key immediately receives 401 with no restart
        res_triage_revoked = client.post(
            "/triage",
            headers={"X-API-Key": raw_key},
            json={"ticket_id": "T-REVOKED-2", "body": "Where is my parcel?"},
        )
        assert res_triage_revoked.status_code == 401

        # 7. GET /admin/keys reflects active: False
        res_get_after = client.get("/admin/keys", headers=ADMIN_HEADERS)
        keys_after = res_get_after.json()
        item_after = [k for k in keys_after if k["name"] == "key-test-client-alpha"][0]
        assert item_after["active"] is False
        assert item_after["revoked_at"] is not None

        # 8. Duplicate key creation -> 409 Conflict
        res_dup = client.post(
            "/admin/keys",
            headers=ADMIN_HEADERS,
            json={"name": "key-test-client-alpha"},
        )
        assert res_dup.status_code == 409


def test_part_d_key_lifecycle_and_security(caplog: pytest.LogCaptureFixture) -> None:
    """SP-118 Part D Requirements:
    1. Create a key; it works on /triage with no restart.
    2. Revoke the key; the very next request gets 401.
    3. A client key on /admin/* gets 403.
    4. Search the DB rows and logs for the raw key; it must be in neither.
    5. An env key still works (fallback).
    """
    admin_headers = {"X-API-Key": "test-admin-key-1"}
    env_client_headers = {"X-API-Key": "test-key-1"}

    with TestClient(app) as client:
        store = client.app.state.feedback_store

        # 3. A client key on /admin/* gets 403 (testing env client key)
        res_admin_list = client.get("/admin/keys", headers=env_client_headers)
        assert res_admin_list.status_code == 403
        assert "Forbidden" in res_admin_list.json()["detail"]

        res_admin_create = client.post(
            "/admin/keys",
            headers=env_client_headers,
            json={"name": "forbidden-attempt"},
        )
        assert res_admin_create.status_code == 403
        assert "Forbidden" in res_admin_create.json()["detail"]

        res_admin_revoke = client.post(
            "/admin/keys/sample/revoke",
            headers=env_client_headers,
        )
        assert res_admin_revoke.status_code == 403
        assert "Forbidden" in res_admin_revoke.json()["detail"]

        # 5. An env key still works (fallback)
        res_env = client.post(
            "/triage",
            headers=env_client_headers,
            json={"ticket_id": "T-ENV-PART-D", "body": "Need refund"},
        )
        assert res_env.status_code == 200

        # 1. Create a key; it works on /triage with no restart
        caplog.clear()
        with caplog.at_level(logging.INFO):
            res_create = client.post(
                "/admin/keys",
                headers=admin_headers,
                json={"name": "part-d-client-key", "role": "client"},
            )
        assert res_create.status_code == 201
        created_payload = res_create.json()
        raw_key = created_payload["key"]
        hash_prefix = created_payload["key_hash"]
        assert raw_key and len(raw_key) > 20
        assert len(hash_prefix) == 8

        # Works on /triage immediately with no restart
        res_triage_new = client.post(
            "/triage",
            headers={"X-API-Key": raw_key},
            json={"ticket_id": "T-NEW-PART-D", "body": "Where is my parcel?"},
        )
        assert res_triage_new.status_code == 200

        # A DB client key on /admin/* also gets 403
        db_client_headers = {"X-API-Key": raw_key}
        res_db_client_admin = client.get("/admin/keys", headers=db_client_headers)
        assert res_db_client_admin.status_code == 403
        assert "Forbidden" in res_db_client_admin.json()["detail"]

        # 2. Revoke the key; the very next request gets 401
        res_revoke = client.post(
            "/admin/keys/part-d-client-key/revoke",
            headers=admin_headers,
        )
        assert res_revoke.status_code == 200
        assert res_revoke.json()["status"] == "revoked"

        # The very next request gets 401
        res_triage_revoked = client.post(
            "/triage",
            headers={"X-API-Key": raw_key},
            json={"ticket_id": "T-REVOKED-PART-D", "body": "Where is my parcel?"},
        )
        assert res_triage_revoked.status_code == 401

        # 4. Search the DB rows and logs for the raw key; it must be in neither
        # A) Search logs
        all_logs = caplog.text
        assert (
            raw_key not in all_logs
        ), f"Security breach: raw key {raw_key} found in application logs!"

        # B) Search DB rows across all tables
        with store.engine.connect() as conn:
            for table_name in ["api_keys", "api_key_usage", "predictions", "feedback"]:
                rows = conn.execute(text(f"SELECT * FROM {table_name}")).fetchall()
                for row in rows:
                    for col in row:
                        assert (
                            raw_key not in str(col)
                        ), f"Security breach: raw key found in {table_name} column: {col}"

        # 5. An env key still works (fallback) after revoking DB key
        res_env_after = client.post(
            "/triage",
            headers=env_client_headers,
            json={"ticket_id": "T-ENV-FINAL", "body": "Password reset"},
        )
        assert res_env_after.status_code == 200
