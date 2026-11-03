"""Unit tests for api_keys table and FeedbackStore key management methods (SP-118 Part A)."""

from __future__ import annotations

import pytest

from supportpilot.feedback.store import get_feedback_store


def test_api_keys_table_crud_methods() -> None:
    """Verify create_key, get_key_by_hash, list_keys, revoke_key, and update_key_last_used."""
    store = get_feedback_store()

    # 1. Create a client key
    key1 = store.create_key(
        name="test-client-1",
        key_hash="hash_client_111",
        role="client",
    )
    assert key1["name"] == "test-client-1"
    assert key1["key_hash"] == "hash_client_111"
    assert key1["role"] == "client"
    assert key1["revoked_at"] is None
    assert key1["last_used_at"] is None

    # 2. Create an admin key
    key2 = store.create_key(
        name="test-admin-1",
        key_hash="hash_admin_222",
        role="admin",
    )
    assert key2["name"] == "test-admin-1"
    assert key2["role"] == "admin"

    # 3. Find by hash
    found = store.get_key_by_hash("hash_client_111")
    assert found is not None
    assert found["name"] == "test-client-1"
    assert found["role"] == "client"
    assert found["revoked_at"] is None

    # Unknown hash returns None
    assert store.get_key_by_hash("nonexistent_hash") is None

    # 4. List keys
    all_keys = store.list_keys()
    names = [k["name"] for k in all_keys]
    assert "test-client-1" in names
    assert "test-admin-1" in names
    for k in all_keys:
        assert "key_hash" in k
        assert "role" in k
        assert "created_at" in k

    # 5. Update last used
    assert found["last_used_at"] is None
    store.update_key_last_used("hash_client_111")
    updated = store.get_key_by_hash("hash_client_111")
    assert updated is not None
    assert updated["last_used_at"] is not None

    # 6. Revoke key
    assert store.revoke_key("test-client-1") is True
    revoked = store.get_key_by_hash("hash_client_111")
    assert revoked is not None
    assert revoked["revoked_at"] is not None

    # Revoking already revoked key returns False
    assert store.revoke_key("test-client-1") is False

    # Revoking non-existent key returns False
    assert store.revoke_key("nonexistent_key") is False


def test_api_keys_validation() -> None:
    """Verify input validation on role, empty name, and length."""
    store = get_feedback_store()

    # Invalid role
    with pytest.raises(ValueError, match="Invalid role"):
        store.create_key(name="test-invalid", key_hash="h1", role="superuser")

    # Empty name
    with pytest.raises(ValueError, match="cannot be empty"):
        store.create_key(name="   ", key_hash="h2", role="client")

    # Name exceeding 64 chars
    with pytest.raises(ValueError, match="exceeds maximum length"):
        store.create_key(name="a" * 65, key_hash="h3", role="client")
