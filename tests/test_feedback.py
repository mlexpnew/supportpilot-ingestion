"""Tests for SupportPilot Feedback Loop (SP-110).

Covers:
  - Prediction saved on /triage and store
  - Upsert on duplicate feedback (one row per ticket)
  - 404 on unknown ticket_id (not triaged)
  - 422 / ValueError on invalid label
  - Accuracy math with known numbers
  - Null (None) live_accuracy when empty
  - Zero ticket body / text anywhere in SQLite DB
  - SQL-injection strings stored safely via ? placeholders
  - Fail-fast at startup and in endpoints when feedback_store is missing
  - Graceful degradation when DB write fails during triage
  - Clean single DB access path on app.state
"""

from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from supportpilot.api.main import app
from supportpilot.feedback import FeedbackRequest, FeedbackStore, init_db
from supportpilot.feedback.__main__ import main as cli_main


def test_init_db_creates_both_tables_and_indices(tmp_path: Path) -> None:
    """Verify init_db creates predictions and feedback tables with correct columns."""
    db_file = tmp_path / "init_check.db"
    assert not db_file.exists()

    init_db(db_file)
    assert db_file.exists()

    conn = sqlite3.connect(str(db_file))
    cur = conn.cursor()
    cur.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name;")
    tables = {r[0] for r in cur.fetchall()}
    assert "predictions" in tables
    assert "feedback" in tables

    # Verify predictions table columns
    cur.execute("PRAGMA table_info(predictions);")
    pred_cols = [r[1] for r in cur.fetchall()]
    assert pred_cols == [
        "ticket_id",
        "predicted_label",
        "confidence",
        "model_version",
        "created_at",
    ]

    # Verify feedback table columns and primary key (ticket_id is PK)
    cur.execute("PRAGMA table_info(feedback);")
    fb_info = cur.fetchall()
    fb_cols = [r[1] for r in fb_info]
    assert fb_cols == ["ticket_id", "correct_label", "agent_id", "created_at"]
    fb_pk = [r[1] for r in fb_info if r[5] > 0]
    assert fb_pk == ["ticket_id"]

    conn.close()


def test_prediction_saved_in_store_and_via_triage(
    tmp_path: Path, auth_headers: dict[str, str]
) -> None:
    """Verify triage predictions are saved to the predictions table."""
    db_file = tmp_path / "pred_test.db"
    store = FeedbackStore(db_file)

    # 1. Direct store call
    store.save_prediction(
        ticket_id="T-STORE-PRED-1",
        predicted_label="delivery",
        confidence=0.88,
        model_version="triage-v1.0.0-20261022",
    )
    pred = store.get_prediction("T-STORE-PRED-1")
    assert pred is not None
    assert pred["ticket_id"] == "T-STORE-PRED-1"
    assert pred["predicted_label"] == "delivery"
    assert abs(pred["confidence"] - 0.88) < 1e-4
    assert pred["model_version"] == "triage-v1.0.0-20261022"
    assert "created_at" in pred

    # 2. Via /triage API call
    with TestClient(app) as client:
        res = client.post(
            "/triage",
            headers=auth_headers,
            json={
                "ticket_id": "T-API-PRED-1",
                "body": "Where is my parcel? Delivery was scheduled today.",
            },
        )
        assert res.status_code == 200
        data = res.json()

        # Query the feedback store on app.state
        saved = app.state.feedback_store.get_prediction("T-API-PRED-1")
        assert saved is not None
        assert saved["ticket_id"] == "T-API-PRED-1"
        assert saved["predicted_label"] == data["label"]
        assert abs(saved["confidence"] - data["confidence"]) < 1e-4
        assert saved["model_version"] == data["model_version"]


def test_feedback_upsert_on_duplicate(
    tmp_path: Path, auth_headers: dict[str, str]
) -> None:
    """Verify duplicate feedback on the same ticket updates in place (one row per ticket)."""
    db_file = tmp_path / "upsert_test.db"
    store = FeedbackStore(db_file)

    store.save_prediction("T-UP-1", "delivery", 0.9, "triage-v1.0.0-20261022")

    # First feedback
    store.save_feedback("T-UP-1", "delivery", "agent-alice")
    fb1 = store.get_feedback("T-UP-1")
    assert fb1 is not None
    assert fb1["correct_label"] == "delivery"
    assert fb1["agent_id"] == "agent-alice"

    # Second feedback for same ticket (agent corrected or changed mind)
    store.save_feedback("T-UP-1", "billing", "agent-bob")
    fb2 = store.get_feedback("T-UP-1")
    assert fb2 is not None
    assert fb2["correct_label"] == "billing"
    assert fb2["agent_id"] == "agent-bob"

    # Verify row count is still exactly 1 in feedback table
    total, _ = store.get_live_accuracy()
    assert total == 1

    # Verify API level upsert
    with TestClient(app) as client:
        # Triage ticket first
        client.post(
            "/triage",
            headers=auth_headers,
            json={"ticket_id": "T-API-UP-1", "body": "Need invoice copy."},
        )

        # First feedback via API
        res1 = client.post(
            "/feedback",
            headers=auth_headers,
            json={
                "ticket_id": "T-API-UP-1",
                "correct_label": "billing",
                "agent_id": "agent-1",
            },
        )
        assert res1.status_code == 200
        assert res1.json()["status"] == "ok"

        # Update feedback via API
        res2 = client.post(
            "/feedback",
            headers=auth_headers,
            json={
                "ticket_id": "T-API-UP-1",
                "correct_label": "refund",
                "agent_id": "agent-2",
            },
        )
        assert res2.status_code == 200

        updated = app.state.feedback_store.get_feedback("T-API-UP-1")
        assert updated is not None
        assert updated["correct_label"] == "refund"
        assert updated["agent_id"] == "agent-2"


def test_retriage_after_feedback_preserves_original_prediction(
    tmp_path: Path, auth_headers: dict[str, str]
) -> None:
    """Verify triage -> feedback -> re-triage leaves the stored prediction unchanged (Rohan's decision)."""
    db_file = tmp_path / "retriage_test.db"
    store = FeedbackStore(db_file)

    # 1. Triage: initial prediction
    store.save_prediction(
        ticket_id="T-RE-1",
        predicted_label="delivery",
        confidence=0.85,
        model_version="v1",
    )
    pred_orig = store.get_prediction("T-RE-1")
    assert pred_orig is not None
    assert pred_orig["predicted_label"] == "delivery"

    # Re-triage with no feedback -> should overwrite
    store.save_prediction(
        ticket_id="T-RE-1",
        predicted_label="billing",
        confidence=0.92,
        model_version="v2",
    )
    pred_overwritten = store.get_prediction("T-RE-1")
    assert pred_overwritten is not None
    assert pred_overwritten["predicted_label"] == "billing"
    assert abs(pred_overwritten["confidence"] - 0.92) < 1e-4

    # 2. Agent gives feedback
    store.save_feedback("T-RE-1", "delivery", "agent-rohan")

    # 3. Re-triage after feedback -> stored prediction must remain unchanged
    store.save_prediction(
        ticket_id="T-RE-1",
        predicted_label="refund",
        confidence=0.60,
        model_version="v3",
    )
    pred_after_fb = store.get_prediction("T-RE-1")
    assert pred_after_fb is not None
    assert pred_after_fb["predicted_label"] == "billing"
    assert abs(pred_after_fb["confidence"] - 0.92) < 1e-4
    assert pred_after_fb["model_version"] == "v2"

    # Also test end-to-end via API
    with TestClient(app) as client:
        # Step A: First triage
        res_t1 = client.post(
            "/triage",
            headers=auth_headers,
            json={"ticket_id": "T-API-RE-1", "body": "Where is my package delivery?"},
        )
        assert res_t1.status_code == 200
        first_label = res_t1.json()["label"]
        stored_pred_1 = app.state.feedback_store.get_prediction("T-API-RE-1")
        assert stored_pred_1 is not None
        assert stored_pred_1["predicted_label"] == first_label

        # Step B: Agent submits feedback
        res_fb = client.post(
            "/feedback",
            headers=auth_headers,
            json={
                "ticket_id": "T-API-RE-1",
                "correct_label": "delivery",
                "agent_id": "agent-1",
            },
        )
        assert res_fb.status_code == 200

        # Step C: Re-triage with different text
        res_t2 = client.post(
            "/triage",
            headers=auth_headers,
            json={
                "ticket_id": "T-API-RE-1",
                "body": "Need refund and billing help right now immediately.",
            },
        )
        assert res_t2.status_code == 200

        # The stored prediction in the DB must remain the original one that the agent saw!
        stored_pred_after = app.state.feedback_store.get_prediction("T-API-RE-1")
        assert stored_pred_after is not None
        assert stored_pred_after["predicted_label"] == stored_pred_1["predicted_label"]
        assert stored_pred_after["created_at"] == stored_pred_1["created_at"]


def test_feedback_unknown_ticket_404(
    tmp_path: Path, auth_headers: dict[str, str]
) -> None:
    """Verify feedback on an untriaged ticket raises KeyError and returns 404."""
    db_file = tmp_path / "unknown_test.db"
    store = FeedbackStore(db_file)

    # Store level raises KeyError
    with pytest.raises(KeyError, match="has not been triaged"):
        store.save_feedback("T-NEVER-TRIAGED-999", "delivery", "agent-1")

    # API level returns 404
    with TestClient(app) as client:
        res = client.post(
            "/feedback",
            headers=auth_headers,
            json={
                "ticket_id": "T-NEVER-TRIAGED-999",
                "correct_label": "delivery",
                "agent_id": "agent-1",
            },
        )
        assert res.status_code == 404
        assert "has not been triaged" in res.json()["detail"]


def test_feedback_invalid_label_422(
    tmp_path: Path, auth_headers: dict[str, str]
) -> None:
    """Verify invalid correct_label raises ValueError in store and returns 422 in API."""
    db_file = tmp_path / "invalid_label_test.db"
    store = FeedbackStore(db_file)
    store.save_prediction("T-LBL-1", "delivery", 0.9, "triage-v1.0.0-20261022")

    # Store level raises ValueError
    with pytest.raises(ValueError, match="Invalid correct_label"):
        store.save_feedback("T-LBL-1", "fraudulent_charge", "agent-1")

    # API level returns 422
    with TestClient(app) as client:
        client.post(
            "/triage",
            headers=auth_headers,
            json={"ticket_id": "T-API-LBL-1", "body": "Test ticket."},
        )

        res = client.post(
            "/feedback",
            headers=auth_headers,
            json={
                "ticket_id": "T-API-LBL-1",
                "correct_label": "nonexistent_label",
                "agent_id": "agent-1",
            },
        )
        assert res.status_code == 422

        # Empty ticket_id or agent_id returns 422
        res_empty_tid = client.post(
            "/feedback",
            headers=auth_headers,
            json={
                "ticket_id": "   ",
                "correct_label": "delivery",
                "agent_id": "agent-1",
            },
        )
        assert res_empty_tid.status_code == 422

        res_empty_agent = client.post(
            "/feedback",
            headers=auth_headers,
            json={
                "ticket_id": "T-API-LBL-1",
                "correct_label": "delivery",
                "agent_id": "   ",
            },
        )
        assert res_empty_agent.status_code == 422


def test_feedback_request_pydantic_model() -> None:
    """Verify FeedbackRequest Pydantic model contract and field validation."""
    req = FeedbackRequest(
        ticket_id="T-999",
        correct_label="delivery",
        agent_id="agent-sana",
    )
    assert req.ticket_id == "T-999"
    assert req.correct_label == "delivery"
    assert req.agent_id == "agent-sana"

    # Case normalization
    req_upper = FeedbackRequest(
        ticket_id="T-999",
        correct_label="DELIVERY",
        agent_id="agent-sana",
    )
    assert req_upper.correct_label == "delivery"

    # Validation errors
    with pytest.raises(ValueError, match="ticket_id cannot be empty"):
        FeedbackRequest(ticket_id="   ", correct_label="delivery", agent_id="agent-1")

    with pytest.raises(ValueError, match="agent_id cannot be empty"):
        FeedbackRequest(ticket_id="T-1", correct_label="delivery", agent_id="   ")

    with pytest.raises(ValueError, match="Invalid correct_label"):
        FeedbackRequest(
            ticket_id="T-1", correct_label="invalid_label", agent_id="agent-1"
        )


def test_accuracy_math_known_numbers(tmp_path: Path) -> None:
    """Verify live accuracy calculation against exact known numbers."""
    db_file = tmp_path / "acc_math.db"
    store = FeedbackStore(db_file)

    # 4 tickets:
    # T1: predicted delivery, feedback delivery (agreed)
    # T2: predicted billing, feedback billing (agreed)
    # T3: predicted refund, feedback cancellation (disagreed)
    # T4: predicted account, feedback other (disagreed)
    store.save_prediction("T1", "delivery", 0.9, "v1")
    store.save_prediction("T2", "billing", 0.85, "v1")
    store.save_prediction("T3", "refund", 0.70, "v1")
    store.save_prediction("T4", "account", 0.65, "v1")

    store.save_feedback("T1", "delivery", "a1")
    store.save_feedback("T2", "billing", "a1")
    store.save_feedback("T3", "cancellation", "a2")
    store.save_feedback("T4", "other", "a2")

    # Math: total = 4, agreed = 2 -> 2/4 = 0.50
    total, live_acc = store.get_live_accuracy()
    assert total == 4
    assert live_acc == 0.50

    # Per-label breakdown
    label_metrics = store.get_label_metrics()
    assert label_metrics["delivery"] == {
        "total": 1,
        "agreed": 1,
        "accuracy": 1.0,
    }
    assert label_metrics["billing"] == {"total": 1, "agreed": 1, "accuracy": 1.0}
    assert label_metrics["cancellation"] == {
        "total": 1,
        "agreed": 0,
        "accuracy": 0.0,
    }
    assert label_metrics["other"] == {"total": 1, "agreed": 0, "accuracy": 0.0}
    # Unused labels have total 0, accuracy None
    assert label_metrics["refund"] == {
        "total": 0,
        "agreed": 0,
        "accuracy": None,
    }
    assert label_metrics["account"] == {
        "total": 0,
        "agreed": 0,
        "accuracy": None,
    }


def test_live_accuracy_null_when_empty(tmp_path: Path) -> None:
    """Verify live_accuracy returns None (null in JSON) when no feedback exists."""
    db_file = tmp_path / "empty_acc.db"
    store = FeedbackStore(db_file)

    store.save_prediction("T-NO-FB", "delivery", 0.9, "v1")
    total, live_acc = store.get_live_accuracy()
    assert total == 0
    assert live_acc is None


def test_zero_ticket_body_in_db(tmp_path: Path, auth_headers: dict[str, str]) -> None:
    """Strict privacy test: verify ticket body and subject are NEVER stored in the DB.

    Tests schema table definitions, row data dumps, and binary database file bytes.
    """
    db_file = tmp_path / "privacy_test.db"

    secret_body = "SENSITIVE_CUSTOMER_PAYMENT_CARD_4111222233334444_SECRET"
    secret_subject = "CONFIDENTIAL_PASSWORD_RESET_TOKEN_XYZ98765"

    with TestClient(app) as client:
        # Override DB for this client
        app.state.feedback_store = FeedbackStore(db_file)

        res = client.post(
            "/triage",
            headers=auth_headers,
            json={
                "ticket_id": "T-PRIVACY-1",
                "subject": secret_subject,
                "body": secret_body,
            },
        )
        assert res.status_code == 200

        # Submit feedback
        res_fb = client.post(
            "/feedback",
            headers=auth_headers,
            json={
                "ticket_id": "T-PRIVACY-1",
                "correct_label": "billing",
                "agent_id": "agent-auditor",
            },
        )
        assert res_fb.status_code == 200

    # 1. Inspect table schemas: no body or subject columns
    conn = sqlite3.connect(str(db_file))
    cur = conn.cursor()
    cur.execute("SELECT sql FROM sqlite_master WHERE type='table';")
    schemas = [r[0].lower() for r in cur.fetchall()]
    for s in schemas:
        assert "body" not in s
        assert "subject" not in s
        assert "secret" not in s

    # 2. Inspect all row values across both tables
    for table in ("predictions", "feedback"):
        cur.execute(f"SELECT * FROM {table};")
        rows = cur.fetchall()
        for row in rows:
            for val in row:
                val_str = str(val)
                assert secret_body not in val_str
                assert secret_subject not in val_str

    conn.close()

    # 3. Inspect raw file bytes of the database file
    db_bytes = db_file.read_bytes()
    assert secret_body.encode() not in db_bytes
    assert secret_subject.encode() not in db_bytes


def test_sql_injection_string_stored_safely(
    tmp_path: Path, auth_headers: dict[str, str]
) -> None:
    """Verify ? parameterization prevents SQL injection payloads from executing."""
    db_file = tmp_path / "sqli.db"
    store = FeedbackStore(db_file)

    sqli_ticket_1 = "T-INJECT-1'; DROP TABLE predictions; --"
    sqli_ticket_2 = "' OR '1'='1"
    sqli_agent = "agent-hack'; DELETE FROM feedback; --"

    # Save predictions with injection strings
    store.save_prediction(sqli_ticket_1, "delivery", 0.95, "v1")
    store.save_prediction(sqli_ticket_2, "billing", 0.80, "v1")

    # Verify tables still exist
    conn = sqlite3.connect(str(db_file))
    cur = conn.cursor()
    cur.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = {r[0] for r in cur.fetchall()}
    assert "predictions" in tables
    assert "feedback" in tables
    conn.close()

    # Verify predictions are retrievable by exact literal ticket ID
    pred1 = store.get_prediction(sqli_ticket_1)
    assert pred1 is not None
    assert pred1["ticket_id"] == sqli_ticket_1

    # Save feedback with injection payload
    store.save_feedback(sqli_ticket_1, "delivery", sqli_agent)
    fb1 = store.get_feedback(sqli_ticket_1)
    assert fb1 is not None
    assert fb1["ticket_id"] == sqli_ticket_1
    assert fb1["agent_id"] == sqli_agent

    # Test via API
    with TestClient(app) as client:
        app.state.feedback_store = store

        res_triage = client.post(
            "/triage",
            headers=auth_headers,
            json={
                "ticket_id": "T-API-INJECT'; DROP TABLE feedback; --",
                "body": "Need urgent help.",
            },
        )
        assert res_triage.status_code == 200

        res_fb = client.post(
            "/feedback",
            headers=auth_headers,
            json={
                "ticket_id": "T-API-INJECT'; DROP TABLE feedback; --",
                "correct_label": "account",
                "agent_id": "agent-sqli",
            },
        )
        assert res_fb.status_code == 200


def test_fail_fast_when_feedback_store_missing(auth_headers: dict[str, str]) -> None:
    """Verify /triage and /feedback fail fast with RuntimeError if feedback_store is missing."""
    with TestClient(app) as client:
        # Deliberately remove feedback_store
        app.state.feedback_store = None

        with pytest.raises(RuntimeError, match="Feedback store is not initialized"):
            client.post(
                "/triage",
                headers=auth_headers,
                json={"ticket_id": "T-FAIL-1", "body": "Order problem."},
            )

        with pytest.raises(RuntimeError, match="Feedback store is not initialized"):
            client.post(
                "/feedback",
                headers=auth_headers,
                json={
                    "ticket_id": "T-FAIL-1",
                    "correct_label": "delivery",
                    "agent_id": "agent-1",
                },
            )


def test_triage_answers_when_db_write_fails(auth_headers: dict[str, str]) -> None:
    """Verify triage still returns predictions if DB write encounters an operational error."""
    with TestClient(app) as client:
        mock_store = MagicMock()
        mock_store.save_prediction.side_effect = sqlite3.OperationalError(
            "database is locked"
        )
        app.state.feedback_store = mock_store

        res = client.post(
            "/triage",
            headers=auth_headers,
            json={"ticket_id": "T-RECOVERY-1", "body": "Where is my parcel?"},
        )
        # Triage succeeds despite DB error (Sana's design rule)
        assert res.status_code == 200
        data = res.json()
        assert data["ticket_id"] == "T-RECOVERY-1"
        assert data["label"] == "delivery"


def test_app_state_has_only_feedback_store_no_aliases() -> None:
    """Verify lifespan maintains clean app.state: only feedback_store, no db connection or aliases."""
    with TestClient(app):
        assert hasattr(app.state, "feedback_store")
        assert app.state.feedback_store is not None
        assert isinstance(app.state.feedback_store, FeedbackStore)

        # Aliases and connections must NOT exist
        assert not hasattr(app.state, "db") or app.state.db is None
        assert not hasattr(app.state, "feedback_db") or app.state.feedback_db is None
        assert not hasattr(app.state, "db_path") or app.state.db_path is None
        assert (
            not hasattr(app.state, "feedback_db_path")
            or app.state.feedback_db_path is None
        )


def test_export_feedback_cli(tmp_path: Path) -> None:
    """Verify CLI export command writes ticket_id and correct_label only."""
    db_file = tmp_path / "cli_test.db"
    store = FeedbackStore(db_file)
    store.save_prediction("T-EXP-1", "delivery", 0.9, "v1")
    store.save_prediction("T-EXP-2", "billing", 0.8, "v1")
    store.save_feedback("T-EXP-1", "delivery", "agent-1")
    store.save_feedback("T-EXP-2", "refund", "agent-2")

    export_file = tmp_path / "export.jsonl"
    argv = [
        "__main__.py",
        "export",
        "--output",
        str(export_file),
        "--db-path",
        str(db_file),
    ]

    orig_argv = sys.argv
    try:
        sys.argv = argv
        exit_code = cli_main()
        assert exit_code == 0
    finally:
        sys.argv = orig_argv

    assert export_file.exists()
    lines = [line.strip() for line in export_file.read_text().splitlines() if line]
    assert len(lines) == 2

    rec1 = json.loads(lines[0])
    assert rec1 == {"ticket_id": "T-EXP-1", "correct_label": "delivery"}
    rec2 = json.loads(lines[1])
    assert rec2 == {"ticket_id": "T-EXP-2", "correct_label": "refund"}
