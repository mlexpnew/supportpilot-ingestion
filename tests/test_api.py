"""Tests for SupportPilot Triage FastAPI service using TestClient.

Covers:
- GET /health
- POST /triage happy path (with & without subject)
- 422 Unprocessable Entity (empty body, whitespace, missing fields)
- 413 Payload Too Large (body over 10 KB)
- Email and PII never echoed back in response
- Model lifecycle (loaded once at startup)
- Fail-fast on missing model file (no silent training)
- Fail-fast on invalid TRIAGE_THRESHOLD (no silent fallback)
- Environment variable configuration (TRIAGE_MODEL_PATH, TRIAGE_THRESHOLD)
- Privacy-safe logging (ticket_id, label, latency only; never ticket body)
"""

from __future__ import annotations

import logging
from pathlib import Path
import pytest
from fastapi.testclient import TestClient

from supportpilot.api.main import MAX_BODY_BYTES, app
from supportpilot.triage import LABELS
from supportpilot.triage.model import train_model


# =====================================================================
# 1. Health Check Test
# =====================================================================


def test_health() -> None:
    with TestClient(app) as client:
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert data["model_loaded"] is True


# =====================================================================
# 2. Happy Path Tests
# =====================================================================


def test_happy_path_with_subject_and_body() -> None:
    with TestClient(app) as client:
        payload = {
            "ticket_id": "T-1001",
            "subject": "Tracking question",
            "body": "Where is my delivery package?",
        }
        response = client.post("/triage", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert set(data.keys()) == {
            "ticket_id",
            "label",
            "confidence",
            "needs_review",
        }
        assert data["ticket_id"] == "T-1001"
        assert data["label"] in LABELS
        assert isinstance(data["confidence"], float)
        assert data["confidence"] > 0.0
        assert isinstance(data["needs_review"], bool)


def test_happy_path_with_body_only() -> None:
    with TestClient(app) as client:
        payload = {
            "ticket_id": "T-1002",
            "body": "I was charged twice on my invoice",
        }
        response = client.post("/triage", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert set(data.keys()) == {
            "ticket_id",
            "label",
            "confidence",
            "needs_review",
        }
        assert data["ticket_id"] == "T-1002"
        assert data["label"] in LABELS
        assert isinstance(data["confidence"], float)
        assert isinstance(data["needs_review"], bool)


# =====================================================================
# 3. 422 Validation Error Tests (Empty body, whitespace, missing fields)
# =====================================================================


def test_triage_422_validation_errors() -> None:
    with TestClient(app) as client:
        # Empty body string -> 422
        res = client.post("/triage", json={"ticket_id": "T-1003", "body": ""})
        assert res.status_code == 422

        # Whitespace-only body string -> 422
        res = client.post("/triage", json={"ticket_id": "T-1003", "body": "   \n\t  "})
        assert res.status_code == 422

        # Missing body field -> 422
        res = client.post("/triage", json={"ticket_id": "T-1003"})
        assert res.status_code == 422

        # Missing ticket_id field -> 422
        res = client.post("/triage", json={"body": "Where is my refund?"})
        assert res.status_code == 422

        # Empty ticket_id string -> 422
        res = client.post(
            "/triage", json={"ticket_id": "", "body": "Where is my refund?"}
        )
        assert res.status_code == 422

        # Empty HTTP request payload -> 422
        res = client.post("/triage", content=b"")
        assert res.status_code == 422


# =====================================================================
# 4. 413 Payload Too Large Tests (Body over 10 KB)
# =====================================================================


def test_triage_413_body_over_10kb() -> None:
    with TestClient(app) as client:
        # Body strictly greater than 10 KB (10241 bytes) -> 413
        large_body = "x" * (MAX_BODY_BYTES + 1)
        res = client.post(
            "/triage",
            json={"ticket_id": "T-LARGE", "body": large_body},
        )
        assert res.status_code == 413
        assert "Body exceeds 10 KB limit" in res.json()["detail"]

        # Exactly 10 KB (10240 bytes) body is accepted -> 200
        exact_10kb = "x" * MAX_BODY_BYTES
        res_ok = client.post(
            "/triage",
            json={"ticket_id": "T-EXACT", "body": exact_10kb},
        )
        assert res_ok.status_code == 200


# =====================================================================
# 5. PII & Email Never Echoed Back Tests
# =====================================================================


def test_triage_email_in_body_never_echoed_back() -> None:
    with TestClient(app) as client:
        sensitive_email = "customer.confidential@secret-corp.com"
        payload = {
            "ticket_id": "T-EMAIL-LEAK-TEST",
            "subject": "Account help",
            "body": f"Please reset password for {sensitive_email} right away.",
        }
        response = client.post("/triage", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert set(data.keys()) == {
            "ticket_id",
            "label",
            "confidence",
            "needs_review",
        }
        assert data["ticket_id"] == "T-EMAIL-LEAK-TEST"
        assert sensitive_email not in response.text
        assert "customer.confidential" not in response.text


def test_triage_pii_never_echoed_back() -> None:
    with TestClient(app) as client:
        sensitive_email = "sarah.connor@cyberdyne-defense.com"
        sensitive_phone = "+1-555-019-2834"
        sensitive_card = "4111 2222 3333 4444"

        payload = {
            "ticket_id": "T-PII-ZERO-LEAK",
            "subject": f"Invoice dispute for card {sensitive_card}",
            "body": (
                f"Please contact customer at {sensitive_email} or "
                f"{sensitive_phone}. I was charged twice on my invoice."
            ),
        }

        response = client.post("/triage", json=payload)
        assert response.status_code == 200

        data = response.json()
        assert set(data.keys()) == {
            "ticket_id",
            "label",
            "confidence",
            "needs_review",
        }
        assert data["ticket_id"] == "T-PII-ZERO-LEAK"
        assert data["label"] in LABELS

        raw_text = response.text
        assert sensitive_email not in raw_text
        assert sensitive_phone not in raw_text
        assert sensitive_card not in raw_text
        assert "sarah.connor" not in raw_text
        assert "cyberdyne" not in raw_text
        assert "4111" not in raw_text


def test_triage_pipeline_redacts_pii_before_predict(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify SP-102 redaction occurs before text reaches the classifier."""
    with TestClient(app) as client:
        model = app.state.model
        assert model is not None

        captured_inputs = []
        original_predict_one = model.predict_one

        def mock_predict_one(text: str, *args, **kwargs):
            captured_inputs.append(text)
            return original_predict_one(text, *args, **kwargs)

        monkeypatch.setattr(model, "predict_one", mock_predict_one)

        payload = {
            "ticket_id": "T-PII-99",
            "subject": "Contact user at john.doe@example.com",
            "body": "Call me on +1-555-123-4567 about my lost shipment parcel",
        }
        response = client.post("/triage", json=payload)
        assert response.status_code == 200

        assert len(captured_inputs) == 1
        classified_text = captured_inputs[0]
        assert "john.doe@example.com" not in classified_text
        assert "+1-555-123-4567" not in classified_text
        assert "[EMAIL]" in classified_text
        assert "[PHONE]" in classified_text


# =====================================================================
# 6. Model Lifecycle & Fail-Fast Startup Tests
# =====================================================================


def test_model_loaded_once_at_startup_not_per_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify model is loaded once at startup and reused for all requests."""
    from supportpilot.triage.model import TriageClassifier

    load_call_count = 0
    original_load = TriageClassifier.load

    def counting_load(*args, **kwargs):
        nonlocal load_call_count
        load_call_count += 1
        return original_load(*args, **kwargs)

    monkeypatch.setattr(TriageClassifier, "load", counting_load)

    with TestClient(app) as client:
        assert load_call_count == 1

        for i in range(5):
            res = client.post(
                "/triage",
                json={
                    "ticket_id": f"T-REUSE-{i}",
                    "body": "Where is my parcel tracking update?",
                },
            )
            assert res.status_code == 200

        assert load_call_count == 1


def test_missing_model_file_fails_fast_at_startup(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Verify service fails fast at startup if model file does not exist."""
    nonexistent = tmp_path / "nonexistent_model.joblib"
    monkeypatch.setenv("TRIAGE_MODEL_PATH", str(nonexistent))
    with pytest.raises(FileNotFoundError, match="Model file not found"):
        with TestClient(app):
            pass


def test_bad_triage_threshold_fails_fast_at_startup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify service fails fast at startup on non-numeric TRIAGE_THRESHOLD."""
    monkeypatch.setenv("TRIAGE_THRESHOLD", "invalid_threshold")
    with pytest.raises(ValueError, match="Invalid TRIAGE_THRESHOLD"):
        with TestClient(app):
            pass


def test_out_of_range_triage_threshold_fails_fast(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify service fails fast at startup on out-of-range TRIAGE_THRESHOLD."""
    monkeypatch.setenv("TRIAGE_THRESHOLD", "1.5")
    with pytest.raises(ValueError, match="Invalid TRIAGE_THRESHOLD"):
        with TestClient(app):
            pass


def test_triage_model_not_loaded_error() -> None:
    """Verify HTTP 503 is returned if model is not loaded."""
    with TestClient(app) as client:
        saved_app_model = getattr(app.state, "model", None)
        try:
            app.state.model = None
            response = client.post(
                "/triage",
                json={"ticket_id": "T-1004", "body": "Where is my order?"},
            )
            assert response.status_code == 503
            assert response.json()["detail"] == "Model is not loaded"
        finally:
            app.state.model = saved_app_model


# =====================================================================
# 7. Environment Variable Configuration Tests
# =====================================================================


def test_triage_threshold_env_var(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TRIAGE_THRESHOLD", "0.9999")
    with TestClient(app) as client:
        res = client.post(
            "/triage",
            json={
                "ticket_id": "T-HIGH-TH",
                "body": "Where is my parcel tracking update?",
            },
        )
        assert res.status_code == 200
        data = res.json()
        assert data["needs_review"] is True

    monkeypatch.setenv("TRIAGE_THRESHOLD", "0.0001")
    with TestClient(app) as client:
        res = client.post(
            "/triage",
            json={
                "ticket_id": "T-LOW-TH",
                "body": "Where is my parcel tracking update?",
            },
        )
        assert res.status_code == 200
        data = res.json()
        assert data["needs_review"] is False


def test_triage_model_path_env_var(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    custom_model_file = tmp_path / "custom_triage.joblib"
    train_model("data/triage_labeled.jsonl", save_path=custom_model_file)
    assert custom_model_file.exists()

    monkeypatch.setenv("TRIAGE_MODEL_PATH", str(custom_model_file))
    from supportpilot.api.main import get_model_path

    assert get_model_path() == custom_model_file

    with TestClient(app) as client:
        res = client.get("/health")
        assert res.status_code == 200
        assert res.json()["model_loaded"] is True

        res_triage = client.post(
            "/triage",
            json={
                "ticket_id": "T-CUSTOM",
                "body": "Cancel my order before shipment",
            },
        )
        assert res_triage.status_code == 200
        assert res_triage.json()["label"] in LABELS


# =====================================================================
# 8. Privacy-Safe Logging Tests
# =====================================================================


def test_triage_logging_only_ticket_id_label_latency_and_never_body(
    caplog: pytest.LogCaptureFixture,
) -> None:
    secret_body_text = "CONFIDENTIAL_PAYLOAD_DO_NOT_LOG_THIS_BODY_12345"
    payload = {
        "ticket_id": "T-AUDIT-999",
        "subject": "Help with shipping",
        "body": f"Where is my package? {secret_body_text}",
    }

    with caplog.at_level(logging.INFO, logger="supportpilot.api.main"):
        with TestClient(app) as client:
            res = client.post("/triage", json=payload)
            assert res.status_code == 200

    api_records = [r for r in caplog.records if r.name == "supportpilot.api.main"]
    assert len(api_records) >= 1

    triage_record = api_records[-1]
    msg = triage_record.getMessage()

    assert "ticket_id=T-AUDIT-999" in msg
    assert "label=" in msg
    assert "latency=" in msg

    assert secret_body_text not in msg
    assert "CONFIDENTIAL" not in msg
    assert payload["body"] not in msg
    assert "Where is my package?" not in msg
    assert "Help with shipping" not in msg
    for record in api_records:
        assert secret_body_text not in record.getMessage()


# =====================================================================
# 9. In-Memory Counters Tests (requests, errors, latency, needs_review)
# =====================================================================


def test_counters_initial_state() -> None:
    """Verify in-memory counters start at zero upon service startup."""
    with TestClient(app) as client:
        res = client.get("/metrics")
        assert res.status_code == 200
        data = res.json()
        assert data == {
            "requests": 0,
            "errors": 0,
            "latency": 0.0,
            "needs_review": 0,
        }

        # /stats alias returns identical counters
        res_stats = client.get("/stats")
        assert res_stats.status_code == 200
        assert res_stats.json() == data

        # app.state.counters contains the same values
        assert app.state.counters["requests"] == 0
        assert app.state.counters["errors"] == 0
        assert app.state.counters["latency"] == 0.0
        assert app.state.counters["needs_review"] == 0


def test_counters_increment_requests_and_latency_on_success() -> None:
    """Verify requests and cumulative latency increment on successful triage."""
    with TestClient(app) as client:
        payload = {
            "ticket_id": "T-CTR-1",
            "body": "Where is my ordered parcel?",
        }
        res1 = client.post("/triage", json=payload)
        assert res1.status_code == 200

        data1 = client.get("/metrics").json()
        assert data1["requests"] == 1
        assert data1["errors"] == 0
        assert isinstance(data1["latency"], float)
        assert data1["latency"] > 0.0

        # Second request increments requests to 2 and accumulates latency
        res2 = client.post(
            "/triage",
            json={"ticket_id": "T-CTR-2", "body": "Need refund on invoice"},
        )
        assert res2.status_code == 200

        data2 = client.get("/metrics").json()
        assert data2["requests"] == 2
        assert data2["errors"] == 0
        assert data2["latency"] >= data1["latency"]


def test_counters_increment_needs_review(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify needs_review counter increments when prediction triggers review."""
    monkeypatch.setenv("TRIAGE_THRESHOLD", "0.9999")
    with TestClient(app) as client:
        payload = {
            "ticket_id": "T-CTR-NR",
            "body": "Ambiguous support inquiry about multiple things",
        }
        res = client.post("/triage", json=payload)
        assert res.status_code == 200
        assert res.json()["needs_review"] is True

        data = client.get("/metrics").json()
        assert data["requests"] == 1
        assert data["errors"] == 0
        assert data["needs_review"] == 1


def test_counters_increment_errors_on_422_validation() -> None:
    """Verify 422 input validation errors increment both requests and errors."""
    with TestClient(app) as client:
        res = client.post("/triage", json={"ticket_id": "T-BAD", "body": ""})
        assert res.status_code == 422

        data = client.get("/metrics").json()
        assert data["requests"] == 1
        assert data["errors"] == 1
        assert data["needs_review"] == 0


def test_counters_increment_errors_on_413_payload_too_large() -> None:
    """Verify 413 payload too large errors increment both requests and errors."""
    with TestClient(app) as client:
        res = client.post(
            "/triage",
            json={"ticket_id": "T-TOO-BIG", "body": "x" * (MAX_BODY_BYTES + 1)},
        )
        assert res.status_code == 413

        data = client.get("/metrics").json()
        assert data["requests"] == 1
        assert data["errors"] == 1


def test_counters_increment_errors_on_503_model_not_loaded() -> None:
    """Verify 503 service unavailable increments both requests and errors."""
    with TestClient(app) as client:
        saved_model = app.state.model
        try:
            app.state.model = None
            res = client.post(
                "/triage",
                json={"ticket_id": "T-UNAVAIL", "body": "Where is my item?"},
            )
            assert res.status_code == 503

            data = client.get("/metrics").json()
            assert data["requests"] == 1
            assert data["errors"] == 1
        finally:
            app.state.model = saved_model


def test_counters_excluded_paths_do_not_increment_metrics() -> None:
    """Verify health and metric probe requests do not pollute counters."""
    with TestClient(app) as client:
        for _ in range(5):
            res_health = client.get("/health")
            assert res_health.status_code == 200

        for _ in range(3):
            client.get("/metrics")
            client.get("/stats")

        # Counters should still be completely unpolluted
        data = client.get("/metrics").json()
        assert data["requests"] == 0
        assert data["errors"] == 0
        assert data["latency"] == 0.0
        assert data["needs_review"] == 0


def test_counters_exact_output_contract() -> None:
    """Verify exact 4-key output contract and types for counters."""
    with TestClient(app) as client:
        res = client.get("/metrics")
        assert res.status_code == 200
        data = res.json()
        assert set(data.keys()) == {
            "requests",
            "errors",
            "latency",
            "needs_review",
        }
        assert isinstance(data["requests"], int)
        assert isinstance(data["errors"], int)
        assert isinstance(data["latency"], float)
        assert isinstance(data["needs_review"], int)
