"""Tests for SP-112 Part A: Structured JSON logging, LOG_LEVEL parsing, and privacy filtering."""

from __future__ import annotations

import io
import json
import logging
import sys
from datetime import datetime
from typing import Any

import pytest
from fastapi.testclient import TestClient

from supportpilot.api.main import app
from supportpilot.logging_config import (
    JSONFormatter,
    SensitiveDataFilter,
    get_log_level,
    setup_logging,
)


@pytest.fixture(autouse=True)
def reset_logging_environment(monkeypatch: pytest.MonkeyPatch) -> Any:
    """Ensure logging stream and LOG_LEVEL are cleanly reset between tests."""
    yield
    monkeypatch.delenv("LOG_LEVEL", raising=False)
    setup_logging(stream=sys.stdout)


# =====================================================================
# 1. JSON Formatter Tests (valid JSON, required fields, extra fields)
# =====================================================================


def test_json_formatter_produces_valid_json_with_required_fields() -> None:
    """Verify JSONFormatter emits a valid JSON string with required fields on every line."""
    formatter = JSONFormatter()
    record = logging.LogRecord(
        name="supportpilot.test",
        level=logging.INFO,
        pathname="test.py",
        lineno=10,
        msg="Test message with parameter: %s",
        args=("sample_param",),
        exc_info=None,
    )

    formatted = formatter.format(record)
    assert "\n" not in formatted.strip()

    data = json.loads(formatted)
    assert isinstance(data, dict)

    # Required fields on every line: timestamp (UTC), level, logger, message
    assert "timestamp" in data
    assert "level" in data
    assert "logger" in data
    assert "message" in data

    # Verify field values
    assert data["level"] == "INFO"
    assert data["logger"] == "supportpilot.test"
    assert data["message"] == "Test message with parameter: sample_param"

    # Verify timestamp is valid ISO 8601 UTC timestamp
    dt = datetime.fromisoformat(data["timestamp"])
    assert dt.tzinfo is not None


def test_json_formatter_includes_arbitrary_extra_fields() -> None:
    """Verify JSONFormatter includes arbitrary extra fields in JSON output."""
    formatter = JSONFormatter()
    record = logging.LogRecord(
        name="supportpilot.api.main",
        level=logging.WARNING,
        pathname="main.py",
        lineno=50,
        msg="Operational warning",
        args=(),
        exc_info=None,
    )
    record.request_id = "req-uuid-12345"
    record.path = "/triage"
    record.method = "POST"
    record.status = 400
    record.latency_ms = 12.34
    record.custom_metric = "custom_value"

    formatted = formatter.format(record)
    data = json.loads(formatted)

    assert data["request_id"] == "req-uuid-12345"
    assert data["path"] == "/triage"
    assert data["method"] == "POST"
    assert data["status"] == 400
    assert data["latency_ms"] == 12.34
    assert data["custom_metric"] == "custom_value"


def test_json_formatter_serializes_exceptions_safely() -> None:
    """Verify JSONFormatter captures and formats exception stack traces safely."""
    formatter = JSONFormatter()
    try:
        raise ValueError("Simulated operational failure")
    except ValueError:
        import sys

        exc_info = sys.exc_info()

    record = logging.LogRecord(
        name="supportpilot.test",
        level=logging.ERROR,
        pathname="test.py",
        lineno=80,
        msg="Exception occurred",
        args=(),
        exc_info=exc_info,
    )

    formatted = formatter.format(record)
    data = json.loads(formatted)

    assert "exception" in data
    assert "ValueError: Simulated operational failure" in data["exception"]


# =====================================================================
# 2. LOG_LEVEL Environment Variable Tests (default INFO, fail fast)
# =====================================================================


def test_get_log_level_default_info(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify LOG_LEVEL defaults to INFO when environment variable is unset."""
    monkeypatch.delenv("LOG_LEVEL", raising=False)
    assert get_log_level() == logging.INFO


@pytest.mark.parametrize(
    "raw_value, expected_level",
    [
        ("DEBUG", logging.DEBUG),
        ("debug", logging.DEBUG),
        ("INFO", logging.INFO),
        ("info", logging.INFO),
        ("WARNING", logging.WARNING),
        ("warning", logging.WARNING),
        ("ERROR", logging.ERROR),
        ("error", logging.ERROR),
        ("CRITICAL", logging.CRITICAL),
        ("critical", logging.CRITICAL),
    ],
)
def test_get_log_level_valid_levels(
    monkeypatch: pytest.MonkeyPatch, raw_value: str, expected_level: int
) -> None:
    """Verify get_log_level parses all standard case-insensitive log levels."""
    monkeypatch.setenv("LOG_LEVEL", raw_value)
    assert get_log_level() == expected_level


@pytest.mark.parametrize(
    "invalid_value",
    [
        "VERBOSE",
        "TRACE",
        "FOOBAR",
        "123",
        "",
        "   ",
        "NONE",
    ],
)
def test_get_log_level_invalid_value_fails_fast(
    monkeypatch: pytest.MonkeyPatch, invalid_value: str
) -> None:
    """Verify invalid LOG_LEVEL raises ValueError fail-fast."""
    monkeypatch.setenv("LOG_LEVEL", invalid_value)
    with pytest.raises(ValueError, match="Invalid LOG_LEVEL"):
        get_log_level()


# =====================================================================
# 3. Setup Lifecycle Tests (lifespan startup, not import time)
# =====================================================================


def test_setup_logging_not_called_at_import_time(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify that importing logging_config or api.main does not invoke setup_logging."""
    import importlib
    import supportpilot.logging_config

    monkeypatch.setenv("LOG_LEVEL", "INVALID_UNPARSEABLE_LEVEL")
    # Re-importing must not fail because setup_logging is not called at import time
    importlib.reload(supportpilot.logging_config)


def test_lifespan_calls_setup_and_fails_fast_on_invalid_log_level(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify FastAPI lifespan invokes setup_logging at startup and fails fast on invalid LOG_LEVEL."""
    monkeypatch.setenv("LOG_LEVEL", "INVALID_STARTUP_LEVEL")
    with pytest.raises(ValueError, match="Invalid LOG_LEVEL"):
        with TestClient(app):
            pass


# =====================================================================
# 4. Request-ID Middleware Tests (one line per request with all fields)
# =====================================================================


def test_request_id_middleware_logs_one_line_per_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify request_id middleware logs exactly one structured line per request with required fields."""
    log_stream = io.StringIO()
    monkeypatch.setenv("LOG_LEVEL", "INFO")
    setup_logging(stream=log_stream)

    custom_id = "test-req-id-structured-999"
    with TestClient(app) as client:
        log_stream.seek(0)
        log_stream.truncate(0)

        res = client.get("/health", headers={"X-Request-ID": custom_id})
        assert res.status_code == 200

    log_output = log_stream.getvalue().strip()
    lines = [ln for ln in log_output.split("\n") if ln.strip()]

    # Filter lines that match our custom request_id
    matching_lines: list[dict[str, Any]] = []
    for ln in lines:
        try:
            parsed = json.loads(ln)
            if parsed.get("request_id") == custom_id:
                matching_lines.append(parsed)
        except json.JSONDecodeError:
            pytest.fail(f"Log line is not valid JSON: {ln}")

    # Exactly ONE line per request must be logged by the middleware
    assert len(matching_lines) == 1
    req_log = matching_lines[0]

    # Required middleware fields: request_id, path, method, status, latency_ms
    assert req_log["request_id"] == custom_id
    assert req_log["path"] == "/health"
    assert req_log["method"] == "GET"
    assert req_log["status"] == 200
    assert isinstance(req_log["latency_ms"], (int, float))
    assert req_log["latency_ms"] >= 0.0

    # Every line must also have timestamp, level, logger, message
    assert "timestamp" in req_log
    assert req_log["level"] == "INFO"
    assert req_log["logger"] == "supportpilot.api.main"
    assert "message" in req_log


# =====================================================================
# 5. Sensitive Data Filter Tests (masking fake key, URL, and body)
# =====================================================================


def test_sensitive_data_filter_masks_fake_key_url_and_body() -> None:
    """Verify filter masks anything resembling X-API-Key, DATABASE_URL, or body fields."""
    fake_key = "sk-live-secret-test-key-99999"
    fake_url = "postgresql://user:super_secret_pw@pg.cluster.render.com:5432/pilot_db"
    fake_body = "CONFIDENTIAL customer ticket content with private issue details 8888"

    filter_instance = SensitiveDataFilter()
    formatter = JSONFormatter()

    # 1. Test in log message strings
    record_msg = logging.LogRecord(
        name="supportpilot.test",
        level=logging.INFO,
        pathname="test.py",
        lineno=100,
        msg="Connecting to %s with X-API-Key: %s and body: %s",
        args=(fake_url, fake_key, fake_body),
        exc_info=None,
    )

    assert filter_instance.filter(record_msg) is True
    formatted_msg = formatter.format(record_msg)
    data_msg = json.loads(formatted_msg)

    # Valid JSON & required fields present
    assert "timestamp" in data_msg
    assert "level" in data_msg
    assert "logger" in data_msg
    assert "message" in data_msg

    # Fake key, URL, password, and body must NEVER appear in captured log output
    assert fake_key not in formatted_msg
    assert fake_url not in formatted_msg
    assert "super_secret_pw" not in formatted_msg
    assert fake_body not in formatted_msg

    # 2. Test in extra attributes
    record_extra = logging.LogRecord(
        name="supportpilot.test",
        level=logging.INFO,
        pathname="test.py",
        lineno=120,
        msg="Request metadata processed",
        args=(),
        exc_info=None,
    )
    record_extra.api_key = fake_key
    record_extra.database_url = fake_url
    record_extra.body = fake_body

    assert filter_instance.filter(record_extra) is True
    formatted_extra = formatter.format(record_extra)
    data_extra = json.loads(formatted_extra)

    assert fake_key not in formatted_extra
    assert fake_url not in formatted_extra
    assert fake_body not in formatted_extra
    assert data_extra["api_key"] == "[REDACTED]"
    assert data_extra["database_url"] == "[REDACTED]"
    assert data_extra["body"] == "[REDACTED]"


def test_triage_request_never_leaks_fake_key_url_or_body_in_captured_logs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify live HTTP request with fake API key and ticket body never leaks them into captured logs."""
    fake_key = "secret-unauthorized-test-token-77777"
    fake_body = "SECRET_PAYLOAD_PRIVATE_DIAGNOSTIC_DATA_55555"
    fake_url = "postgresql://user:db_pass_xyz@db.internal:5432/live_db"

    log_stream = io.StringIO()
    monkeypatch.setenv("LOG_LEVEL", "DEBUG")
    setup_logging(stream=log_stream)

    test_logger = logging.getLogger("supportpilot.api.main")

    with TestClient(app) as client:
        # Intentionally attempt logging with simulated sensitive content
        test_logger.info(
            "DATABASE_URL: %s - Processing ticket with body: %s",
            fake_url,
            fake_body,
        )

        # Make triage call with sensitive header and body
        client.post(
            "/triage",
            headers={"X-API-Key": fake_key},
            json={"ticket_id": "T-PRIVACY-1", "body": fake_body},
        )

    captured_logs = log_stream.getvalue()

    # Never a ticket body, API key or DB URL in captured logs
    assert fake_key not in captured_logs
    assert fake_body not in captured_logs
    assert fake_url not in captured_logs
    assert "db_pass_xyz" not in captured_logs

    # Every captured line must be valid JSON
    lines = [ln for ln in captured_logs.split("\n") if ln.strip()]
    assert len(lines) > 0
    for ln in lines:
        parsed = json.loads(ln)
        assert "timestamp" in parsed
        assert "level" in parsed
        assert "logger" in parsed
        assert "message" in parsed


def test_api_key_hash_audit_is_preserved_while_raw_key_is_redacted() -> None:
    """Verify api_key_hash is preserved for audit logs while raw api_key is masked."""
    filter_instance = SensitiveDataFilter()
    formatter = JSONFormatter()

    record = logging.LogRecord(
        name="supportpilot.api.main",
        level=logging.INFO,
        pathname="main.py",
        lineno=150,
        msg="Authentication event: api_key_hash=8a7b6c5d for X-API-Key: raw-secret-key-12345",
        args=(),
        exc_info=None,
    )
    record.api_key_hash = "8a7b6c5d"
    record.x_api_key = "raw-secret-key-12345"

    assert filter_instance.filter(record) is True
    formatted = formatter.format(record)
    data = json.loads(formatted)

    assert "8a7b6c5d" in formatted
    assert data["api_key_hash"] == "8a7b6c5d"
    assert "raw-secret-key-12345" not in formatted
    assert data["x_api_key"] == "[REDACTED]"


def test_sensitive_filter_drops_records_when_drop_is_enabled() -> None:
    """Verify SensitiveDataFilter drops the record when drop=True."""
    filter_drop = SensitiveDataFilter(drop=True)

    record_clean = logging.LogRecord(
        name="test",
        level=logging.INFO,
        pathname="test.py",
        lineno=1,
        msg="Harmless operational log",
        args=(),
        exc_info=None,
    )
    record_dirty = logging.LogRecord(
        name="test",
        level=logging.INFO,
        pathname="test.py",
        lineno=2,
        msg="X-API-Key: secret-key-to-drop",
        args=(),
        exc_info=None,
    )

    assert filter_drop.filter(record_clean) is True
    assert filter_drop.filter(record_dirty) is False
