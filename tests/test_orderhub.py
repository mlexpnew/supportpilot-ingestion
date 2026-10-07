"""Tests for the OrderHub client."""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from supportpilot.integrations import orderhub


def _response(status: int, body: object, retry_after: str | None = None):
    """Build a fake urllib response."""
    payload = body if isinstance(body, bytes) else json.dumps(body).encode("utf-8")

    headers = {}
    if retry_after is not None:
        headers["Retry-After"] = retry_after

    return SimpleNamespace(
        status=status,
        headers=headers,
        read=lambda: payload,
        __enter__=lambda self: self,
        __exit__=lambda self, *args: None,
    )


def _order_response(order_id: str = "55231"):
    """Build a safe OrderHub response."""
    return {
        "order_id": order_id,
        "status": "shipped",
        "carrier": "BlueDart",
        "eta": "2025-01-17",
        "customer_email": "secret@example.com",
        "ship_to": "12 MG Road, Bengaluru",
    }


@pytest.fixture
def env(monkeypatch):
    """Provide valid OrderHub configuration."""
    monkeypatch.setenv(
        "ORDERHUB_BASE_URL",
        "http://127.0.0.1:8099",
    )
    monkeypatch.setenv("ORDERHUB_API_KEY", "dev-key")


def test_success_returns_exactly_four_fields(env):
    """Successful responses expose only the approved fields."""
    response = _response(200, _order_response())

    with patch(
        "supportpilot.integrations.orderhub._request_once",
        return_value=(response.status, response.read(), None),
    ):
        result = orderhub.get_order_status("55231")

    assert result.__dataclass_fields__.keys() == {
        "order_id",
        "status",
        "carrier",
        "eta",
    }

    assert result.order_id == "55231"
    assert result.status == "shipped"
    assert result.carrier == "BlueDart"
    assert result.eta == "2025-01-17"


def test_success_does_not_expose_pii_or_key(env):
    """PII and the API key never appear in the returned object."""
    response = _response(200, _order_response())

    with patch(
        "supportpilot.integrations.orderhub._request_once",
        return_value=(response.status, response.read(), None),
    ):
        result = orderhub.get_order_status("55231")

    rendered = repr(result)

    assert "secret@example.com" not in rendered
    assert "12 MG Road" not in rendered
    assert "dev-key" not in rendered


def test_404_is_not_retried(env):
    """404 immediately raises OrderNotFound."""
    with patch(
        "supportpilot.integrations.orderhub._request_once",
        return_value=(404, b'{"error":"not_found"}', None),
    ) as request:
        with pytest.raises(orderhub.OrderNotFound):
            orderhub.get_order_status("99999")

    assert request.call_count == 1


def test_401_is_not_retried(env):
    """401 immediately raises AuthenticationError."""
    with patch(
        "supportpilot.integrations.orderhub._request_once",
        return_value=(401, b'{"error":"unauthorized"}', None),
    ) as request:
        with pytest.raises(orderhub.AuthenticationError):
            orderhub.get_order_status("55231")

    assert request.call_count == 1


def test_500_is_retried_and_bounded(env):
    """Repeated 500 responses stop after the maximum attempts."""
    with patch(
        "supportpilot.integrations.orderhub._request_once",
        return_value=(500, b'{"error":"internal"}', None),
    ) as request, patch("supportpilot.integrations.orderhub._sleep"):
        with pytest.raises(orderhub.ServiceUnavailable):
            orderhub.get_order_status("55233")

    assert request.call_count == 3


def test_429_retries_then_succeeds(env):
    """429 responses are retried before a successful response."""
    responses = [
        (429, b'{"error":"rate_limited"}', "0"),
        (429, b'{"error":"rate_limited"}', "0"),
        (
            200,
            json.dumps(_order_response("55234")).encode("utf-8"),
            None,
        ),
    ]

    with patch(
        "supportpilot.integrations.orderhub._request_once",
        side_effect=responses,
    ) as request, patch("supportpilot.integrations.orderhub._sleep") as sleep:
        result = orderhub.get_order_status("55234")

    assert result.status == "shipped"
    assert request.call_count == 3
    assert sleep.call_count == 2


def test_retry_after_is_used(env):
    """Numeric Retry-After values are respected."""
    responses = [
        (429, b'{"error":"rate_limited"}', "0.25"),
        (
            200,
            json.dumps(_order_response("55234")).encode("utf-8"),
            None,
        ),
    ]

    with patch(
        "supportpilot.integrations.orderhub._request_once",
        side_effect=responses,
    ), patch("supportpilot.integrations.orderhub._sleep") as sleep:
        orderhub.get_order_status("55234")

    sleep.assert_called_once()
    assert sleep.call_args.args[0] == pytest.approx(0.25)


def test_retry_after_cannot_exceed_remaining_deadline(env):
    """A Retry-After larger than the remaining budget is rejected."""
    with patch(
        "supportpilot.integrations.orderhub._request_once",
        return_value=(429, b'{"error":"rate_limited"}', "30"),
    ), patch("supportpilot.integrations.orderhub._sleep") as sleep:
        with pytest.raises(orderhub.ServiceUnavailable):
            orderhub.get_order_status(
                "55234",
                deadline_seconds=0.1,
            )

    sleep.assert_not_called()


def test_invalid_json_raises_service_unavailable(env):
    """A 200 response with invalid JSON is rejected safely."""
    with patch(
        "supportpilot.integrations.orderhub._request_once",
        return_value=(200, b"<html>bad gateway</html>", None),
    ):
        with pytest.raises(orderhub.ServiceUnavailable) as exc_info:
            orderhub.get_order_status("55236")

    assert "<html>" not in str(exc_info.value)
    assert "bad gateway" not in str(exc_info.value)


def test_missing_fields_raise_service_unavailable(env):
    """A JSON response missing required fields is rejected."""
    payload = {
        "order_id": "55231",
        "status": "shipped",
    }

    with patch(
        "supportpilot.integrations.orderhub._request_once",
        return_value=(
            200,
            json.dumps(payload).encode("utf-8"),
            None,
        ),
    ):
        with pytest.raises(orderhub.ServiceUnavailable):
            orderhub.get_order_status("55231")


def test_order_id_path_injection_is_rejected_before_network(env):
    """Path-injection attempts never reach the network."""
    with patch("supportpilot.integrations.orderhub._request_once") as request:
        with pytest.raises(ValueError):
            orderhub.get_order_status("../admin")

    request.assert_not_called()


def test_order_id_query_injection_is_rejected_before_network(env):
    """Query-string injection never reaches the network."""
    with patch("supportpilot.integrations.orderhub._request_once") as request:
        with pytest.raises(ValueError):
            orderhub.get_order_status("55231?x=1")

    request.assert_not_called()


def test_empty_order_id_is_rejected_before_network(env):
    """Empty order IDs are rejected."""
    with patch("supportpilot.integrations.orderhub._request_once") as request:
        with pytest.raises(ValueError):
            orderhub.get_order_status("")

    request.assert_not_called()


def test_very_long_order_id_is_rejected_before_network(env):
    """Oversized order IDs are rejected."""
    order_id = "A" * 10_000

    with patch("supportpilot.integrations.orderhub._request_once") as request:
        with pytest.raises(ValueError):
            orderhub.get_order_status(order_id)

    request.assert_not_called()


def test_invalid_order_id_type_is_rejected_before_network(env):
    """Non-string order IDs are rejected."""
    with patch("supportpilot.integrations.orderhub._request_once") as request:
        with pytest.raises(ValueError):
            orderhub.get_order_status(55231)  # type: ignore[arg-type]

    request.assert_not_called()


def test_connection_failure_is_retried(env):
    """Connection failures are retried and eventually become unavailable."""
    with patch(
        "supportpilot.integrations.orderhub._request_once",
        side_effect=orderhub.URLError("connection failed"),
    ) as request, patch("supportpilot.integrations.orderhub._sleep"):
        with pytest.raises(orderhub.ServiceUnavailable):
            orderhub.get_order_status("55231")

    assert request.call_count == 3


def test_timeout_is_retried(env):
    """Timeouts are retried and eventually become unavailable."""
    with patch(
        "supportpilot.integrations.orderhub._request_once",
        side_effect=TimeoutError,
    ) as request, patch("supportpilot.integrations.orderhub._sleep"):
        with pytest.raises(orderhub.ServiceUnavailable):
            orderhub.get_order_status("55235")

    assert request.call_count == 3


def test_response_order_id_mismatch_is_rejected(env):
    """A response for another order is rejected."""
    payload = _order_response("99999")

    with patch(
        "supportpilot.integrations.orderhub._request_once",
        return_value=(
            200,
            json.dumps(payload).encode("utf-8"),
            None,
        ),
    ):
        with pytest.raises(orderhub.ServiceUnavailable):
            orderhub.get_order_status("55231")


def test_api_key_never_appears_in_exception(env):
    """The configured API key never leaks through exceptions."""
    with patch(
        "supportpilot.integrations.orderhub._request_once",
        return_value=(401, b'{"error":"unauthorized"}', None),
    ):
        with pytest.raises(orderhub.AuthenticationError) as exc_info:
            orderhub.get_order_status("55231")

    assert "dev-key" not in str(exc_info.value)
