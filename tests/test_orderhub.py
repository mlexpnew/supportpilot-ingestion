"""Tests for the OrderHub client."""

from __future__ import annotations

import json
import threading
import time
from http.client import BadStatusLine, IncompleteRead, LineTooLong
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

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


def test_redirect_handler_refuses_redirect():
    """_NoRedirectHandler returns None to prevent forwarding requests or headers."""
    handler = orderhub._NoRedirectHandler()
    req = orderhub.Request("http://127.0.0.1:8099/orders/55231")
    assert (
        handler.redirect_request(req, None, 302, "Found", {}, "https://evil.com")
        is None
    )


@pytest.mark.parametrize("code", [301, 302, 307, 308])
def test_3xx_redirect_codes_raise_service_unavailable(env, code):
    """3xx redirects end as ServiceUnavailable with exactly one request."""
    with patch(
        "supportpilot.integrations.orderhub._request_once",
        return_value=(code, b"", None),
    ) as request:
        with pytest.raises(orderhub.ServiceUnavailable):
            orderhub.get_order_status("55231")

    assert request.call_count == 1


def test_http_non_localhost_base_url_rejected(monkeypatch):
    """Non-localhost base URLs must use HTTPS."""
    monkeypatch.setenv("ORDERHUB_BASE_URL", "http://api.orderhub.example.com")
    monkeypatch.setenv("ORDERHUB_API_KEY", "dev-key")

    with patch("supportpilot.integrations.orderhub._request_once") as request:
        with pytest.raises(orderhub.ConfigurationError):
            orderhub.get_order_status("55231")

    request.assert_not_called()


def test_body_read_respects_deadline_during_drip():
    """A response dripping one byte at a time raises deadline error before finishing."""
    clock = 100.0

    def fake_monotonic():
        return clock

    class SlowStream:
        def read(self, n):
            nonlocal clock
            clock += 0.5
            return b"x"

    class FakeResponse:
        def __init__(self, stream):
            self.status = 200
            self.headers = {}
            self.stream = stream

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def read(self, n):
            return self.stream.read(n)

    fake_response = FakeResponse(SlowStream())

    with patch("time.monotonic", side_effect=fake_monotonic), patch(
        "supportpilot.integrations.orderhub._OPENER.open",
        return_value=fake_response,
    ):
        with pytest.raises(orderhub.ServiceUnavailable) as exc_info:
            orderhub._request_once(
                "http://127.0.0.1:8099/orders/55231",
                "dev-key",
                3.0,
                deadline=102.0,
            )

    assert "deadline" in str(exc_info.value).lower()


def test_body_size_limit_exact_and_overflow():
    """A body of exactly the limit passes, and limit + 1 fails without reading further."""
    import io

    limit = orderhub.MAX_BODY_BYTES

    # Exactly limit passes
    stream_exact = io.BytesIO(b"a" * limit)
    result = orderhub._read_body_bounded(stream_exact, deadline=float("inf"))
    assert len(result) == limit

    # Limit + 1 fails without reading further
    reads_called = 0

    class OverflowStream:
        def __init__(self):
            self.data = io.BytesIO(b"a" * (limit + 500))

        def read(self, n):
            nonlocal reads_called
            reads_called += 1
            return self.data.read(n)

    stream_over = OverflowStream()
    with pytest.raises(orderhub.ServiceUnavailable) as exc_info:
        orderhub._read_body_bounded(stream_over, deadline=float("inf"))

    assert "size limit" in str(exc_info.value).lower()
    # It must stop reading as soon as total exceeds limit
    assert stream_over.data.tell() <= limit + 1


def test_error_response_never_reads_unbounded_body():
    """Error responses do not read an unbounded body and close the connection."""
    read_called = False
    close_called = False

    class UnboundedStream:
        def read(self, *args):
            nonlocal read_called
            read_called = True
            return b"infinite"

        def close(self):
            nonlocal close_called
            close_called = True

    exc = orderhub.HTTPError(
        url="http://127.0.0.1:8099/orders/55231",
        code=500,
        msg="Internal Server Error",
        hdrs={},
        fp=UnboundedStream(),
    )

    with patch("supportpilot.integrations.orderhub._OPENER.open", side_effect=exc):
        code, body, _ = orderhub._request_once(
            "http://127.0.0.1:8099/orders/55231",
            "dev-key",
            3.0,
            deadline=float("inf"),
        )

    assert code == 500
    assert body == b""
    assert not read_called
    assert close_called


@pytest.mark.parametrize(
    "invalid_id",
    [
        "55231\n",
        "55231\r\n",
        "55231 ",
        "55231\t",
        "55231\x00",
        "５５２３１",
        "A" * 65,
    ],
)
def test_order_id_trailing_newlines_control_and_fullwidth_rejected(env, invalid_id):
    """Trailing newlines, control chars, full-width digits, and 65-char IDs are rejected pre-network."""
    with patch("supportpilot.integrations.orderhub._request_once") as request:
        with pytest.raises(ValueError):
            orderhub.get_order_status(invalid_id)

    request.assert_not_called()


def test_order_id_64_characters_passes(env):
    """A valid 64-character order ID passes validation and reaches the network."""
    valid_id = "A" * 64
    fake_order = _order_response(valid_id)
    with patch(
        "supportpilot.integrations.orderhub._request_once",
        return_value=(200, json.dumps(fake_order).encode("utf-8"), None),
    ) as request:
        result = orderhub.get_order_status(valid_id)

    assert result.order_id == valid_id
    assert request.call_count == 1


@pytest.mark.parametrize(
    "payload",
    [
        {
            "order_id": "55231",
            "status": "shipped",
            "carrier": 12345,
            "eta": None,
        },
        {
            "order_id": "55231",
            "status": ["shipped"],
            "carrier": "BlueDart",
            "eta": None,
        },
    ],
)
def test_response_wrong_types_for_carrier_and_status(env, payload):
    """Wrong types like integer carrier or list status end in ServiceUnavailable."""
    with patch(
        "supportpilot.integrations.orderhub._request_once",
        return_value=(200, json.dumps(payload).encode("utf-8"), None),
    ):
        with pytest.raises(orderhub.ServiceUnavailable) as exc_info:
            orderhub.get_order_status("55231")
        assert str(payload["status"]) not in str(exc_info.value)
        assert str(payload["carrier"]) not in str(exc_info.value)


@pytest.mark.parametrize("field", ["status", "carrier", "order_id"])
def test_response_field_string_5000_chars_rejected(env, field):
    """Fields with 5,000-character strings end in ServiceUnavailable without leaking text."""
    long_str = "A" * 5000
    payload = _order_response("55231")
    payload[field] = long_str
    with patch(
        "supportpilot.integrations.orderhub._request_once",
        return_value=(200, json.dumps(payload).encode("utf-8"), None),
    ):
        with pytest.raises(orderhub.ServiceUnavailable) as exc_info:
            orderhub.get_order_status("55231")
        assert "AAAAA" not in str(exc_info.value)


@pytest.mark.parametrize(
    "bad_eta",
    ["tomorrow", "2025-13-45", "", "2025-02-30", "२०२५-01-17"],
)
def test_response_invalid_eta_rejected(env, bad_eta):
    """Non-ISO, out-of-range calendar, or non-ASCII ETAs end in ServiceUnavailable."""
    payload = _order_response("55231")
    payload["eta"] = bad_eta
    with patch(
        "supportpilot.integrations.orderhub._request_once",
        return_value=(200, json.dumps(payload).encode("utf-8"), None),
    ):
        with pytest.raises(orderhub.ServiceUnavailable) as exc_info:
            orderhub.get_order_status("55231")
        if bad_eta:
            assert bad_eta not in str(exc_info.value)


@pytest.mark.parametrize(
    "status,carrier",
    [
        ("out_for_delivery", "Blue Dart Express"),
        ("in_transit", "System Logistics"),
        ("delivered", "DTDC"),
    ],
)
def test_response_valid_status_and_carrier_pass(env, status, carrier):
    """Valid tokens like out_for_delivery and multi-word carriers like Blue Dart Express pass."""
    payload = _order_response("55231")
    payload["status"] = status
    payload["carrier"] = carrier
    with patch(
        "supportpilot.integrations.orderhub._request_once",
        return_value=(200, json.dumps(payload).encode("utf-8"), None),
    ):
        result = orderhub.get_order_status("55231")

    assert result.status == status
    assert result.carrier == carrier


@pytest.mark.parametrize("field", ["status", "carrier"])
def test_response_empty_string_fields_rejected(env, field):
    """Empty string status or carrier ends in ServiceUnavailable."""
    payload = _order_response("55231")
    payload[field] = ""
    with patch(
        "supportpilot.integrations.orderhub._request_once",
        return_value=(200, json.dumps(payload).encode("utf-8"), None),
    ):
        with pytest.raises(orderhub.ServiceUnavailable):
            orderhub.get_order_status("55231")


def test_incomplete_read_raises_service_unavailable(env):
    """Truncated HTTP body raising IncompleteRead ends in ServiceUnavailable."""
    fake_stream = MagicMock()
    fake_stream.status = 200
    fake_stream.headers = {}
    fake_stream.read.side_effect = IncompleteRead(b"partial", expected=100)
    fake_stream.__enter__.return_value = fake_stream

    with patch(
        "supportpilot.integrations.orderhub._OPENER.open",
        return_value=fake_stream,
    ):
        with pytest.raises(orderhub.ServiceUnavailable) as exc_info:
            orderhub.get_order_status("55231")

    assert "incomplete" in str(exc_info.value).lower()


@pytest.mark.parametrize(
    "http_exc",
    [
        BadStatusLine("garbage"),
        LineTooLong("header line too long"),
    ],
)
def test_bad_status_line_and_line_too_long_raise_service_unavailable(env, http_exc):
    """Garbage status line or oversized header from opener raises ServiceUnavailable."""
    with patch(
        "supportpilot.integrations.orderhub._OPENER.open",
        side_effect=http_exc,
    ) as mock_open:
        with pytest.raises(orderhub.ServiceUnavailable):
            orderhub.get_order_status("55231")

    mock_open.assert_called_once()


def test_deeply_nested_json_recursion_error_raises_service_unavailable(env):
    """Deeply nested JSON triggering real RecursionError without mock ends in ServiceUnavailable."""
    # 20,000 bytes of nested brackets, well within 64 KB MAX_BODY_BYTES
    deeply_nested = b"[" * 10000 + b"]" * 10000
    with patch(
        "supportpilot.integrations.orderhub._request_once",
        return_value=(200, deeply_nested, None),
    ):
        with pytest.raises(orderhub.ServiceUnavailable) as exc_info:
            orderhub.get_order_status("55231")

    assert "invalid json" in str(exc_info.value).lower()


@pytest.mark.parametrize(
    "bad_url",
    [
        "http://[invalid-ipv6",
        "http://localhost:99999999",
        "http://localhost:notaport",
    ],
)
def test_malformed_base_url_raises_configuration_error(monkeypatch, bad_url):
    """Malformed ORDERHUB_BASE_URL raising ValueError ends in ConfigurationError."""
    monkeypatch.setenv("ORDERHUB_BASE_URL", bad_url)
    monkeypatch.setenv("ORDERHUB_API_KEY", "dev-key")

    with patch("supportpilot.integrations.orderhub._request_once") as request:
        with pytest.raises(orderhub.ConfigurationError):
            orderhub.get_order_status("55231")

    request.assert_not_called()


def test_missing_api_key_raises_configuration_error(monkeypatch):
    """Missing ORDERHUB_API_KEY raises ConfigurationError before network access."""
    monkeypatch.setenv("ORDERHUB_BASE_URL", "http://127.0.0.1:8099")
    monkeypatch.delenv("ORDERHUB_API_KEY", raising=False)

    with patch("supportpilot.integrations.orderhub._request_once") as request:
        with pytest.raises(orderhub.ConfigurationError):
            orderhub.get_order_status("55231")

    request.assert_not_called()


def test_missing_base_url_raises_configuration_error(monkeypatch):
    """Missing ORDERHUB_BASE_URL raises ConfigurationError before network access."""
    monkeypatch.delenv("ORDERHUB_BASE_URL", raising=False)
    monkeypatch.setenv("ORDERHUB_API_KEY", "dev-key")

    with patch("supportpilot.integrations.orderhub._request_once") as request:
        with pytest.raises(orderhub.ConfigurationError):
            orderhub.get_order_status("55231")

    request.assert_not_called()


@pytest.mark.parametrize("bad_retry_after", ["nan", "inf", "-1"])
def test_retry_after_non_finite_or_negative_falls_back_to_backoff(env, bad_retry_after):
    """Non-finite or negative Retry-After headers fall back to standard backoff."""
    responses = [
        (429, b'{"error":"rate_limited"}', bad_retry_after),
        (200, json.dumps(_order_response("55234")).encode("utf-8"), None),
    ]

    with patch(
        "supportpilot.integrations.orderhub._request_once",
        side_effect=responses,
    ), patch("supportpilot.integrations.orderhub._sleep") as sleep:
        res = orderhub.get_order_status("55234")

    assert res.order_id == "55234"
    sleep.assert_called_once()
    assert 0.0 < sleep.call_args.args[0] <= 0.25


def test_total_elapsed_deadline_with_fake_clock(env):
    """Retry loop aborts when fake clock advances past total deadline."""
    clock = 100.0

    def fake_monotonic():
        return clock

    def fake_sleep(duration):
        nonlocal clock
        clock += duration

    with patch("time.monotonic", side_effect=fake_monotonic), patch(
        "time.sleep", side_effect=fake_sleep
    ), patch(
        "supportpilot.integrations.orderhub._request_once",
        side_effect=orderhub.URLError("transient error"),
    ):
        with pytest.raises(orderhub.ServiceUnavailable) as exc_info:
            orderhub.get_order_status("55231", deadline_seconds=0.15)

    assert "deadline" in str(exc_info.value).lower()


@pytest.fixture
def fake_orderhub_server():
    """Run an ephemeral fake OrderHub server for live protocol tests."""
    from http.server import ThreadingHTTPServer

    from tools.fake_orderhub import Handler

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{port}"
    server.shutdown()


def test_live_fake_server_redirect_route(fake_orderhub_server, monkeypatch):
    """A 302 redirect on fake server raises ServiceUnavailable with key never sent to 2nd host."""
    from tools.fake_orderhub import HITS

    monkeypatch.setenv("ORDERHUB_BASE_URL", fake_orderhub_server)
    monkeypatch.setenv("ORDERHUB_API_KEY", "dev-key")
    HITS.clear()

    with pytest.raises(orderhub.ServiceUnavailable):
        orderhub.get_order_status("redirect")

    assert HITS.get("leak", 0) == 0
    assert HITS.get("redirect", 0) == 1


def test_live_fake_server_drip_route(fake_orderhub_server, monkeypatch):
    """Slow drip on fake server gives up near the configured deadline."""
    monkeypatch.setenv("ORDERHUB_BASE_URL", fake_orderhub_server)
    monkeypatch.setenv("ORDERHUB_API_KEY", "dev-key")

    t0 = time.perf_counter()
    with pytest.raises(orderhub.ServiceUnavailable) as exc_info:
        orderhub.get_order_status("drip", deadline_seconds=0.5)
    elapsed = time.perf_counter() - t0

    assert "deadline" in str(exc_info.value).lower()
    assert 0.45 <= elapsed <= 1.5
