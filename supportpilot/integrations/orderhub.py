"""Safe OrderHub order-status client."""

from __future__ import annotations

import json
import os
import random
import re
import time
from dataclasses import dataclass
from datetime import date
from http.client import HTTPException
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import HTTPRedirectHandler, Request, build_opener


DEFAULT_DEADLINE_SECONDS = 3.0
MAX_ATTEMPTS = 3
MAX_ORDER_ID_LENGTH = 64
MAX_STATUS_LENGTH = 32
MAX_CARRIER_LENGTH = 64
BASE_BACKOFF_SECONDS = 0.1

MAX_BODY_BYTES = 65_536
ORDER_ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]+$")
STATUS_PATTERN = re.compile(r"^[A-Za-z0-9_-]+$")
CARRIER_PATTERN = re.compile(r"^[A-Za-z0-9 ._-]+$")
PROMPT_INJECTION_PATTERN = re.compile(r"(?i)\b(?:ignore|system|prompt|instruction)\b")


class OrderHubError(Exception):
    """Base exception for OrderHub failures."""


class OrderNotFound(OrderHubError):
    """Raised when the requested order does not exist."""


class AuthenticationError(OrderHubError):
    """Raised when authentication with OrderHub fails."""


class ServiceUnavailable(OrderHubError):
    """Raised when OrderHub cannot provide a valid response."""


class ConfigurationError(ServiceUnavailable):
    """Raised when OrderHub configuration is invalid or missing."""


# Backward compatibility aliases
OrderHubOrderError = OrderHubError
OrderHubTransientError = ServiceUnavailable
ServiceUnavailableOrderError = ServiceUnavailable


class _NoRedirectHandler(HTTPRedirectHandler):
    """Refuse HTTP redirects to prevent leaking authentication headers."""

    def redirect_request(
        self,
        req: Any,
        fp: Any,
        code: int,
        msg: str,
        headers: Any,
        newurl: str,
    ) -> None:
        return None


_OPENER = build_opener(_NoRedirectHandler)


@dataclass(frozen=True)
class OrderStatus:
    """Safe order-status data returned to callers."""

    order_id: str
    status: str
    carrier: str | None
    eta: str | None


def _validate_order_id(order_id: str) -> None:
    """Validate an untrusted order ID before any network access."""
    if not isinstance(order_id, str):
        raise ValueError("order_id must be a string")

    if not order_id:
        raise ValueError("order_id must not be empty")

    if len(order_id) > MAX_ORDER_ID_LENGTH:
        raise ValueError("order_id is too long")

    if ORDER_ID_PATTERN.fullmatch(order_id) is None:
        raise ValueError("invalid order_id")


def _get_config() -> tuple[str, str]:
    """Read and validate OrderHub configuration."""
    base_url = os.environ.get("ORDERHUB_BASE_URL")
    api_key = os.environ.get("ORDERHUB_API_KEY")

    if not base_url:
        raise ConfigurationError("OrderHub configuration is unavailable")

    if not api_key:
        raise AuthenticationError("OrderHub authentication is not configured")

    base_url = base_url.rstrip("/")

    if not base_url:
        raise ConfigurationError("OrderHub configuration is unavailable")

    try:
        parsed = urlparse(base_url)
        scheme = parsed.scheme
        netloc = parsed.netloc
        hostname = parsed.hostname
        _ = parsed.port
    except (ValueError, Exception) as exc:
        raise ConfigurationError("OrderHub configuration is unavailable") from exc

    if scheme not in ("http", "https") or not netloc or not hostname:
        raise ConfigurationError("OrderHub configuration is unavailable")

    if scheme == "http" and hostname not in ("127.0.0.1", "localhost"):
        raise ConfigurationError(
            "OrderHub base URL must use HTTPS for non-localhost hosts"
        )

    return base_url, api_key


def _remaining_time(deadline: float) -> float:
    """Return seconds remaining before the request deadline."""
    return max(0.0, deadline - time.monotonic())


def _backoff_seconds(attempt: int, remaining: float) -> float:
    """Calculate bounded exponential backoff with jitter."""
    if remaining <= 0:
        return 0.0

    exponential = BASE_BACKOFF_SECONDS * (2 ** (attempt - 1))
    jitter = random.uniform(0.0, exponential * 0.25)

    return min(exponential + jitter, remaining)


def _sleep(delay: float) -> None:
    """Sleep for a bounded retry delay."""
    if delay > 0:
        time.sleep(delay)


def _build_url(base_url: str, order_id: str) -> str:
    """Build the OrderHub URL from validated components."""
    return f"{base_url}/orders/{order_id}"


def _parse_response(payload: bytes, requested_order_id: str) -> OrderStatus:
    """Parse only the approved fields from an OrderHub response."""
    try:
        data: Any = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError, RecursionError) as exc:
        raise ServiceUnavailable("OrderHub returned invalid JSON") from exc

    if not isinstance(data, dict):
        raise ServiceUnavailable("OrderHub returned invalid data")

    required_fields = ("order_id", "status", "carrier", "eta")

    for field in required_fields:
        if field not in data:
            raise ServiceUnavailable("OrderHub response is missing required fields")

    response_order_id = data["order_id"]
    status = data["status"]
    carrier = data["carrier"]
    eta = data["eta"]

    if response_order_id != requested_order_id:
        raise ServiceUnavailable("OrderHub returned an unexpected order")

    if (
        not isinstance(response_order_id, str)
        or not response_order_id
        or len(response_order_id) > MAX_ORDER_ID_LENGTH
        or ORDER_ID_PATTERN.fullmatch(response_order_id) is None
    ):
        raise ServiceUnavailable("OrderHub returned invalid order data")

    if (
        not isinstance(status, str)
        or not status
        or len(status) > MAX_STATUS_LENGTH
        or STATUS_PATTERN.fullmatch(status) is None
    ):
        raise ServiceUnavailable("OrderHub returned invalid status data")

    if carrier is not None:
        if (
            not isinstance(carrier, str)
            or not carrier
            or len(carrier) > MAX_CARRIER_LENGTH
            or CARRIER_PATTERN.fullmatch(carrier) is None
            or PROMPT_INJECTION_PATTERN.search(carrier) is not None
        ):
            raise ServiceUnavailable("OrderHub returned invalid carrier data")

    if eta is not None:
        if (
            not isinstance(eta, str)
            or not eta
            or len(eta) != 10
            or not re.fullmatch(r"^\d{4}-\d{2}-\d{2}$", eta)
        ):
            raise ServiceUnavailable("OrderHub returned invalid ETA data")
        try:
            date.fromisoformat(eta)
        except ValueError:
            raise ServiceUnavailable("OrderHub returned invalid ETA data")

    return OrderStatus(
        order_id=response_order_id,
        status=status,
        carrier=carrier,
        eta=eta,
    )


def _read_body_bounded(
    stream: Any,
    deadline: float,
    max_bytes: int = MAX_BODY_BYTES,
    chunk_size: int = 4096,
) -> bytes:
    """Read stream in bounded chunks enforcing size limit and deadline."""
    chunks: list[bytes] = []
    total = 0

    while True:
        if time.monotonic() >= deadline:
            raise ServiceUnavailable("OrderHub request deadline exceeded")

        remaining_budget = max_bytes - total + 1
        to_read = min(chunk_size, remaining_budget)
        chunk = stream.read(to_read)
        if not chunk:
            break

        total += len(chunk)
        if total > max_bytes:
            raise ServiceUnavailable("OrderHub response body exceeded size limit")

        chunks.append(chunk)

    return b"".join(chunks)


def _request_once(
    url: str,
    api_key: str,
    timeout: float,
    deadline: float | None = None,
) -> tuple[int, bytes, str | None]:
    """Perform one HTTP request and return safe response metadata."""
    try:
        request = Request(
            url,
            method="GET",
            headers={"X-Api-Key": api_key},
        )
    except ValueError as exc:
        raise ConfigurationError("OrderHub configuration is unavailable") from exc

    effective_deadline = (
        deadline if deadline is not None else (time.monotonic() + timeout)
    )

    try:
        with _OPENER.open(request, timeout=timeout) as response:
            status = response.status
            retry_after = response.headers.get("Retry-After")
            body = _read_body_bounded(response, deadline=effective_deadline)
            return status, body, retry_after
    except HTTPError as exc:
        try:
            retry_after = exc.headers.get("Retry-After")
            # Do not read unbounded error bodies; response text is never exposed
            return exc.code, b"", retry_after
        finally:
            exc.close()
    except HTTPException as exc:
        raise ServiceUnavailable(
            "OrderHub response was incomplete or invalid HTTP"
        ) from exc


def get_order_status(
    order_id: str,
    *,
    deadline_seconds: float = DEFAULT_DEADLINE_SECONDS,
) -> OrderStatus:
    """Fetch an order status while enforcing security and time limits."""
    _validate_order_id(order_id)

    if deadline_seconds <= 0:
        raise ValueError("deadline_seconds must be positive")

    base_url, api_key = _get_config()
    deadline = time.monotonic() + deadline_seconds
    url = _build_url(base_url, order_id)

    for attempt in range(1, MAX_ATTEMPTS + 1):
        remaining = _remaining_time(deadline)

        if remaining <= 0:
            raise ServiceUnavailable("OrderHub request deadline exceeded")

        try:
            status_code, body, retry_after = _request_once(
                url,
                api_key,
                remaining,
                deadline=deadline,
            )

        except (TimeoutError, URLError, OSError):
            if attempt == MAX_ATTEMPTS:
                raise ServiceUnavailable("OrderHub service is unavailable") from None

            remaining = _remaining_time(deadline)

            if remaining <= 0:
                raise ServiceUnavailable("OrderHub request deadline exceeded")

            delay = _backoff_seconds(attempt, remaining)
            _sleep(delay)
            continue
        except ValueError as exc:
            raise ConfigurationError("OrderHub configuration is unavailable") from exc

        if status_code == 200:
            return _parse_response(body, order_id)

        if status_code == 404:
            raise OrderNotFound("Order not found")

        if status_code == 401:
            raise AuthenticationError("OrderHub authentication failed")

        if status_code == 429:
            if attempt == MAX_ATTEMPTS:
                raise ServiceUnavailable("OrderHub service is unavailable")

            remaining = _remaining_time(deadline)

            if remaining <= 0:
                raise ServiceUnavailable("OrderHub request deadline exceeded")

            retry_delay = _parse_retry_after(retry_after)
            if retry_delay is None:
                retry_delay = _backoff_seconds(attempt, remaining)

            if retry_delay >= remaining:
                raise ServiceUnavailable("OrderHub request deadline exceeded")

            _sleep(retry_delay)
            continue

        if 500 <= status_code <= 599:
            if attempt == MAX_ATTEMPTS:
                raise ServiceUnavailable("OrderHub service is unavailable")

            remaining = _remaining_time(deadline)

            if remaining <= 0:
                raise ServiceUnavailable("OrderHub request deadline exceeded")

            delay = _backoff_seconds(attempt, remaining)
            _sleep(delay)
            continue

        raise ServiceUnavailable("OrderHub returned an unexpected response")

    raise ServiceUnavailable("OrderHub service is unavailable")


def _parse_retry_after(value: str | None) -> float | None:
    """Parse a numeric Retry-After value without exposing its contents."""
    if value is None:
        return None

    try:
        delay = float(value)
    except ValueError:
        return None

    if delay < 0:
        return None

    return delay
