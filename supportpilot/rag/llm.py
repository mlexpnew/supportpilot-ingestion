"""LLM clients for ReplyPilot drafting (SP-124).

One small interface, three implementations, chosen by ``REPLYPILOT_LLM``:

    ollama     (default) local Ollama server, free, no key
    anthropic  Anthropic Messages API, key from ANTHROPIC_API_KEY only
    fake       scripted replies for tests: no network, no key

Environment variables:
    REPLYPILOT_LLM          ollama | anthropic | fake
    REPLYPILOT_MODEL        model override (defaults below)
    REPLYPILOT_LLM_TIMEOUT  seconds per attempt (default 60)
    OLLAMA_HOST             default http://localhost:11434
    ANTHROPIC_API_KEY       required for anthropic; never logged

Every network call gets a timeout and exactly one retry on timeouts,
connection errors, 429 and 5xx. 4xx errors (bad key, bad request) are not
retried. Logs carry provider, model, latency and token counts only: never the
API key and never the ticket text (it may contain personal data).
"""

from __future__ import annotations

import json
import logging
import os
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any, Protocol, TypeVar
from urllib.error import HTTPError, URLError
from urllib.request import HTTPRedirectHandler, Request, build_opener

logger = logging.getLogger(__name__)

DEFAULT_PROVIDER = "ollama"
DEFAULT_MODELS: dict[str, str] = {
    "ollama": "llama3.2:3b",
    "anthropic": "claude-haiku-5-5",
    "fake": "fake-llm",
}
DEFAULT_TIMEOUT_SECONDS = 60.0
DEFAULT_MAX_TOKENS = 400
DEFAULT_TEMPERATURE = 0.2
MAX_ATTEMPTS = 2  # first try + 1 retry
RETRY_BACKOFF_SECONDS = 0.5
ANTHROPIC_URL = "https://api.anthropic.com/v1/messages"
ANTHROPIC_VERSION = "2023-06-01"

T = TypeVar("T")


class LLMError(RuntimeError):
    """The LLM call failed and should not be retried (or retries ran out)."""


class LLMTimeout(LLMError):
    """The LLM did not answer within the timeout, even after the retry."""


class LLMConfigError(LLMError):
    """Missing or invalid configuration (unknown provider, missing key)."""


class _Retryable(Exception):
    """Internal marker: this attempt may be retried."""


@dataclass(frozen=True)
class LLMResponse:
    text: str
    model: str
    input_tokens: int
    output_tokens: int
    latency_ms: int


class LLMClient(Protocol):
    provider: str
    model: str

    def generate(self, system: str, user: str) -> LLMResponse: ...


# --------------------------------------------------------------------------
# Retry + HTTP helpers
# --------------------------------------------------------------------------


def call_with_retry(
    fn: Callable[[], T],
    attempts: int = MAX_ATTEMPTS,
    backoff: float = RETRY_BACKOFF_SECONDS,
    sleep: Callable[[float], None] | None = None,
) -> T:
    """Run ``fn``; retry once on timeout/transient errors.

    ``fn`` raises ``TimeoutError`` or ``_Retryable`` for retryable failures.
    After the last attempt a timeout becomes ``LLMTimeout`` and anything else
    retryable becomes ``LLMError``.
    """
    last: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            return fn()
        except (TimeoutError, _Retryable) as exc:
            last = exc
            logger.warning("llm attempt %d/%d failed: %s", attempt, attempts, exc)
            if attempt < attempts:
                (sleep or time.sleep)(backoff)
    if isinstance(last, TimeoutError):
        raise LLMTimeout(f"LLM timed out after {attempts} attempts") from last
    raise LLMError(f"LLM failed after {attempts} attempts: {last}") from last


class _NoRedirect(HTTPRedirectHandler):
    """Never follow redirects (they could leak the API key header)."""

    def redirect_request(self, *args: Any, **kwargs: Any) -> None:
        return None


_OPENER = build_opener(_NoRedirect)


def post_json(
    url: str, payload: dict[str, Any], headers: dict[str, str], timeout: float
) -> dict[str, Any]:
    """POST JSON once. Raises TimeoutError / _Retryable / LLMError."""
    body = json.dumps(payload).encode("utf-8")
    req = Request(
        url,
        data=body,
        method="POST",
        headers={"content-type": "application/json", **headers},
    )
    try:
        with _OPENER.open(req, timeout=timeout) as resp:
            data: dict[str, Any] = json.loads(resp.read().decode("utf-8"))
            return data
    except HTTPError as exc:
        if exc.code == 429 or exc.code >= 500:
            raise _Retryable(f"HTTP {exc.code}") from exc
        # 4xx: do not echo the response body, it can repeat request headers
        raise LLMError(f"LLM request rejected: HTTP {exc.code}") from exc
    except TimeoutError as exc:
        raise TimeoutError("request timed out") from exc
    except URLError as exc:
        if isinstance(exc.reason, TimeoutError):
            raise TimeoutError("request timed out") from exc
        raise _Retryable(f"connection failed: {exc.reason}") from exc
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise LLMError("LLM returned invalid JSON") from exc


def _timeout_from_env() -> float:
    raw = os.getenv("REPLYPILOT_LLM_TIMEOUT", "")
    try:
        value = float(raw) if raw.strip() else DEFAULT_TIMEOUT_SECONDS
    except ValueError as exc:
        raise LLMConfigError("REPLYPILOT_LLM_TIMEOUT must be a number") from exc
    if value <= 0:
        raise LLMConfigError("REPLYPILOT_LLM_TIMEOUT must be positive")
    return value


def _log_call(resp: LLMResponse, provider: str) -> None:
    logger.info(
        "llm call provider=%s model=%s latency_ms=%d in_tokens=%d out_tokens=%d",
        provider,
        resp.model,
        resp.latency_ms,
        resp.input_tokens,
        resp.output_tokens,
    )


# --------------------------------------------------------------------------
# Clients
# --------------------------------------------------------------------------


class AnthropicClient:
    provider = "anthropic"

    def __init__(
        self,
        model: str | None = None,
        timeout: float | None = None,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        post: Callable[..., dict[str, Any]] = post_json,
    ) -> None:
        api_key = os.getenv("ANTHROPIC_API_KEY", "").strip()
        if not api_key:
            raise LLMConfigError("ANTHROPIC_API_KEY is not set")
        self._api_key = api_key  # kept private, never logged or repr'd
        self.model = model or DEFAULT_MODELS["anthropic"]
        self.timeout = timeout or _timeout_from_env()
        self.max_tokens = max_tokens
        self._post = post

    def __repr__(self) -> str:
        return f"AnthropicClient(model={self.model!r}, api_key=***)"

    def generate(self, system: str, user: str) -> LLMResponse:
        payload = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "temperature": DEFAULT_TEMPERATURE,
            "system": system,
            "messages": [{"role": "user", "content": user}],
        }
        headers = {"x-api-key": self._api_key, "anthropic-version": ANTHROPIC_VERSION}
        start = time.perf_counter()
        data = call_with_retry(
            lambda: self._post(ANTHROPIC_URL, payload, headers, self.timeout)
        )
        try:
            text = "".join(
                b.get("text", "") for b in data["content"] if b.get("type") == "text"
            )
            usage = data.get("usage", {})
            resp = LLMResponse(
                text=text.strip(),
                model=str(data.get("model", self.model)),
                input_tokens=int(usage.get("input_tokens", 0)),
                output_tokens=int(usage.get("output_tokens", 0)),
                latency_ms=int((time.perf_counter() - start) * 1000),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise LLMError("unexpected Anthropic response shape") from exc
        _log_call(resp, self.provider)
        return resp


class OllamaClient:
    provider = "ollama"

    def __init__(
        self,
        model: str | None = None,
        host: str | None = None,
        timeout: float | None = None,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        post: Callable[..., dict[str, Any]] = post_json,
    ) -> None:
        self.model = model or DEFAULT_MODELS["ollama"]
        self.host = (
            host or os.getenv("OLLAMA_HOST") or "http://localhost:11434"
        ).rstrip("/")
        if not self.host.startswith(("http://", "https://")):
            self.host = "http://" + self.host
        self.timeout = timeout or _timeout_from_env()
        self.max_tokens = max_tokens
        self._post = post

    def __repr__(self) -> str:
        return f"OllamaClient(model={self.model!r}, host={self.host!r})"

    def generate(self, system: str, user: str) -> LLMResponse:
        payload = {
            "model": self.model,
            "stream": False,
            "options": {
                "temperature": DEFAULT_TEMPERATURE,
                "num_predict": self.max_tokens,
            },
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        }
        start = time.perf_counter()
        data = call_with_retry(
            lambda: self._post(f"{self.host}/api/chat", payload, {}, self.timeout)
        )
        try:
            resp = LLMResponse(
                text=str(data["message"]["content"]).strip(),
                model=str(data.get("model", self.model)),
                input_tokens=int(data.get("prompt_eval_count", 0)),
                output_tokens=int(data.get("eval_count", 0)),
                latency_ms=int((time.perf_counter() - start) * 1000),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise LLMError("unexpected Ollama response shape") from exc
        _log_call(resp, self.provider)
        return resp


class FakeLLMClient:
    """Deterministic client for tests. No network, no key.

    ``replies`` items are returned in order (the last one repeats). An item
    that is an Exception instance is raised instead, e.g. ``LLMTimeout()``.
    Every (system, user) pair is recorded in ``calls``.
    """

    provider = "fake"

    def __init__(
        self,
        replies: Iterable[str | Exception] = ("Thanks for reaching out.",),
        model: str = DEFAULT_MODELS["fake"],
    ) -> None:
        self._replies = list(replies) or ["Thanks for reaching out."]
        self.model = model
        self.calls: list[tuple[str, str]] = []

    def generate(self, system: str, user: str) -> LLMResponse:
        self.calls.append((system, user))
        idx = min(len(self.calls) - 1, len(self._replies) - 1)
        reply = self._replies[idx]
        if isinstance(reply, Exception):
            raise reply
        return LLMResponse(
            text=reply,
            model=self.model,
            input_tokens=len((system + user).split()),
            output_tokens=len(reply.split()),
            latency_ms=0,
        )


def get_llm_client(provider: str | None = None) -> LLMClient:
    """Build the client named by ``provider`` or ``REPLYPILOT_LLM``."""
    name = (provider or os.getenv("REPLYPILOT_LLM") or DEFAULT_PROVIDER).strip().lower()
    model = os.getenv("REPLYPILOT_MODEL", "").strip() or None
    if name == "ollama":
        return OllamaClient(model=model)
    if name == "anthropic":
        return AnthropicClient(model=model)
    if name == "fake":
        return FakeLLMClient(model=model or DEFAULT_MODELS["fake"])
    raise LLMConfigError(
        f"unknown REPLYPILOT_LLM {name!r} (use ollama, anthropic or fake)"
    )
