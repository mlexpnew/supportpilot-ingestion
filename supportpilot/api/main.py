from __future__ import annotations

import hashlib
import logging
import math
import os
import re
import secrets
import time
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import Depends, FastAPI, HTTPException, Request, Response, Security
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, Field, ValidationError, field_validator

from supportpilot.api.alerts import (
    AlertThresholds,
    AlertsResponse,
    evaluate_alerts,
    get_alert_thresholds,
)
from supportpilot.api.models import (
    BatchItemResult,
    BatchSummary,
    BatchTriageRequest,
    BatchTriageResponse,
    KeyUsageItem,
    MetricsResponse,
    VersionResponse,
)
from supportpilot.feedback import (
    MAX_ID_LENGTH,
    FeedbackRequest,
    FeedbackResponse,
    FeedbackStore,
    LabelMetric,
    normalize_ticket_id,
)
from supportpilot.logging_config import setup_logging
from supportpilot.preprocessing.redact import redact_text
from supportpilot.triage.model import (
    DEFAULT_CONFIDENCE_THRESHOLD,
    DEFAULT_MODEL_PATH,
    TriageClassifier,
    is_valid_model_version,
)

MAX_BODY_BYTES: int = 10 * 1024  # 10 KB limit
MAX_BATCH_SIZE: int = 100  # Maximum tickets allowed in a single batch request
RATE_LIMIT_WINDOW_SECONDS: float = 60.0  # 60 second sliding window
RATE_LIMIT_MAX_REQUESTS: int = 60  # Maximum requests allowed per sliding window
TICKET_RATE_LIMIT_DEFAULT: int = 1000  # Default tickets allowed per minute per key
TICKET_RATE_LIMIT_WINDOW_SECONDS: float = 60.0  # Sliding window for ticket rate limiter
logger = logging.getLogger(__name__)

REQUEST_ID_REGEX = re.compile(r"^[A-Za-z0-9-]{1,64}$")

EXCLUDED_METRIC_PATHS: set[str] = {
    "/health",
    "/version",
    "/alerts",
    "/metrics",
    "/metrics/labels",
    "/usage",
    "/docs",
    "/openapi.json",
    "/redoc",
}


@dataclass
class Counters:
    """In-memory operational counters tracking triage traffic and outcomes."""

    requests_total: int = 0
    triage_total: int = 0
    errors_total: int = 0
    total_latency_ms: float = 0.0
    needs_review_total: int = 0
    rate_limited_total: int = 0


def get_git_sha() -> str:
    """Resolve Git commit SHA: check RENDER_GIT_COMMIT first, then GIT_SHA, else 'unknown'.

    Returns the first 7 characters if known, or 'unknown'.
    """
    raw_sha = os.getenv("RENDER_GIT_COMMIT") or os.getenv("GIT_SHA") or "unknown"
    raw_sha = raw_sha.strip()
    if not raw_sha or raw_sha == "unknown":
        return "unknown"
    return raw_sha[:7]


def get_model_path() -> Path:
    return Path(os.getenv("TRIAGE_MODEL_PATH", str(DEFAULT_MODEL_PATH)))


def get_threshold() -> float:
    raw_val = os.getenv("TRIAGE_THRESHOLD")
    if raw_val is not None:
        try:
            val = float(raw_val)
        except ValueError as exc:
            raise ValueError(
                f"Invalid TRIAGE_THRESHOLD '{raw_val}': must be a valid float."
            ) from exc
        if not (0.0 <= val <= 1.0):
            raise ValueError(
                f"Invalid TRIAGE_THRESHOLD '{raw_val}': must be between 0.0 and 1.0."
            )
        return val
    return DEFAULT_CONFIDENCE_THRESHOLD


def get_api_keys() -> set[str]:
    raw_keys = os.getenv("TRIAGE_API_KEYS")
    if raw_keys is None or not raw_keys.strip():
        raise ValueError(
            "No API keys configured: TRIAGE_API_KEYS environment variable "
            "is required."
        )
    keys = {k.strip() for k in raw_keys.split(",") if k.strip()}
    if not keys:
        raise ValueError(
            "No API keys configured: TRIAGE_API_KEYS must contain at "
            "least one valid key."
        )
    return keys


def get_admin_api_keys() -> set[str]:
    raw_keys = os.getenv("ADMIN_API_KEYS")
    if raw_keys is None or not raw_keys.strip():
        raise ValueError(
            "No admin API keys configured: ADMIN_API_KEYS environment variable "
            "is required."
        )
    keys = {k.strip() for k in raw_keys.split(",") if k.strip()}
    if not keys:
        raise ValueError(
            "No admin API keys configured: ADMIN_API_KEYS must contain at "
            "least one valid key."
        )
    return keys


def validate_key_separation(client_keys: set[str], admin_keys: set[str]) -> None:
    """Ensure client keys and admin keys never overlap (least privilege separation)."""
    overlap = client_keys & admin_keys
    if overlap:
        raise ValueError(
            "Security violation: ADMIN_API_KEYS and TRIAGE_API_KEYS must not overlap. "
            f"Overlapping keys detected: {len(overlap)}"
        )


def validate_api_key(
    api_key: str | None,
    valid_keys: set[str] | None,
) -> bool:
    """Compare with secrets.compare_digest against each stored key (constant time)."""
    if not valid_keys:
        return False
    candidate = api_key if isinstance(api_key, str) else ""
    candidate_bytes = candidate.encode("utf-8")
    matched = False
    for expected_key in valid_keys:
        expected_bytes = expected_key.encode("utf-8")
        if secrets.compare_digest(candidate_bytes, expected_bytes):
            matched = True
    return matched and bool(api_key)


def hash_api_key(api_key: str | None) -> str:
    """Return a short SHA-256 hash (first 8 chars) of the API key, or 'none'."""
    if not api_key:
        return "none"
    return hashlib.sha256(api_key.encode("utf-8")).hexdigest()[:8]


def prune_old_timestamps(
    timestamps: list[float],
    now: float,
    window_seconds: float = RATE_LIMIT_WINDOW_SECONDS,
) -> list[float]:
    """Drop timestamps older than window_seconds (60s)."""
    cutoff = now - window_seconds
    return [ts for ts in timestamps if ts > cutoff]


api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def get_api_key(
    request: Request,
    api_key: str | None = Security(api_key_header),
) -> str:
    """FastAPI dependency that reads and validates the X-API-Key header.

    Authenticates via (SP-118 Part C):
    1. Hashes incoming key.
    2. Looks the hash up in the DB (api_keys table), ignoring revoked keys.
    3. Falls back to env TRIAGE_API_KEYS if the DB lookup fails.
    4. Keeps constant-time comparison for env keys.
    """
    key_hash = hash_api_key(api_key)
    request.state.api_key_hash = key_hash

    is_valid = False
    feedback_store: Any = getattr(request.app.state, "feedback_store", None)

    if api_key and isinstance(api_key, str) and api_key.strip():
        # 1. Hash incoming key (full SHA-256)
        full_hash = hashlib.sha256(api_key.encode("utf-8")).hexdigest()

        # 2. Look up hash in DB, ignoring revoked keys
        db_key: dict[str, Any] | None = None
        if feedback_store is not None and hasattr(feedback_store, "get_key_by_hash"):
            try:
                db_key = feedback_store.get_key_by_hash(full_hash)
                if db_key is None:
                    db_key = feedback_store.get_key_by_hash(key_hash)
            except Exception:
                logger.exception(
                    "Error querying api_keys table for key_hash=%s",
                    key_hash,
                    extra={"api_key_hash": key_hash},
                )

        if db_key is not None:
            # Found in DB: check if active (ignoring revoked keys)
            if not db_key.get("revoked_at"):
                is_valid = True
                request.state.api_key_role = db_key.get("role", "client")
                request.state.api_key_name = db_key.get("name")
                try:
                    if hasattr(feedback_store, "update_key_last_used"):
                        feedback_store.update_key_last_used(db_key["key_hash"])
                except Exception:
                    pass
        else:
            # 3. Fall back to env TRIAGE_API_KEYS if DB lookup fails
            # 4. Keeps constant-time compare for env keys
            valid_keys: set[str] = getattr(request.app.state, "api_keys", set())
            is_valid = validate_api_key(api_key, valid_keys)
            if is_valid:
                request.state.api_key_role = "client"

    # Usage tracking (SP-115 Part A & Sana table protection fix)
    try:
        if feedback_store is not None and hasattr(feedback_store, "record_key_usage"):
            tracked_hash = key_hash if is_valid else "unknown"
            feedback_store.record_key_usage(key_hash=tracked_hash, ok=is_valid)
    except Exception:
        logger.exception(
            "Failed to record API key usage for key_hash=%s",
            key_hash,
            extra={"api_key_hash": key_hash},
        )

    if not is_valid:
        logger.warning(
            "api_key_hash=%s authentication failed",
            key_hash,
            extra={"api_key_hash": key_hash},
        )
        raise HTTPException(
            status_code=401,
            detail="Unauthorized",
        )
    logger.info(
        "api_key_hash=%s authenticated",
        key_hash,
        extra={"api_key_hash": key_hash},
    )
    return api_key  # type: ignore[return-value]


def require_admin_key(
    request: Request,
    api_key: str | None = Security(api_key_header),
) -> str:
    """FastAPI dependency requiring a valid admin API key from DB or ADMIN_API_KEYS (SP-118).

    Guarantees:
    - Checks DB on every request; no startup caching.
    - Revoked keys receive 401 immediately.
    - Valid client keys attempting admin endpoints receive 403 Forbidden.
    - Unauthenticated/unknown keys receive 401 Unauthorized.
    """
    key_hash = hash_api_key(api_key)
    request.state.api_key_hash = key_hash

    is_admin = False
    is_client = False
    feedback_store: Any = getattr(request.app.state, "feedback_store", None)

    if api_key and isinstance(api_key, str) and api_key.strip():
        full_hash = hashlib.sha256(api_key.encode("utf-8")).hexdigest()

        # 1. Check DB on every request (no startup caching)
        db_key: dict[str, Any] | None = None
        if feedback_store is not None and hasattr(feedback_store, "get_key_by_hash"):
            try:
                db_key = feedback_store.get_key_by_hash(full_hash)
                if db_key is None:
                    db_key = feedback_store.get_key_by_hash(key_hash)
            except Exception:
                logger.exception(
                    "Error querying api_keys table for admin key_hash=%s",
                    key_hash,
                    extra={"api_key_hash": key_hash},
                )

        if db_key is not None:
            # Check revocation
            if not db_key.get("revoked_at"):
                role = db_key.get("role", "client")
                if role == "admin":
                    is_admin = True
                    request.state.api_key_role = "admin"
                    request.state.api_key_name = db_key.get("name")
                    try:
                        if hasattr(feedback_store, "update_key_last_used"):
                            feedback_store.update_key_last_used(db_key["key_hash"])
                    except Exception:
                        pass
                else:
                    is_client = True
                    request.state.api_key_role = "client"
            # If revoked, neither is_admin nor is_client is True -> will 401
        else:
            # 2. Fall back to environment keys with constant-time compare
            valid_admin_keys: set[str] = getattr(
                request.app.state, "admin_api_keys", set()
            )
            if validate_api_key(api_key, valid_admin_keys):
                is_admin = True
                request.state.api_key_role = "admin"
            else:
                valid_client_keys: set[str] = getattr(
                    request.app.state, "api_keys", set()
                )
                if validate_api_key(api_key, valid_client_keys):
                    is_client = True
                    request.state.api_key_role = "client"

    # Usage tracking
    try:
        if feedback_store is not None and hasattr(feedback_store, "record_key_usage"):
            tracked_hash = key_hash if (is_admin or is_client) else "unknown"
            feedback_store.record_key_usage(
                key_hash=tracked_hash, ok=(is_admin or is_client)
            )
    except Exception:
        logger.exception(
            "Failed to record API key usage for admin key_hash=%s",
            key_hash,
            extra={"api_key_hash": key_hash},
        )

    # Client keys are recognized but forbidden on /admin/* endpoints (403)
    if is_client and not is_admin:
        if request.url.path.startswith("/admin"):
            logger.warning(
                "api_key_hash=%s client key forbidden on admin endpoint",
                key_hash,
                extra={"api_key_hash": key_hash},
            )
            raise HTTPException(
                status_code=403,
                detail="Forbidden: client key cannot access admin endpoints",
            )
        raise HTTPException(
            status_code=401,
            detail="Unauthorized",
        )

    if not is_admin:
        logger.warning(
            "api_key_hash=%s admin authentication failed",
            key_hash,
            extra={"api_key_hash": key_hash},
        )
        raise HTTPException(
            status_code=401,
            detail="Unauthorized",
        )

    logger.info(
        "api_key_hash=%s admin authenticated",
        key_hash,
        extra={"api_key_hash": key_hash},
    )
    return api_key  # type: ignore[return-value]


def rate_limit(
    request: Request,
    api_key: str = Depends(get_api_key),
) -> None:
    """FastAPI dependency enforcing per-key sliding window rate limiting.

    Runs after authentication (only valid keys are counted).
    Note: In-memory limits only work for a single instance.
    """
    key_hash = getattr(request.state, "api_key_hash", None) or hash_api_key(api_key)
    rate_limits: dict[str, list[float]] | None = getattr(
        request.app.state, "rate_limits", None
    )
    if rate_limits is None:
        rate_limits = {}
        request.app.state.rate_limits = rate_limits

    now = time.monotonic()
    # On each request, drop timestamps older than 60 seconds
    active_timestamps = prune_old_timestamps(
        rate_limits.get(key_hash, []),
        now=now,
        window_seconds=RATE_LIMIT_WINDOW_SECONDS,
    )

    max_requests = getattr(
        request.app.state, "rate_limit_max_requests", RATE_LIMIT_MAX_REQUESTS
    )
    if len(active_timestamps) >= max_requests:
        oldest = active_timestamps[0]
        retry_after = max(1, math.ceil((oldest + RATE_LIMIT_WINDOW_SECONDS) - now))
        rate_limits[key_hash] = active_timestamps
        counters: Counters | None = getattr(request.app.state, "counters", None)
        if counters is not None:
            counters.rate_limited_total += 1
        logger.warning(
            "api_key_hash=%s rate limit exceeded retry_after=%ds",
            key_hash,
            retry_after,
            extra={"api_key_hash": key_hash, "retry_after": retry_after},
        )
        raise HTTPException(
            status_code=429,
            detail="Too Many Requests",
            headers={"Retry-After": str(retry_after)},
        )

    # Otherwise add the current monotonic timestamp and continue
    active_timestamps.append(now)
    rate_limits[key_hash] = active_timestamps


def get_ticket_rate_limit(app: FastAPI | None = None) -> int:
    """Return maximum allowed tickets per minute per key."""
    if app is not None and hasattr(app.state, "ticket_rate_limit_per_minute"):
        return int(app.state.ticket_rate_limit_per_minute)
    env_val = os.getenv("TICKET_RATE_LIMIT_PER_MINUTE")
    if env_val:
        try:
            return int(env_val)
        except ValueError:
            pass
    return TICKET_RATE_LIMIT_DEFAULT


def check_ticket_rate_limit(
    request: Request,
    key_hash: str,
    ticket_count: int,
) -> None:
    """Enforce per-key sliding window rate limit on number of tickets processed per minute."""
    if ticket_count <= 0:
        return

    ticket_rate_limits: dict[str, list[tuple[float, int]]] | None = getattr(
        request.app.state, "ticket_rate_limits", None
    )
    if ticket_rate_limits is None:
        ticket_rate_limits = {}
        request.app.state.ticket_rate_limits = ticket_rate_limits

    now = time.monotonic()
    entries = ticket_rate_limits.get(key_hash, [])
    active_entries = [
        (ts, cnt) for ts, cnt in entries if now - ts < TICKET_RATE_LIMIT_WINDOW_SECONDS
    ]

    max_tickets = get_ticket_rate_limit(request.app)
    current_tickets = sum(cnt for _, cnt in active_entries)

    if current_tickets + ticket_count > max_tickets:
        oldest_ts = active_entries[0][0] if active_entries else now
        retry_after = max(
            1, math.ceil((oldest_ts + TICKET_RATE_LIMIT_WINDOW_SECONDS) - now)
        )
        ticket_rate_limits[key_hash] = active_entries
        counters: Counters | None = getattr(request.app.state, "counters", None)
        if counters is not None:
            counters.rate_limited_total += 1
        logger.warning(
            "api_key_hash=%s ticket rate limit exceeded current=%d requested=%d max=%d retry_after=%ds",
            key_hash,
            current_tickets,
            ticket_count,
            max_tickets,
            retry_after,
            extra={
                "api_key_hash": key_hash,
                "current_tickets": current_tickets,
                "requested_tickets": ticket_count,
                "max_tickets": max_tickets,
                "retry_after": retry_after,
            },
        )
        raise HTTPException(
            status_code=429,
            detail="Too Many Requests: ticket rate limit exceeded",
            headers={"Retry-After": str(retry_after)},
        )

    active_entries.append((now, ticket_count))
    ticket_rate_limits[key_hash] = active_entries


class TriageRequest(BaseModel):
    ticket_id: str = Field(min_length=1, max_length=MAX_ID_LENGTH)
    subject: str | None = None
    body: str

    @field_validator("ticket_id", mode="before")
    @classmethod
    def validate_ticket_id(cls, v: object) -> str:
        return normalize_ticket_id(v)

    @field_validator("body")
    @classmethod
    def validate_body(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("body cannot be empty")
        if len(v.encode("utf-8")) > MAX_BODY_BYTES:
            raise HTTPException(
                status_code=413,
                detail=f"Body exceeds 10 KB limit ({MAX_BODY_BYTES} bytes)",
            )
        return v


class TriageResponse(BaseModel):
    ticket_id: str
    label: str
    confidence: float
    needs_review: bool
    model_version: str


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    setup_logging()
    model_file = get_model_path()
    if not model_file.exists():
        raise FileNotFoundError(
            f"Model file not found: {model_file}. "
            "Please train the model before starting the service."
        )

    threshold = get_threshold()
    loaded_model = TriageClassifier.load(model_file)
    loaded_model.confidence_threshold = threshold

    if not getattr(loaded_model, "version", None) or not is_valid_model_version(
        loaded_model.version
    ):
        model_ver = getattr(loaded_model, "version", None)
        raise ValueError(
            f"Invalid or missing model version '{model_ver}' in {model_file}. "
            "Expected format triage-v<semver>-<YYYYMMDD>."
        )

    try:
        app.state.feedback_store = FeedbackStore()
    except Exception as exc:
        raise RuntimeError(f"Failed to initialize feedback storage: {exc}") from exc

    app.state.model = loaded_model
    app.state.threshold = threshold
    app.state.counters = Counters()
    app.state.api_keys = get_api_keys()
    app.state.admin_api_keys = get_admin_api_keys()
    validate_key_separation(app.state.api_keys, app.state.admin_api_keys)
    app.state.rate_limits = {}
    app.state.ticket_rate_limits = {}
    app.state.ticket_rate_limit_per_minute = get_ticket_rate_limit(app)
    app.state.alert_thresholds = get_alert_thresholds()
    app.state.started_at = datetime.now(timezone.utc).isoformat()
    yield
    app.state.model = None
    app.state.threshold = None
    app.state.api_keys = set()
    app.state.admin_api_keys = set()
    app.state.rate_limits = {}
    app.state.alert_thresholds = None
    app.state.started_at = None
    if getattr(app.state, "feedback_store", None) is not None:
        app.state.feedback_store.close()
    app.state.feedback_store = None


app = FastAPI(
    title="SupportPilot Triage API",
    description=(
        "SupportPilot Ticket Triage API. "
        "In-memory limits only work for a single instance."
    ),
    lifespan=lifespan,
)
app.state.counters = Counters()
app.state.api_keys = set()
app.state.admin_api_keys = set()
app.state.rate_limits = {}
app.state.ticket_rate_limits = {}
app.state.ticket_rate_limit_per_minute = TICKET_RATE_LIMIT_DEFAULT
app.state.feedback_store = None
app.state.alert_thresholds = None
app.state.started_at = None


@app.middleware("http")
async def track_counters(request: Request, call_next: Any) -> Response:
    if request.url.path in EXCLUDED_METRIC_PATHS:
        return await call_next(request)

    t0 = time.perf_counter()
    counters: Counters | None = getattr(request.app.state, "counters", None)
    if counters is not None:
        counters.requests_total += 1

    try:
        response: Response = await call_next(request)
    except Exception:
        if counters is not None:
            counters.errors_total += 1
            counters.total_latency_ms += (time.perf_counter() - t0) * 1000.0
        raise

    if counters is not None:
        # Only count 5xx server failures as errors; 4xx are client errors
        if response.status_code >= 500:
            counters.errors_total += 1
        counters.total_latency_ms += (time.perf_counter() - t0) * 1000.0

    return response


@app.middleware("http")
async def request_id_middleware(request: Request, call_next: Any) -> Response:
    incoming_id = request.headers.get("X-Request-ID")
    # Allow only [A-Za-z0-9-] up to 64 chars to prevent log injection
    if incoming_id and REQUEST_ID_REGEX.fullmatch(incoming_id):
        request_id = incoming_id
    else:
        request_id = str(uuid.uuid4())
    request.state.request_id = request_id

    t0 = time.perf_counter()
    status_code = 500
    try:
        response: Response = await call_next(request)
        status_code = response.status_code
        response.headers["X-Request-ID"] = request_id
        return response
    except Exception:
        status_code = 500
        raise
    finally:
        latency_ms = round((time.perf_counter() - t0) * 1000.0, 2)
        logger.info(
            "request_id=%s path=%s method=%s status=%d latency_ms=%.2f",
            request_id,
            request.url.path,
            request.method,
            status_code,
            latency_ms,
            extra={
                "request_id": request_id,
                "path": request.url.path,
                "method": request.method,
                "status": status_code,
                "latency_ms": latency_ms,
            },
        )


@app.get("/health")
def health(req: Request) -> dict[str, object]:
    model = getattr(req.app.state, "model", None)
    version = getattr(model, "version", None) if model is not None else None
    return {
        "status": "ok",
        "model_loaded": model is not None,
        "model_version": version,
    }


@app.get(
    "/version",
    response_model=VersionResponse,
)
def get_version(req: Request) -> VersionResponse:
    """Return Git commit SHA, active model version, and startup timestamp (public)."""
    model = getattr(req.app.state, "model", None)
    model_version = getattr(model, "version", None) if model is not None else None
    git_sha = get_git_sha()
    started_at = getattr(req.app.state, "started_at", None)
    return VersionResponse(
        git_sha=git_sha,
        model_version=model_version,
        started_at=started_at,
    )


@app.get(
    "/metrics",
    response_model=MetricsResponse,
    dependencies=[Depends(rate_limit)],
)
def get_metrics(req: Request) -> MetricsResponse:
    counters: Counters | None = getattr(req.app.state, "counters", None)
    feedback_store: FeedbackStore | None = getattr(
        req.app.state, "feedback_store", None
    )
    feedback_total, live_accuracy = (
        feedback_store.get_live_accuracy() if feedback_store is not None else (0, None)
    )

    if counters is None or counters.requests_total == 0:
        return MetricsResponse(
            requests_total=0 if counters is None else counters.requests_total,
            triage_total=0 if counters is None else counters.triage_total,
            errors_total=0 if counters is None else counters.errors_total,
            avg_latency_ms=0.0,
            needs_review_rate=0.0,
            rate_limited_total=0 if counters is None else counters.rate_limited_total,
            feedback_total=feedback_total,
            live_accuracy=live_accuracy,
        )

    avg_latency = round(counters.total_latency_ms / counters.requests_total, 2)
    review_rate = (
        round(counters.needs_review_total / counters.triage_total, 4)
        if counters.triage_total > 0
        else 0.0
    )

    return MetricsResponse(
        requests_total=counters.requests_total,
        triage_total=counters.triage_total,
        errors_total=counters.errors_total,
        avg_latency_ms=avg_latency,
        needs_review_rate=review_rate,
        rate_limited_total=counters.rate_limited_total,
        feedback_total=feedback_total,
        live_accuracy=live_accuracy,
    )


@app.get(
    "/metrics/labels",
    response_model=dict[str, LabelMetric],
    dependencies=[Depends(get_api_key), Depends(rate_limit)],
)
def get_label_metrics(req: Request) -> dict[str, Any]:
    feedback_store: FeedbackStore | None = getattr(
        req.app.state, "feedback_store", None
    )
    if feedback_store is None:
        raise RuntimeError(
            "Feedback store is not initialized. Service startup invariant violated."
        )
    return feedback_store.get_label_metrics()


@app.get(
    "/alerts",
    response_model=AlertsResponse,
    dependencies=[Depends(get_api_key), Depends(rate_limit)],
)
def get_alerts(req: Request) -> AlertsResponse:
    metrics = get_metrics(req)
    thresholds: AlertThresholds | None = getattr(
        req.app.state, "alert_thresholds", None
    )
    if thresholds is None:
        raise RuntimeError(
            "Alert thresholds are not initialized. Service startup invariant violated."
        )
    feedback_store: FeedbackStore | None = getattr(
        req.app.state, "feedback_store", None
    )
    key_usage = feedback_store.get_key_usage() if feedback_store is not None else []
    client_keys: set[str] = getattr(req.app.state, "api_keys", set())
    client_key_hashes = {hash_api_key(k) for k in client_keys}
    started_at = getattr(req.app.state, "started_at", None)
    active_alerts = evaluate_alerts(
        metrics,
        thresholds,
        key_usage=key_usage,
        client_key_hashes=client_key_hashes,
        started_at=started_at,
    )
    return AlertsResponse(
        alerts=active_alerts,
        checked_at=datetime.now(timezone.utc).isoformat(),
    )


@app.get(
    "/usage",
    response_model=list[KeyUsageItem],
    dependencies=[Depends(require_admin_key)],
)
def get_usage(req: Request) -> list[dict[str, Any]]:
    feedback_store: FeedbackStore | None = getattr(
        req.app.state, "feedback_store", None
    )
    if feedback_store is None:
        raise RuntimeError(
            "Feedback store is not initialized. Service startup invariant violated."
        )
    return feedback_store.get_key_usage()


@app.post(
    "/triage",
    response_model=TriageResponse,
    dependencies=[Depends(rate_limit)],
)
def triage(request: TriageRequest, req: Request) -> TriageResponse:
    start_time = time.perf_counter()

    model = getattr(req.app.state, "model", None)
    if model is None:
        raise HTTPException(status_code=503, detail="Model is not loaded")

    # Pipeline: request -> redact PII (SP-102) -> predict_one() -> response
    redacted_subject = redact_text(request.subject).text if request.subject else None
    redacted_body = redact_text(request.body).text
    text_to_classify = (
        f"{redacted_subject} {redacted_body}".strip()
        if redacted_subject
        else redacted_body
    )

    threshold = getattr(req.app.state, "threshold", DEFAULT_CONFIDENCE_THRESHOLD)

    result = model.predict_one(text_to_classify, threshold=threshold)
    latency = time.perf_counter() - start_time

    counters: Counters | None = getattr(req.app.state, "counters", None)
    if counters is not None:
        counters.triage_total += 1
        if result["needs_review"]:
            counters.needs_review_total += 1

    # Never log the ticket body, only ticket_id, label, latency, and api_key_hash
    request_id = getattr(req.state, "request_id", None)
    key_hash = getattr(req.state, "api_key_hash", None)
    logger.info(
        "ticket_id=%s label=%s latency=%.4fs api_key_hash=%s",
        request.ticket_id,
        result["label"],
        latency,
        key_hash,
        extra={
            "request_id": request_id,
            "ticket_id": request.ticket_id,
            "label": result["label"],
            "latency": latency,
            "api_key_hash": key_hash,
        },
    )

    # Record prediction in feedback store (fail fast if not initialized)
    feedback_store: FeedbackStore | None = getattr(
        req.app.state, "feedback_store", None
    )
    if feedback_store is None:
        raise RuntimeError(
            "Feedback store is not initialized. Service startup invariant violated."
        )

    try:
        feedback_store.save_prediction(
            ticket_id=request.ticket_id,
            predicted_label=result["label"],
            confidence=result["confidence"],
            model_version=model.version,
            needs_review=result["needs_review"],
        )
    except Exception:
        logger.exception(
            "Failed to save prediction for ticket_id=%s",
            request.ticket_id,
            extra={"ticket_id": request.ticket_id},
        )

    return TriageResponse(
        ticket_id=request.ticket_id,
        label=result["label"],
        confidence=result["confidence"],
        needs_review=result["needs_review"],
        model_version=model.version,
    )


def extract_short_validation_error(exc: ValidationError) -> str:
    """Format a short, safe error reason without leaking any input ticket text."""
    errs = exc.errors()
    if not errs:
        return "validation error"
    first = errs[0]
    loc = ".".join(str(part) for part in first.get("loc", []))
    msg = str(first.get("msg", "invalid")).lower()
    err_type = str(first.get("type", "")).lower()

    if "body" in loc:
        if "empty" in msg or "missing" in err_type:
            return "missing or empty body"
        return "invalid body"
    if "ticket_id" in loc:
        if "missing" in err_type:
            return "missing or empty ticket_id"
        return "invalid ticket_id format"
    return f"invalid {loc}" if loc else "validation error"


@app.post(
    "/triage/batch",
    response_model=BatchTriageResponse,
    dependencies=[Depends(rate_limit)],
)
def triage_batch(
    payload: BatchTriageRequest,
    req: Request,
    api_key: str = Depends(get_api_key),
) -> BatchTriageResponse:
    """Triage a batch of tickets (SP-117).

    - Max 100 tickets per request (101 gives 413 Payload Too Large).
    - Enforces sliding window rate limit of 1,000 tickets/minute per key (429).
    - Per-ticket validation: invalid tickets do not fail the batch.
    - Idempotent: cached predictions are returned with cached=True.
    - Summary metrics: total, ok, failed.
    """
    start_time = time.perf_counter()

    # 1. More than 100 tickets -> 413 (check list length before any work)
    if len(payload.tickets) > MAX_BATCH_SIZE:
        raise HTTPException(
            status_code=413,
            detail=f"Batch size ({len(payload.tickets)}) exceeds maximum limit of {MAX_BATCH_SIZE} tickets",
        )

    # 2. Tickets per minute per key rate limiter (default 1,000/min)
    key_hash = getattr(req.state, "api_key_hash", None) or hash_api_key(api_key)
    check_ticket_rate_limit(req, key_hash, len(payload.tickets))

    model = getattr(req.app.state, "model", None)
    if model is None:
        raise HTTPException(status_code=503, detail="Model is not loaded")

    threshold = getattr(req.app.state, "threshold", DEFAULT_CONFIDENCE_THRESHOLD)
    feedback_store: FeedbackStore | None = getattr(
        req.app.state, "feedback_store", None
    )

    results: list[BatchItemResult] = []
    seen_in_batch: set[str] = set()
    valid_tickets: list[tuple[int, TriageRequest]] = []

    # 1. Per-ticket validation pass
    for item in payload.tickets:
        if not isinstance(item, dict):
            results.append(
                BatchItemResult(
                    ticket_id="unknown",
                    status="error",
                    error="ticket must be an object",
                )
            )
            continue

        raw_id = item.get("ticket_id")
        tid_fallback = (
            str(raw_id).strip()[:MAX_ID_LENGTH]
            if raw_id is not None and str(raw_id).strip()
            else "unknown"
        )

        try:
            validated_ticket = TriageRequest.model_validate(item)
        except HTTPException:
            # 10 KB body limit check from TriageRequest validator
            results.append(
                BatchItemResult(
                    ticket_id=tid_fallback,
                    status="error",
                    error=f"body exceeds {MAX_BODY_BYTES} bytes limit",
                )
            )
            continue
        except ValidationError as exc:
            results.append(
                BatchItemResult(
                    ticket_id=tid_fallback,
                    status="error",
                    error=extract_short_validation_error(exc),
                )
            )
            continue
        except Exception:
            results.append(
                BatchItemResult(
                    ticket_id=tid_fallback,
                    status="error",
                    error="invalid ticket format",
                )
            )
            continue

        tid = validated_ticket.ticket_id
        if tid in seen_in_batch:
            results.append(
                BatchItemResult(
                    ticket_id=tid,
                    status="error",
                    error="duplicate_in_batch",
                )
            )
            continue
        seen_in_batch.add(tid)

        # Place placeholder in results to preserve exact input order
        curr_idx = len(results)
        results.append(BatchItemResult(ticket_id=tid, status="ok"))
        valid_tickets.append((curr_idx, validated_ticket))

    # 2. One DB query: fetch existing predictions for all valid ticket IDs (SP-117 Part C)
    cached_predictions: dict[str, dict[str, Any]] = {}
    valid_tids = [vt.ticket_id for _, vt in valid_tickets]
    if feedback_store is not None and valid_tids:
        try:
            cached_predictions = feedback_store.get_predictions_batch(valid_tids)
        except Exception:
            logger.exception("Failed to fetch cached predictions for batch")

    # 3. Partition into cached vs new (unpredicted) tickets
    to_predict_indices: list[int] = []
    to_predict_texts: list[str] = []
    to_predict_tids: list[str] = []

    for res_idx, vt in valid_tickets:
        tid = vt.ticket_id
        if tid in cached_predictions:
            # Already in DB -> return stored prediction with cached: true (no re-prediction, no overwrite)
            cached = cached_predictions[tid]
            results[res_idx] = BatchItemResult(
                ticket_id=tid,
                status="ok",
                label=cached["predicted_label"],
                confidence=cached["confidence"],
                needs_review=cached.get("needs_review", False),
                cached=True,
            )
        else:
            # New ticket -> prepare for model prediction
            red_subj = redact_text(vt.subject).text if vt.subject else None
            red_body = redact_text(vt.body).text
            text_to_classify = (
                f"{red_subj} {red_body}".strip() if red_subj else red_body
            )
            to_predict_indices.append(res_idx)
            to_predict_texts.append(text_to_classify)
            to_predict_tids.append(tid)

    # 4. Predict all new tickets in one single batch call (SP-117 Part D)
    counters: Counters | None = getattr(req.app.state, "counters", None)
    new_predictions_to_save: list[dict[str, Any]] = []

    if to_predict_texts:
        batch_predictions = model.predict_batch_scored(
            to_predict_texts, threshold=threshold
        )
        for idx, tid, pred in zip(
            to_predict_indices, to_predict_tids, batch_predictions
        ):
            results[idx] = BatchItemResult(
                ticket_id=tid,
                status="ok",
                label=pred["label"],
                confidence=pred["confidence"],
                needs_review=pred["needs_review"],
                cached=False,
            )
            new_predictions_to_save.append(
                {
                    "ticket_id": tid,
                    "predicted_label": pred["label"],
                    "confidence": pred["confidence"],
                    "model_version": model.version,
                    "needs_review": pred["needs_review"],
                }
            )

        if counters is not None:
            counters.triage_total += len(batch_predictions)
            counters.needs_review_total += sum(
                1 for p in batch_predictions if p["needs_review"]
            )

    # 5. Store all new predictions in a single database transaction (SP-117 Part D)
    if feedback_store is not None and new_predictions_to_save:
        try:
            feedback_store.save_predictions_batch(new_predictions_to_save)
        except Exception:
            logger.exception("Failed to save batch predictions in database")

    ok_count = sum(1 for r in results if r.status == "ok")
    failed_count = sum(1 for r in results if r.status == "error")
    cached_count = sum(1 for r in results if r.cached is True)
    summary = BatchSummary(
        total=len(payload.tickets),
        ok=ok_count,
        failed=failed_count,
    )

    latency_ms = round((time.perf_counter() - start_time) * 1000, 2)
    request_id = getattr(req.state, "request_id", None)
    logger.info(
        "request_id=%s total=%d ok=%d failed=%d cached=%d latency_ms=%.2f",
        request_id,
        summary.total,
        summary.ok,
        summary.failed,
        cached_count,
        latency_ms,
        extra={
            "request_id": request_id,
            "total": summary.total,
            "ok": summary.ok,
            "failed": summary.failed,
            "cached": cached_count,
            "latency_ms": latency_ms,
            "api_key_hash": key_hash,
        },
    )

    return BatchTriageResponse(results=results, summary=summary)


@app.post(
    "/feedback",
    response_model=FeedbackResponse,
    dependencies=[Depends(get_api_key), Depends(rate_limit)],
)
def submit_feedback(
    request: FeedbackRequest,
    req: Request,
) -> FeedbackResponse:
    feedback_store: FeedbackStore | None = getattr(
        req.app.state, "feedback_store", None
    )
    if feedback_store is None:
        raise RuntimeError(
            "Feedback store is not initialized. Service startup invariant violated."
        )

    try:
        result = feedback_store.save_feedback(
            ticket_id=request.ticket_id,
            correct_label=request.correct_label,
            agent_id=request.agent_id,
        )
    except KeyError as exc:
        raise HTTPException(
            status_code=404,
            detail=f"Ticket '{request.ticket_id}' has not been triaged.",
        ) from exc
    except ValueError as exc:
        raise HTTPException(
            status_code=422,
            detail=str(exc),
        ) from exc

    logger.info(
        "ticket_id=%s agent_id=%s agreed=%s",
        request.ticket_id,
        request.agent_id,
        result["agreed"],
        extra={
            "ticket_id": request.ticket_id,
            "agent_id": request.agent_id,
            "agreed": result["agreed"],
        },
    )

    return FeedbackResponse(
        ticket_id=result["ticket_id"],
        correct_label=result["correct_label"],
        predicted_label=result["predicted_label"],
        agreed=result["agreed"],
    )
