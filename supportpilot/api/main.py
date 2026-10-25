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
from pydantic import BaseModel, Field, field_validator

from supportpilot.api.alerts import (
    AlertsResponse,
    evaluate_alerts,
    get_alert_thresholds,
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
RATE_LIMIT_WINDOW_SECONDS: float = 60.0  # 60 second sliding window
RATE_LIMIT_MAX_REQUESTS: int = 60  # Maximum requests allowed per sliding window
logger = logging.getLogger(__name__)

REQUEST_ID_REGEX = re.compile(r"^[A-Za-z0-9-]{1,64}$")

EXCLUDED_METRIC_PATHS: set[str] = {
    "/health",
    "/alerts",
    "/metrics",
    "/metrics/labels",
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
    """FastAPI dependency that reads and validates the X-API-Key header."""
    key_hash = hash_api_key(api_key)
    request.state.api_key_hash = key_hash
    valid_keys: set[str] = getattr(request.app.state, "api_keys", set())
    if not validate_api_key(api_key, valid_keys):
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


class MetricsResponse(BaseModel):
    requests_total: int
    triage_total: int
    errors_total: int
    avg_latency_ms: float
    needs_review_rate: float
    rate_limited_total: int
    feedback_total: int = 0
    live_accuracy: float | None = None


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
    app.state.rate_limits = {}
    app.state.alert_thresholds = get_alert_thresholds()
    yield
    app.state.model = None
    app.state.threshold = None
    app.state.api_keys = set()
    app.state.rate_limits = {}
    app.state.alert_thresholds = None
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
app.state.rate_limits = {}
app.state.feedback_store = None
app.state.alert_thresholds = None


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
    thresholds = getattr(req.app.state, "alert_thresholds", None)
    if thresholds is None:
        thresholds = get_alert_thresholds()
    active_alerts = evaluate_alerts(metrics, thresholds)
    return AlertsResponse(
        alerts=active_alerts,
        checked_at=datetime.now(timezone.utc).isoformat(),
    )


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
