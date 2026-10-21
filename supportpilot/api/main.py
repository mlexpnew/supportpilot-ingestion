from __future__ import annotations

import logging
import os
import re
import secrets
import time
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request, Response, Security
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, field_validator

from supportpilot.preprocessing.redact import redact_text
from supportpilot.triage.model import (
    DEFAULT_CONFIDENCE_THRESHOLD,
    DEFAULT_MODEL_PATH,
    TriageClassifier,
)

MAX_BODY_BYTES: int = 10 * 1024  # 10 KB limit
logger = logging.getLogger(__name__)

REQUEST_ID_REGEX = re.compile(r"^[A-Za-z0-9-]{1,64}$")

EXCLUDED_METRIC_PATHS: set[str] = {
    "/health",
    "/metrics",
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
    matched = False
    for expected_key in valid_keys:
        if secrets.compare_digest(candidate, expected_key):
            matched = True
    return matched and bool(api_key)


api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def get_api_key(
    request: Request,
    api_key: str | None = Security(api_key_header),
) -> str:
    """FastAPI dependency that reads and validates the X-API-Key header."""
    valid_keys: set[str] = getattr(request.app.state, "api_keys", set())
    if not validate_api_key(api_key, valid_keys):
        raise HTTPException(
            status_code=401,
            detail="Unauthorized",
        )
    return api_key  # type: ignore[return-value]


verify_api_key = get_api_key


class TriageRequest(BaseModel):
    ticket_id: str
    subject: str | None = None
    body: str

    @field_validator("ticket_id")
    @classmethod
    def validate_ticket_id(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("ticket_id cannot be empty")
        return v

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


class MetricsResponse(BaseModel):
    requests_total: int
    triage_total: int
    errors_total: int
    avg_latency_ms: float
    needs_review_rate: float


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    model_file = get_model_path()
    if not model_file.exists():
        raise FileNotFoundError(
            f"Model file not found: {model_file}. "
            "Please train the model before starting the service."
        )

    threshold = get_threshold()
    loaded_model = TriageClassifier.load(model_file)
    loaded_model.confidence_threshold = threshold

    app.state.model = loaded_model
    app.state.threshold = threshold
    app.state.counters = Counters()
    app.state.api_keys = get_api_keys()
    yield
    app.state.model = None
    app.state.threshold = None
    app.state.api_keys = set()


app = FastAPI(title="SupportPilot Triage API", lifespan=lifespan)
app.state.counters = Counters()
app.state.api_keys = set()


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

    logger.info(
        "request_id=%s method=%s path=%s",
        request_id,
        request.method,
        request.url.path,
        extra={
            "request_id": request_id,
            "method": request.method,
            "path": request.url.path,
        },
    )

    try:
        response: Response = await call_next(request)
    except Exception:
        logger.error(
            "request_id=%s method=%s path=%s unhandled error",
            request_id,
            request.method,
            request.url.path,
            extra={"request_id": request_id},
        )
        raise

    response.headers["X-Request-ID"] = request_id
    return response


@app.get("/health")
def health(req: Request) -> dict[str, object]:
    model = getattr(req.app.state, "model", None)
    return {"status": "ok", "model_loaded": model is not None}


@app.get("/metrics", response_model=MetricsResponse)
def get_metrics(req: Request) -> MetricsResponse:
    counters: Counters | None = getattr(req.app.state, "counters", None)
    if counters is None or counters.requests_total == 0:
        return MetricsResponse(
            requests_total=0 if counters is None else counters.requests_total,
            triage_total=0 if counters is None else counters.triage_total,
            errors_total=0 if counters is None else counters.errors_total,
            avg_latency_ms=0.0,
            needs_review_rate=0.0,
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
    )


@app.post("/triage", response_model=TriageResponse)
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

    # Never log the ticket body, only ticket_id, label and latency
    request_id = getattr(req.state, "request_id", None)
    logger.info(
        "ticket_id=%s label=%s latency=%.4fs",
        request.ticket_id,
        result["label"],
        latency,
        extra={
            "request_id": request_id,
            "ticket_id": request.ticket_id,
            "label": result["label"],
            "latency": latency,
        },
    )

    return TriageResponse(
        ticket_id=request.ticket_id,
        label=result["label"],
        confidence=result["confidence"],
        needs_review=result["needs_review"],
    )
