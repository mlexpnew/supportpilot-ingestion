from __future__ import annotations

import logging
import os
import time
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request, Response
from pydantic import BaseModel, field_validator

from supportpilot.preprocessing.redact import redact_text
from supportpilot.triage.model import (
    DEFAULT_CONFIDENCE_THRESHOLD,
    DEFAULT_MODEL_PATH,
    TriageClassifier,
)

MAX_BODY_BYTES: int = 10 * 1024  # 10 KB limit
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

EXCLUDED_METRIC_PATHS: set[str] = {
    "/health",
    "/metrics",
    "/stats",
    "/docs",
    "/openapi.json",
    "/redoc",
}


class Counters(dict[str, Any]):
    """In-memory counters tracking requests, errors, latency, and needs_review."""

    def __init__(
        self,
        requests: int = 0,
        errors: int = 0,
        latency: float = 0.0,
        needs_review: int = 0,
    ) -> None:
        super().__init__(
            requests=requests,
            errors=errors,
            latency=latency,
            needs_review=needs_review,
        )

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError:
            raise AttributeError(
                f"'{self.__class__.__name__}' object has no attribute '{name}'"
            )

    def __setattr__(self, name: str, value: Any) -> None:
        self[name] = value


def init_counters() -> Counters:
    return Counters(requests=0, errors=0, latency=0.0, needs_review=0)


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
    requests: int
    errors: int
    latency: float
    needs_review: int


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
    counters = init_counters()
    app.state.counters = counters
    app.state.metrics = counters
    yield
    app.state.model = None
    app.state.threshold = None


app = FastAPI(title="SupportPilot Triage API", lifespan=lifespan)
_initial_counters = init_counters()
app.state.counters = _initial_counters
app.state.metrics = _initial_counters


@app.middleware("http")
async def track_counters(request: Request, call_next: Any) -> Response:
    if request.url.path in EXCLUDED_METRIC_PATHS:
        return await call_next(request)

    t0 = time.perf_counter()
    counters: Counters | None = getattr(request.app.state, "counters", None)
    if counters is not None:
        counters["requests"] += 1

    try:
        response: Response = await call_next(request)
    except Exception:
        if counters is not None:
            counters["errors"] += 1
            counters["latency"] = round(
                counters["latency"] + (time.perf_counter() - t0), 4
            )
        raise

    if counters is not None:
        if response.status_code >= 400:
            counters["errors"] += 1
        counters["latency"] = round(counters["latency"] + (time.perf_counter() - t0), 4)

    return response


@app.middleware("http")
async def request_id_middleware(request: Request, call_next: Any) -> Response:
    incoming_id = request.headers.get("X-Request-ID")
    request_id = (
        incoming_id.strip()
        if incoming_id and incoming_id.strip()
        else str(uuid.uuid4())
    )
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
    if counters is None:
        return MetricsResponse(requests=0, errors=0, latency=0.0, needs_review=0)
    return MetricsResponse(
        requests=counters["requests"],
        errors=counters["errors"],
        latency=counters["latency"],
        needs_review=counters["needs_review"],
    )


@app.get("/stats", response_model=MetricsResponse)
def get_stats(req: Request) -> MetricsResponse:
    return get_metrics(req)


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
    if counters is not None and result["needs_review"]:
        counters["needs_review"] += 1

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
