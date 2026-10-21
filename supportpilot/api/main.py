from __future__ import annotations

import logging
import os
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, field_validator

from supportpilot.preprocessing.redact import redact_text
from supportpilot.triage.model import (
    DEFAULT_CONFIDENCE_THRESHOLD,
    DEFAULT_MODEL_PATH,
    TriageClassifier,
)

MAX_BODY_BYTES: int = 10 * 1024  # 10 KB limit
logger = logging.getLogger(__name__)


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
    yield
    app.state.model = None
    app.state.threshold = None


app = FastAPI(title="SupportPilot Triage API", lifespan=lifespan)


@app.get("/health")
def health(req: Request) -> dict[str, object]:
    model = getattr(req.app.state, "model", None)
    return {"status": "ok", "model_loaded": model is not None}


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

    # Never log the ticket body, only ticket_id, label and latency
    logger.info(
        "ticket_id=%s label=%s latency=%.4fs",
        request.ticket_id,
        result["label"],
        latency,
        extra={
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
