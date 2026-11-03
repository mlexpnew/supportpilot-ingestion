"""API response models and schemas."""

from __future__ import annotations

from pydantic import BaseModel


class MetricsResponse(BaseModel):
    """Aggregate metrics and telemetry counters."""

    requests_total: int
    triage_total: int
    errors_total: int
    avg_latency_ms: float
    needs_review_rate: float
    rate_limited_total: int
    feedback_total: int = 0
    live_accuracy: float | None = None


class VersionResponse(BaseModel):
    """Runtime service version and startup metadata."""

    git_sha: str
    model_version: str | None
    started_at: str | None = None


class KeyUsageItem(BaseModel):
    """Per-API-key usage tracking telemetry (SP-115)."""

    key_hash: str
    first_seen: str
    last_seen: str
    ok_count: int
    auth_fail_count: int


class BatchSummary(BaseModel):
    """Batch execution counts summary."""

    total: int
    ok: int
    failed: int


class BatchItemResult(BaseModel):
    """Individual ticket triage result within a batch."""

    ticket_id: str
    status: str  # "ok" or "error"
    label: str | None = None
    confidence: float | None = None
    needs_review: bool | None = None
    cached: bool | None = None
    error: str | None = None


class BatchTriageRequest(BaseModel):
    """Request payload for batch triage endpoint."""

    tickets: list[object] = []


class BatchTriageResponse(BaseModel):
    """Response schema for batch triage endpoint."""

    results: list[BatchItemResult]
    summary: BatchSummary
