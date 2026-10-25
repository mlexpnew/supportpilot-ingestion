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
