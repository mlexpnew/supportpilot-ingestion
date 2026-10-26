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
    """Deployment version and runtime metadata.

    Note on started_at vs deployed_at:
        started_at represents the UTC ISO timestamp when the container process started up
        in lifespan. On platforms like Render, restarting the service refreshes started_at
        without a new code deployment. deployed_at is preserved as a backward-compatible alias.
    """

    git_sha: str
    model_version: str | None
    started_at: str | None = None
    deployed_at: str | None = None
