"""API response models and schemas."""

from __future__ import annotations

from pydantic import BaseModel, Field


class EndpointLatency(BaseModel):
    """Latency percentiles and sample count for an endpoint (SP-119 Part B)."""

    p50_ms: float = 0.0
    p95_ms: float = 0.0
    p50: float = 0.0
    p95: float = 0.0
    samples: int = 0
    sample_count: int = 0


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
    latencies: dict[str, EndpointLatency] = Field(default_factory=dict)


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


class QualityLabelStats(BaseModel):
    """Per-label performance metrics and sample counts (SP-120)."""

    precision: float | None = None
    recall: float | None = None
    predicted_count: int = 0
    actual_count: int = 0
    true_positives: int = 0


class QualityResponse(BaseModel):
    """Model quality metrics and confusion matrix response (SP-120)."""

    status: str  # "ok" or "insufficient_data"
    message: str | None = None
    days: int = 7
    sample_size: int = 0
    samples: int = 0
    accuracy: float | None = None
    precision: dict[str, float | None] | None = None
    recall: dict[str, float | None] | None = None
    confusion_matrix: dict[str, dict[str, int]] | None = None
    matrix: dict[str, dict[str, int]] | None = None
    per_label: dict[str, QualityLabelStats] | None = None


class CreateKeyRequest(BaseModel):
    """Request payload to create a new API key (SP-118)."""

    name: str
    role: str = "client"


class CreateKeyResponse(BaseModel):
    """Response containing the generated raw key, returned once only (SP-118)."""

    name: str
    key: str
    key_hash: str
    role: str
    created_at: str


class KeyItemResponse(BaseModel):
    """Safe key listing item (never contains raw key) (SP-118)."""

    name: str
    role: str
    hash_prefix: str
    created_at: str
    revoked_at: str | None = None
    last_used_at: str | None = None
    active: bool


class RevokeKeyResponse(BaseModel):
    """Response returned upon revoking a key (SP-118)."""

    name: str
    status: str
    revoked_at: str


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
