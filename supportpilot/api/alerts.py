"""Model health alerts evaluation and threshold configuration (SP-112 Part C)."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, Field

from supportpilot.api.models import MetricsResponse

logger = logging.getLogger(__name__)

ALERT_NAME_ACCURACY: str = "accuracy"
ALERT_NAME_REVIEW_RATE: str = "review_rate"
ALERT_NAME_ERROR_RATE: str = "error_rate"
ALERT_NAME_NO_CLIENT_TRAFFIC: str = "no_client_traffic"
ALERT_NAME_NEVER_SEEN: str = "client_never_seen"

SEVERITY_WARNING: str = "warning"
SEVERITY_CRITICAL: str = "critical"

DEFAULT_MIN_ACCURACY: float = 0.80
DEFAULT_MAX_REVIEW_RATE: float = 0.30
DEFAULT_MAX_ERROR_RATE: float = 0.05
DEFAULT_MIN_FEEDBACK: int = 20
DEFAULT_MIN_TRIAGE: int = 20
DEFAULT_MIN_REQUESTS: int = 50


class Alert(BaseModel):
    """Active operational or model health alert."""

    name: str = Field(
        description="Alert identifier (e.g. accuracy, review_rate, error_rate, no_client_traffic, client_never_seen)"
    )
    value: float = Field(description="Observed metric value")
    threshold: float = Field(description="Configured boundary threshold value")
    severity: str = Field(description="Alert severity level: warning or critical")
    message: str | None = Field(default=None, description="Optional diagnostic message")


class AlertsResponse(BaseModel):
    """Payload returned by GET /alerts."""

    alerts: list[Alert] = Field(
        default_factory=list, description="Active health alerts"
    )
    checked_at: str = Field(description="UTC ISO 8601 timestamp of alert evaluation")


@dataclass(frozen=True)
class AlertThresholds:
    """Thresholds determining when health alerts trigger."""

    min_accuracy: float = DEFAULT_MIN_ACCURACY
    max_review_rate: float = DEFAULT_MAX_REVIEW_RATE
    max_error_rate: float = DEFAULT_MAX_ERROR_RATE
    min_feedback: int = DEFAULT_MIN_FEEDBACK
    min_triage: int = DEFAULT_MIN_TRIAGE
    min_requests: int = DEFAULT_MIN_REQUESTS


def _parse_float_env(
    var_name: str, default: float, min_val: float = 0.0, max_val: float = 1.0
) -> float:
    raw = os.getenv(var_name)
    if raw is None or not raw.strip():
        return default
    try:
        val = float(raw.strip())
    except ValueError as exc:
        raise ValueError(
            f"Invalid {var_name} '{raw}': must be a valid float between {min_val} and {max_val}."
        ) from exc

    if not (min_val <= val <= max_val):
        raise ValueError(
            f"Invalid {var_name} '{raw}': value {val} must be between {min_val} and {max_val}."
        )
    return val


def _parse_int_env(var_name: str, default: int, min_val: int = 0) -> int:
    raw = os.getenv(var_name)
    if raw is None or not raw.strip():
        return default
    try:
        val = int(raw.strip())
    except ValueError as exc:
        raise ValueError(
            f"Invalid {var_name} '{raw}': must be a valid integer >= {min_val}."
        ) from exc

    if val < min_val:
        raise ValueError(
            f"Invalid {var_name} '{raw}': value {val} must be >= {min_val}."
        )
    return val


def get_alert_thresholds() -> AlertThresholds:
    """Read alert thresholds from environment variables with fail-fast validation.

    Supported env vars:
      - ALERT_MIN_ACCURACY (default: 0.80)
      - ALERT_MAX_REVIEW_RATE (default: 0.30)
      - ALERT_MAX_ERROR_RATE (default: 0.05)
      - ALERT_MIN_FEEDBACK (default: 20)
      - ALERT_MIN_TRIAGE (default: 20)
      - ALERT_MIN_REQUESTS (default: 50)
    """
    return AlertThresholds(
        min_accuracy=_parse_float_env("ALERT_MIN_ACCURACY", DEFAULT_MIN_ACCURACY),
        max_review_rate=_parse_float_env(
            "ALERT_MAX_REVIEW_RATE", DEFAULT_MAX_REVIEW_RATE
        ),
        max_error_rate=_parse_float_env("ALERT_MAX_ERROR_RATE", DEFAULT_MAX_ERROR_RATE),
        min_feedback=_parse_int_env("ALERT_MIN_FEEDBACK", DEFAULT_MIN_FEEDBACK),
        min_triage=_parse_int_env("ALERT_MIN_TRIAGE", DEFAULT_MIN_TRIAGE),
        min_requests=_parse_int_env("ALERT_MIN_REQUESTS", DEFAULT_MIN_REQUESTS),
    )


def evaluate_alerts(
    metrics: MetricsResponse,
    thresholds: AlertThresholds,
    key_usage: list[dict[str, Any]] | None = None,
    client_key_hashes: set[str] | None = None,
    now: datetime | None = None,
    started_at: str | datetime | None = None,
) -> list[Alert]:
    """Pure function evaluating current operational metrics against boundary thresholds.

    Rules:
      1. Accuracy alert: only evaluated if feedback_total >= min_feedback (default 20).
         Triggers when live_accuracy < min_accuracy.
         Severity: critical if accuracy < min_accuracy - 0.10 (e.g. < 0.70); otherwise warning.

      2. Needs Review Rate alert: only evaluated if triage_total >= min_triage (default 20).
         Triggers when needs_review_rate > max_review_rate.
         Severity: critical if needs_review_rate >= max_review_rate + 0.10 (e.g. >= 0.40); otherwise warning.

      3. Error Rate alert: only evaluated if requests_total >= min_requests (default 50).
         Triggers when error_rate (errors_total / requests_total) > max_error_rate.
         Severity: critical if error_rate >= max_error_rate * 2.0 (e.g. >= 0.10); otherwise warning.

      4. Client Traffic alerts (SP-115 Part B & Part D):
         - Stale key: any known client key with last_seen older than 24 hours -> warning: no_client_traffic.
         - Never seen: any known client key with no row once service has been up 24 h -> warning: client_never_seen.
    """
    alerts: list[Alert] = []

    # 1. Accuracy Alert (only if feedback_total >= min_feedback)
    if (
        metrics.feedback_total >= thresholds.min_feedback
        and metrics.live_accuracy is not None
    ):
        acc = float(metrics.live_accuracy)
        if acc < thresholds.min_accuracy:
            critical_boundary = thresholds.min_accuracy - 0.10
            severity = (
                SEVERITY_CRITICAL if acc < critical_boundary else SEVERITY_WARNING
            )
            alerts.append(
                Alert(
                    name=ALERT_NAME_ACCURACY,
                    value=round(acc, 4),
                    threshold=thresholds.min_accuracy,
                    severity=severity,
                )
            )

    # 2. Needs Review Rate Alert (only if triage_total >= min_triage)
    if metrics.triage_total >= thresholds.min_triage:
        review_rate = float(metrics.needs_review_rate)
        if review_rate > thresholds.max_review_rate:
            critical_boundary = thresholds.max_review_rate + 0.10
            severity = (
                SEVERITY_CRITICAL
                if review_rate >= critical_boundary
                else SEVERITY_WARNING
            )
            alerts.append(
                Alert(
                    name=ALERT_NAME_REVIEW_RATE,
                    value=round(review_rate, 4),
                    threshold=thresholds.max_review_rate,
                    severity=severity,
                )
            )

    # 3. Error Rate Alert (only if requests_total >= min_requests)
    if metrics.requests_total >= thresholds.min_requests:
        error_rate = round(metrics.errors_total / metrics.requests_total, 4)
        if error_rate > thresholds.max_error_rate:
            critical_boundary = thresholds.max_error_rate * 2.0
            severity = (
                SEVERITY_CRITICAL
                if error_rate >= critical_boundary
                else SEVERITY_WARNING
            )
            alerts.append(
                Alert(
                    name=ALERT_NAME_ERROR_RATE,
                    value=error_rate,
                    threshold=thresholds.max_error_rate,
                    severity=severity,
                )
            )

    # 4. Client Traffic Alerts (SP-115 Part B & Never Seen fix)
    if client_key_hashes:
        usage_list = key_usage or []
        usage_by_hash = {r["key_hash"]: r for r in usage_list if "key_hash" in r}
        curr_now = now or datetime.now(timezone.utc)

        uptime_hours: float | None = None
        if started_at is not None:
            try:
                if isinstance(started_at, datetime):
                    started_dt = started_at
                else:
                    s_str = str(started_at).replace("Z", "+00:00")
                    started_dt = datetime.fromisoformat(s_str)
                if started_dt.tzinfo is None:
                    started_dt = started_dt.replace(tzinfo=timezone.utc)
                uptime_hours = (curr_now - started_dt).total_seconds() / 3600.0
            except Exception as exc:
                logger.warning("Failed parsing started_at=%s: %s", started_at, exc)

        for kh in sorted(client_key_hashes):
            r = usage_by_hash.get(kh)
            if r and r.get("last_seen"):
                try:
                    ls_str = str(r["last_seen"]).replace("Z", "+00:00")
                    ls_dt = datetime.fromisoformat(ls_str)
                    if ls_dt.tzinfo is None:
                        ls_dt = ls_dt.replace(tzinfo=timezone.utc)
                    hours_inactive = (curr_now - ls_dt).total_seconds() / 3600.0
                    if hours_inactive >= 24.0:
                        alerts.append(
                            Alert(
                                name=ALERT_NAME_NO_CLIENT_TRAFFIC,
                                value=round(hours_inactive, 1),
                                threshold=24.0,
                                severity=SEVERITY_WARNING,
                                message="stale traffic",
                            )
                        )
                except Exception as exc:
                    logger.warning(
                        "Failed evaluating client traffic alert for key_hash=%s: %s",
                        kh,
                        exc,
                    )
            else:
                # Key has no row (never seen) -> alert "never seen" if service up >= 24 h
                if uptime_hours is not None and uptime_hours >= 24.0:
                    alerts.append(
                        Alert(
                            name=ALERT_NAME_NEVER_SEEN,
                            value=round(uptime_hours, 1),
                            threshold=24.0,
                            severity=SEVERITY_WARNING,
                            message="never seen",
                        )
                    )

    return alerts
