"""Model health alerts evaluation and threshold configuration (SP-112 Part C)."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, Field

ALERT_NAME_ACCURACY: str = "accuracy"
ALERT_NAME_REVIEW_RATE: str = "review_rate"
ALERT_NAME_ERROR_RATE: str = "error_rate"

SEVERITY_WARNING: str = "warning"
SEVERITY_CRITICAL: str = "critical"

DEFAULT_MIN_ACCURACY: float = 0.80
DEFAULT_MAX_REVIEW_RATE: float = 0.30
DEFAULT_MAX_ERROR_RATE: float = 0.05
DEFAULT_MIN_FEEDBACK: int = 20


class Alert(BaseModel):
    """Active operational or model health alert."""

    name: str = Field(
        description="Alert identifier (e.g. accuracy, review_rate, error_rate)"
    )
    value: float = Field(description="Observed metric value")
    threshold: float = Field(description="Configured boundary threshold value")
    severity: str = Field(description="Alert severity level: warning or critical")


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
    """
    return AlertThresholds(
        min_accuracy=_parse_float_env("ALERT_MIN_ACCURACY", DEFAULT_MIN_ACCURACY),
        max_review_rate=_parse_float_env(
            "ALERT_MAX_REVIEW_RATE", DEFAULT_MAX_REVIEW_RATE
        ),
        max_error_rate=_parse_float_env("ALERT_MAX_ERROR_RATE", DEFAULT_MAX_ERROR_RATE),
        min_feedback=_parse_int_env("ALERT_MIN_FEEDBACK", DEFAULT_MIN_FEEDBACK),
    )


def _extract_metric(metrics: Any, key: str, default: Any = None) -> Any:
    if isinstance(metrics, dict):
        return metrics.get(key, default)
    return getattr(metrics, key, default)


def evaluate_alerts(
    metrics: Any,
    thresholds: AlertThresholds,
) -> list[Alert]:
    """Pure function evaluating current operational metrics against boundary thresholds.

    Rules:
      1. Accuracy alert: only evaluated if feedback_total >= min_feedback (default 20).
         Triggers when live_accuracy < min_accuracy.
         Severity: critical if accuracy < min_accuracy - 0.10 (e.g. < 0.70); otherwise warning.

      2. Needs Review Rate alert: only evaluated if triage_total > 0.
         Triggers when needs_review_rate > max_review_rate.
         Severity: critical if needs_review_rate >= max_review_rate + 0.10 (e.g. >= 0.40); otherwise warning.

      3. Error Rate alert: only evaluated if requests_total > 0.
         Triggers when error_rate (errors_total / requests_total) > max_error_rate.
         Severity: critical if error_rate >= max_error_rate * 2 (e.g. >= 0.10); otherwise warning.
    """
    alerts: list[Alert] = []

    # 1. Accuracy Alert
    feedback_total: int = int(_extract_metric(metrics, "feedback_total", 0) or 0)
    live_accuracy = _extract_metric(metrics, "live_accuracy", None)

    if feedback_total >= thresholds.min_feedback and live_accuracy is not None:
        acc_float = float(live_accuracy)
        if acc_float < thresholds.min_accuracy:
            # Slightly past threshold -> warning; far past (e.g. accuracy < 0.70) -> critical
            critical_boundary = thresholds.min_accuracy - 0.10
            severity = (
                SEVERITY_CRITICAL if acc_float < critical_boundary else SEVERITY_WARNING
            )
            alerts.append(
                Alert(
                    name=ALERT_NAME_ACCURACY,
                    value=round(acc_float, 4),
                    threshold=thresholds.min_accuracy,
                    severity=severity,
                )
            )

    # 2. Needs Review Rate Alert
    triage_total: int = int(_extract_metric(metrics, "triage_total", 0) or 0)
    needs_review_rate = _extract_metric(metrics, "needs_review_rate", None)

    if triage_total > 0 and needs_review_rate is not None:
        review_rate_float = float(needs_review_rate)
        if review_rate_float > thresholds.max_review_rate:
            critical_boundary = thresholds.max_review_rate + 0.10
            severity = (
                SEVERITY_CRITICAL
                if review_rate_float >= critical_boundary
                else SEVERITY_WARNING
            )
            alerts.append(
                Alert(
                    name=ALERT_NAME_REVIEW_RATE,
                    value=round(review_rate_float, 4),
                    threshold=thresholds.max_review_rate,
                    severity=severity,
                )
            )

    # 3. Error Rate Alert
    requests_total: int = int(_extract_metric(metrics, "requests_total", 0) or 0)
    errors_total: int = int(_extract_metric(metrics, "errors_total", 0) or 0)

    if requests_total > 0:
        error_rate_float = round(errors_total / requests_total, 4)
        if error_rate_float > thresholds.max_error_rate:
            critical_boundary = thresholds.max_error_rate * 2.0
            severity = (
                SEVERITY_CRITICAL
                if error_rate_float >= critical_boundary
                else SEVERITY_WARNING
            )
            alerts.append(
                Alert(
                    name=ALERT_NAME_ERROR_RATE,
                    value=error_rate_float,
                    threshold=thresholds.max_error_rate,
                    severity=severity,
                )
            )

    return alerts
