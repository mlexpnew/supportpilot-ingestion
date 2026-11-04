"""Model health alerts evaluation and threshold configuration (SP-112 Part C)."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, Field

from supportpilot.api.models import MetricsResponse
from supportpilot.monitoring.quality import compute_accuracy, compute_psi

logger = logging.getLogger(__name__)

ALERT_NAME_ACCURACY: str = "accuracy"
ALERT_NAME_REVIEW_RATE: str = "review_rate"
ALERT_NAME_ERROR_RATE: str = "error_rate"
ALERT_NAME_NO_CLIENT_TRAFFIC: str = "no_client_traffic"
ALERT_NAME_NEVER_SEEN: str = "client_never_seen"
ALERT_NAME_LATENCY_SLO_BREACH: str = "latency_slo_breach"
ALERT_NAME_LABEL_DRIFT: str = "label_drift"
ALERT_NAME_LOW_LIVE_ACCURACY: str = "low_live_accuracy"

SEVERITY_WARNING: str = "warning"
SEVERITY_CRITICAL: str = "critical"

DEFAULT_MIN_ACCURACY: float = 0.80
DEFAULT_MAX_REVIEW_RATE: float = 0.30
DEFAULT_MAX_ERROR_RATE: float = 0.05
DEFAULT_MIN_FEEDBACK: int = 20
DEFAULT_MIN_TRIAGE: int = 20
DEFAULT_MIN_REQUESTS: int = 50
DEFAULT_MAX_P95_LATENCY_MS: float = 500.0
DEFAULT_MIN_LATENCY_SAMPLES: int = 20
DEFAULT_MAX_PSI: float = 0.20
DEFAULT_MIN_DRIFT_PREDICTIONS: int = 50
DEFAULT_MIN_LIVE_ACCURACY: float = 0.75
DEFAULT_MIN_QUALITY_FEEDBACK: int = 30


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
    max_p95_latency_ms: float = DEFAULT_MAX_P95_LATENCY_MS
    min_latency_samples: int = DEFAULT_MIN_LATENCY_SAMPLES
    max_psi: float = DEFAULT_MAX_PSI
    min_drift_predictions: int = DEFAULT_MIN_DRIFT_PREDICTIONS
    min_live_accuracy: float = DEFAULT_MIN_LIVE_ACCURACY
    min_quality_feedback: int = DEFAULT_MIN_QUALITY_FEEDBACK


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
      - ALERT_MAX_P95_LATENCY_MS (default: 500.0)
      - ALERT_MIN_LATENCY_SAMPLES (default: 20)
      - ALERT_MAX_PSI (default: 0.20)
      - ALERT_MIN_DRIFT_PREDICTIONS (default: 50)
      - ALERT_MIN_LIVE_ACCURACY (default: 0.75)
      - ALERT_MIN_QUALITY_FEEDBACK (default: 30)
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
        max_p95_latency_ms=_parse_float_env(
            "ALERT_MAX_P95_LATENCY_MS",
            DEFAULT_MAX_P95_LATENCY_MS,
            min_val=0.0,
            max_val=100000.0,
        ),
        min_latency_samples=_parse_int_env(
            "ALERT_MIN_LATENCY_SAMPLES", DEFAULT_MIN_LATENCY_SAMPLES, min_val=1
        ),
        max_psi=_parse_float_env(
            "ALERT_MAX_PSI", DEFAULT_MAX_PSI, min_val=0.0, max_val=100.0
        ),
        min_drift_predictions=_parse_int_env(
            "ALERT_MIN_DRIFT_PREDICTIONS", DEFAULT_MIN_DRIFT_PREDICTIONS, min_val=1
        ),
        min_live_accuracy=_parse_float_env(
            "ALERT_MIN_LIVE_ACCURACY", DEFAULT_MIN_LIVE_ACCURACY
        ),
        min_quality_feedback=_parse_int_env(
            "ALERT_MIN_QUALITY_FEEDBACK", DEFAULT_MIN_QUALITY_FEEDBACK, min_val=1
        ),
    )


def evaluate_alerts(
    metrics: MetricsResponse,
    thresholds: AlertThresholds,
    key_usage: list[dict[str, Any]] | None = None,
    client_key_hashes: set[str] | None = None,
    now: datetime | None = None,
    started_at: str | datetime | None = None,
    feedback_store: Any | None = None,
    recent_predictions_counts: dict[str, int] | None = None,
    psi: float | None = None,
    total_predictions: int | None = None,
    feedback_rows: int | None = None,
    live_accuracy: float | None = None,
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

      5. Latency SLO Breach alert (SP-119 Part B):
         - p95 latency > 500 ms with at least 20 samples -> latency_slo_breach.

      6. Low Live Accuracy alert (SP-120 Part C):
         - accuracy < 0.75 with at least 30 feedback rows -> low_live_accuracy.

      7. Label Drift alert (SP-120 Part B):
         - PSI > 0.20 against frozen eval set baseline with at least 50 predictions in last 7 days -> label_drift.
    """
    alerts: list[Alert] = []
    curr_now = now or datetime.now(timezone.utc)

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

        # 5. Latency SLO Breach Alert (SP-119 Part B)
        # Fires when p95 is over 500 ms and there are at least 20 samples.
    if metrics.latencies:
        for ep, lat in sorted(metrics.latencies.items()):
            samples = lat.samples if hasattr(lat, "samples") else lat.get("samples", 0)
            p95_val = (
                lat.p95_ms
                if hasattr(lat, "p95_ms")
                else lat.get("p95_ms", lat.get("p95", 0.0))
            )
            if samples >= thresholds.min_latency_samples:
                if p95_val > thresholds.max_p95_latency_ms:
                    critical_boundary = thresholds.max_p95_latency_ms * 2.0
                    severity = (
                        SEVERITY_CRITICAL
                        if p95_val >= critical_boundary
                        else SEVERITY_WARNING
                    )
                    alerts.append(
                        Alert(
                            name=ALERT_NAME_LATENCY_SLO_BREACH,
                            value=round(p95_val, 2),
                            threshold=thresholds.max_p95_latency_ms,
                            severity=severity,
                            message=f"{ep} p95 latency {round(p95_val, 2)}ms exceeds {thresholds.max_p95_latency_ms}ms SLO ({samples} samples)",
                        )
                    )

    # 6. Low Live Accuracy Alert (SP-120 Part C)
    # Fires when accuracy is under 0.75 and there are at least 30 feedback rows.
    fb_total: int | None = feedback_rows
    acc_candidate: float | None = live_accuracy

    if fb_total is None and acc_candidate is None and feedback_store is not None:
        try:
            pairs = feedback_store.get_prediction_feedback_pairs()
            fb_total = len(pairs)
            if fb_total >= thresholds.min_quality_feedback:
                acc_candidate = compute_accuracy(pairs)
        except Exception as exc:
            logger.warning("Failed computing quality accuracy for alert: %s", exc)

    if (
        fb_total is not None
        and fb_total >= thresholds.min_quality_feedback
        and acc_candidate is not None
    ):
        acc_f = float(acc_candidate)
        if acc_f < thresholds.min_live_accuracy:
            critical_boundary = thresholds.min_live_accuracy - 0.10
            severity = (
                SEVERITY_CRITICAL if acc_f < critical_boundary else SEVERITY_WARNING
            )
            alerts.append(
                Alert(
                    name=ALERT_NAME_LOW_LIVE_ACCURACY,
                    value=round(acc_f, 4),
                    threshold=thresholds.min_live_accuracy,
                    severity=severity,
                    message=f"live accuracy {round(acc_f, 4)} is below threshold {thresholds.min_live_accuracy} ({fb_total} feedback rows)",
                )
            )

    # 7. Label Drift Alert (SP-120 Part B)
    # Compares last 7 days' predicted label mix with frozen eval set via PSI.
    # Fires when PSI is over 0.2 and there are at least 50 predictions.
    psi_val: float | None = psi
    n_preds: int | None = total_predictions

    if psi_val is None and recent_predictions_counts is not None:
        n_preds = sum(recent_predictions_counts.values())
        if n_preds >= thresholds.min_drift_predictions:
            psi_val = compute_psi(recent_predictions_counts)
    elif psi_val is None and feedback_store is not None:
        try:
            counts = feedback_store.get_predicted_label_counts_last_7_days(now=curr_now)
            n_preds = sum(counts.values())
            if n_preds >= thresholds.min_drift_predictions:
                psi_val = compute_psi(counts)
        except Exception as exc:
            logger.warning("Failed computing PSI for label drift alert: %s", exc)

    if (
        n_preds is not None
        and n_preds >= thresholds.min_drift_predictions
        and psi_val is not None
    ):
        if psi_val > thresholds.max_psi:
            critical_boundary = thresholds.max_psi * 2.0
            severity = (
                SEVERITY_CRITICAL if psi_val >= critical_boundary else SEVERITY_WARNING
            )
            alerts.append(
                Alert(
                    name=ALERT_NAME_LABEL_DRIFT,
                    value=round(psi_val, 4),
                    threshold=thresholds.max_psi,
                    severity=severity,
                    message=f"label_drift PSI {round(psi_val, 4)} exceeds threshold {thresholds.max_psi} ({n_preds} predictions)",
                )
            )

    return alerts
