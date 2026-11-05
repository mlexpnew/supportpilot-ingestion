"""Model quality mathematics and monitoring metrics (SP-120).

Provides pure, testable functions for evaluating real-world model quality
from human agent feedback and production prediction distributions:
  - Confusion Matrix: counts every (predicted, actual) pair across all labels.
  - Live Accuracy: correct predictions divided by total feedback rows.
  - Per-label Precision & Recall: derived directly from the confusion matrix,
    returning None (null) when the denominator is 0.
  - Population Stability Index (PSI): compares 7-day predicted label distribution
    against the sacred frozen evaluation set baseline distribution.
"""

from __future__ import annotations

import json
import logging
import math
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from supportpilot.triage import LABELS

logger = logging.getLogger(__name__)

DEFAULT_DRIFT_BASELINE_PATH: Path = Path(__file__).parent / "drift_baseline.json"
DEFAULT_EPSILON: float = 0.0001


def get_drift_baseline_path() -> Path:
    """Return configured or default path to drift_baseline.json inside the package."""
    env_p = os.getenv("DRIFT_BASELINE_PATH")
    if env_p and env_p.strip():
        return Path(env_p.strip())
    return DEFAULT_DRIFT_BASELINE_PATH


def load_drift_baseline(
    path: Path | str | None = None,
    labels: Sequence[str] = LABELS,
) -> dict[str, int] | None:
    """Load baseline predicted label counts from drift_baseline.json.

    Returns:
        dict mapping label -> count, or None if the baseline file does not exist or is invalid.
        Never falls back silently: logs a warning when the baseline is missing.
    """
    target = Path(path) if path is not None else get_drift_baseline_path()
    if not target.exists():
        logger.warning(
            "Drift baseline file '%s' does not exist (baseline_missing). Never falling back silently.",
            target,
        )
        return None

    try:
        with open(target, "r", encoding="utf-8") as f:
            data = json.load(f)

        if (
            isinstance(data, dict)
            and "counts" in data
            and isinstance(data["counts"], dict)
        ):
            raw_counts = data["counts"]
        elif isinstance(data, dict):
            raw_counts = data
        else:
            logger.warning(
                "Invalid drift baseline format in '%s': expected JSON object, got %s",
                target,
                type(data).__name__,
            )
            return None

        counts = {lbl: int(raw_counts.get(lbl, 0)) for lbl in labels}
        if sum(counts.values()) <= 0:
            logger.warning(
                "Drift baseline in '%s' contains zero total samples.", target
            )
            return None
        return counts
    except Exception as exc:
        logger.warning("Failed to load drift baseline file '%s': %s", target, exc)
        return None


class ConfusionMatrix(dict[str, dict[str, int]]):
    """Confusion matrix representation: matrix[predicted][actual] = count.

    Also supports 2-tuple indexing: matrix[(predicted, actual)] = count.
    Serializes directly to JSON as a nested dictionary.
    """

    def __init__(
        self,
        matrix_dict: Mapping[str, Mapping[str, int]] | None = None,
        labels: Sequence[str] = LABELS,
    ) -> None:
        self.labels: list[str] = list(labels)
        base: dict[str, dict[str, int]] = {
            p: {a: 0 for a in self.labels} for p in self.labels
        }
        if matrix_dict:
            for p, inner in matrix_dict.items():
                if p not in base:
                    base[p] = {a: 0 for a in self.labels}
                for a, cnt in inner.items():
                    base[p][a] = int(cnt)
        super().__init__(base)

    def __getitem__(self, key: Any) -> Any:
        if isinstance(key, tuple) and len(key) == 2:
            pred, actual = key
            return super().__getitem__(str(pred))[str(actual)]
        return super().__getitem__(key)

    def __setitem__(self, key: Any, value: Any) -> None:
        if isinstance(key, tuple) and len(key) == 2:
            pred, actual = key
            super().__getitem__(str(pred))[str(actual)] = int(value)
            return
        super().__setitem__(key, value)

    @property
    def total_samples(self) -> int:
        """Total number of evaluated feedback samples in the matrix."""
        return sum(self[p][a] for p in self.keys() for a in self[p].keys())


def compute_confusion_matrix(
    pairs: Sequence[Mapping[str, str] | tuple[str, str]],
    labels: Sequence[str] = LABELS,
) -> ConfusionMatrix:
    """Count every (predicted, actual) pair across all labels.

    Args:
        pairs: Sequence of items, either dicts with ('predicted_label', 'correct_label')
               or 2-tuples (predicted, actual).
        labels: Sequence of all known label categories.

    Returns:
        ConfusionMatrix mapping matrix[predicted][actual] = count.
    """
    matrix = ConfusionMatrix(labels=labels)

    for item in pairs:
        if isinstance(item, tuple) and len(item) == 2:
            pred, actual = str(item[0]), str(item[1])
        elif isinstance(item, Mapping):
            pred = str(item.get("predicted_label") or item.get("predicted") or "")
            actual = str(item.get("correct_label") or item.get("actual") or "")
        else:
            continue

        if not pred or not actual:
            continue

        if pred not in matrix:
            matrix[pred] = {a: 0 for a in matrix.labels}
        if actual not in matrix[pred]:
            matrix[pred][actual] = 0

        matrix[pred][actual] += 1

    return matrix


def compute_accuracy(
    data: (
        Sequence[Mapping[str, str] | tuple[str, str]] | Mapping[str, Mapping[str, int]]
    ),
    total: int | None = None,
) -> float | None:
    """Calculate overall accuracy: correct predictions divided by total feedback rows.

    Returns:
        Accuracy as a float rounded to 4 decimals, or None if total rows == 0.
    """
    if isinstance(data, Mapping) and any(isinstance(v, Mapping) for v in data.values()):
        # Confusion matrix provided
        agreed = sum(
            data[lbl].get(lbl, 0) for lbl in data if isinstance(data[lbl], Mapping)
        )
        sample_total = (
            total
            if total is not None
            else sum(
                sum(inner.values())
                for inner in data.values()
                if isinstance(inner, Mapping)
            )
        )
    else:
        # Sequence of (predicted, actual) pairs
        pairs = data  # type: ignore[assignment]
        sample_total = total if total is not None else len(pairs)
        agreed = 0
        for item in pairs:
            if isinstance(item, tuple) and len(item) == 2:
                p, a = item[0], item[1]
            elif isinstance(item, Mapping):
                p = item.get("predicted_label") or item.get("predicted")
                a = item.get("correct_label") or item.get("actual")
            else:
                continue
            if p == a and p is not None:
                agreed += 1

    if sample_total == 0:
        return None
    return round(float(agreed / sample_total), 4)


def compute_precision_recall(
    matrix: Mapping[str, Mapping[str, int]],
    labels: Sequence[str] = LABELS,
) -> tuple[dict[str, float | None], dict[str, float | None]]:
    """Work out per-label precision and recall directly from the confusion matrix.

    Precision(L) = TP(L) / Total Predicted as L
    Recall(L) = TP(L) / Total Actual L

    Returns:
        (precision_by_label, recall_by_label), with None for any class with 0 samples.
    """
    precision: dict[str, float | None] = {}
    recall: dict[str, float | None] = {}

    for lbl in labels:
        tp = matrix.get(lbl, {}).get(lbl, 0)

        # Total times model predicted `lbl` (row sum across all actual labels)
        total_predicted = sum(matrix.get(lbl, {}).values())
        if total_predicted == 0:
            precision[lbl] = None
        else:
            precision[lbl] = round(float(tp / total_predicted), 4)

        # Total times ground truth was `lbl` (column sum across all predicted labels)
        total_actual = sum(
            matrix.get(p, {}).get(lbl, 0)
            for p in matrix.keys()
            if isinstance(matrix.get(p), Mapping)
        )
        if total_actual == 0:
            recall[lbl] = None
        else:
            recall[lbl] = round(float(tp / total_actual), 4)

    return precision, recall


def compute_precision(
    matrix: Mapping[str, Mapping[str, int]],
    labels: Sequence[str] = LABELS,
) -> dict[str, float | None]:
    """Work out precision per label from the confusion matrix."""
    prec, _ = compute_precision_recall(matrix, labels=labels)
    return prec


def compute_recall(
    matrix: Mapping[str, Mapping[str, int]],
    labels: Sequence[str] = LABELS,
) -> dict[str, float | None]:
    """Work out recall per label from the confusion matrix."""
    _, rec = compute_precision_recall(matrix, labels=labels)
    return rec


def compute_label_metrics(
    matrix: Mapping[str, Mapping[str, int]],
    labels: Sequence[str] = LABELS,
) -> dict[str, dict[str, Any]]:
    """Work out comprehensive per-label stats including sample sizes.

    For each label returns:
      - precision: float | None
      - recall: float | None
      - predicted_count (precision sample size): int
      - actual_count (recall sample size): int
      - true_positives: int
    """
    stats: dict[str, dict[str, Any]] = {}
    precision, recall = compute_precision_recall(matrix, labels=labels)

    for lbl in labels:
        tp = matrix.get(lbl, {}).get(lbl, 0)
        total_pred = sum(matrix.get(lbl, {}).values())
        total_act = sum(
            matrix.get(p, {}).get(lbl, 0)
            for p in matrix.keys()
            if isinstance(matrix.get(p), Mapping)
        )
        stats[lbl] = {
            "precision": precision[lbl],
            "recall": recall[lbl],
            "predicted_count": total_pred,
            "actual_count": total_act,
            "true_positives": tp,
        }

    return stats


def find_most_confused_pair(
    matrix: Mapping[str, Mapping[str, int]],
    labels: Sequence[str] = LABELS,
) -> tuple[str, str, int] | None:
    """Find the most frequently confused (predicted, actual) label pair where predicted != actual.

    Returns:
        (predicted_label, actual_label, count) or None if no off-diagonal errors exist.
    """
    max_count = 0
    top_pair: tuple[str, str, int] | None = None

    for p in labels:
        for a in labels:
            if p == a:
                continue
            cnt = matrix.get(p, {}).get(a, 0)
            if cnt > max_count:
                max_count = cnt
                top_pair = (p, a, cnt)

    return top_pair


def get_frozen_eval_label_counts(
    eval_path: Path | str | None = None,
    labels: Sequence[str] = LABELS,
) -> dict[str, int] | None:
    """Get the label counts of the baseline file or eval set.

    Returns None if missing (never falls back silently).
    """
    return load_drift_baseline(path=eval_path, labels=labels)


def compute_psi(
    actual_counts: Mapping[str, int],
    expected_counts: Mapping[str, int] | None = None,
    eps: float = DEFAULT_EPSILON,
    labels: Sequence[str] = LABELS,
    baseline_path: Path | str | None = None,
) -> float | None:
    """Compute Population Stability Index (PSI) between actual and expected distributions.

    Formula:
        For each label i:
          actual_share A_i = actual_counts[i] / total_actual (replaced with eps if 0)
          expected_share E_i = expected_counts[i] / total_expected (replaced with eps if 0)
          term_i = (A_i - E_i) * ln(A_i / E_i)
        PSI = sum(term_i across all labels)

    Args:
        actual_counts: Observed prediction counts (e.g., from the last 7 days).
        expected_counts: Baseline counts from drift_baseline.json. If None, loaded from baseline.
        eps: Small non-zero floor to replace 0 shares and avoid log(0) or division by zero.
        labels: All labels to evaluate.
        baseline_path: Optional explicit path to baseline JSON file.

    Returns:
        PSI as a float rounded to 4 decimals, or None if actual total count is 0 or baseline is missing.
    """
    total_actual = sum(actual_counts.get(lbl, 0) for lbl in labels)
    if total_actual == 0:
        return None

    if expected_counts is None:
        expected_counts = load_drift_baseline(path=baseline_path, labels=labels)
        if expected_counts is None:
            logger.warning(
                "Cannot compute PSI: baseline_missing (supportpilot/monitoring/drift_baseline.json not found). Never falling back silently."
            )
            return None

    total_expected = sum(expected_counts.get(lbl, 0) for lbl in labels)
    if total_expected == 0:
        logger.warning("Cannot compute PSI: expected baseline total count is 0.")
        return None

    psi_total = 0.0

    for lbl in labels:
        act_cnt = actual_counts.get(lbl, 0)
        exp_cnt = expected_counts.get(lbl, 0)

        act_share = act_cnt / total_actual
        exp_share = exp_cnt / total_expected

        # Replace any 0 share with epsilon to avoid log(0)
        a_i = max(act_share, eps)
        e_i = max(exp_share, eps)

        term = (a_i - e_i) * math.log(a_i / e_i)
        psi_total += term

    return round(float(psi_total), 4)


def get_drift_status(
    actual_counts: Mapping[str, int] | None = None,
    expected_counts: Mapping[str, int] | None = None,
    min_predictions: int = 50,
    max_psi: float = 0.20,
    baseline_path: Path | str | None = None,
    labels: Sequence[str] = LABELS,
) -> tuple[str, float | None]:
    """Evaluate drift status and PSI value against baseline.

    Returns:
        (status, psi):
            status: "baseline_missing" | "insufficient_data" | "drift_detected" | "stable"
            psi: float | None
    """
    if expected_counts is None:
        expected_counts = load_drift_baseline(path=baseline_path, labels=labels)
        if expected_counts is None:
            return "baseline_missing", None

    if actual_counts is None:
        return "insufficient_data", None

    total_actual = sum(actual_counts.get(lbl, 0) for lbl in labels)
    if total_actual < min_predictions:
        return "insufficient_data", None

    psi = compute_psi(
        actual_counts=actual_counts,
        expected_counts=expected_counts,
        labels=labels,
    )
    if psi is None:
        return "baseline_missing", None

    if psi > max_psi:
        return "drift_detected", psi
    return "stable", psi


def compute_quality_metrics(
    pairs: Sequence[Mapping[str, str] | tuple[str, str]],
    recent_predictions_counts: Mapping[str, int] | None = None,
    expected_eval_counts: Mapping[str, int] | None = None,
    labels: Sequence[str] = LABELS,
    baseline_path: Path | str | None = None,
) -> dict[str, Any]:
    """Compute all quality metrics: confusion matrix, accuracy, precision, recall, and PSI."""
    matrix = compute_confusion_matrix(pairs, labels=labels)
    total_feedback = len(pairs)
    accuracy = compute_accuracy(matrix, total=total_feedback)
    precision, recall = compute_precision_recall(matrix, labels=labels)
    label_metrics = compute_label_metrics(matrix, labels=labels)

    drift_status = "baseline_missing"
    psi = None
    if recent_predictions_counts is not None:
        drift_status, psi = get_drift_status(
            actual_counts=recent_predictions_counts,
            expected_counts=expected_eval_counts,
            baseline_path=baseline_path,
            labels=labels,
        )

    most_confused = find_most_confused_pair(matrix, labels=labels)

    return {
        "sample_size": total_feedback,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "per_label": label_metrics,
        "confusion_matrix": matrix,
        "most_confused_pair": (
            {
                "predicted": most_confused[0],
                "actual": most_confused[1],
                "count": most_confused[2],
            }
            if most_confused
            else None
        ),
        "psi": psi,
        "drift_status": drift_status,
    }

    return {
        "sample_size": total_feedback,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "per_label": label_metrics,
        "confusion_matrix": matrix,
        "most_confused_pair": (
            {
                "predicted": most_confused[0],
                "actual": most_confused[1],
                "count": most_confused[2],
            }
            if most_confused
            else None
        ),
        "psi": psi,
    }
