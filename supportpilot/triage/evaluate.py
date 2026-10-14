"""Evaluation and benchmarking module comparing rule-based and ML triage models."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Sequence

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split

from . import LABELS
from .model import (
    DEFAULT_CONFIDENCE_THRESHOLD,
    TriageClassifier,
    load_dataset,
)
from .rules import predict_rule


def evaluate_predictions(
    y_true: Sequence[str], y_pred: Sequence[str]
) -> dict[str, Any]:
    """Calculate accuracy, macro metrics, confusion matrix, and per-class stats.

    Args:
        y_true: Ground truth labels.
        y_pred: Predicted labels.

    Returns:
        Dictionary containing overall accuracy, macro metrics, per-class stats,
        and confusion matrix.
    """
    acc = float(accuracy_score(y_true, y_pred))
    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )

    p_per, r_per, f1_per, sup_per = precision_recall_fscore_support(
        y_true, y_pred, labels=list(LABELS), average=None, zero_division=0
    )

    cm = confusion_matrix(y_true, y_pred, labels=list(LABELS)).tolist()

    per_class = {}
    for idx, label in enumerate(LABELS):
        per_class[label] = {
            "precision": round(float(p_per[idx]), 4),
            "recall": round(float(r_per[idx]), 4),
            "f1": round(float(f1_per[idx]), 4),
            "support": int(sup_per[idx]),
        }

    return {
        "accuracy": round(acc, 4),
        "macro_precision": round(float(p_macro), 4),
        "macro_recall": round(float(r_macro), 4),
        "macro_f1": round(float(f1_macro), 4),
        "total_samples": len(y_true),
        "confusion_matrix": cm,
        "per_class": per_class,
    }


def evaluate_rules(texts: Sequence[str], labels: Sequence[str]) -> dict[str, Any]:
    """Evaluate rule-based classifier against ground truth labels."""
    preds = [predict_rule(t) for t in texts]
    return evaluate_predictions(labels, preds)


def evaluate_model(
    model: TriageClassifier, texts: Sequence[str], labels: Sequence[str]
) -> dict[str, Any]:
    """Evaluate trained ML model against ground truth labels."""
    preds = model.predict_batch(texts)
    return evaluate_predictions(labels, preds)


def evaluate_cross_validation(
    texts: Sequence[str], labels: Sequence[str], n_splits: int = 5
) -> dict[str, float]:
    """Run stratified K-fold cross-validation on the ML pipeline."""
    clf = TriageClassifier()
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    cv_res = cross_validate(
        clf.pipeline,
        texts,
        labels,
        cv=cv,
        scoring=["accuracy", "f1_macro"],
    )
    return {
        "cv_accuracy_mean": round(float(cv_res["test_accuracy"].mean()), 4),
        "cv_accuracy_std": round(float(cv_res["test_accuracy"].std()), 4),
        "cv_f1_macro_mean": round(float(cv_res["test_f1_macro"].mean()), 4),
        "cv_f1_macro_std": round(float(cv_res["test_f1_macro"].std()), 4),
    }


def compare_classifiers(
    data_path: Path | str = "data/triage_labeled.jsonl",
    test_size: float = 0.3,
    random_state: int = 42,
    confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
) -> dict[str, Any]:
    """70/30 train/test split and compare Rule-Based vs ML classifier."""
    _, texts, labels = load_dataset(data_path)

    X_train, X_test, y_train, y_test = train_test_split(
        texts,
        labels,
        test_size=test_size,
        random_state=random_state,
        stratify=labels,
    )

    # 1. Rule-based evaluation on test set
    rule_metrics = evaluate_rules(X_test, y_test)

    # 2. ML model trained on train set, evaluated on test set
    model = TriageClassifier(confidence_threshold=confidence_threshold)
    model.fit(X_train, y_train)
    ml_metrics = evaluate_model(model, X_test, y_test)

    # 3. Confidence fallback analysis (predict_one & needs_review)
    one_preds = [model.predict_one(t, threshold=confidence_threshold) for t in X_test]
    flagged = [res["needs_review"] for res in one_preds]
    flagged_count = sum(flagged)
    flagged_pct = round(flagged_count / len(X_test), 4)

    confident_correct = sum(
        1
        for res, true_l in zip(one_preds, y_test)
        if not res["needs_review"] and res["label"] == true_l
    )
    confident_total = len(X_test) - flagged_count
    confident_acc = (
        round(confident_correct / confident_total, 4) if confident_total > 0 else 0.0
    )

    fallback_stats = {
        "threshold": confidence_threshold,
        "total_test": len(X_test),
        "flagged_for_review": flagged_count,
        "flagged_percentage": flagged_pct,
        "confident_count": confident_total,
        "confident_accuracy": confident_acc,
    }

    # 4. ML cross-validation over entire dataset
    cv_metrics = evaluate_cross_validation(texts, labels)

    return {
        "test_size": test_size,
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "rule_baseline": rule_metrics,
        "ml_model": ml_metrics,
        "fallback_analysis": fallback_stats,
        "ml_cross_validation": cv_metrics,
    }


def format_comparison_table(results: dict[str, Any]) -> str:
    """Format evaluation comparison results as a human-readable table."""
    rule = results["rule_baseline"]
    ml = results["ml_model"]
    cv = results["ml_cross_validation"]
    fb = results["fallback_analysis"]

    train_n = results["train_samples"]
    test_n = results["test_samples"]
    split_pct_train = int(round((train_n / (train_n + test_n)) * 100))
    split_pct_test = int(round((test_n / (train_n + test_n)) * 100))

    lines = [
        "=" * 72,
        "           SUPPORTPILOT TICKET TRIAGE EVALUATION REPORT",
        "=" * 72,
        f" Dataset Split: {split_pct_train}/{split_pct_test} ({train_n} Train | {test_n} Test | Total: {train_n + test_n})",
        "-" * 72,
        f" {'Metric':<26} | {'Rule Baseline':<18} | {'ML Model (Held-Out)':<18}",
        "-" * 72,
        f" {'Accuracy':<26} | {rule['accuracy']:<18.2%} | {ml['accuracy']:<18.2%}",
        f" {'Macro Precision':<26} | {rule['macro_precision']:<18.4f} | {ml['macro_precision']:<18.4f}",
        f" {'Macro Recall':<26} | {rule['macro_recall']:<18.4f} | {ml['macro_recall']:<18.4f}",
        f" {'Macro F1':<26} | {rule['macro_f1']:<18.4f} | {ml['macro_f1']:<18.4f}",
        "-" * 72,
        f" Confidence Fallback Threshold: {fb['threshold']:.2f}",
        f" Flagged for Human Review:      {fb['flagged_for_review']}/{fb['total_test']} ({fb['flagged_percentage']:.1%})",
        f" Confident Predictions Accuracy: {fb['confident_accuracy']:.2%} ({fb['confident_count']} tickets)",
        "-" * 72,
        f" ML 5-Fold CV Accuracy:         {cv['cv_accuracy_mean']:.2%} (+/- {cv['cv_accuracy_std']:.2%})",
        f" ML 5-Fold CV Macro F1:         {cv['cv_f1_macro_mean']:.4f} (+/- {cv['cv_f1_macro_std']:.4f})",
        "-" * 72,
        " Per-Class Performance (Held-Out 30% Test Set):",
        f" {'Label':<15} | {'Support':<8} | {'Rule F1':<12} | {'ML Precision':<12} | {'ML Recall':<10} | {'ML F1':<10}",
        "-" * 72,
    ]

    for label in LABELS:
        r_f1 = rule["per_class"][label]["f1"]
        m_p = ml["per_class"][label]["precision"]
        m_r = ml["per_class"][label]["recall"]
        m_f1 = ml["per_class"][label]["f1"]
        sup = rule["per_class"][label]["support"]
        lines.append(
            f" {label:<15} | {sup:<8} | {r_f1:<12.4f} | {m_p:<12.4f} | {m_r:<10.4f} | {m_f1:<10.4f}"
        )

    lines.append("-" * 72)
    lines.append(" ML Confusion Matrix (Rows: Ground Truth, Columns: Predicted):")
    header_labels = " ".join(f"{lbl[:5]:>6}" for lbl in LABELS)
    lines.append(f" {'Actual / Pred':<15} | {header_labels}")
    lines.append("-" * 72)

    cm = ml["confusion_matrix"]
    for idx, label in enumerate(LABELS):
        row_str = " ".join(f"{cm[idx][j]:>6d}" for j in range(len(LABELS)))
        lines.append(f" {label:<15} | {row_str}")

    lines.append("=" * 72)
    return "\n".join(lines)


def main() -> int:
    """CLI execution for evaluation comparison."""
    parser = argparse.ArgumentParser(
        description="Evaluate SupportPilot ticket triage classifiers (SP-105)."
    )
    parser.add_argument(
        "--data",
        default="data/triage_labeled.jsonl",
        type=Path,
        help="Path to labeled JSONL dataset.",
    )
    parser.add_argument(
        "--test-size",
        default=0.3,
        type=float,
        help="Fraction of dataset to reserve for test split (default: 0.3 for 70/30 split).",
    )
    parser.add_argument(
        "--threshold",
        default=DEFAULT_CONFIDENCE_THRESHOLD,
        type=float,
        help="Confidence threshold for human review fallback (default: 0.35).",
    )
    args = parser.parse_args()

    results = compare_classifiers(
        data_path=args.data,
        test_size=args.test_size,
        confidence_threshold=args.threshold,
    )
    print(format_comparison_table(results))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
