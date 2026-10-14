"""Evaluation and benchmarking module comparing rule-based and ML triage models."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Sequence

from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split

from . import LABELS
from .model import TriageClassifier, load_dataset
from .rules import predict_rule


def evaluate_predictions(
    y_true: Sequence[str], y_pred: Sequence[str]
) -> dict[str, Any]:
    """Calculate accuracy, macro metrics, and per-class precision/recall/F1.

    Args:
        y_true: Ground truth labels.
        y_pred: Predicted labels.

    Returns:
        Dictionary containing overall accuracy, macro metrics, and per-class stats.
    """
    acc = float(accuracy_score(y_true, y_pred))
    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )

    p_per, r_per, f1_per, sup_per = precision_recall_fscore_support(
        y_true, y_pred, labels=list(LABELS), average=None, zero_division=0
    )

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
    test_size: float = 0.2,
    random_state: int = 42,
) -> dict[str, Any]:
    """Train/test split and compare Rule-Based vs ML classifier."""
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
    model = TriageClassifier()
    model.fit(X_train, y_train)
    ml_metrics = evaluate_model(model, X_test, y_test)

    # 3. ML cross-validation over entire dataset
    cv_metrics = evaluate_cross_validation(texts, labels)

    return {
        "test_size": test_size,
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "rule_baseline": rule_metrics,
        "ml_model": ml_metrics,
        "ml_cross_validation": cv_metrics,
    }


def format_comparison_table(results: dict[str, Any]) -> str:
    """Format evaluation comparison results as a human-readable table."""
    rule = results["rule_baseline"]
    ml = results["ml_model"]
    cv = results["ml_cross_validation"]

    lines = [
        "=" * 68,
        "           SUPPORTPILOT TICKET TRIAGE EVALUATION REPORT",
        "=" * 68,
        f" Dataset: {results['train_samples']} Train | {results['test_samples']} Test (Total: {results['train_samples'] + results['test_samples']})",
        "-" * 68,
        f" {'Metric':<24} | {'Rule Baseline':<18} | {'ML Model (Held-Out)':<18}",
        "-" * 68,
        f" {'Accuracy':<24} | {rule['accuracy']:<18.2%} | {ml['accuracy']:<18.2%}",
        f" {'Macro Precision':<24} | {rule['macro_precision']:<18.4f} | {ml['macro_precision']:<18.4f}",
        f" {'Macro Recall':<24} | {rule['macro_recall']:<18.4f} | {ml['macro_recall']:<18.4f}",
        f" {'Macro F1':<24} | {rule['macro_f1']:<18.4f} | {ml['macro_f1']:<18.4f}",
        "-" * 68,
        f" ML 5-Fold CV Accuracy: {cv['cv_accuracy_mean']:.2%} (+/- {cv['cv_accuracy_std']:.2%})",
        f" ML 5-Fold CV Macro F1: {cv['cv_f1_macro_mean']:.4f} (+/- {cv['cv_f1_macro_std']:.4f})",
        "-" * 68,
        " Per-Class F1 Breakdown (Held-Out Test Set):",
        f" {'Label':<14} | {'Support':<8} | {'Rule F1':<12} | {'ML F1':<12}",
        "-" * 68,
    ]

    for label in LABELS:
        r_f1 = rule["per_class"][label]["f1"]
        m_f1 = ml["per_class"][label]["f1"]
        sup = rule["per_class"][label]["support"]
        lines.append(f" {label:<14} | {sup:<8} | {r_f1:<12.4f} | {m_f1:<12.4f}")

    lines.append("=" * 68)
    return "\n".join(lines)


def main() -> int:
    """CLI execution for evaluation comparison."""
    parser = argparse.ArgumentParser(
        description="Evaluate SupportPilot ticket triage classifiers."
    )
    parser.add_argument(
        "--data",
        default="data/triage_labeled.jsonl",
        type=Path,
        help="Path to labeled JSONL dataset.",
    )
    parser.add_argument(
        "--test-size",
        default=0.2,
        type=float,
        help="Fraction of dataset to reserve for test split (default: 0.2).",
    )
    args = parser.parse_args()

    results = compare_classifiers(data_path=args.data, test_size=args.test_size)
    print(format_comparison_table(results))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
