"""Retraining pipeline with Champion/Challenger evaluation and promotion (SP-113).

Usage:
    python -m supportpilot.triage.retrain --base data/triage_labeled.jsonl --feedback data/feedback_export.jsonl --pilot data/sample_tickets.jsonl --out models/ --expected-champion-version "$CURRENT_VER"
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import logging
import os
from pathlib import Path
import re
import tempfile
from typing import Any

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
)

from supportpilot.feedback.retrain import join_feedback_for_retraining
from supportpilot.feedback.store import FeedbackStore
from supportpilot.triage import LABELS
from supportpilot.triage.model import (
    DEFAULT_CONFIDENCE_THRESHOLD,
    DEFAULT_MODEL_PATH,
    TriageClassifier,
    is_valid_model_version,
    load_dataset,
    train_model,
)

logger = logging.getLogger(__name__)

DEFAULT_EVAL_PATH = Path("data/eval_frozen.jsonl")
DEFAULT_REPORTS_DIR = Path("reports")


@dataclass
class PromotionDecision:
    """Outcome and detailed explanation of the Champion/Challenger promotion rule."""

    promoted: bool
    f1_diff: float
    min_recall_diff: float
    f1_improved: bool
    no_recall_regressed: bool
    regressed_labels: list[str]
    reasons: list[str]

    @property
    def winner(self) -> str:
        return "challenger" if self.promoted else "champion"


def normalize_text(text: str) -> str:
    """Normalize text for leakage detection: lowercase and collapse consecutive whitespace."""
    return " ".join(text.lower().split())


def bump_model_version(
    champion_version: str | None = None,
    date_str: str | None = None,
) -> str:
    """Bump minor version automatically adhering to triage-v<semver>-<YYYYMMDD>.

    Example:
        triage-v1.0.0-20261026 -> triage-v1.1.0-YYYYMMDD
    """
    if not champion_version or not champion_version.strip():
        raise ValueError(
            "Champion version is required to bump version. Cannot be empty or None."
        )

    if date_str is None:
        today = datetime.now(timezone.utc).strftime("%Y%m%d")
    else:
        today = date_str.replace("-", "").strip()

    match = re.search(r"(\d+)\.(\d+)\.(\d+)", champion_version)
    if match:
        major = int(match.group(1))
        minor = int(match.group(2))
        return f"triage-v{major}.{minor + 1}.0-{today}"

    raise ValueError(
        f"Champion version '{champion_version}' does not contain a valid semver pattern (X.Y.Z)."
    )


def verify_eval_frozen_isolation(
    training_path: Path | str,
    eval_path: Path | str,
) -> None:
    """Verify that the frozen evaluation set has zero overlap with training data.

    Rohan's rule: 'Never evaluate on data the model trained on. The frozen set is sacred.'
    Raises RuntimeError immediately if any ticket ID or body text leaks into training.
    """
    t_path = Path(training_path)
    e_path = Path(eval_path)
    if not t_path.exists():
        raise FileNotFoundError(
            f"Training dataset not found for leakage check: {t_path}"
        )
    if not e_path.exists():
        raise FileNotFoundError(
            f"Frozen evaluation dataset not found for leakage check: {e_path}"
        )

    train_ids: set[str] = set()
    train_texts: set[str] = set()
    with open(t_path, "r", encoding="utf-8") as f:
        for line in f:
            line_str = line.strip()
            if line_str:
                item = json.loads(line_str)
                if "ticket_id" in item and item["ticket_id"]:
                    train_ids.add(str(item["ticket_id"]).strip())
                if "body" in item and item["body"]:
                    train_texts.add(normalize_text(item["body"]))

    with open(e_path, "r", encoding="utf-8") as f:
        for line in f:
            line_str = line.strip()
            if line_str:
                item = json.loads(line_str)
                tid = str(item["ticket_id"]).strip() if item.get("ticket_id") else None
                body = item.get("body")
                if tid and tid in train_ids:
                    raise RuntimeError(
                        f"Data leakage detected! Sacred evaluation ticket {tid} "
                        f"found in training data {t_path}."
                    )
                if body and normalize_text(body) in train_texts:
                    raise RuntimeError(
                        f"Data leakage detected! Sacred evaluation ticket text "
                        f"found in training data {t_path}."
                    )


def evaluate_model_on_dataset(
    model: TriageClassifier,
    dataset_path: Path | str,
    threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
) -> dict[str, Any]:
    """Evaluate a triage classifier on a labeled evaluation dataset.

    Returns:
        dict containing macro F1, macro recall, per-label recall, needs_review_rate,
        accuracy, and total samples.
    """
    path = Path(dataset_path)
    if not path.exists():
        raise FileNotFoundError(f"Evaluation dataset not found: {path}")

    _, texts, labels = load_dataset(path)
    if not texts:
        raise ValueError(f"Evaluation dataset is empty: {path}")

    preds = model.predict_batch(texts)
    acc = float(accuracy_score(labels, preds))

    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(
        labels, preds, average="macro", zero_division=0
    )
    p_per, r_per, f1_per, sup_per = precision_recall_fscore_support(
        labels, preds, labels=list(LABELS), average=None, zero_division=0
    )

    per_label_recall = {
        label: round(float(r_per[i]), 4) for i, label in enumerate(LABELS)
    }
    per_label_precision = {
        label: round(float(p_per[i]), 4) for i, label in enumerate(LABELS)
    }
    per_label_f1 = {label: round(float(f1_per[i]), 4) for i, label in enumerate(LABELS)}
    per_label_support = {label: int(sup_per[i]) for i, label in enumerate(LABELS)}

    needs_review_count = 0
    for text in texts:
        res = model.predict_one(text, threshold=threshold)
        if res.get("needs_review", False):
            needs_review_count += 1

    needs_review_rate = round(needs_review_count / len(texts), 4)

    return {
        "total_samples": len(texts),
        "accuracy": round(acc, 4),
        "macro_precision": round(float(p_macro), 4),
        "macro_recall": round(float(r_macro), 4),
        "macro_f1": round(float(f1_macro), 4),
        "per_label_recall": per_label_recall,
        "per_label_precision": per_label_precision,
        "per_label_f1": per_label_f1,
        "per_label_support": per_label_support,
        "needs_review_count": needs_review_count,
        "needs_review_rate": needs_review_rate,
        "confidence_threshold": threshold,
    }


def evaluate_promotion_rule(
    champion_metrics: dict[str, Any],
    challenger_metrics: dict[str, Any],
    min_f1_improvement: float = 0.02,
    max_recall_drop: float = 0.05,
) -> PromotionDecision:
    """Evaluate whether the challenger model should replace the champion.

    Promotion rule (Rohan):
        The challenger wins only if:
        1. Macro F1 improves by >= 0.02.
        2. No label's recall drops by more than 0.05.
    """
    champ_f1 = float(champion_metrics.get("macro_f1", 0.0))
    chall_f1 = float(challenger_metrics.get("macro_f1", 0.0))
    f1_diff = round(chall_f1 - champ_f1, 4)

    f1_improved = f1_diff >= min_f1_improvement

    champ_recalls = champion_metrics.get("per_label_recall", {})
    chall_recalls = challenger_metrics.get("per_label_recall", {})

    regressed_labels: list[str] = []
    recall_diffs: list[float] = []
    reasons: list[str] = []

    for label in LABELS:
        c_rec = float(champ_recalls.get(label, 0.0))
        ch_rec = float(chall_recalls.get(label, 0.0))
        diff = round(ch_rec - c_rec, 4)
        recall_diffs.append(diff)
        drop = round(c_rec - ch_rec, 4)
        if drop > max_recall_drop:
            regressed_labels.append(label)

    no_recall_regressed = len(regressed_labels) == 0
    min_recall_diff = min(recall_diffs) if recall_diffs else 0.0
    promoted = f1_improved and no_recall_regressed

    if not f1_improved:
        reasons.append(
            f"Macro F1 improvement ({f1_diff:+.4f}) is below required threshold (+{min_f1_improvement:.4f})"
        )
    else:
        reasons.append(
            f"Macro F1 improved by {f1_diff:+.4f} (>= +{min_f1_improvement:.4f})"
        )

    if not no_recall_regressed:
        reasons.append(
            f"Recall dropped by > {max_recall_drop:.2%} on label(s): {', '.join(regressed_labels)}"
        )
    else:
        reasons.append(f"No label recall dropped by more than {max_recall_drop:.2%}")

    return PromotionDecision(
        promoted=promoted,
        f1_diff=f1_diff,
        min_recall_diff=min_recall_diff,
        f1_improved=f1_improved,
        no_recall_regressed=no_recall_regressed,
        regressed_labels=regressed_labels,
        reasons=reasons,
    )


def format_summary_table(
    champion_version: str,
    champion_metrics: dict[str, Any],
    challenger_version: str,
    challenger_metrics: dict[str, Any],
    decision: PromotionDecision,
    eval_dataset: str,
) -> str:
    """Format evaluation comparison results into a clean, readable text table."""
    c_f1 = champion_metrics["macro_f1"]
    ch_f1 = challenger_metrics["macro_f1"]
    f1_delta = decision.f1_diff

    c_rec = champion_metrics["macro_recall"]
    ch_rec = challenger_metrics["macro_recall"]
    rec_delta = round(ch_rec - c_rec, 4)

    c_prec = champion_metrics["macro_precision"]
    ch_prec = challenger_metrics["macro_precision"]
    prec_delta = round(ch_prec - c_prec, 4)

    c_rev = champion_metrics["needs_review_rate"]
    ch_rev = challenger_metrics["needs_review_rate"]
    rev_delta = round(ch_rev - c_rev, 4)

    lines = [
        "=" * 80,
        "                 SUPPORTPILOT RETRAINING EVALUATION REPORT",
        "=" * 80,
        f" Frozen Evaluation Set: {eval_dataset} ({champion_metrics['total_samples']} samples)",
        f" Champion:   {champion_version}",
        f" Challenger: {challenger_version}",
        "-" * 80,
        f" {'Metric':<22} | {'Champion':<14} | {'Challenger':<14} | {'Delta':<18}",
        "-" * 80,
        f" {'Macro F1':<22} | {c_f1:<14.2%} | {ch_f1:<14.2%} | {f1_delta:>+7.2%} (min +2.00%)",
        f" {'Macro Recall':<22} | {c_rec:<14.2%} | {ch_rec:<14.2%} | {rec_delta:>+7.2%}",
        f" {'Macro Precision':<22} | {c_prec:<14.2%} | {ch_prec:<14.2%} | {prec_delta:>+7.2%}",
        f" {'Needs Review Rate':<22} | {c_rev:<14.2%} | {ch_rev:<14.2%} | {rev_delta:>+7.2%}",
        "-" * 80,
        " Per-Label Recall Comparison:",
        f" {'Label':<15} | {'Champion':<10} | {'Challenger':<10} | {'Delta':<10} | {'Status':<16}",
        "-" * 80,
    ]

    c_recalls = champion_metrics["per_label_recall"]
    ch_recalls = challenger_metrics["per_label_recall"]

    for label in LABELS:
        c_r = c_recalls.get(label, 0.0)
        ch_r = ch_recalls.get(label, 0.0)
        d = round(ch_r - c_r, 4)
        status = "FAIL (Drop > 5%)" if label in decision.regressed_labels else "PASS"
        lines.append(
            f" {label:<15} | {c_r:<10.2%} | {ch_r:<10.2%} | {d:>+8.2%} | {status:<16}"
        )

    lines.append("-" * 80)
    decision_header = (
        "PROMOTED (Challenger WON)"
        if decision.promoted
        else "REJECTED (Champion RETAINED)"
    )
    lines.append(f" PROMOTION DECISION: {decision_header}")
    for r in decision.reasons:
        lines.append(f"   • {r}")
    lines.append("=" * 80)

    return "\n".join(lines)


def retrain_pipeline(
    base_dataset: Path | str = "data/triage_labeled.jsonl",
    feedback_export: Path | str | None = None,
    pilot_tickets: Path | str | None = None,
    output_dir: Path | str = "models",
    eval_dataset: Path | str = DEFAULT_EVAL_PATH,
    champion_path: Path | str = DEFAULT_MODEL_PATH,
    reports_dir: Path | str = DEFAULT_REPORTS_DIR,
    expected_champion_version: str | None = None,
) -> dict[str, Any]:
    """Execute end-to-end retraining pipeline with Champion/Challenger evaluation.

    1. Export feedback from database if needed.
    2. Join feedback with base and pilot tickets.
    3. Train challenger model and bump version automatically.
    4. Save challenger next to champion (never overwrite champion).
    5. Evaluate both models on sacred frozen eval set.
    6. Apply promotion rule and write comparison report.
    """
    base_p = Path(base_dataset)
    if not base_p.exists():
        raise FileNotFoundError(f"Base dataset not found: {base_p}")

    eval_p = Path(eval_dataset)
    if not eval_p.exists():
        raise FileNotFoundError(f"Frozen evaluation dataset not found: {eval_p}")

    out_d = Path(output_dir)
    out_d.mkdir(parents=True, exist_ok=True)

    rep_d = Path(reports_dir)
    rep_d.mkdir(parents=True, exist_ok=True)

    # 1. Prepare training data (join feedback if provided)
    training_data_path = base_p
    cleanup_temp_data = False

    if feedback_export is not None and pilot_tickets is not None:
        fb_p = Path(feedback_export)
        pilot_p = Path(pilot_tickets)

        # Export feedback if file does not exist yet
        if not fb_p.exists():
            store = FeedbackStore()
            try:
                count = store.export_feedback(fb_p)
                logger.info("Exported %d feedback rows to %s", count, fb_p)
            finally:
                store.close()

        joined_fd, joined_file = tempfile.mkstemp(
            prefix="triage_retrain_joined_", suffix=".jsonl", dir=str(out_d)
        )
        os.close(joined_fd)
        join_feedback_for_retraining(
            original_dataset=base_p,
            feedback_export=fb_p,
            redacted_tickets=pilot_p,
            output_dataset=joined_file,
        )
        training_data_path = Path(joined_file)
        cleanup_temp_data = True

    try:
        # 2. Sacred frozen evaluation set verification (fail-fast if data leaks)
        verify_eval_frozen_isolation(training_data_path, eval_p)

        # 3. Load active champion (must exist; never invent a baseline during retraining)
        champ_p = Path(champion_path)
        if not champ_p.exists():
            raise FileNotFoundError(
                f"Champion model not found at: {champ_p}. "
                "Retraining requires comparing against an existing production champion model. "
                "Refusing to invent an unverified baseline champion."
            )

        champion = TriageClassifier.load(champ_p)
        champion_version = champion.version
        if not champion_version or not champion_version.strip():
            raise ValueError(
                f"Champion model at '{champ_p}' is missing a version in its metadata. "
                "Corrupt or unversioned champion."
            )

        if not expected_champion_version or not expected_champion_version.strip():
            logger.warning(
                "SAFETY WARNING: 'expected_champion_version' is missing! "
                "Retraining without verifying against the production champion version is dangerous."
            )
        elif champion_version != expected_champion_version.strip():
            raise ValueError(
                f"Champion version mismatch: expected '{expected_champion_version.strip()}' "
                f"(copied from production /version), but champion model at '{champ_p}' "
                f"has version '{champion_version}'. The comparison must use the exact "
                "model that production serves."
            )

        # 4. Bump version automatically for challenger
        challenger_version = bump_model_version(champion_version)

        # 5. Determine challenger save path (never overwrite champion)
        challenger_path = out_d / f"{challenger_version}.joblib"
        if challenger_path.resolve() == champ_p.resolve():
            challenger_path = out_d / f"challenger_{challenger_version}.joblib"

        # 6. Train challenger on updated dataset
        logger.info(
            "Training challenger %s on %s...", challenger_version, training_data_path
        )
        challenger = train_model(
            data_path=training_data_path,
            save_path=challenger_path,
            version=challenger_version,
        )

        # 7. Startup checks on challenger
        if not is_valid_model_version(challenger.version):
            raise ValueError(
                f"Challenger version '{challenger.version}' failed startup format validation."
            )
        challenger_labels = challenger.metadata.get("labels", [])
        if set(challenger_labels) != set(LABELS):
            raise ValueError(
                f"Challenger labels {challenger_labels} do not match service LABELS {list(LABELS)}."
            )

        # 8. Fair evaluation on frozen evaluation set
        champion_metrics = evaluate_model_on_dataset(champion, eval_p)
        challenger_metrics = evaluate_model_on_dataset(challenger, eval_p)

        # 9. Evaluate promotion rule
        decision = evaluate_promotion_rule(champion_metrics, challenger_metrics)

        # 10. Write comparison report
        today_date = datetime.now(timezone.utc).strftime("%Y%m%d")
        report_file = rep_d / f"compare_{today_date}.json"

        report_payload = {
            "date": datetime.now(timezone.utc).isoformat(),
            "eval_dataset": str(eval_p),
            "eval_samples": champion_metrics["total_samples"],
            "champion": {
                "version": champion_version,
                "path": str(champ_p),
                "macro_f1": champion_metrics["macro_f1"],
                "macro_recall": champion_metrics["macro_recall"],
                "macro_precision": champion_metrics["macro_precision"],
                "needs_review_rate": champion_metrics["needs_review_rate"],
                "per_label_recall": champion_metrics["per_label_recall"],
            },
            "challenger": {
                "version": challenger_version,
                "path": str(challenger_path),
                "macro_f1": challenger_metrics["macro_f1"],
                "macro_recall": challenger_metrics["macro_recall"],
                "macro_precision": challenger_metrics["macro_precision"],
                "needs_review_rate": challenger_metrics["needs_review_rate"],
                "per_label_recall": challenger_metrics["per_label_recall"],
            },
            "comparison": {
                "f1_improvement": decision.f1_diff,
                "f1_improved": decision.f1_improved,
                "no_recall_regressed": decision.no_recall_regressed,
                "regressed_labels": decision.regressed_labels,
                "per_label_recall_diff": {
                    lbl: round(
                        challenger_metrics["per_label_recall"][lbl]
                        - champion_metrics["per_label_recall"][lbl],
                        4,
                    )
                    for lbl in LABELS
                },
            },
            "decision": {
                "promoted": decision.promoted,
                "winner": decision.winner,
                "reasons": decision.reasons,
            },
        }

        with open(report_file, "w", encoding="utf-8") as rf:
            json.dump(report_payload, rf, indent=2, ensure_ascii=False)

        summary_table = format_summary_table(
            champion_version=champion_version,
            champion_metrics=champion_metrics,
            challenger_version=challenger_version,
            challenger_metrics=challenger_metrics,
            decision=decision,
            eval_dataset=str(eval_p),
        )

        return {
            "champion_version": champion_version,
            "challenger_version": challenger_version,
            "challenger_path": str(challenger_path),
            "champion_path": str(champ_p),
            "promoted": decision.promoted,
            "winner": decision.winner,
            "report_file": str(report_file),
            "summary_table": summary_table,
            "report_data": report_payload,
        }

    finally:
        if cleanup_temp_data and training_data_path.exists():
            try:
                os.unlink(training_data_path)
            except OSError:
                pass


def main() -> int:
    """CLI for SupportPilot ticket triage retraining pipeline."""
    parser = argparse.ArgumentParser(
        description="Retrain SupportPilot triage classifier with Champion/Challenger evaluation (SP-113)."
    )
    parser.add_argument(
        "--base",
        type=Path,
        default=Path("data/triage_labeled.jsonl"),
        help="Path to base labeled JSONL dataset.",
    )
    parser.add_argument(
        "--feedback",
        type=Path,
        default=None,
        help="Path to feedback export JSONL file.",
    )
    parser.add_argument(
        "--pilot",
        type=Path,
        default=None,
        help="Path to redacted pilot tickets JSONL file.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("models"),
        help="Destination directory for challenger model.",
    )
    parser.add_argument(
        "--eval",
        type=Path,
        default=DEFAULT_EVAL_PATH,
        help="Path to sacred frozen evaluation JSONL dataset.",
    )
    parser.add_argument(
        "--champion",
        type=Path,
        default=DEFAULT_MODEL_PATH,
        help="Path to active champion model file.",
    )
    parser.add_argument(
        "--reports",
        type=Path,
        default=DEFAULT_REPORTS_DIR,
        help="Directory to save compare_<date>.json reports.",
    )
    parser.add_argument(
        "--expected-champion-version",
        type=str,
        required=True,
        help="Expected model version of production champion (copied from /version). Retraining aborts on mismatch.",
    )

    args = parser.parse_args()

    results = retrain_pipeline(
        base_dataset=args.base,
        feedback_export=args.feedback,
        pilot_tickets=args.pilot,
        output_dir=args.out,
        eval_dataset=args.eval,
        champion_path=args.champion,
        reports_dir=args.reports,
        expected_champion_version=args.expected_champion_version,
    )

    print(results["summary_table"])
    print(f"\nDetailed JSON report written to: {results['report_file']}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
