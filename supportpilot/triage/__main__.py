"""CLI entry point for SupportPilot ticket triage classification (SP-105)."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

from . import LABELS
from .evaluate import compare_classifiers, format_comparison_table
from .model import DEFAULT_MODEL_PATH, TriageClassifier, train_model
from .rules import predict_rule


def process_triage_file(
    input_path: Path,
    output_dir: Path,
    method: str = "model",
    model_path: Path | None = None,
) -> dict[str, Any]:
    """Process a JSONL file of tickets and write triaged outputs.

    Args:
        input_path: Path to input JSONL file.
        output_dir: Directory where triaged.jsonl and triage_report.json are written.
        method: Classification method ('model' or 'rules').
        model_path: Optional path to serialized model file.

    Returns:
        Summary report dictionary.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    triaged_file = output_dir / "triaged.jsonl"
    report_file = output_dir / "triage_report.json"

    classifier: TriageClassifier | None = None
    if method == "model":
        m_path = model_path or DEFAULT_MODEL_PATH
        if m_path.exists():
            classifier = TriageClassifier.load(m_path)
        else:
            # Auto-train default model if labeled dataset exists
            dataset_path = Path("data/triage_labeled.jsonl")
            if dataset_path.exists():
                classifier = train_model(dataset_path, save_path=m_path)
            else:
                raise FileNotFoundError(
                    f"Model file not found at {m_path} and dataset data/triage_labeled.jsonl is missing."
                )

    counts: dict[str, int] = {label: 0 for label in LABELS}
    total_records = 0

    tmp_triaged = output_dir / ".triaged.jsonl.tmp"

    with open(input_path, "r", encoding="utf-8") as in_f, open(
        tmp_triaged, "w", encoding="utf-8"
    ) as out_f:
        for idx, line in enumerate(in_f, 1):
            line_str = line.strip()
            if not line_str:
                continue

            try:
                ticket = json.loads(line_str)
            except json.JSONDecodeError as exc:
                tmp_triaged.unlink(missing_ok=True)
                raise ValueError(f"Line {idx}: Invalid JSON: {exc}") from exc

            subject = ticket.get("subject")
            body = ticket.get("body", "")
            text_to_classify = f"{subject} {body}".strip() if subject else body

            if method == "rules":
                pred_label = predict_rule(text_to_classify)
                confidence = None
            else:
                assert classifier is not None
                pred_label = classifier.predict(text_to_classify)
                probs = classifier.predict_proba(text_to_classify)
                confidence = round(probs.get(pred_label, 0.0), 4)

            ticket["predicted_label"] = pred_label
            if confidence is not None:
                ticket["confidence"] = confidence

            counts[pred_label] = counts.get(pred_label, 0) + 1
            total_records += 1

            out_f.write(json.dumps(ticket, ensure_ascii=False) + "\n")

    # Atomic rename
    tmp_triaged.replace(triaged_file)

    report = {
        "total_records": total_records,
        "method": method,
        "counts": counts,
        "output_file": str(triaged_file),
    }

    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    return report


def main() -> int:
    """Run SupportPilot triage CLI."""
    parser = argparse.ArgumentParser(
        description="Classify SupportPilot tickets into triage categories (SP-105)."
    )

    parser.add_argument(
        "--input",
        type=Path,
        help="Path to input JSONL tickets file.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Directory to write triaged.jsonl and triage_report.json.",
    )
    parser.add_argument(
        "--method",
        choices=["model", "rules"],
        default="model",
        help="Classification method to use (default: model).",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=DEFAULT_MODEL_PATH,
        help="Path to trained model file (default: models/triage_model.joblib).",
    )
    parser.add_argument(
        "--predict",
        type=str,
        help="Predict triage label for a single ticket body string.",
    )
    parser.add_argument(
        "--train",
        action="store_true",
        help="Train and save model from --data.",
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("data/triage_labeled.jsonl"),
        help="Dataset path for training or evaluation (default: data/triage_labeled.jsonl).",
    )
    parser.add_argument(
        "--evaluate",
        action="store_true",
        help="Run benchmark evaluation comparing Rule vs ML classifiers.",
    )

    args = parser.parse_args()

    # Mode 1: Single string prediction
    if args.predict:
        if args.method == "rules":
            pred = predict_rule(args.predict)
            output = {"text": args.predict, "label": pred, "method": "rules"}
        else:
            if not args.model_path.exists():
                train_model(args.data, save_path=args.model_path)
            classifier = TriageClassifier.load(args.model_path)
            pred = classifier.predict(args.predict)
            probs = classifier.predict_proba(args.predict)
            output = {
                "text": args.predict,
                "label": pred,
                "confidence": round(probs.get(pred, 0.0), 4),
                "method": "model",
            }
        print(json.dumps(output, ensure_ascii=False))
        return 0

    # Mode 2: Train model
    if args.train:
        start_time = time.monotonic()
        train_model(args.data, save_path=args.model_path)
        elapsed = round(time.monotonic() - start_time, 4)
        print(
            f"Triage model trained on {args.data} and saved to {args.model_path} in {elapsed}s."
        )
        return 0

    # Mode 3: Evaluate models
    if args.evaluate:
        results = compare_classifiers(data_path=args.data)
        print(format_comparison_table(results))
        return 0

    # Mode 4: Batch file processing
    if args.input and args.output_dir:
        if not args.input.exists():
            print(f"Error: input file not found: {args.input}", file=sys.stderr)
            return 1
        if not args.input.is_file():
            print(f"Error: input path is not a file: {args.input}", file=sys.stderr)
            return 1

        try:
            report = process_triage_file(
                input_path=args.input,
                output_dir=args.output_dir,
                method=args.method,
                model_path=args.model_path,
            )
        except Exception as exc:
            print(f"Triage error: {exc}", file=sys.stderr)
            return 1

        print("Ticket triage completed successfully.")
        print(f"Method: {report['method']}")
        print(f"Total records: {report['total_records']}")
        for label, count in report["counts"].items():
            print(f"  {label:<12}: {count}")
        print(f"Output directory: {args.output_dir}")
        return 0

    parser.print_help(sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
