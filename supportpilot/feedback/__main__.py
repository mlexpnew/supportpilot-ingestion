"""CLI entry point for SupportPilot feedback loop operations (SP-110 Part D).

Usage:
    python -m supportpilot.feedback export --output data/feedback_export.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .retrain import join_feedback_for_retraining
from .store import FeedbackStore, get_feedback_db_path


def main() -> int:
    parser = argparse.ArgumentParser(
        description="SupportPilot agent feedback management CLI (SP-110)."
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # export command
    export_parser = subparsers.add_parser(
        "export", help="Export agent feedback records for model retraining."
    )
    export_parser.add_argument(
        "--output",
        "-o",
        type=Path,
        required=True,
        help="Path to output JSONL file (e.g. data/feedback_export.jsonl).",
    )
    export_parser.add_argument(
        "--db-path",
        type=Path,
        default=None,
        help="Path to SQLite feedback database (defaults to FEEDBACK_DB_PATH).",
    )

    # join command
    join_parser = subparsers.add_parser(
        "join",
        help="Join feedback export with original and redacted pilot tickets for retraining.",
    )
    join_parser.add_argument(
        "--feedback",
        "-f",
        type=Path,
        required=True,
        help="Path to feedback export JSONL file (e.g. data/feedback_export.jsonl).",
    )
    join_parser.add_argument(
        "--original",
        type=Path,
        default=Path("data/triage_labeled.jsonl"),
        help="Path to original labeled dataset (default: data/triage_labeled.jsonl).",
    )
    join_parser.add_argument(
        "--redacted",
        "-r",
        type=Path,
        default=None,
        help="Path to redacted pilot tickets (default: out_redacted/redacted.jsonl if present).",
    )
    join_parser.add_argument(
        "--output",
        "-o",
        type=Path,
        required=True,
        help="Path to destination retraining dataset JSONL.",
    )

    args = parser.parse_args()

    if args.command == "export":
        db_path = args.db_path or get_feedback_db_path()
        store = FeedbackStore(db_path=db_path)
        try:
            count = store.export_feedback(args.output)
            print(
                f"Exported {count} feedback rows to {args.output}",
                file=sys.stderr,
            )
            summary = {
                "command": "export",
                "exported_records": count,
                "output": str(args.output),
                "status": "ok",
            }
            print(json.dumps(summary, ensure_ascii=False))
            return 0
        except Exception as exc:  # noqa: BLE001
            print(f"Error exporting feedback: {exc}", file=sys.stderr)
            return 1

    if args.command == "join":
        try:
            report = join_feedback_for_retraining(
                original_dataset=args.original,
                feedback_export=args.feedback,
                output_dataset=args.output,
                redacted_tickets=args.redacted,
            )
            summary = {
                "command": "join",
                "matched": report.matched,
                "appended": report.appended,
                "unmatched": report.unmatched,
                "output": str(args.output),
                "status": "ok",
            }
            print(json.dumps(summary, ensure_ascii=False))
            return 0
        except Exception as exc:  # noqa: BLE001
            print(f"Error executing retraining join: {exc}", file=sys.stderr)
            return 1

    parser.print_help(sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
