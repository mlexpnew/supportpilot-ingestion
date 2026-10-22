"""CLI entry point for SupportPilot feedback loop operations (SP-110 Part D).

Usage:
    python -m supportpilot.feedback export --output data/feedback_export.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .store import FeedbackStore, get_feedback_db_path


def main() -> int:
    parser = argparse.ArgumentParser(
        description="SupportPilot agent feedback management CLI (SP-110)."
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

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

    args = parser.parse_args()

    if args.command == "export":
        db_path = args.db_path or get_feedback_db_path()
        store = FeedbackStore(db_path=db_path)
        try:
            count = store.export_feedback(args.output)
            print(
                f"Exported {count} feedback record(s) to {args.output}",
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

    parser.print_help(sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
