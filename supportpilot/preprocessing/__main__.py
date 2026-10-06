"""CLI entry point for SupportPilot PII redaction (SP-102)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .redact import process_file


def main() -> int:
    """Run the PII redaction CLI."""
    parser = argparse.ArgumentParser(
        description="Redact PII from SupportPilot ticket text (SP-102)."
    )

    parser.add_argument(
        "--input",
        required=True,
        type=Path,
        help="Path to the input JSONL file.",
    )

    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="Directory where output files will be written.",
    )

    args = parser.parse_args()

    if not args.input.exists():
        print(f"Error: input file not found: {args.input}", file=sys.stderr)
        return 1

    if not args.input.is_file():
        print(f"Error: input path is not a file: {args.input}", file=sys.stderr)
        return 1

    try:
        report = process_file(args.input, args.output_dir)
    except ValueError as exc:
        print(f"Path conflict: {exc}", file=sys.stderr)
        return 1
    except OSError as exc:
        print(f"Error processing files: {exc}", file=sys.stderr)
        return 1

    print("PII redaction completed successfully.")
    print(f"Total records: {report['total_records']}")
    print(f"Total redactions: {report['total_redactions']}")
    print(f"Card redactions: {report['card']}")
    print(f"Email redactions: {report['email']}")
    print(f"Phone redactions: {report['phone']}")
    print(f"Output directory: {args.output_dir}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
