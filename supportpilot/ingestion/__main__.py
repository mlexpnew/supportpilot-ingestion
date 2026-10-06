import argparse
import sys
from pathlib import Path

from .loader import process_file


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate and ingest SupportPilot ticket JSONL data."
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

    parser.add_argument(
        "--fail-on-rejects",
        action="store_true",
        help="Exit with non-zero status (code 2) if any invalid records are encountered.",
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
    except (OSError, ValueError) as exc:
        print(f"Error processing files: {exc}", file=sys.stderr)
        return 1

    print("Ticket ingestion completed successfully.")
    print(f"Total records: {report['total_records']}")
    print(f"Valid records: {report['valid_records']}")
    print(f"Invalid records: {report['invalid_records']}")
    print(f"Blank lines skipped: {report['blank_lines_skipped']}")
    print(f"Output directory: {args.output_dir}")

    if args.fail_on_rejects and report["invalid_records"] > 0:
        return 2

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
