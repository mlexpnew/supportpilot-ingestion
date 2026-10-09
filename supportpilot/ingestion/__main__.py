import argparse
import json
import sys
import time
from pathlib import Path

from .loader import process_file

EXIT_OK = 0
EXIT_CRASH = 1
EXIT_BAD_OR_EMPTY_FILE = 2
EXIT_REJECT_RATE_TOO_HIGH = 3


def _emit_summary(
    total: int,
    valid: int,
    invalid: int,
    reject_rate: float,
    elapsed: float,
    exit_reason: str,
) -> None:
    """Print one JSON summary line to stdout containing operational metrics only."""
    summary = {
        "total": total,
        "valid": valid,
        "invalid": invalid,
        "reject_rate": round(reject_rate, 4),
        "elapsed": round(elapsed, 4),
        "exit_reason": exit_reason,
    }
    print(json.dumps(summary, ensure_ascii=False))


def main() -> int:
    start_time = time.monotonic()

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
        "--max-reject-rate",
        type=float,
        default=0.5,
        help="Maximum allowed fraction of rejected records (default: 0.5).",
    )

    parser.add_argument(
        "--min-records",
        type=int,
        default=20,
        help="Minimum total records required before evaluating --max-reject-rate (default: 20).",
    )

    parser.add_argument(
        "--fail-on-rejects",
        action="store_true",
        help="Exit with non-zero status (code 3) if any invalid records are encountered.",
    )

    args = parser.parse_args()

    if not args.input.exists():
        elapsed = time.monotonic() - start_time
        print(f"Error: input file not found: {args.input}", file=sys.stderr)
        _emit_summary(0, 0, 0, 0.0, elapsed, "bad_file")
        return EXIT_BAD_OR_EMPTY_FILE

    if not args.input.is_file():
        elapsed = time.monotonic() - start_time
        print(f"Error: input path is not a file: {args.input}", file=sys.stderr)
        _emit_summary(0, 0, 0, 0.0, elapsed, "bad_file")
        return EXIT_BAD_OR_EMPTY_FILE

    try:
        report = process_file(args.input, args.output_dir)
    except ValueError as exc:
        elapsed = time.monotonic() - start_time
        print(f"File error: {exc}", file=sys.stderr)
        msg = str(exc).lower()
        if "empty" in msg:
            reason = "empty_file"
        elif "blank" in msg:
            reason = "blank_only_file"
        elif "utf-16 without bom" in msg:
            reason = "unsupported_encoding"
        else:
            reason = "bad_file"
        _emit_summary(0, 0, 0, 0.0, elapsed, reason)
        return EXIT_BAD_OR_EMPTY_FILE
    except OSError as exc:
        elapsed = time.monotonic() - start_time
        print(f"Error processing files: {exc}", file=sys.stderr)
        _emit_summary(0, 0, 0, 0.0, elapsed, "bad_file")
        return EXIT_BAD_OR_EMPTY_FILE
    except Exception as exc:
        elapsed = time.monotonic() - start_time
        print(f"Fatal crash: {exc}", file=sys.stderr)
        _emit_summary(0, 0, 0, 0.0, elapsed, "crash")
        return EXIT_CRASH

    total = report["total_records"]
    valid = report["valid_records"]
    invalid = report["invalid_records"]
    reject_rate = round(invalid / total, 4) if total > 0 else 0.0
    elapsed = time.monotonic() - start_time

    print("Ticket ingestion completed successfully.")
    print(f"Total records: {total}")
    print(f"Valid records: {valid}")
    print(f"Invalid records: {invalid}")
    print(f"Reject rate: {reject_rate:.4f}")
    print(f"Blank lines skipped: {report['blank_lines_skipped']}")
    print(f"Output directory: {args.output_dir}")

    if args.fail_on_rejects and invalid > 0:
        print(
            "Rejection guard triggered: invalid records encountered with --fail-on-rejects.",
            file=sys.stderr,
        )
        _emit_summary(total, valid, invalid, reject_rate, elapsed, "fail_on_rejects")
        return EXIT_REJECT_RATE_TOO_HIGH

    if total >= args.min_records and reject_rate > args.max_reject_rate:
        print(
            f"Rejection guard triggered: reject rate {reject_rate:.4f} exceeds threshold {args.max_reject_rate}.",
            file=sys.stderr,
        )
        _emit_summary(
            total, valid, invalid, reject_rate, elapsed, "reject_rate_too_high"
        )
        return EXIT_REJECT_RATE_TOO_HIGH

    _emit_summary(total, valid, invalid, reject_rate, elapsed, "ok")
    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
