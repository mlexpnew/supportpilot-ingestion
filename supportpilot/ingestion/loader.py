import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from .models import Ticket


def _reject(
    line_number: int,
    error_type: str,
    field: str | None = None,
) -> dict[str, Any]:
    """Create a sanitized rejection record without exposing PII."""
    return {
        "line_number": line_number,
        "field": field,
        "error_type": error_type,
    }


def _validation_rejection(
    line_number: int,
    error: ValidationError,
) -> dict[str, Any]:
    """Convert Pydantic errors into sanitized rejection information."""
    errors = error.errors()

    first_error = errors[0] if errors else {}
    location = first_error.get("loc", ())
    field = str(location[0]) if location else None

    error_type = first_error.get("type", "validation_error")

    return _reject(
        line_number=line_number,
        field=field,
        error_type=error_type,
    )


def validate_lines(
    input_path: Path,
) -> Iterator[tuple[int, Ticket | None, dict[str, Any] | None]]:
    """
    Stream and validate tickets from a JSONL file.

    Yields:
        (line_number, ticket, rejection)
    """
    seen_ticket_ids: set[str] = set()

    with input_path.open("r", encoding="utf-8") as file:
        for line_number, raw_line in enumerate(file, start=1):
            if not raw_line.strip():
                continue

            try:
                data = json.loads(raw_line)
            except json.JSONDecodeError:
                yield (
                    line_number,
                    None,
                    _reject(line_number, "invalid_json"),
                )
                continue

            if not isinstance(data, dict):
                yield (
                    line_number,
                    None,
                    _reject(line_number, "invalid_record_type"),
                )
                continue

            try:
                ticket = Ticket.model_validate(data)
            except ValidationError as error:
                yield (
                    line_number,
                    None,
                    _validation_rejection(line_number, error),
                )
                continue

            if ticket.ticket_id in seen_ticket_ids:
                yield (
                    line_number,
                    None,
                    _reject(
                        line_number,
                        "duplicate_ticket_id",
                        "ticket_id",
                    ),
                )
                continue

            seen_ticket_ids.add(ticket.ticket_id)

            yield line_number, ticket, None


def process_file(
    input_path: Path,
    output_dir: Path,
) -> dict[str, int]:
    """
    Process a JSONL ticket file using streaming input/output.

    Returns aggregate processing statistics.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    valid_path = output_dir / "valid.jsonl"
    rejects_path = output_dir / "rejects.jsonl"

    total_records = 0
    valid_records = 0
    invalid_records = 0
    blank_lines_skipped = 0

    with (
        input_path.open("r", encoding="utf-8") as input_file,
        valid_path.open("w", encoding="utf-8") as valid_file,
        rejects_path.open("w", encoding="utf-8") as rejects_file,
    ):
        seen_ticket_ids: set[str] = set()

        for line_number, raw_line in enumerate(input_file, start=1):
            if not raw_line.strip():
                blank_lines_skipped += 1
                continue

            total_records += 1

            try:
                data = json.loads(raw_line)
            except json.JSONDecodeError:
                invalid_records += 1
                rejects_file.write(
                    json.dumps(
                        _reject(line_number, "invalid_json"),
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                continue

            if not isinstance(data, dict):
                invalid_records += 1
                rejects_file.write(
                    json.dumps(
                        _reject(line_number, "invalid_record_type"),
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                continue

            try:
                ticket = Ticket.model_validate(data)
            except ValidationError as error:
                invalid_records += 1
                rejects_file.write(
                    json.dumps(
                        _validation_rejection(line_number, error),
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                continue

            if ticket.ticket_id in seen_ticket_ids:
                invalid_records += 1
                rejects_file.write(
                    json.dumps(
                        _reject(
                            line_number,
                            "duplicate_ticket_id",
                            "ticket_id",
                        ),
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                continue

            seen_ticket_ids.add(ticket.ticket_id)

            valid_records += 1
            valid_file.write(
                json.dumps(
                    ticket.model_dump(mode="json"),
                    ensure_ascii=False,
                )
                + "\n"
            )

    report = {
        "total_records": total_records,
        "valid_records": valid_records,
        "invalid_records": invalid_records,
        "blank_lines_skipped": blank_lines_skipped,
    }

    with (output_dir / "report.json").open(
        "w",
        encoding="utf-8",
    ) as report_file:
        json.dump(report, report_file, indent=2)

    return report
