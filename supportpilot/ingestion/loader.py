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
    ticket_id: str | None = None,
) -> dict[str, Any]:
    """Create a sanitized rejection record without exposing PII."""
    reject = {
        "line_number": line_number,
        "field": field,
        "error_type": error_type,
    }

    if ticket_id is not None:
        reject["ticket_id"] = ticket_id

    return reject


def _validation_rejection(
    line_number: int,
    error: ValidationError,
) -> dict[str, Any]:
    """Convert Pydantic errors into stable application-level errors."""
    errors = error.errors()

    first_error = errors[0] if errors else {}
    location = first_error.get("loc", ())
    field = str(location[0]) if location else None
    pydantic_type = first_error.get("type", "")

    if field == "created_at":
        error_type = "invalid_timestamp"
    elif field == "channel":
        error_type = "invalid_channel"
    elif field == "body" and pydantic_type == "value_error":
        error_type = "empty_body"
    elif pydantic_type == "missing":
        error_type = "missing_field"
    else:
        error_type = "validation_error"

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
                        ticket_id=ticket.ticket_id,
                    ),
                )
                continue

            seen_ticket_ids.add(ticket.ticket_id)

            yield line_number, ticket, None


def process_file(
    input_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
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
    rejection_reasons: dict[str, int] = {}

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

                rejection_reasons["invalid_json"] = (
                    rejection_reasons.get("invalid_json", 0) + 1
                )

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

                rejection_reasons["invalid_record_type"] = (
                    rejection_reasons.get("invalid_record_type", 0) + 1
                )

                rejects_file.write(
                    json.dumps(
                        _reject(
                            line_number,
                            "invalid_record_type",
                        ),
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                continue

            try:
                ticket = Ticket.model_validate(data)
            except ValidationError as error:
                invalid_records += 1

                reject = _validation_rejection(
                    line_number,
                    error,
                )

                rejection_reasons[reject["error_type"]] = (
                    rejection_reasons.get(
                        reject["error_type"],
                        0,
                    )
                    + 1
                )

                rejects_file.write(
                    json.dumps(
                        reject,
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                continue

            if ticket.ticket_id in seen_ticket_ids:
                invalid_records += 1

                rejection_reasons["duplicate_ticket_id"] = (
                    rejection_reasons.get(
                        "duplicate_ticket_id",
                        0,
                    )
                    + 1
                )

                rejects_file.write(
                    json.dumps(
                        _reject(
                            line_number,
                            "duplicate_ticket_id",
                            "ticket_id",
                            ticket_id=ticket.ticket_id,
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
        "blank_lines_skipped": blank_lines_skipped,
        "invalid_records": invalid_records,
        "rejection_reasons": dict(sorted(rejection_reasons.items())),
        "total_records": total_records,
        "valid_records": valid_records,
    }

    with (output_dir / "report.json").open(
        "w",
        encoding="utf-8",
    ) as report_file:
        json.dump(
            report,
            report_file,
            indent=2,
            ensure_ascii=False,
            sort_keys=True,
        )
        report_file.write("\n")

    return report
