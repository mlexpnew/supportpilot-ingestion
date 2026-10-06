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
    reject: dict[str, Any] = {
        "line_number": line_number,
        "field": field,
        "error_type": error_type,
    }

    if ticket_id is not None:
        reject["ticket_id"] = ticket_id

    return reject


def _extract_ticket_id(data: dict[str, Any]) -> str | None:
    """Extract candidate ticket_id if it is a bounded string (1-64 chars)."""
    raw_id = data.get("ticket_id")
    if isinstance(raw_id, str):
        cleaned = raw_id.strip()
        if 1 <= len(cleaned) <= 64:
            return cleaned
    return None


def _validation_rejection(
    line_number: int,
    error: ValidationError,
    ticket_id: str | None = None,
) -> dict[str, Any]:
    """Convert Pydantic errors into stable application-level errors."""
    errors = error.errors()

    first_error = errors[0] if errors else {}
    location = first_error.get("loc", ())
    field = str(location[0]) if location else None
    pydantic_type = first_error.get("type", "")

    if pydantic_type == "missing":
        error_type = "missing_field"
    elif field == "created_at":
        error_type = "invalid_timestamp"
    elif field == "channel":
        error_type = "invalid_channel"
    elif field == "body" and pydantic_type in ("value_error", "string_too_short"):
        error_type = "empty_body"
    elif pydantic_type in ("value_error", "string_too_short"):
        error_type = "empty_field"
    elif pydantic_type in (
        "string_type",
        "int_type",
        "bool_type",
        "float_type",
        "dict_type",
        "list_type",
    ) or pydantic_type.endswith("_type"):
        error_type = "invalid_type"
    else:
        error_type = "validation_error"

    return _reject(
        line_number=line_number,
        field=field,
        error_type=error_type,
        ticket_id=ticket_id,
    )


def validate_lines(
    input_path: Path,
) -> Iterator[tuple[int, Ticket | None, dict[str, Any] | None]]:
    """
    Stream and validate tickets from a JSONL file.

    Yields:
        (line_number, ticket, rejection)
        For blank lines, both ticket and rejection are None.
    """
    seen_ticket_ids: set[str] = set()

    with input_path.open("rb") as file:
        for line_number, raw_bytes in enumerate(file, start=1):
            if line_number == 1 and raw_bytes.startswith(b"\xef\xbb\xbf"):
                raw_bytes = raw_bytes[3:]

            try:
                raw_line = raw_bytes.decode("utf-8")
            except UnicodeDecodeError:
                yield (
                    line_number,
                    None,
                    _reject(line_number, "invalid_encoding"),
                )
                continue

            if not raw_line.strip():
                yield (line_number, None, None)
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

            candidate_ticket_id = _extract_ticket_id(data)

            try:
                ticket = Ticket.model_validate(data)
            except ValidationError as error:
                yield (
                    line_number,
                    None,
                    _validation_rejection(
                        line_number,
                        error,
                        ticket_id=candidate_ticket_id,
                    ),
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
                        ticket_id=candidate_ticket_id or ticket.ticket_id,
                    ),
                )
                continue

            seen_ticket_ids.add(ticket.ticket_id)

            yield (line_number, ticket, None)


def process_file(
    input_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    """
    Process a JSONL ticket file using streaming input/output.

    Returns aggregate processing statistics.
    """
    resolved_input = input_path.resolve()
    resolved_output_dir = output_dir.resolve()
    conflicting_outputs = {
        (resolved_output_dir / "valid.jsonl").resolve(),
        (resolved_output_dir / "rejects.jsonl").resolve(),
        (resolved_output_dir / "report.json").resolve(),
    }
    if resolved_input in conflicting_outputs:
        raise ValueError(
            f"Input file '{input_path}' conflicts with output files in '{output_dir}'."
        )

    output_dir.mkdir(parents=True, exist_ok=True)

    valid_path = output_dir / "valid.jsonl"
    rejects_path = output_dir / "rejects.jsonl"

    total_records = 0
    valid_records = 0
    invalid_records = 0
    blank_lines_skipped = 0
    rejection_reasons: dict[str, int] = {}

    with (
        valid_path.open("w", encoding="utf-8") as valid_file,
        rejects_path.open("w", encoding="utf-8") as rejects_file,
    ):
        for line_number, ticket, reject in validate_lines(input_path):
            if ticket is None and reject is None:
                blank_lines_skipped += 1
                continue

            total_records += 1

            if reject is not None:
                invalid_records += 1
                error_type = reject["error_type"]
                rejection_reasons[error_type] = rejection_reasons.get(error_type, 0) + 1
                rejects_file.write(json.dumps(reject, ensure_ascii=False) + "\n")
            elif ticket is not None:
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
