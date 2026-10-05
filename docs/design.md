# SP-101: Ticket Data Loader & Validator

## Module structure

Pydantic models are defined in `models.py` and file processing is implemented in `loader.py`. `__main__.py` provides the CLI. Report generation maintains aggregate counters during processing and writes `report.json` at the end.

## Duplicate ticket_id decision

`ticket_id` must be unique within the input file. The first successfully validated occurrence is accepted and its ID is stored in an in-memory set. Later occurrences are rejected as duplicates. Invalid records do not reserve their `ticket_id`.

The set stores only ticket IDs rather than complete records, which keeps memory usage significantly lower than retaining the full dataset. For substantially larger-scale ingestion, an external key-value store or database could be considered for deduplication.

## Corrupt JSON line decision

Invalid JSON lines are rejected with the line number and a sanitized error type such as `invalid_json`. Raw parser or validation error messages are not written because they may contain ticket PII.

## Whitespace in body decision

The `body` value is stripped before validation. Records with an empty body after stripping are rejected. The normalized body is written for accepted records.

Null or non-string body values are rejected as validation errors.

## Blank lines

Blank lines are skipped and counted separately as `blank_lines_skipped`. They are not considered ticket records, so they are excluded from the `valid_records + invalid_records == total_records` invariant.

## Timestamp handling

`created_at` must be a valid ISO 8601 timestamp and must include timezone information. Non-ISO timestamps and timezone-naive timestamps are rejected.

Strict timezone handling prevents ambiguous ticket ordering and incorrect SLA calculations. Neha should be aware that rejecting timezone-naive records may reduce the accepted-record count, but protects downstream analytics from incorrect event timing.

## Reject reason structure

Rejected records contain only the line number, field when applicable, and sanitized error type.

Example:

```json
{
  "line_number": 12,
  "field": "channel",
  "error_type": "validation_error"
}