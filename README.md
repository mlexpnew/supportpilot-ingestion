# SupportPilot Ingestion

Streaming JSONL ticket ingestion, validation, and deduplication service built with Python and Pydantic v2.

## Python Version

- **Python 3.12+** required (tested on Python 3.12.4).

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## CLI Usage

```bash
python -m supportpilot.ingestion --input data/sample_tickets.jsonl --output-dir out
```

Optional flags:
- `--fail-on-rejects`: Exits with non-zero status (code 2) if any invalid records are encountered. By default, the CLI exits `0` upon completing processing and writes invalid records to `rejects.jsonl`.

## Rejection Error Codes

Rejected records use a stable, sanitized application-level error vocabulary:

| Error Code | Meaning |
|---|---|
| `invalid_encoding` | Line contains non-UTF-8 bytes and cannot be decoded |
| `invalid_json` | Line is not valid JSON |
| `invalid_record_type` | Valid JSON but not a JSON object (e.g. array or primitive) |
| `missing_field` | A required field is missing from the record |
| `invalid_type` | Field value has an incorrect data type (e.g. numeric body, non-string ID) |
| `empty_body` | `body` is empty (`""`) or contains only whitespace |
| `empty_field` | A required string field (e.g. `ticket_id`, `customer_id`) is empty or whitespace |
| `invalid_channel` | `channel` is not one of the supported enum values (`email`, `chat`, `phone`, `social`) |
| `invalid_timestamp` | `created_at` is not an ISO 8601 string, is numeric, or lacks required timezone info |
| `duplicate_ticket_id` | `ticket_id` was already accepted earlier in the file |
| `validation_error` | Other schema validation failure |

### Error Reporting Determinism
When a record violates multiple constraints, SupportPilot reports the first validation failure deterministically based on Pydantic's `errors()[0]`.

### Sanitized Rejection Output & PII Protection
Rejections written to `rejects.jsonl` contain only:
- `line_number` (int)
- `field` (string or null)
- `error_type` (string)
- `ticket_id` (string, optional): Included only if the record parsed as a JSON object and has a valid string `ticket_id` between 1 and 64 characters in length. Arbitrary or excessively long strings are omitted.

Message bodies, customer data, and ticket text are **never** written to `rejects.jsonl` or stderr.

## Blank Line Rule

Blank lines and lines containing only whitespace are skipped during processing:
- Tracked separately in `report.json` under `blank_lines_skipped`.
- Invariant: `total_records == valid_records + invalid_records`. Blank lines do not represent ticket records and are excluded from `total_records`.

## Deduplication Memory Measurements & 2 GB Projection

- **Current Implementation**: In-memory Python `set[str]` storing unique accepted `ticket_id` values.
- **Empirical Measurements (`/usr/bin/time -l`)**:
  - Baseline (1 record): **32.88 MB** RSS (Python runtime + imports baseline).
  - 1,000,000 records (139 MB file): **134.14 MB** RSS, processed in **8.45 s** (~118,000 records/s).
  - Net memory delta for 1M IDs: `134.14 MB - 32.88 MB` = **101.26 MB**.
  - **Per-ID Cost**: ~`101.25 bytes/ID` (consistent with CPython's 64-byte `PyASCIIObject` for ~8-char IDs + 16 bytes hash table entry + allocator padding).
- **2 GB RAM Projection**:
  - Available memory budget: `(2,000 MB - 32.88 MB) / 101.25 bytes/ID` ≈ **~19.4 million records**.
  - A standard 2 GB container will reach its memory ceiling at approximately 19–20 million unique ticket IDs.
- **Alternative for Greater Scale**: For datasets exceeding 10–20 million records or in severely memory-constrained environments (<512 MB RAM), replace the in-memory set with:
  - An external key-value store (e.g. Redis `SET` / `SETNX` commands).
  - An embedded disk-backed index (such as SQLite with an indexed ID table or RocksDB / LMDB).
  - A Bloom filter for $O(1)$ constant-memory probabilistic pre-filtering.

## Timezone Handling & Product Decision (Flagged for Neha)

- **Timestamp Rule**: `created_at` must be an ISO 8601 formatted string with an explicit UTC offset or timezone specifier (e.g. `2025-01-14T09:15:00Z` or `2025-01-14T10:30:00+05:30`). Timezone-naive timestamps (e.g. `2025-01-14T09:15:00`) are rejected as `invalid_timestamp`.
- **Note for Neha (SLA Timing)**: Rejecting timezone-naive timestamps ensures correct event ordering and SLA metrics downstream. However, if upstream data sources omit timezone offsets, those records will be rejected. If upstream systems cannot be updated, team alignment is needed on whether to assume a default timezone (e.g. UTC).
- **Note for Neha (CLI Exit Code)**: By default, the CLI exits `0` when ingestion finishes, even if records are rejected, allowing pipelines to inspect `report.json` and `rejects.jsonl`. For pipelines that require immediate failure on any reject, `--fail-on-rejects` exits with code `2`.

## Encoding & BOM Handling

- **UTF-8 BOM**: Files starting with a UTF-8 Byte Order Mark (`\xef\xbb\xbf`), commonly exported by Excel or Windows tooling, are automatically supported; the BOM is stripped from the first line without error.
- **Non-UTF-8 Bytes**: Input files are streamed line-by-line in binary mode. Lines containing invalid UTF-8 bytes are rejected as `invalid_encoding` without crashing the ingestion process or dumping raw bytes to stderr.

## Source-Overwrite Protection

The service refuses execution if `--input` resolves to any of the destination files (`valid.jsonl`, `rejects.jsonl`, `report.json`) within `--output-dir`, preventing accidental data loss.