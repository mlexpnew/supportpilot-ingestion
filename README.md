# SupportPilot Ingestion

Streaming JSONL ticket ingestion, validation, and deduplication service built with Python and Pydantic v2.

## Python Version

- **Python 3.11+** required (tested on Python 3.11 and 3.12).

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

For development and test tooling:
```bash
pip install -r requirements-dev.txt
```

## CLI Usage

### Ingestion & Validation (SP-101 / SP-104)

```bash
python -m supportpilot.ingestion --input data/sample_tickets.jsonl --output-dir out
```

Optional flags:
- `--max-reject-rate`: Maximum allowed fraction of rejected records (default: `0.5`). Evaluated when total records $\ge 20$. If `reject_rate > max_reject_rate`, exits with code `3`.
- `--min-records`: Minimum record count required before evaluating `--max-reject-rate` (default: `20`). Prevents small sample variance from causing false alarms.
- `--fail-on-rejects`: Exits with code `3` if any invalid records are encountered.

### CLI Exit Codes

| Exit Code | Meaning | Triggers |
|---|---|---|
| `0` | OK | Completed normally with reject rate $\le$ threshold |
| `1` | Crash | Unhandled fatal exception or crash |
| `2` | Bad or Empty File | File not found, unreadable, 0-byte file, blank-only file, or UTF-16 without BOM |
| `3` | Reject Rate Too High | `reject_rate > max_reject_rate` (for $\ge 20$ records) or `--fail-on-rejects` with invalid lines |

### Operational Summary Line

The ingestion CLI prints exactly one JSON line to `stdout` upon termination:
```json
{"total": 10, "valid": 4, "invalid": 6, "reject_rate": 0.6, "elapsed": 0.0032, "exit_reason": "ok"}
```
Strict Zero-Leakage: Contains only numeric operational metrics and status enums. Never contains ticket bodies, customer data, ticket IDs, or file paths.

### PII Redaction (SP-102)

```bash
python -m supportpilot.preprocessing --input out/valid.jsonl --output-dir out_redacted
```

Reads tickets line by line, redacting sensitive customer PII from `body` and `subject` fields:
- `redacted.jsonl`: Sanitized records with `[EMAIL]`, `[PHONE]`, and `[CARD]` placeholders.
- `redaction_report.json`: Sorted-key JSON report with per-type counts nested under `counts`:

```json
{
  "counts": {
    "card": 1,
    "email": 2,
    "phone": 3
  },
  "total_records": 8,
  "total_redactions": 6
}
```

## OrderHub Client (SP-103)

Programmatic client for querying the merchant's OrderHub service:

```python
from supportpilot.integrations.orderhub import (
    get_order_status,
    ConfigurationError,
    OrderNotFound,
    AuthenticationError,
    ServiceUnavailable,
)

# Fetch status safely (defaults to a 3.0s total deadline across all retries)
try:
    status = get_order_status("55231")
    print(status.order_id, status.status, status.carrier, status.eta)
except ConfigurationError:
    print("OrderHub client misconfigured")
except OrderNotFound:
    print("Order does not exist")
except AuthenticationError:
    print("OrderHub API key rejected by server")
except ServiceUnavailable:
    print("OrderHub service unavailable or deadline exceeded")
```

### Key Guarantees
- **Strict 3.0 s Deadline**: Enforces an absolute 3.0-second timeout budget across all attempts (at most 3 attempts).
- **PII Zero-Leakage**: Strips customer email and physical shipping addresses from the upstream response. Returns only an `OrderStatus` dataclass containing `order_id`, `status`, `carrier`, and `eta`.
- **Pre-Network Validation**: Rejects path traversal (`../admin`), query string injection (`55231?x=1`), empty strings, and oversized IDs with `ValueError` before any network call.
- **Typed Exceptions**: Disambiguates between `ConfigurationError` (missing key or malformed URL), `OrderNotFound` (404), `AuthenticationError` (401), and `ServiceUnavailable` (5xx, timeouts, 429 exhaustion, corrupt JSON).

## Schema & Extraneous Fields (`extra="ignore"`)

The ticket model is configured with `extra="ignore"`. Any unexpected additional fields provided by clients or upstream systems are safely accepted and ignored, ensuring ingestion does not break when clients send non-standard metadata, while omitting extra keys from the normalized output in `valid.jsonl`.

## Rejection Error Codes

Rejected records use a stable, sanitized application-level error vocabulary:

| Error Code | Meaning |
|---|---|
| `invalid_encoding` | Line contains non-UTF-8 bytes and cannot be decoded |
| `line_too_long` | Line exceeds the 1 MB buffer limit |
| `invalid_json` | Line is not valid JSON or exceeds recursion limits |
| `invalid_record_type` | Valid JSON but not a JSON object (e.g. array or primitive) |
| `missing_field` | A required field is missing from the record |
| `invalid_type` | Field value has an incorrect data type (e.g. numeric body, non-string ID) |
| `empty_body` | `body` is empty (`""`) or contains only whitespace |
| `empty_field` | A required string field (e.g. `ticket_id`, `customer_id`) is empty or whitespace |
| `invalid_channel` | `channel` is not one of the supported enum values (`email`, `chat`, `phone`, `social`) |
| `invalid_status` | `status` is not one of the supported enum values (`open`, `pending`, `resolved`, `escalated`) |
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

Message bodies, customer data, and ticket text are **never** written to `rejects.jsonl` or stderr. Pydantic is configured with `hide_input_in_errors=True` to prevent sensitive payloads from appearing in error representations.

## Blank Line Rule

Blank lines and lines containing only whitespace are skipped during processing:
- Tracked separately in `report.json` under `blank_lines_skipped`.
- Invariant: `total_records == valid_records + invalid_records`. Blank lines do not represent ticket records and are excluded from `total_records`.

## Deduplication Memory Measurements & Projections

- **Current Implementation**: In-memory Python `set[str]` storing unique accepted `ticket_id` values.
- **Workbook Projection (2 GB limit)**:
  - At ~109 bytes per ID (`108.87 bytes/ID` net delta), a 2 GB memory boundary can accommodate approximately **18 million records** (or ~14–15 million records leaving generous runtime and OS headroom).
- **Empirical Measurements (`/usr/bin/time -l`)**:
  - Baseline (1 record): **31.77 MB** RSS (`31,768,576 bytes`).
  - 1,000,000 records (139 MB file): **140.64 MB** RSS (`140,640,256 bytes`), processed in **9.38 s** (~106,600 records/s).
  - Net memory delta for 1M IDs: `140,640,256 bytes - 31,768,576 bytes` = **108,871,680 bytes** (~`108.87 bytes/ID`, or ~`109 bytes/ID`).
- **Alternative for Greater Scale**: For datasets exceeding 10–14 million records or in severely memory-constrained environments (<512 MB RAM), replace the in-memory set with:
  - An external key-value store (e.g. Redis `SET` / `SETNX` commands).
  - An embedded disk-backed index (such as SQLite with an indexed ID table or RocksDB / LMDB).
  - A Bloom filter for $O(1)$ constant-memory probabilistic pre-filtering.

## Product Decisions Flagged for Neha

1. **Strict Timezone Enforcement**:
   - `created_at` must be an ISO 8601 formatted string with an explicit UTC offset or timezone specifier (e.g. `2025-01-14T09:15:00Z` or `2025-01-14T10:30:00+05:30`). Timezone-naive timestamps (e.g. `2025-01-14T09:15:00`) and numeric timestamps are strictly rejected as `invalid_timestamp`.
   - *Trade-off*: Strict rejection protects downstream SLA metrics and analytics from timing inaccuracies. If upstream data sources omit timezone offsets, those records will be rejected. Neha should confirm if upstream sources can supply timezone offsets or if a default assumption (e.g. UTC) should be introduced.
2. **CLI Exit Code**:
   - By default, the CLI exits `0` when ingestion finishes, even if records are rejected, allowing pipelines to inspect `report.json` and `rejects.jsonl`. For CI/CD pipelines that require immediate failure on any reject, `--fail-on-rejects` exits with code `2`.
3. **Client-Controlled `ticket_id` on Rejections**:
   - Sanitized `ticket_id` values (capped at 64 characters) are preserved in `rejects.jsonl` to assist debugging. We assume ticket IDs are non-sensitive identifiers; however, if client sources ever embed PII (e.g. customer email addresses) in `ticket_id`, Neha should decide whether to mask or omit `ticket_id` from reject logs.

## Streaming Line Limits & Encoding

- **Line Length Guard**: Lines are read in chunks up to 1 MB (`MAX_LINE_BYTES = 1_048_576`). Lines exceeding 1 MB are discarded without buffering and rejected with `line_too_long`, preventing unbounded memory growth.
- **UTF-8 BOM**: Files starting with a UTF-8 Byte Order Mark (`\xef\xbb\xbf`) are automatically supported; the BOM is stripped from the first line without error.
- **UTF-16 Support**: Automatically detects UTF-16LE (`\xff\xfe`) and UTF-16BE (`\xfe\xff`) by peeking the initial 2 bytes without buffering the whole file in memory. Lines are streamed and decoded incrementally on 2-byte-aligned newline delimiters.
- **UTF-16 without BOM Rejection**: UTF-16 files lacking a BOM (detected via alternating null bytes in the header) are rejected loudly at the file level with a `ValueError` and exit code `2`.
- **Non-UTF-8 Bytes**: Input files are streamed in binary mode. Lines containing invalid bytes are rejected as `invalid_encoding` without crashing or dumping raw bytes to stderr.
- **Hostile JSON**: Deeply nested JSON that triggers parser recursion errors is caught and marked as `invalid_json`.

## Crash Safety & Atomic Outputs

To prevent stale or mismatched outputs after an unexpected crash or termination:
- Pre-existing `report.json` files in `--output-dir` are removed at the start of ingestion.
- Output lines are written to temporary files (`.valid.jsonl.tmp`, `.rejects.jsonl.tmp`, `.report.json.tmp`) and atomically promoted on completion, ensuring partial runs never leave corrupt or uncoordinated state.

## Source-Overwrite Protection

The service refuses execution if `--input` resolves to any destination file (`valid.jsonl`, `rejects.jsonl`, `report.json`) within `--output-dir`, preventing accidental data loss.