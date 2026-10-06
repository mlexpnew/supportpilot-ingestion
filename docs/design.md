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
```

# SP-102 — PII Redaction Design

## Architecture & Approach

SupportPilot ticket text must be sanitized before transmission to downstream LLM APIs. The redaction pipeline operates with strict zero-PII leakage guarantees across output files, logs, and CLI reports.

1. **Candidate Extraction & Algorithmic Validation**:
   - **Email (`[EMAIL]`)**: Detected via bounded RFC 5322-compliant expression handling `+` tags, subdomains, and multi-part TLDs (e.g. `example.co.in`).
   - **Phone (`[PHONE]`)**: Matches US (e.g. `(415) 555-0132`, `+1-415-555-0132`) and Indian (e.g. `+91 98765 43210`, `9876543210`) formats with or without country codes and standard punctuation.
   - **Payment Card (`[CARD]`)**: Extracted using a bounded pattern (`\d(?:[ -]?\d){12,18}`) spanning 13 to 19 digits. Extracted candidates are stripped of spaces/dashes and validated via the **Luhn checksum algorithm** (ISO/IEC 7812). Only candidates passing Luhn are redacted.

2. **Execution Order**:
   - Email addresses are redacted first to ensure numeric characters in local parts (e.g. `user9876543210@domain.com`) are not misidentified as phone numbers or card fragments.
   - Phone numbers are redacted second.
   - Payment cards are redacted third, validating remaining long digit sequences with the Luhn checksum.

3. **ReDoS & Backtracking Safety**:
   - All regular expressions avoid nested quantifiers (`(a+)+`), possessive overlaps, or unbounded whitespace sequences.
   - Card matching requires explicit digit boundaries `(?<![\d-])` and `(?![\d-])` and ends on a required digit `\d`, ensuring $O(N)$ linear-time evaluation even on hostile inputs (e.g. 1 MB of repeated digits and dashes).

4. **Streaming & Crash Resilience**:
   - CLI processes `valid.jsonl` line by line with constant $O(1)$ resident memory.
   - Output files are written to `.redacted.jsonl.tmp` and `.redaction_report.json.tmp`.
   - Pre-existing reports are deleted at startup, and temporary files are atomically promoted via `os.replace` on clean completion.
   - Refuses to overwrite input files when `--input` matches destination paths.

## Decisions on Open Questions

### 1. 16-Digit Number Failing Luhn: Redact or Leave?
- **Decision**: Leave unredacted.
- **Rationale**: Real payment cards adhere to the Luhn checksum standard. Redacting invalid-Luhn numbers creates catastrophic false positives on tracking numbers, order references, parcel IDs, and device serial numbers. For fintech security compliance, legitimate card numbers will always pass the checksum.

### 2. Order Numbers, Ticket IDs, and Tracking Numbers
- **Decision**: Preserved intact without redaction.
- **Rationale**:
  - Order numbers (e.g. `#55231`) and ticket IDs (e.g. `T-1001`) contain fewer than 13 digits and have non-digit prefixes (`#`, `T-`).
  - Tracking numbers (e.g. 13-digit `1234567890123` in sample S-4) fail the Luhn checksum and remain intact without requiring fragile keyword allowlists.

### 3. Bank Account Numbers and National IDs
- **Decision**: Out of scope for SP-102; flagged for Neha.
- **Rationale**: Bank account numbers (9–18 digits depending on country) and national identifiers (such as US SSN, Indian PAN / Aadhaar) do not have a uniform global checksum and risk massive false positives if matched naively. This has been explicitly flagged as an open product question for Neha prior to expanding the PII taxonomy.

### 4. Non-String Values and `null` Fields
- **Decision**: `null` values (such as `subject: null` or `body: null`) are preserved as `null` with 0 redactions. Non-string, non-null values raise a `TypeError` in `redact_field`, aligning with the SP-101 schema boundary where ticket bodies and subjects must be valid strings or null.

### 5. Idempotence
- **Decision**: The redaction process is strictly idempotent.
- **Rationale**: Existing placeholders `[EMAIL]`, `[PHONE]`, and `[CARD]` are enclosed in square brackets and do not match the email, phone, or card regexes. Running redaction multiple times over already-redacted text yields identical text and zero additional redactions.

## Reporting & Privacy Invariants

- `redaction_report.json` contains only numeric counts, sorted keys, and a trailing newline:
  - `card`: Count of redacted payment cards.
  - `email`: Count of redacted emails.
  - `phone`: Count of redacted phone numbers.
  - `total_records`: Number of records processed.
  - `total_redactions`: Sum of all redactions.
  - `counts`: Dictionary of redaction counts by type.
- Original customer text, ticket IDs, and matched PII values are **never** recorded in the report, CLI logs, stdout, or stderr.
