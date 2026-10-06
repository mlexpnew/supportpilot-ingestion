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
- **Decision**: Fail closed for card-shaped and context-accompanied numbers.
- **Immediate Context Window Specification**:
  - **Window Size**: Up to 25 characters immediately preceding the candidate number and up to 25 characters immediately following the candidate number within the same clause.
  - **Direction**: Both preceding and following context within the same clause.
  - **Clause Boundary Isolation**: The window strictly terminates at clause punctuation (English `.`, `\n`, `;`, `!`, `?` and Devanagari danda `।`, `॥`) and does not cross existing redaction placeholder tokens (`[CARD]`, `[EMAIL]`, `[PHONE]`).
  - **Multilingual Keywords & Non-Latin Word Boundary Safety**:
    - **Keyword List**: English (`card`, `visa`, `mastercard`, `amex`) and Hindi / Devanagari (`कार्ड`, `क्रेडिट`, `डेबिट`, covering `क्रेडिट कार्ड` and `डेबिट कार्ड`).
    - **Token & Boundary Safety**: Matching uses negative lookaround boundaries `(?<!\[)(?<![A-Za-z\u0900-\u0963\u0970-\u097F])(?:...)(?![A-Za-z\u0900-\u0963\u0970-\u097F])(?!\])`. In Python's `re`, standard `\b` fails between Devanagari combining characters and adjacent ASCII digits (e.g. `कार्ड4111`). Explicit letter-exclusion lookarounds allow keywords touching numbers, colons, danda, or whitespace, while preventing false matches inside unrelated words (such as Hindi `रिकॉर्ड` or English `discard`) and ensuring `[CARD]` placeholders never re-trigger matches.
- **Rationale & Accepted Risks**:
  - For fintech clients whose enterprise contracts prohibit customer payment card data from ever reaching external LLM APIs, a false negative is a contractual and regulatory breach, while a false positive merely costs minor LLM contextual visibility.
  - Mistyped cards (such as `4111 1111 1111 1112` in sample S-4 where a single digit is transposed or mistyped) fail the Luhn checksum but are unmistakably card-formatted.
  - To prevent leaks of customer card numbers due to typos, SupportPilot adopts a hybrid fail-closed strategy:
    1. Numbers formatted in card-like groups (e.g. 4-4-4-4 such as `4111 1111 1111 1112` or Amex 4-6-5) are redacted as `[CARD]` even if they fail Luhn.
    2. Numbers accompanied by card keywords (`card`, `visa`, `mastercard`, `amex`, `कार्ड`, `क्रेडिट`, `डेबिट`) in the immediate 25-character clause window are redacted as `[CARD]` even if they fail Luhn.
    3. Ungrouped solid runs of 13–19 digits without card keywords (such as courier tracking numbers like `1234567890123`) must pass the Luhn checksum to be redacted; if they fail Luhn, they remain unredacted.
  - **Accepted False-Positive Risk**: A tracking number or serial number formatted in 4-4-4-4 or immediately adjacent to a card keyword (e.g. "card" or "कार्ड") will be redacted as `[CARD]`. We accept this minor false-positive risk because preventing card data exfiltration is non-negotiable for fintech compliance.
  - **Known Gap (Residual Leak)**: A customer who writes an ungrouped, mistyped 16-digit card that fails Luhn without any nearby card keywords in English or Hindi (e.g. `"my number is 4111111111111112"` or `"मेरा नंबर 4111111111111112 है"`) will survive redaction unredacted. This false-negative gap is **flagged for Neha (Product & Compliance) and pending her decision**. Until formal product/compliance sign-off, accountability is held by engineering (the author and PR reviewers: Arjun, Sana). This trade-off is proposed because indiscriminately redacting all ungrouped 13–19 digit strings would destroy 100% of order numbers, shipment barcodes, and courier tracking numbers across all non-card tickets.
  - **Multilingual Scope Flagged for Neha**: Clients' customer bases write across multiple Indian languages (e.g. Tamil, Telugu, Bengali, Marathi, Kannada). Without approved lexicons for each language, card keyword detection in non-Hindi Indian languages remains an open gap, pending Neha's determination on multilingual coverage.

### 2. Order Numbers, Ticket IDs, Tracking Numbers, and Numeric References
- **Rules Governing Numeric Identifiers**:
  - **Prefix-Protected Identifiers**: Order numbers and ticket IDs carrying a non-digit prefix (e.g. `#55231`, `T-1001`) survive intact because they do not match digit-only extraction patterns.
  - **Digit-Length Rules**:
    - **Fewer than 10 digits**: Survive intact (below phone and card length thresholds).
    - **Bare 10-digit runs**: Redacted as `[PHONE]` (e.g. `Order 1234567890 shipped` $\to$ `Order [PHONE] shipped`). Standalone 10-digit runs are indistinguishable from standard US / Indian phone numbers, and fail closed under phone sanitization.
    - **11 to 12 digits**: Ungrouped runs without phone formatting bypass phone matching and fall below the 13-digit card minimum, surviving intact.
    - **13 to 19 digits**: Preserved only if they are ungrouped, lack card context keywords, and fail the Luhn checksum (e.g. 13-digit tracking number `1234567890123`, 14-digit timestamp `Ref 20250114103000`, 14-digit amount `Amount 12345678901234 INR`).
- **Accepted False-Positive Risks & Statistical Trade-Offs**:
  1. **Bare 10-Digit Order Numbers Redacted as `[PHONE]`**:
     - *Risk*: Clients whose systems generate bare 10-digit order IDs will have them redacted as `[PHONE]`.
     - *Flagged for Neha*: This false-positive risk is explicitly **flagged for Neha**. If enterprise clients utilize bare 10-digit numeric order IDs, a merchant-specific allowlist or structural prefix rule will be required in a future iteration.
  2. **~10% Statistical Luhn Collision Rate for Long Numeric Runs**:
     - Long numeric references (such as tracking IDs like `1234567890123`, 14-digit timestamps like `20250114103000`, or bare amounts like `12345678901234`) survive because they happen to fail the Luhn checksum.
     - Because Luhn is a mod-10 checksum algorithm, approximately **1 in 10 random digit sequences (~10%)** will pass Luhn purely by chance and will be redacted as `[CARD]`. This ~10% false-positive rate is proposed as an operational compromise, flagged for Neha's confirmation, to guarantee that all valid payment cards are redacted without maintaining brittle merchant-specific tracking format allowlists.

### 3. Bank Account Numbers and National IDs
- **Decision**: Out of scope for SP-102; flagged for Neha.
- **Rationale**: Bank account numbers (9–18 digits depending on country) and national identifiers (such as US SSN, Indian PAN / Aadhaar) do not have a uniform global checksum and risk massive false positives if matched naively. This has been explicitly flagged as an open product question for Neha prior to expanding the PII taxonomy.

### 4. Non-String Values, Malformed Fields, and CLI Batch Failure Decision
- **Two-Layer Architecture**:
  - **Core Layer (`redact_field`)**: Preserves `null` values as `null` with 0 redactions. Non-string, non-null values raise a `TypeError("PII-redactable fields must be strings or null")`.
  - **CLI Layer (`process_file` / `__main__.py`)**: Catches `TypeError`, annotates with the 1-indexed line number, prints a clean error message (`Error at line {line_number}: PII-redactable fields must be strings or null`) to `stderr`, immediately unlinks temporary output files (`.redacted.jsonl.tmp`, `.redaction_report.json.tmp`), and exits with code `1`. Raw tracebacks and record field values are strictly suppressed to guarantee zero PII leakage on failure.
- **Product Decision (Fail-Stop Batch Run)**:
  - Encountering a single malformed row fails the entire batch run rather than skipping or writing a sanitized reject record.
  - **Accepted Trade-Off**: In a production nightly batch, a single corrupt record will halt the processing of 100k tickets. This fail-stop policy is proposed for SP-102, flagged for Neha's confirmation, because preprocessing operates strictly on `out/valid.jsonl` (the output of the SP-101 ticket validator), which already enforces data types and shunts invalid records to `rejects.jsonl`. A non-string field reaching SP-102 indicates an upstream pipeline breach or severe data corruption, justifying an immediate abort before sending compromised data to external LLMs. If per-record reject tolerance is required in the future, it can be designed as a coordinated extension with Priya and Neha.

### 5. Idempotence & Data Structure Invariants
- **Decision**: The redaction process is strictly idempotent across multiple passes.
- **Rationale**:
  - Existing placeholders `[EMAIL]`, `[PHONE]`, and `[CARD]` are enclosed in square brackets and do not match the email, phone, or card regexes.
  - Redaction tokens are explicitly bounded so they are not recognized as card context keywords in subsequent passes.
  - `RedactionResult` is implemented as a `NamedTuple`, ensuring backward-compatible indexing (`result[0]`), attribute access (`result.text`), and tuple unpacking (`text, counts = redact_text(...)`).
  - Running redaction multiple times over already-redacted text yields identical text and zero additional redactions.

## Reporting & Privacy Invariants

- `redaction_report.json` contains only numeric counts nested under a single canonical `counts` dictionary, alongside top-level metadata, with sorted keys and a trailing newline:
  - `counts`: Dictionary mapping PII type to count (`card`, `email`, `phone`).
  - `total_records`: Number of records processed.
  - `total_redactions`: Sum of all redactions.
- To prevent drifting sources of truth and key collisions, individual entity counts are not duplicated at the top level.
- Original customer text, ticket IDs, and matched PII values are **never** recorded in the report, CLI logs, stdout, or stderr.
