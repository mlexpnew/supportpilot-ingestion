# supportpilot-ingestion
# SupportPilot Ingestion

## Overview

Streaming JSONL ticket ingestion and validation service built with Python and Pydantic v2.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
## Rejection Error Codes

Rejected records use a stable application-level error vocabulary:

| Error code | Meaning |
|---|---|
| `invalid_timestamp` | `created_at` is invalid or missing required timezone information |
| `invalid_channel` | `channel` is not one of the supported values |
| `empty_body` | `body` is empty or contains only whitespace |
| `duplicate_ticket_id` | `ticket_id` was already accepted earlier in the file |
| `missing_field` | A required field is missing |
| `invalid_json` | The input line is not valid JSON |
| `invalid_record_type` | Valid JSON but not a JSON object |
| `validation_error` | Other schema validation failure |

Reject records contain only the line number, field name when available, and error code. Ticket values and message bodies are never included.