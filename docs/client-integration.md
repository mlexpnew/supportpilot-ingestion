# SupportPilot Client Integration & Onboarding Guide

Welcome to the SupportPilot API integration guide. This document provides everything required to connect your customer support platform (e.g. Zendesk, OrderHub, custom CRM) to the SupportPilot ML triage and feedback pipeline.

---

## 1. Quick Start

### Target Endpoints & Base URLs

| Environment | Base URL | Notes |
| :--- | :--- | :--- |
| **Production** | `https://supportpilot-ingestion.onrender.com` | Live production cloud deployment |
| **Local / Dev** | `http://localhost:8000` | Local container or development server |

### Authentication Header

All protected endpoints require an API key passed via the `X-API-Key` HTTP header.

```http
X-API-Key: $SUPPORTPILOT_API_KEY
```

> [!IMPORTANT]
> **Zero-PII & Secret Security**: Never hardcode API keys in source code or client repositories. Store your key as an environment variable (`SUPPORTPILOT_API_KEY`) or in a secure secret manager.

---

## 2. Integration Verification (`examples/client_check.py`)

Before routing production ticket traffic, run the automated integration diagnostic script:

```bash
export SUPPORTPILOT_API_KEY="your-assigned-api-key"
python examples/client_check.py
# Or verify batch triage capabilities:
python examples/client_check.py --batch
```

The script performs:
1. Public liveness probe (`GET /health`)
2. Authenticated ticket triage (`POST /triage`)
3. Optional batch triage check (`POST /triage/batch` when `--batch` is passed)
4. Validation of response schema and predictions

A successful run outputs:
```text
==================================================
  SupportPilot Integration Diagnostic Check
==================================================
Target Base URL: https://supportpilot-ingestion.onrender.com
API Key:         ********xxxx
--------------------------------------------------

[1/2] Checking API service liveness (/health)...
✅ /health check passed: Service is alive and healthy.

[2/2] Checking authenticated ticket triage (/triage)...
✅ /triage check passed: Successfully classified test ticket!
   Predicted Label: order_status
   Confidence:      0.9241
   Needs Review:    False

--------------------------------------------------
🎉 All checks passed! Your integration is ready for production traffic.
==================================================
```

---

## 3. Endpoints

### 3.1 Public Liveness Probe: `GET /health`

Use this endpoint for health checks, uptime monitoring, and cold start detection. No authentication required.

#### Request
```bash
curl -i https://supportpilot-ingestion.onrender.com/health
```

#### Response (`200 OK`)
```json
{
  "status": "ok"
}
```

---

### 3.2 Automated Ticket Triage: `POST /triage`

Classifies customer support tickets into intent categories.

#### Request Headers
- `Content-Type: application/json`
- `X-API-Key: $SUPPORTPILOT_API_KEY`

#### Request Payload Schema
| Field | Type | Required | Description |
| :--- | :--- | :--- | :--- |
| `ticket_id` | string | **Yes** | Unique ticket identifier (1–64 characters). |
| `subject` | string | Optional | Email or ticket subject line. |
| `body` | string | **Yes** | Main inquiry text or customer message. Maximum total payload size is 10 KB. |

#### Classification Categories (`label`)
- `billing`: Billing discrepancies, charge issues, invoice requests.
- `cancellation`: Order or subscription cancellation requests.
- `inquiry`: General product or service inquiries.
- `order_status`: Delivery tracking, shipping status, delayed packages.
- `refund`: Return requests, refund inquiries.
- `technical`: Account access, bug reports, software errors.

#### Example Request (`cURL`)
```bash
curl -s -X POST "https://supportpilot-ingestion.onrender.com/triage" \
  -H "Content-Type: application/json" \
  -H "X-API-Key: $SUPPORTPILOT_API_KEY" \
  -d '{
    "ticket_id": "TICKET-10492",
    "subject": "Where is my package?",
    "body": "Hi, I placed an order three days ago and the tracking link shows pending. Can you update me?"
  }'
```

#### Example Response (`200 OK`)
```json
{
  "ticket_id": "TICKET-10492",
  "label": "order_status",
  "confidence": 0.8842,
  "needs_review": false,
  "model_version": "v1.0.0"
}
```

#### Python Integration Example
```python
import os
import requests

api_key = os.environ["SUPPORTPILOT_API_KEY"]
url = "https://supportpilot-ingestion.onrender.com/triage"

payload = {
    "ticket_id": "TICKET-10492",
    "subject": "Where is my package?",
    "body": "Hi, I placed an order three days ago and the tracking link shows pending.",
}

response = requests.post(
    url,
    headers={"X-API-Key": api_key, "Content-Type": "application/json"},
    json=payload,
    timeout=10,
)
response.raise_for_status()
triage_result = response.json()

if triage_result["needs_review"]:
    # Route to human review queue
    print(f"Flagged for human review (confidence: {triage_result['confidence']:.2f})")
else:
    # Auto-route or apply automated workflow
    print(f"Auto-routed to {triage_result['label']}")
```

---

### 3.3 Batch Ticket Triage: `POST /triage/batch`

Enables high-throughput bulk classification for historical backlogs (e.g. ingesting 5,000 backlog tickets) and high-volume burst ingestion.

#### Key Features & Guarantees
- **Vectorized Performance**: Batch inference is executed in a single vectorized model call rather than ticket-by-ticket (100 tickets finish in < 2 seconds).
- **Single Database Transaction**: All newly predicted tickets are persisted atomically in one database write.
- **Partial Failure Resiliency**: A malformed ticket in a batch does not fail the entire batch. Valid tickets are processed normally; invalid items return status `"error"` with a concise description (e.g. `invalid ticket_id`, `body is empty`). The request returns **HTTP 200** as long as the request structure itself is valid.
- **Idempotency & Safe Retries**:
  - **Duplicate within batch**: If the same `ticket_id` appears more than once in the same request payload, subsequent occurrences are marked as errors with `"error": "duplicate_in_batch"`.
  - **Existing tickets cached**: If a `ticket_id` was already triaged before, SupportPilot immediately returns the stored prediction with `"cached": true` without re-evaluating the model or modifying the database.
  - Clients can safely re-send batches upon network timeouts or transient errors without duplicate predictions or corrupting telemetry counters.

#### Batch Limits & Throttling
| Limit | Threshold | Status Code | Behavior |
| :--- | :--- | :--- | :--- |
| **Max Batch Size** | **100 tickets / request** | `413 Payload Too Large` | Checked before processing. Batches with 101+ tickets are rejected immediately. |
| **Request Rate Limit** | **60 requests / minute** | `429 Too Many Requests` | A batch request of up to 100 tickets counts as **1 request** against the 60 req/min limit. |
| **Ticket Rate Limit** | **1,000 tickets / minute** | `429 Too Many Requests` | Sliding-window limiter on cumulative ticket count per API key. Returns `Retry-After: <seconds>`. |

#### Request Payload Schema
```json
{
  "tickets": [
    {
      "ticket_id": "TICKET-1001",
      "subject": "Delivery tracking inquiry",
      "body": "Where is my shipment? Tracking says delayed."
    },
    {
      "ticket_id": "TICKET-1002",
      "subject": "Cancel order",
      "body": "Please cancel order #48291 and process refund."
    }
  ]
}
```

#### Response Payload Schema (`200 OK`)
```json
{
  "results": [
    {
      "ticket_id": "TICKET-1001",
      "status": "ok",
      "label": "order_status",
      "confidence": 0.9412,
      "needs_review": false,
      "cached": false
    },
    {
      "ticket_id": "TICKET-1002",
      "status": "ok",
      "label": "cancellation",
      "confidence": 0.9105,
      "needs_review": false,
      "cached": false
    }
  ],
  "summary": {
    "total": 2,
    "ok": 2,
    "failed": 0
  }
}
```

#### Example Request (`cURL`)
```bash
curl -s -X POST "https://supportpilot-ingestion.onrender.com/triage/batch" \
  -H "Content-Type: application/json" \
  -H "X-API-Key: $SUPPORTPILOT_API_KEY" \
  -d '{
    "tickets": [
      {
        "ticket_id": "TICKET-2001",
        "subject": "Delayed shipment",
        "body": "My package has not moved in four days."
      },
      {
        "ticket_id": "TICKET-2002",
        "body": "Can I return this item for a full refund?"
      }
    ]
  }'
```

#### Recommended Backlog Ingestion Pattern (Python)
When ingesting large historical archives (e.g. 5,000 tickets):
1. Chunk your ticket list into batches of **100 tickets**.
2. Respect the **1,000 tickets/minute** rate limit (e.g. send up to 10 batches per minute, or 1 batch every 6 seconds).
3. If receiving **HTTP 429**, inspect the `Retry-After` response header, sleep for that duration, and retry the batch safely (idempotent caching ensures zero duplicate work).

```python
import os
import time
import requests

api_key = os.environ["SUPPORTPILOT_API_KEY"]
url = "https://supportpilot-ingestion.onrender.com/triage/batch"


def chunked(items, n=100):
    for i in range(0, len(items), n):
        yield items[i : i + n]


def ingest_backlog(all_tickets):
    headers = {"X-API-Key": api_key, "Content-Type": "application/json"}

    for batch in chunked(all_tickets, 100):
        while True:
            resp = requests.post(
                url, json={"tickets": batch}, headers=headers, timeout=30
            )
            if resp.status_code == 429:
                wait_time = int(resp.headers.get("Retry-After", 10))
                print(f"Rate limited (tickets/min); waiting {wait_time}s...")
                time.sleep(wait_time)
                continue
            resp.raise_for_status()
            data = resp.json()
            print(
                f"Batch processed: ok={data['summary']['ok']} failed={data['summary']['failed']}"
            )
            break
        # Pace throughput to ~1,000 tickets/min (100 tickets every 6s)
        time.sleep(6.0)
```

---

### 3.4 Human Feedback: `POST /feedback`

Submits human agent confirmation or correction for closed tickets. Used to compute live model accuracy and drive continuous model retraining.

#### Request Payload Schema
| Field | Type | Required | Description |
| :--- | :--- | :--- | :--- |
| `ticket_id` | string | **Yes** | Matching ticket identifier originally sent to `/triage`. |
| `correct_label` | string | **Yes** | Ground-truth category confirmed by human agent. |
| `agent_id` | string | **Yes** | Opaque identifier of human reviewer (e.g. `AGENT-42`, 1–64 characters). |

#### Example Request (`cURL`)
```bash
curl -s -X POST "https://supportpilot-ingestion.onrender.com/feedback" \
  -H "Content-Type: application/json" \
  -H "X-API-Key: $SUPPORTPILOT_API_KEY" \
  -d '{
    "ticket_id": "TICKET-10492",
    "correct_label": "order_status",
    "agent_id": "AGENT-42"
  }'
```

#### Example Response (`200 OK`)
```json
{
  "ticket_id": "TICKET-10492",
  "status": "accepted"
}
```

---

## 4. HTTP Error Codes & Troubleshooting

| HTTP Status | Error Reason | Root Cause | Resolution / Fix |
| :--- | :--- | :--- | :--- |
| **401 Unauthorized** | Invalid or missing API key | Header `X-API-Key` is missing or value does not match provisioned client keys. | Verify that `$SUPPORTPILOT_API_KEY` is loaded in your environment. Ensure no whitespace or quotes are in the header value. Contact SupportPilot team to verify key provisioning. |
| **413 Payload Too Large** | Request exceeds size limit | In `/triage`: payload exceeds `MAX_BODY_BYTES` (10,240 bytes). In `/triage/batch`: batch contains more than 100 tickets. | In `/triage`, truncate email reply history. In `/triage/batch`, partition tickets into batches of $\le 100$ items. |
| **422 Unprocessable Entity** | Schema validation failed | Required fields missing or malformed JSON syntax. | Validate JSON structure against OpenAPI contract. (Note: `/triage/batch` accepts loose item dicts to report per-ticket partial failures under 200 OK). |
| **429 Too Many Requests** | Rate limit exceeded | Exceeded quota of 60 requests per 60 seconds, or exceeded 1,000 tickets per minute for batch endpoints. | Inspect `Retry-After` header. Implement exponential backoff and pacing (e.g. 1 batch of 100 every 6 seconds). Safe to retry due to idempotent caching. |
| **503 Service Unavailable** | Cold start or initialization | Free-tier container spins down after 15 min of inactivity; spin-up takes ~50s. | Retry request after 30–60 seconds. Uptime monitor keeps container warm during business hours. |

---

## 5. Security & Privacy Guarantees

- **Zero-PII Storage**: All incoming ticket text is scrubbed through PII redaction filters before inference. No raw ticket bodies, customer names, email addresses, or phone numbers are ever persisted in databases or application logs.
- **Key Hashing**: API keys are authenticated using constant-time byte comparisons. Usage tracking records only an 8-character cryptographic SHA-256 hash prefix (`key_hash`), never raw keys.

