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
```

The script performs:
1. Public liveness probe (`GET /health`)
2. Authenticated ticket triage (`POST /triage`)
3. Validation of response schema and predictions

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

### 3.3 Human Feedback: `POST /feedback`

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
| **413 Payload Too Large** | Request body exceeds 10 KB | Ingestion payload exceeds `MAX_BODY_BYTES` (10,240 bytes). Usually caused by attaching full email thread history or raw base64 signatures. | Truncate previous email reply history before sending. Send only the subject and recent customer inquiry text. |
| **422 Unprocessable Entity** | Schema validation failed | Required fields missing (`ticket_id` or `body`), `ticket_id` > 64 characters, or malformed JSON syntax. | Validate JSON structure against OpenAPI contract. Ensure `ticket_id` is a non-empty string between 1 and 64 characters. |
| **429 Too Many Requests** | Rate limit exceeded | Exceeded quota of 60 requests per 60 seconds per API key. | Inspect `Retry-After: 60` response header. Implement exponential backoff and jitter. If your throughput requirement exceeds 1 req/sec, contact SupportPilot for rate limit increases. |
| **503 Service Unavailable** | Cold start or initialization | Free-tier container spins down after 15 min of inactivity; spin-up takes ~50s. | Retry request after 30–60 seconds. Uptime monitor keeps container warm during business hours. |

---

## 5. Security & Privacy Guarantees

- **Zero-PII Storage**: All incoming ticket text is scrubbed through PII redaction filters before inference. No raw ticket bodies, customer names, email addresses, or phone numbers are ever persisted in databases or application logs.
- **Key Hashing**: API keys are authenticated using constant-time byte comparisons. Usage tracking records only an 8-character cryptographic SHA-256 hash prefix (`key_hash`), never raw keys.
