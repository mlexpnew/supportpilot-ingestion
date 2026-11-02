#!/usr/bin/env python3
"""SupportPilot Client Integration Diagnostic Tool (SP-115).

Verifies connectivity and authentication with SupportPilot API.
Zero hardcoded keys: reads credentials strictly from $SUPPORTPILOT_API_KEY.

Usage:
    export SUPPORTPILOT_API_KEY="your-client-api-key"
    python examples/client_check.py
"""

from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.request


def main() -> int:
    api_key = os.getenv("SUPPORTPILOT_API_KEY")
    if not api_key or not api_key.strip():
        print("❌ Error: SUPPORTPILOT_API_KEY environment variable is not set.")
        print("   Please export your key before running this script:")
        print("     export SUPPORTPILOT_API_KEY='your-assigned-api-key'")
        return 1

    api_key = api_key.strip()
    base_url = os.getenv(
        "SUPPORTPILOT_BASE_URL", "https://supportpilot-ingestion.onrender.com"
    ).rstrip("/")

    print("==================================================")
    print("  SupportPilot Integration Diagnostic Check")
    print("==================================================")
    print(f"Target Base URL: {base_url}")
    print(
        f"API Key:         {'*' * (len(api_key) - 4) + api_key[-4:] if len(api_key) >= 4 else '****'}"
    )
    print("--------------------------------------------------")

    # Step 1: Health Check (Public Liveness Probe)
    print("\n[1/2] Checking API service liveness (/health)...")
    health_url = f"{base_url}/health"
    try:
        req = urllib.request.Request(health_url, method="GET")
        with urllib.request.urlopen(req, timeout=30) as resp:
            status_code = resp.status
            body = resp.read().decode("utf-8")
            data = json.loads(body) if body else {}

        if status_code == 200 and data.get("status") == "ok":
            print("✅ /health check passed: Service is alive and healthy.")
        else:
            print(
                f"❌ /health returned unexpected status code {status_code} or body: {body}"
            )
            return 1
    except urllib.error.HTTPError as e:
        print(f"❌ /health failed with HTTP {e.code}: {e.reason}")
        return 1
    except urllib.error.URLError as e:
        print(f"❌ /health connection failed: {e.reason}")
        print(
            "   Hint: If hosting on Render free tier, the service may be spinning up from cold start."
        )
        print("   Please wait 60 seconds and run the script again.")
        return 1
    except Exception as e:
        print(f"❌ Unexpected error connecting to /health: {e}")
        return 1

    # Step 2: Triage Check (Authenticated POST /triage)
    print("\n[2/2] Checking authenticated ticket triage (/triage)...")
    triage_url = f"{base_url}/triage"
    test_payload = {
        "ticket_id": "TEST-INTEGRATION-CHECK-001",
        "subject": "Integration test: order tracking question",
        "body": "Hello SupportPilot team, this is an automated client integration test payload.",
    }
    encoded_data = json.dumps(test_payload).encode("utf-8")

    headers = {
        "Content-Type": "application/json",
        "X-API-Key": api_key,
        "User-Agent": "SupportPilot-ClientCheck/1.0",
    }

    try:
        req = urllib.request.Request(
            triage_url, data=encoded_data, headers=headers, method="POST"
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            status_code = resp.status
            resp_body = resp.read().decode("utf-8")
            result = json.loads(resp_body) if resp_body else {}

        if status_code == 200:
            label = result.get("label", "unknown")
            confidence = result.get("confidence", 0.0)
            needs_review = result.get("needs_review", False)
            print("✅ /triage check passed: Successfully classified test ticket!")
            print(f"   Predicted Label: {label}")
            print(f"   Confidence:      {confidence:.4f}")
            print(f"   Needs Review:    {needs_review}")
        else:
            print(
                f"❌ /triage returned unexpected HTTP status {status_code}: {resp_body}"
            )
            return 1
    except urllib.error.HTTPError as e:
        raw_error = e.read().decode("utf-8", errors="replace")
        print(f"❌ /triage failed with HTTP {e.code}: {e.reason}")
        if e.code == 401:
            print("   👉 Cause: Authentication failed. Your API key was rejected.")
            print(
                "   👉 Fix:   Verify that $SUPPORTPILOT_API_KEY matches your assigned key with no trailing spaces."
            )
        elif e.code == 413:
            print(
                "   👉 Cause: Payload Too Large. Ticket body exceeded the 10 KB limit."
            )
            print("   👉 Fix:   Truncate request body before sending.")
        elif e.code == 422:
            print(
                "   👉 Cause: Unprocessable Entity. Request schema or fields are malformed."
            )
            print(f"   👉 Server detail: {raw_error}")
        elif e.code == 429:
            print("   👉 Cause: Rate limit exceeded (limit is 60 req/min).")
            print(
                "   👉 Fix:   Slow down requests and respect the Retry-After response header."
            )
        else:
            print(f"   👉 Server response: {raw_error}")
        return 1
    except urllib.error.URLError as e:
        print(f"❌ /triage connection failed: {e.reason}")
        return 1
    except Exception as e:
        print(f"❌ Unexpected error connecting to /triage: {e}")
        return 1

    print("\n--------------------------------------------------")
    print("🎉 All checks passed! Your integration is ready for production traffic.")
    print("==================================================")
    return 0


if __name__ == "__main__":
    sys.exit(main())
