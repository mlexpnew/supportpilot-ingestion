#!/usr/bin/env python3
"""SupportPilot Client Integration Diagnostic Tool (SP-115).

Verifies connectivity and authentication with the SupportPilot API.
Zero hardcoded keys: reads credentials strictly from $SUPPORTPILOT_API_KEY.

Usage:
    export SUPPORTPILOT_API_KEY="your-assigned-key"
    python examples/client_check.py
"""

from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.request


def main() -> int:
    # 1. Read SUPPORTPILOT_API_KEY from environment
    api_key = os.getenv("SUPPORTPILOT_API_KEY")
    if not api_key or not api_key.strip():
        print("❌ Error: SUPPORTPILOT_API_KEY environment variable is missing.")
        print("   Set it with: export SUPPORTPILOT_API_KEY='your-key-here'")
        return 1

    api_key = api_key.strip()
    base_url = os.getenv(
        "SUPPORTPILOT_BASE_URL", "https://supportpilot-ingestion.onrender.com"
    ).rstrip("/")

    # 2. Call /health -> print ✅/❌
    health_url = f"{base_url}/health"
    try:
        req = urllib.request.Request(health_url, method="GET")
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            if resp.status == 200 and data.get("status") == "ok":
                print("✅ /health")
            else:
                print(f"❌ /health (status {resp.status})")
                return 1
    except Exception as e:
        print(f"❌ /health ({e})")
        return 1

    # 3. Send one test /triage -> print ✅ + label, or ❌ + fix for 401/422/429
    triage_url = f"{base_url}/triage"
    test_payload = {
        "ticket_id": "TEST-ONBOARDING-001",
        "subject": "Integration test: order status check",
        "body": "Where is my package? I ordered it three days ago and haven't received a tracking update.",
    }
    encoded = json.dumps(test_payload).encode("utf-8")
    headers = {
        "Content-Type": "application/json",
        "X-API-Key": api_key,
    }

    try:
        req = urllib.request.Request(
            triage_url, data=encoded, headers=headers, method="POST"
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            result = json.loads(resp.read().decode("utf-8"))
            label = result.get("label", "unknown")
            print(f"✅ /triage (label: {label})")
            return 0
    except urllib.error.HTTPError as e:
        if e.code == 401:
            print("❌ /triage (401 Unauthorized)")
            print(
                "   Fix: Check that $SUPPORTPILOT_API_KEY matches your assigned key with no trailing spaces."
            )
        elif e.code == 422:
            print("❌ /triage (422 Unprocessable Entity)")
            print(
                "   Fix: Validate payload format. Ensure ticket_id (1-64 chars) and non-empty body are present."
            )
        elif e.code == 429:
            print("❌ /triage (429 Too Many Requests)")
            print(
                "   Fix: Rate limit exceeded (60 req/min). Wait and retry according to Retry-After header."
            )
        elif e.code == 413:
            print("❌ /triage (413 Payload Too Large)")
            print(
                "   Fix: Request body exceeds 10 KB limit. Truncate previous email thread history."
            )
        else:
            print(f"❌ /triage (HTTP {e.code})")
        return 1
    except Exception as e:
        print(f"❌ /triage ({e})")
        return 1


if __name__ == "__main__":
    sys.exit(main())
