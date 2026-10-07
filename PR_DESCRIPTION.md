## SP-103: Resilient OrderHub Client

### Summary

Implements a resilient, zero-PII client for querying the merchant's OrderHub order-status API (`get_order_status(order_id)`), designed to withstand peak sales instability and strict chat SLAs.
- **Strict Latency Budget**: Guarantees completion within an absolute 3.0-second deadline across up to 3 bounded retry attempts using dynamic timeout budgets and exponential backoff with jitter.
- **Zero PII Leakage**: Strips customer email, shipping addresses, and API keys from upstream responses; returns a frozen `OrderStatus` containing only `order_id`, `status`, `carrier`, and `eta`. Zero personal data in return values, `repr`, logs, or exceptions.
- **Pre-Network Guard**: Rejects path traversal (`../admin`), query injection (`?x=1`), empty strings, and inputs > 64 chars before opening network sockets.
- **Typed Error Disambiguation**: Categorizes failures cleanly into `OrderNotFound` (404), `AuthenticationError` (401), and `ServiceUnavailable` (5xx, timeouts, 429 exhaustion, corrupt responses).
- **Stdlib Only**: Built entirely using Python standard library `urllib.request` with zero third-party networking dependencies.

---

### Evidence Table (Priya's Acceptance Criteria)

Tested live against `tools/fake_orderhub.py` on `127.0.0.1:8099`:

| Acceptance Criterion | Fake Server Behavior | Actual Measured Output / Outcome | Elapsed | Server Hits | Status |
| :--- | :--- | :--- | :--- | :--- | :---: |
| **`55231` Works** | 200 OK | Returns `OrderStatus(order_id='55231', status='shipped', carrier='BlueDart', eta='2025-01-17')`. Exactly 4 fields; zero email or address in object or `repr`. | `0.001 s` | 1 | ✅ PASS |
| **`99999` Not Found** | 404 Not Found | Raises `OrderNotFound('Order not found')`. Exactly 1 request; no retries attempted. | `0.001 s` | 1 | ✅ PASS |
| **`55233` Always 500** | Repeated 500 | Raises `ServiceUnavailable('OrderHub service is unavailable')`. Bounded to exactly 3 attempts; finishes inside 3.0 s. | `0.366 s` | 3 | ✅ PASS |
| **`55234` Throttled (429)** | Two 429s, then 200 | Respects `Retry-After: 1` twice, succeeds on attempt 3: `OrderStatus(order_id='55234', status='delivered', carrier='Delhivery', eta=None)`. | `2.013 s` | 3 | ✅ PASS |
| **`55235` Hung Server** | Hangs 10 s | Dynamic deadline budget cancels socket: raises `ServiceUnavailable('OrderHub request deadline exceeded')` at 3.0 s, not 10 s. | `3.002 s` | 1 | ✅ PASS |
| **`55236` Gateway Error** | HTML instead of JSON | Raises `ServiceUnavailable('OrderHub returned invalid JSON')`. Zero HTML or gateway body text exposed in message. | `0.001 s` | 1 | ✅ PASS |
| **Wrong API Key** | 401 Unauthorized | Raises `AuthenticationError('OrderHub authentication failed')`. Exactly 1 request; key never leaked in error. | `0.001 s` | 0 | ✅ PASS |
| **Malicious IDs** (`../admin`, `?x=1`, empty, 10k) | Pre-network rejection | All rejected immediately with `ValueError`. Zero network calls made. | `0.000 s` | 0 | ✅ PASS |
| **Fast Test Suite (12+ tests)** | Mocked responses / zero sleep | **19 unit tests passed** in `test_orderhub.py` (71 total in repo) with zero real waiting, completing in **0.03 s** (< 5.0 s threshold). | `0.03 s` | 0 | ✅ PASS |
| **Lint & Formatting** | CI Matrix (3.11 & 3.12) | `ruff check .` (0 errors), `black --check supportpilot/ tests/` (all clean). | `0.11 s` | 0 | ✅ PASS |

---

### Verification Commands

```bash
# Linting & Formatting
.venv/bin/ruff check .
.venv/bin/black --check supportpilot/ tests/

# Test Suite
pytest -q tests/test_orderhub.py
```
