# SP-103 — OrderHub Client Design

## 1. Problem & Context

SupportPilot's chat agent answers customer queries like *"Where is my order?"* by querying the merchant's OrderHub service. Under peak sales loads, OrderHub suffers from 5xx server errors, 429 throttling, multi-second hangs, and unexpected HTML gateway error pages.

Two strict constraints govern this integration:
1. **Chat Latency Budget**: Chat promises responses in under 4 seconds. The OrderHub client is allocated a strict **3.0-second total deadline** across all attempts.
2. **PII Zero-Leakage**: OrderHub responses include customer email and physical shipping address. Because output feeds directly into downstream LLM prompts, no customer personal information, headers, or API keys may escape the module.

---

## 2. Timing Budget & Arithmetic Plan

- **Total Deadline**: 3.0 s absolute deadline measured monotonically from the start of the call.
- **Max Attempts**: Capped at 3 attempts.
- **Per-Attempt Timeout**: Dynamic budget: $\text{timeout} = \max(0, \text{deadline} - \text{monotonic()})$.
- **Backoff Schedule**: Exponential backoff with jitter:
  $$\text{backoff} = \text{BASE\_BACKOFF\_SECONDS} \times 2^{(\text{attempt} - 1)} + \text{jitter}$$
  where $\text{BASE\_BACKOFF\_SECONDS} = 0.1\,\text{s}$ and $\text{jitter} \in [0, 0.25 \times \text{backoff}]$.
- **Arithmetic Guarantee**:
  - Attempt 1: timeout $\le 3.0\,\text{s}$.
  - Attempt 2: sleep $\approx 0.10\text{--}0.125\,\text{s}$, remaining budget $\le 2.9\,\text{s}$.
  - Attempt 3: sleep $\approx 0.20\text{--}0.25\,\text{s}$, remaining budget $\le 2.65\,\text{s}$.
  - Every sleep is bounded: $\text{sleep\_duration} = \min(\text{backoff}, \text{remaining\_budget})$. If $\text{remaining\_budget} \le 0$, the loop aborts immediately with `ServiceUnavailable`. Total execution cannot exceed 3.0 s.

---

## 3. The Six Architectural Decisions

### Decision 1: Which failures are retried, and which aren't?
- **Retried**: Transient errors where an immediate subsequent call may succeed:
  - HTTP `429` (Rate Limited).
  - HTTP `5xx` (500–599 server errors).
  - Network-layer errors: connection dropped (`URLError`), OS socket errors (`OSError`), and timeouts (`TimeoutError`).
- **Not Retried**: Deterministic client failures:
  - HTTP `404` (`OrderNotFound`): An order that does not exist will not exist milliseconds later. Immediate fail-fast.
  - HTTP `401` (`AuthenticationError`): A bad or missing API key requires operator intervention. Retrying wastes chat budget.

### Decision 2: Retry-After says 30 s but 1.5 s remain in budget. What happens?
- If $\text{retry\_delay} \ge \text{remaining\_budget}$, sleeping would guarantee a deadline breach.
- The client aborts immediately and raises `ServiceUnavailable("OrderHub request deadline exceeded")` without sleeping.

### Decision 3: What happens on a 200 response with a non-JSON body or missing fields?
- Upstream returned unexpected data (e.g., an HTML gateway error page or truncated payload).
- Raises `ServiceUnavailable`.
- Raw bodies, HTML snippets, and field error details are **never** echoed in exception messages to avoid HTML injection and PII leakage.

### Decision 4: What counts as a valid `order_id`?
- **Rules**: Must be a non-empty `str`, length $\le 64$ characters, matching `re.fullmatch(r"^[A-Za-z0-9_-]+$", order_id)`.
- **Why `fullmatch`**: Standard `re.match` with `$` treats a trailing newline `\n` as matching immediately before the end of the line. Using `re.fullmatch` strictly enforces that every character from index 0 to length matches the allowed ASCII set, categorically rejecting trailing newlines (`"55231\n"`), carriage returns (`"55231\r\n"`), spaces, tabs, null bytes (`"55231\x00"`), and Unicode full-width digits (`"５５２３１"`).
- **Enforcement**:
  - Path traversal (`../admin`): rejected before network.
  - Query parameter injection (`55231?x=1`): rejected before network.
  - Trailing newlines, tabs, and control characters: rejected before network.
  - Empty string (`""`) or non-string (`int`, `None`): rejected before network.
  - Oversized input (e.g., 65 characters or 10,000 characters): rejected before network.
- Raises `ValueError` prior to opening any socket or URL connection.

### Decision 5: Standard library `urllib` or an external library?
- **Decision**: Python standard library `urllib.request` only.
- **Reason**: The client performs straightforward HTTP `GET` queries. Stdlib avoids adding dependencies (`httpx`, `requests`), reduces supply-chain attack surface, and minimizes docker container image overhead.

### Decision 6: What do you log, and what must never be logged?
- **Allowed in Logs**: High-level event outcome (success/failure), HTTP status code, attempt count, and elapsed latency.
- **Forbidden from Logs**: Customer email, shipping address, recipient name, raw HTTP response payloads, request/response headers, and API keys (`ORDERHUB_API_KEY`).

---

## 4. Output Contract & PII Isolation

`get_order_status(order_id)` returns a frozen dataclass:
```python
@dataclass(frozen=True)
class OrderStatus:
    order_id: str
    status: str
    carrier: str | None
    eta: str | None
```

- Customer email and shipping address returned by the server are discarded during JSON parsing and never stored on the object.
- `OrderStatus.__repr__` contains only these four fields, guaranteeing downstream LLM prompts receive zero PII.