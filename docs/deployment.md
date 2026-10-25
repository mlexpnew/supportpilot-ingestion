# Deployment & Operations Guide (SP-108)

This guide documents the production deployment lifecycle for the SupportPilot Ticket Triage API service, including container builds, local execution, environment variables, Render cloud deployment, and rollback procedures.

---

## 1. Build

The service packages a multi-stage Docker container (`Dockerfile`) designed to keep image size under 400 MB and minimize runtime attack surface.

### Architecture
- **Builder Stage (`python:3.11-slim`)**:
  - Compiles virtual environment in `/opt/venv` from pinned dependencies in `requirements.txt`.
  - Trains the baseline triage classification model from dataset artifacts (`python -m supportpilot.triage --train`).
  - Cleans up compiler artifacts, package caches, and strips unneeded binary debug symbols (`strip --strip-unneeded`).
- **Runner Stage (`python:3.11-slim`)**:
  - Runs as an unprivileged non-root user (`appuser:appuser`, UID/GID system-allocated).
  - Copies strictly the `/opt/venv` runtime environment and the serialized model (`models/triage_model.joblib`).
  - Removes build tools and `curl` from the runtime image to reduce attack surface.
  - Native Python standard library healthcheck (`urllib.request`) polls `/health`.

### Build Context & `.dockerignore`
The `.dockerignore` file excludes local environments, test files, and git history to guarantee clean, reproducible builds:
```text
.git
.venv
out*/
models/
tests/
```

### Build Command
```bash
docker build -t supportpilot-api:latest .
```

### Image Size Verification
```bash
docker images supportpilot-api:latest
```
*Current production size: **380 MB** (strictly beneath the 400 MB constraint).*

---

## 2. Run

### Local Container Execution
Run the container in detached mode mapping port 8000 (at least one valid key is required in `TRIAGE_API_KEYS`):
```bash
docker run -d --name supportpilot-api -p 8000:8000 \
  -e TRIAGE_API_KEYS="secret-key-1,secret-key-2" \
  supportpilot-api:latest
```

### Running with Custom Configuration
```bash
docker run -d --name supportpilot-api -p 8000:8000 \
  -e TRIAGE_API_KEYS="secret-key-1,secret-key-2" \
  -e TRIAGE_THRESHOLD=0.35 \
  supportpilot-api:latest
```

### Inspecting Logs
Follow live server logs:
```bash
docker logs -f supportpilot-api
```

### Verifying Service Health
Query the public `/health` endpoint (no API key required):
```bash
curl -i http://localhost:8000/health
```
Expected response (`HTTP 200 OK`):
```json
{"status":"ok","model_loaded":true,"model_version":"triage-v1.0.0-20261022"}
```

### Unauthenticated Request Rejection
Requests to `/triage` without an `X-API-Key` header are rejected before schema parsing:
```bash
curl -i -X POST http://localhost:8000/triage \
  -H "Content-Type: application/json" \
  -d '{"ticket_id":"T-TEST-1","body":"Where is my order?"}'
```
Expected response (`HTTP 401 Unauthorized`):
```json
{"detail":"Unauthorized"}
```

### Testing Authenticated Triage Prediction
Provide a valid API key in the `X-API-Key` header:
```bash
curl -s -X POST http://localhost:8000/triage \
  -H "X-API-Key: secret-key-1" \
  -H "Content-Type: application/json" \
  -d '{"ticket_id":"T-TEST-1","body":"Where is my order? Delivery was expected yesterday."}'
```
Expected response (`HTTP 200 OK`):
```json
{
  "ticket_id": "T-TEST-1",
  "label": "delivery",
  "confidence": 0.3172,
  "needs_review": true,
  "model_version": "triage-v1.0.0-20261022"
}
```

### Querying Operational Telemetry
Telemetry endpoint `/metrics` requires API key authentication:
```bash
curl -s http://localhost:8000/metrics \
  -H "X-API-Key: secret-key-1"
```
Expected response:
```json
{
  "requests_total": 1,
  "triage_total": 1,
  "errors_total": 0,
  "avg_latency_ms": 3.12,
  "needs_review_rate": 1.0,
  "rate_limited_total": 0
}
```

---

## 3. Environment Variables

All configuration settings are read dynamically at application startup:

| Variable | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `TRIAGE_API_KEYS` | `string` | *(Required)* | Comma-separated list of authorized API keys. At least one non-empty key is required. |
| `DATABASE_URL` | `string` | *(Required in Prod)* | PostgreSQL connection URI for feedback & predictions storage. Must start with `postgresql://` or `postgres://`. |
| `ALLOW_SQLITE_FALLBACK` | `int` | `0` | If set to `1`, allows fallback to local SQLite when `DATABASE_URL` is omitted (local dev only). Missing `DATABASE_URL` fails fast in production. |
| `TRIAGE_THRESHOLD` | `float` | `0.35` | Minimum classification confidence required to avoid manual human review. Predictions below this cutoff mark `needs_review: true`. Must be between `0.0` and `1.0`. |
| `TRIAGE_MODEL_PATH`| `string`| `models/triage_model.joblib` | Filesystem path to the serialized trained model artifact. |
| `LOG_LEVEL` | `string` | `INFO` | Application and server log level (`DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL`). Invalid values fail fast at startup. |
| `PORT` | `int` | `8000` | Port on which the Uvicorn ASGI server listens. |

### Validation & Fail-Fast Guarantees
- **Missing or Empty API Keys**: If `TRIAGE_API_KEYS` is missing, blank, or contains no non-empty comma-separated keys, startup fails immediately with `ValueError`.
- **Missing Database URL**: In production, if `DATABASE_URL` is missing and `ALLOW_SQLITE_FALLBACK != 1`, startup fails immediately with `RuntimeError`.
- **Invalid Log Level**: If `LOG_LEVEL` is not a valid standard level, startup fails immediately with `ValueError`.
- **Invalid Float / Format**: If `TRIAGE_THRESHOLD` is not a valid float, startup fails immediately with `ValueError`.
- **Out of Range**: If `TRIAGE_THRESHOLD` is outside `[0.0, 1.0]`, startup fails fast with `ValueError`.
- **Missing Model File**: If the model artifact cannot be located at `TRIAGE_MODEL_PATH`, startup fails fast with `FileNotFoundError`.

### Structured Logging & Observability
- **JSON Format**: Application logs are formatted as single-line JSON objects with `timestamp` (UTC ISO 8601), `level`, `logger`, `message`, and any contextual `extra` fields.
- **Request Logging**: Uvicorn access logging is disabled (`--no-access-log`). The request-ID middleware logs exactly one structured JSON line per request with `request_id`, `path`, `method`, `status`, and `latency_ms`.
- **Alembic Startup Logs**: The pre-server migration command (`alembic upgrade head`) prints plain-text startup migration status lines to stdout prior to Uvicorn booting. This is normal and acceptable.
- **Privacy Sanitization**: Logs pass through a sensitive data filter that automatically masks credentials, database URLs, API keys (`X-API-Key`), and ticket bodies (`[REDACTED]`), while preserving audit `api_key_hash`.

---

## 4. API Key Management & Security

### Generating Strong API Keys
Generate cryptographically random 256-bit URL-safe tokens (43 characters in base64url) using Python's standard `secrets` module:
```bash
python -c "import secrets; print(secrets.token_urlsafe(32))"
```

### Secret Hygiene & Distribution Rules
- **Never Commit Secrets**: Never commit raw API keys, `.env`, or `.secrets` files to Git. Secrets must remain ignored in `.gitignore`.
- **Private Channels Only**: Distribute client and team API keys (e.g., Neha's key) **strictly through private, encrypted channels** (such as 1Password vaults or direct encrypted messages) — **never in chat transcripts, issue trackers, or Git commits**.
- **Audit Logging**: The API server hashes API keys with SHA-256 and logs only the first 8 characters (`api_key_hash=<hash[:8]>`). Raw secret strings never appear in captured server logs.

### Adding a New Key
To add a new key, append it to the comma-separated `TRIAGE_API_KEYS` environment variable:
```bash
TRIAGE_API_KEYS=existing-key-1,new-key-2
```
Save and redeploy. Both keys are immediately active.

### Zero-Downtime Key Rotation
To rotate an API key without service downtime:
1. **Generate**: Generate a new strong key using `python -c "import secrets; print(secrets.token_urlsafe(32))"`.
2. **Dual-Key Stage**: Update `TRIAGE_API_KEYS` in the Render environment settings to contain both the old and new keys separated by a comma (e.g., `TRIAGE_API_KEYS=<old_key>,<new_key>`).
3. **Deploy**: Save and trigger a redeploy. While the deployment rolls out, both old and new keys will authenticate successfully.
4. **Distribute & Migrate**: Transmit `<new_key>` to the consumer (e.g. Neha) via a secure private channel. Update downstream client configurations to use `<new_key>`.
5. **Revoke Old Key**: Once clients have migrated, update `TRIAGE_API_KEYS` to remove `<old_key>` (e.g., `TRIAGE_API_KEYS=<new_key>`) and redeploy. The old key is permanently revoked, and any requests still using it will receive `401 Unauthorized`.

### Emergency Key Revocation
If an API key is leaked or compromised:
1. Open the **Render Dashboard** → **Environment Settings**.
2. Remove the compromised key from `TRIAGE_API_KEYS`.
3. Save and trigger a redeployment immediately.
4. Once deployed, the compromised key will fail authentication instantly with `401 Unauthorized` without consuming rate limit quota or leaking schema details.

---

## 5. Rate Limiting Rules & Telemetry

### Rate Limit Specification
- **Window**: 60-second sliding window (`RATE_LIMIT_WINDOW_SECONDS = 60.0`).
- **Quota**: Up to 60 requests (`RATE_LIMIT_MAX_REQUESTS = 60`) per window per API key.
- **Execution Order**: The rate limit check executes *after* successful authentication. Unauthenticated requests are rejected immediately with 401 and do not consume quota.
- **Clock Precision**: Tracks timestamps using `time.monotonic()` rather than wall-clock time (`time.time()`), preventing window corruption from server NTP adjustments or leap seconds.
- **HTTP 429 Too Many Requests**: When the 61st request arrives within 60 seconds, the server rejects it with:
  ```http
  HTTP/1.1 429 Too Many Requests
  Retry-After: 42
  Content-Type: application/json

  {"detail":"Rate limit exceeded"}
  ```
  The `Retry-After` header indicates the integer seconds until the oldest request in the window expires and quota becomes available.
- **Metrics Exposure**: Every 429 response increments `rate_limited_total`, visible on `/metrics`.

### Single-Instance Limitation
> ⚠️ **In-Memory Single-Instance Limitation**:
> Rate limiting timestamps are tracked in-memory within FastAPI application state (`app.state.rate_limit_history`) using Python deques.
> - **In-memory limits only work for a single instance**: In a single-instance deployment (e.g. standard Render Web Service), the 60 req/min quota is strictly enforced per key.
> - **Horizontal Scaling**: If the service is scaled to $N$ container instances or deployed behind a round-robin load balancer, each instance maintains independent memory state. A single API key could issue up to $N \times 60$ requests per minute across the fleet.
> - **Multi-Instance Migration**: For horizontal multi-instance production clusters, replace the in-memory deque with a centralized distributed cache (e.g. Redis sliding window using `ZADD` / `ZREMRANGEBYSCORE` or Redis token bucket).

---

## 6. Deploy

### Continuous Integration (GitHub Actions)
Every pull request and push to `main` or `feature/**` triggers `.github/workflows/ci.yml`:
1. **`test` Job**: Runs pytest across Python 3.11 and 3.12, verifying linters (`black --check .`, `ruff check .`).
2. **`docker` Job**:
   - Builds container from Dockerfile context.
   - Enforces the image size guard ($< 400\text{ MB}$).
   - Boots container with `--name api`.
   - Polls `/health` with a 40-second timeout guard.
   - Dumps `docker logs api` on failure (`if: failure()`).

### Render Cloud Deployment
The service is hosted on Render as a Docker Web Service.

#### Pre-Deployment Requirement:
> 🚨 **Critical Deployment Order**: Because `TRIAGE_API_KEYS` is validated at startup with fail-fast enforcement, you **must set `TRIAGE_API_KEYS` in the Render environment variables BEFORE merging PR #15 or deploying**, otherwise the new container will fail to start.

#### Service Configuration:
- **Service Type**: Web Service
- **Repository**: `mlexpnew/supportpilot-ingestion`
- **Branch**: `main` (post-merge)
- **Runtime**: **Docker** (Render uses repo `Dockerfile`)
- **Port**: `8000` (Render auto-detects `EXPOSE 8000`)
- **Health Check Path**: `/health`
- **Instance Type**: Free tier or Starter (512 MB+ RAM)
- **Environment Variables**:
  - `TRIAGE_THRESHOLD`: `0.35`
  - `TRIAGE_API_KEYS`: `<comma-separated-api-keys>`

#### Production Verification Checklist:
Once deployed on Render, verify endpoints:
1. **Health Check (Public)**:
   ```bash
   curl -i https://supportpilot-ingestion.onrender.com/health
   # HTTP/2 200 OK
   # {"status":"ok","model_loaded":true,"model_version":"triage-v1.0.0-YYYYMMDD"}
   ```
2. **Unauthenticated Triage Rejection (401)**:
   ```bash
   curl -i -X POST https://supportpilot-ingestion.onrender.com/triage \
     -H "Content-Type: application/json" \
     -d '{"ticket_id":"T-VERIFY-1","body":"Where is my order?"}'
   # HTTP/2 401 Unauthorized
   # {"detail":"Unauthorized"}
   ```
3. **Authenticated Triage Prediction (200)**:
   ```bash
   curl -i -X POST https://supportpilot-ingestion.onrender.com/triage \
     -H "X-API-Key: <your-key>" \
     -H "Content-Type: application/json" \
     -d '{"ticket_id":"T-VERIFY-1","body":"Where is my order? Delivery was scheduled today."}'
   # HTTP/2 200 OK
   # {"ticket_id":"T-VERIFY-1","label":"delivery","confidence":...,"needs_review":...,"model_version":"triage-v1.0.0-YYYYMMDD"}
   ```
4. **Telemetry (Authenticated)**:
   ```bash
   curl -i https://supportpilot-ingestion.onrender.com/metrics \
     -H "X-API-Key: <your-key>"
   # HTTP/2 200 OK
   ```

#### Production URLs:
- **Base Service**: `https://supportpilot-ingestion.onrender.com`
- **Healthcheck**: `https://supportpilot-ingestion.onrender.com/health`
- **Interactive Documentation**: `https://supportpilot-ingestion.onrender.com/docs`
- **Telemetry**: `https://supportpilot-ingestion.onrender.com/metrics`

---

## 7. Rollback

In the event of an incident or degraded release, use one of the following rollback procedures:

### Option A: Render Dashboard Instant Rollback (Zero Downtime)
1. Open the **SupportPilot** service in the [Render Dashboard](https://dashboard.render.com).
2. Click **Deploys** in the left navigation sidebar.
3. Scroll through the deployment history to locate the last known healthy deployment commit.
4. Click the three dots (`...`) on the right side of the healthy deploy card.
5. Select **Rollback to this deploy**.
6. Render will instantly switch traffic back to the previous healthy container image without re-building.

### Option B: Git Revert (Standard Pipeline)
To revert via source control and re-run standard CI validation:
1. Revert the problematic commit on your local branch:
   ```bash
   git fetch origin
   git checkout main
   git revert <bad-commit-sha>
   ```
2. Push the revert commit:
   ```bash
   git push origin main
   ```
3. GitHub Actions CI will validate the reverted code.
4. Render will detect the new commit and automatically build and deploy the reverted version.

### Option C: Environment Variable Rollback
If an incorrect environment variable (such as an overly strict `TRIAGE_THRESHOLD`) caused degraded behavior:
1. Go to **Environment** in the Render service dashboard.
2. Update the variable to the safe baseline (`TRIAGE_THRESHOLD=0.35`).
3. Click **Save Changes**.
4. Render triggers an automatic rolling redeploy with the corrected environment variable.
