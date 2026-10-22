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
Run the container in detached mode mapping port 8000:
```bash
docker run -d --name supportpilot-api -p 8000:8000 supportpilot-api:latest
```

### Running with Custom Configuration
```bash
docker run -d --name supportpilot-api -p 8000:8000 \
  -e TRIAGE_THRESHOLD=0.35 \
  supportpilot-api:latest
```

### Inspecting Logs
Follow live server logs:
```bash
docker logs -f supportpilot-api
```

### Verifying Service Health
Query the `/health` endpoint:
```bash
curl -i http://localhost:8000/health
```
Expected response (`HTTP 200 OK`):
```json
{"status":"ok","model_loaded":true}
```

### Testing Triage Prediction
```bash
curl -s -X POST http://localhost:8000/triage \
  -H "Content-Type: application/json" \
  -d '{"ticket_id":"T-TEST-1","body":"Where is my order? Delivery was expected yesterday."}'
```
Expected response:
```json
{
  "ticket_id": "T-TEST-1",
  "label": "delivery",
  "confidence": 0.3172,
  "needs_review": true
}
```

### Querying Operational Telemetry
```bash
curl -s http://localhost:8000/metrics
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

> Note: In-memory limits only work for a single instance.

---

## 3. Environment Variables

All configuration settings are read dynamically at application startup:

| Variable | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `TRIAGE_THRESHOLD` | `float` | `0.35` | Minimum classification confidence required to avoid manual human review. Predictions below this cutoff mark `needs_review: true`. Must be between `0.0` and `1.0`. |
| `TRIAGE_MODEL_PATH`| `string`| `models/triage_model.joblib` | Filesystem path to the serialized trained model artifact. |
| `PORT` | `int` | `8000` | Port on which the Uvicorn ASGI server listens. |

### Validation & Fail-Fast Guarantees
- **Invalid Float / Format**: If `TRIAGE_THRESHOLD` is not a valid float, startup fails immediately with `ValueError`.
- **Out of Range**: If `TRIAGE_THRESHOLD` is outside `[0.0, 1.0]`, startup fails fast with `ValueError`.
- **Missing Model File**: If the model artifact cannot be located at `TRIAGE_MODEL_PATH`, startup fails fast with `FileNotFoundError`.

---

## 4. Deploy

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

#### Service Configuration:
- **Service Type**: Web Service
- **Repository**: `mlexpnew/supportpilot-ingestion`
- **Branch**: `feature/SP-108-docker-deploy` (or `main` post-merge)
- **Runtime**: **Docker** (Render uses repo `Dockerfile`)
- **Port**: `8000` (Render auto-detects `EXPOSE 8000`)
- **Health Check Path**: `/health`
- **Instance Type**: Free tier or Starter (512 MB+ RAM)
- **Environment Variables**:
  - `TRIAGE_THRESHOLD`: `0.35`

#### Production URLs:
- **Base Service**: `https://supportpilot-ingestion.onrender.com`
- **Healthcheck**: `https://supportpilot-ingestion.onrender.com/health`
- **Interactive Documentation**: `https://supportpilot-ingestion.onrender.com/docs`
- **Telemetry**: `https://supportpilot-ingestion.onrender.com/metrics`

---

## 5. Rollback

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
