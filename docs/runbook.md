# SupportPilot Operations Runbook

Production operations and emergency incident response procedures for SupportPilot.

- **On call (pilot week 26 Oct – 1 Nov)**: Chandini
- **How to reach**: mlexpnew@gmail.com / Slack: @chandini
- **Backup**: Dev

---


## Quick Operational Links

| Resource | URL / Destination | Notes |
| :--- | :--- | :--- |
| **Render Web Service** | [Render Dashboard](https://dashboard.render.com) (`supportpilot-ingestion`) | Service status, deploy history, manual restart, env vars |
| **Render PostgreSQL** | [Render Dashboard](https://dashboard.render.com) (`supportpilot-postgres`) | PostgreSQL metrics, connections, suspension status |
| **GitHub Actions CI** | [mlexpnew/supportpilot-ingestion/actions](https://github.com/mlexpnew/supportpilot-ingestion/actions) | Automated test, lint, and Docker container verification |
| **Uptime Monitor** | [Uptime Monitor Dashboard](https://uptimerobot.com/dashboard) | External `/health` polling (5 min interval, 2-strike alert) |
| **Live Health Endpoint** | `https://supportpilot-ingestion.onrender.com/health` | Public unauthenticated liveness probe |
| **Live Alerts Endpoint** | `https://supportpilot-ingestion.onrender.com/alerts` | Active health alerts (`X-API-Key` protected) |
| **Live Metrics Endpoint** | `https://supportpilot-ingestion.onrender.com/metrics` | Operational counters, telemetry, and live accuracy |

---

## Incident 1: API Down / 5xx Spike

### Symptoms
- Uptime monitor sends email notification: Service is DOWN (2 consecutive failed checks).
- Public `/health` endpoint times out or returns HTTP 502/503.
- Support agents report inability to triage tickets or submit feedback.
- `GET /alerts` returns active `error_rate` alert (`severity: critical` or `severity: warning`).

### Check
```bash
# 1. Check public service liveness
curl -i https://supportpilot-ingestion.onrender.com/health

# 2. Check active operational alerts (requires API key)
curl -s -H "X-API-Key: $TRIAGE_API_KEY" https://supportpilot-ingestion.onrender.com/alerts | jq .

# 3. Check live JSON logs in Render Dashboard for 5xx errors
# Note: Render log search is plain text (not a query language).
# Search for: '"status": 50' or '"level": "ERROR"'

```

### Fix
1. **Rule out free-tier cold start**: On Render's free tier, the container spins down after 15 minutes of inactivity and requires ~50s to wake up. Wait 60 seconds and re-check `/health`.
2. **Restart Web Service**: In [Render Dashboard](https://dashboard.render.com) -> SupportPilot Web Service -> Click **Manual Deploy** -> Select **Restart service**.
3. **Inspect runtime crash logs**: Review recent deployment logs for unhandled exceptions, memory exhaustion (OOM), or missing environment variables (`TRIAGE_API_KEYS`, `DATABASE_URL`).
4. **Rollback if recent deploy caused outage**: If downtime began immediately after a deployment, proceed immediately to **Incident 5: Bad Deploy (Rollback Procedure)**.

### Notify
- **Dev** (Infrastructure Lead): Investigate Render container health, server logs, and memory/CPU limits.
- **Neha** (Support Lead): Alert support agents of transient downtime and communicate expected resolution time.

---

## Incident 2: Database Unreachable

### Symptoms
- Service logs show repeated warnings: `DATABASE_URL not set; falling back to SQLite storage` or `Failed to save prediction`.
- `POST /feedback` returns HTTP 500 (`Feedback store is not initialized. Service startup invariant violated`).
- Render PostgreSQL dashboard displays status `Suspended` or connection count exceeds quota (>50).
- New feedback entries fail to persist across redeployments.

### Check
```bash
# 1. Test direct PostgreSQL connectivity
# Note: From your laptop, use the External Database URL from the Render Dashboard (the Internal URL only resolves inside Render's private network).
psql "$EXTERNAL_DATABASE_URL" -c "SELECT 1;"

# 2. Inspect Render application logs for database connection timeouts
# Search plain text: "feedback_store" or "OperationalError" or "Connection refused"

# 3. Verify PostgreSQL instance state in Render Dashboard (must show "Available")
```

### Fix
1. **Resume suspended database**: Render free-tier databases automatically suspend after 90 days of inactivity. In [Render Dashboard](https://dashboard.render.com) -> PostgreSQL -> Click **Resume Database**.
2. **Verify connection credentials**: In Web Service Environment settings, verify that `DATABASE_URL` exactly matches the PostgreSQL **Internal Database URL** provided by Render (the web service runs inside Render's private network).
3. **Reset exhausted connection pool**: Render free tier limits connections to 50 concurrent handles. If connection pool leaked, restart the web service to terminate abandoned sockets.
4. **Emergency backup & restore (`pg_dump`)**: If database corruption is suspected, execute an emergency dump before attempting recovery (from your laptop, connect using the **External Database URL**):
   ```bash
   # Emergency backup of existing state (run from laptop with External Database URL)
   pg_dump "$EXTERNAL_DATABASE_URL" --no-owner --no-acl -f emergency_backup_$(date +%Y%m%d_%H%M%S).sql

   # Restore from last verified logical backup
   psql "$EXTERNAL_DATABASE_URL" -f backup_latest.sql
   ```

5. **Run database migrations**: Ensure schema tables (`predictions`, `feedback`) exist:
   ```bash
   alembic upgrade head
   ```

### Notify
- **Dev** (Infrastructure Lead): Check Render Postgres resource consumption, network routing, and connection pool saturation.
- **Meera** (Product Lead): Advise that `/triage` continues operating in degraded mode, but feedback analytics logging is temporarily impaired.

---

## Incident 3: Accuracy Drop

### Symptoms
- `GET /alerts` returns active `accuracy` alert (`severity: warning` when accuracy $< 0.80$; `severity: critical` when accuracy $< 0.70$).
- Support agents report frequent misclassifications for specific ticket categories (e.g. `billing` or `delivery`).
- `GET /metrics/labels` shows category `recall` dropping significantly below baseline targets.

### Check
```bash
# 1. Check active model health alerts
curl -s -H "X-API-Key: $TRIAGE_API_KEY" https://supportpilot-ingestion.onrender.com/alerts | jq .

# 2. Check live global accuracy and feedback count (minimum 20 samples required)
curl -s -H "X-API-Key: $TRIAGE_API_KEY" https://supportpilot-ingestion.onrender.com/metrics | jq '{live_accuracy, feedback_total}'

# 3. Check per-category recall breakdown
curl -s -H "X-API-Key: $TRIAGE_API_KEY" https://supportpilot-ingestion.onrender.com/metrics/labels | jq .
```

### Fix
1. **Verify sample size qualification**: Confirm `feedback_total >= 20` (alerts are suppressed below 20 feedback entries to avoid small-sample noise).
2. **Execute Champion/Challenger retraining pipeline (SP-113)**:
   ```bash
   # Query active production model version from /version
   CURRENT_VER=$(curl -s https://supportpilot-ingestion.onrender.com/version | jq -r .model_version)

   python -m supportpilot.triage.retrain \
     --base data/triage_labeled.jsonl \
     --feedback data/feedback_export.jsonl \
     --pilot out_redacted/redacted.jsonl \
     --out models/ \
     --expected-champion-version "$CURRENT_VER"
   ```
3. **Inspect comparison report**:
   - Check printed summary table and `reports/compare_<YYYYMMDD>.json`.
   - Verify challenger passes promotion rule: $\Delta \text{Macro F1} \ge 0.02$ and no label recall drops by $> 0.05$.
4. **Promote Challenger via single config change (Arjun's rule)**:
   - In [Render Dashboard](https://dashboard.render.com) -> SupportPilot Web Service -> **Environment**.
   - Update `TRIAGE_MODEL_PATH` to the challenger model file:
     ```text
     TRIAGE_MODEL_PATH=models/triage-v1.1.0-YYYYMMDD.joblib
     ```
   - Click **Save Changes** (triggers rolling restart with new model; zero code change).
5. **Post-promotion verification**:
   ```bash
   curl -s https://supportpilot-ingestion.onrender.com/version | jq .
   # Verify model_version reflects the newly promoted challenger
   ```
6. **Instant Rollback Path (if needed)**:
   - In Render Environment, point `TRIAGE_MODEL_PATH` back to the champion:
     ```text
     TRIAGE_MODEL_PATH=models/triage_model.joblib
     ```
   - Click **Save Changes** to instantly restore the champion model.

### Notify
- **Rohan** (ML Lead): Review misclassified ticket clusters, validate class distributions in feedback export, and approve retrained model metrics.
- **Neha** (Support Lead): Notify support team of observed model drift and advise agents to verify low-confidence predictions until retrained model is deployed.

---

## Incident 4: Leaked API Key (Zero-Downtime Key Rotation)

### Symptoms
- API key was inadvertently committed to Git, posted in public communication channels, or shared insecurely.
- Render server logs reveal unexpected requests originating from unrecognized IP addresses with high request velocity.
- Legitimate support requests receive unexpected HTTP 429 (`Too Many Requests`) due to key quota exhaustion.

### Check
```bash
# 1. Copy the suspected leaked key into your clipboard (e.g. from GitHub, Slack, or report),
# then compute its 8-character SHA-256 hash prefix via pbpaste so the secret NEVER enters shell history:
pbpaste | python3 -c "import sys,hashlib;print(hashlib.sha256(sys.stdin.read().strip().encode()).hexdigest()[:8])"

# 2. Search Render logs for plain text: "api_key_hash": "<computed_8_char_prefix>"
```


### Fix (3-Step pbcopy Rotation Method)
1. **Step 1: Generate key directly to clipboard**  
   Generate a cryptographically secure 256-bit URL-safe token directly into system clipboard so the raw secret never prints to terminal stdout, shell history (`~/.zsh_history`), or screen recordings:
   ```bash
   python3 -c "import secrets; print(secrets.token_urlsafe(32), end='')" | pbcopy
   ```
2. **Step 2: Dual-Key Stage in Render (Zero-Downtime Transition)**  
   - Open [Render Dashboard](https://dashboard.render.com) -> SupportPilot Web Service -> **Environment**.
   - Edit `TRIAGE_API_KEYS` to append the new key from clipboard (`Cmd+V`) separated by a comma:
     ```text
     TRIAGE_API_KEYS=<old_key>,<new_key_pasted_from_clipboard>
     ```
   - Click **Save Changes**. Render triggers a rolling deployment. Once live, requests signed with either key authenticate successfully.
3. **Step 3: Securely Distribute & Revoke Old Key**  
   - Transmit `<new_key>` to Neha via encrypted vault (e.g. 1Password shared team vault) — **never via unencrypted chat, PR, or ticket**.
   - After downstream clients update their credentials, return to Render Environment settings.
   - Remove `<old_key>` from `TRIAGE_API_KEYS` so only `<new_key>` remains, then click **Save Changes**.
   - Verify revocation: Confirm requests using `<old_key>` receive `401 Unauthorized`.

### Notify
- **Neha** (Client & Support Lead): Coordinate credential updates with support agents and client system integrators.
- **Dev** (Infrastructure Lead): Inspect audit logs for requests using the compromised `api_key_hash` to assess potential unauthorized exposure.

---

## Incident 5: Bad Deploy (Rollback Procedure)

### Symptoms
- New container build crashes continuously on startup (crash-loop backoff).
- Startup failures logged with `RuntimeError` (e.g. missing required env vars, invalid `TRIAGE_THRESHOLD`, or failed Alembic migration).
- Immediate spike in HTTP 500/502/503 responses following a deployment.
- High triage latency or unexpected regression in core endpoint behavior.
- Deploy stuck after Alembic (missing `"migrations complete"` in startup logs) → check `pg_stat_activity` for locks.

### Check
```bash
# 1. Query health endpoint
curl -i https://supportpilot-ingestion.onrender.com/health

# 2. Inspect latest deploy status in Render Dashboard
# Navigate to: SupportPilot Web Service -> Deploys -> Select latest deploy -> View Logs

# 3. Check GitHub Actions CI run on latest commit for unexpected step failures

# 4. Container stuck after Alembic → check pg_stat_activity for locks:
psql "$EXTERNAL_DATABASE_URL" -c "SELECT pid, usename, state, wait_event_type, wait_event, query FROM pg_stat_activity WHERE wait_event_type = 'Lock' OR state != 'idle';"
```

### Fix

#### Option A: Render Instant Rollback (Zero Downtime — Recommended)
1. Open [Render Dashboard](https://dashboard.render.com) -> SupportPilot Web Service.
2. Click **Deploys** in the left sidebar.
3. Locate the last known green, healthy deployment in the deployment list.
4. Click the three dots (`...`) on the right side of that healthy deploy card.
5. Select **Rollback to this deploy**.
6. Render will instantly switch incoming traffic back to the previous healthy container image without re-building.

#### Option B: Git Revert via PR (Protected Branch Workflow)
Direct push to `main` is blocked by GitHub branch protection. To revert via PR:
1. Pull latest `main` branch and create a revert branch:
   ```bash
   git checkout main && git pull origin main
   git checkout -b revert/rollback-<bad-commit-sha>
   ```
2. Revert the problematic commit (plain `git revert` without `-m`):
   ```bash
   git revert <bad-commit-sha> --no-edit
   ```
3. Push the revert branch and open a pull request:
   ```bash
   git push -u origin revert/rollback-<bad-commit-sha>
   gh pr create --title "revert: rollback <bad-commit-sha>" --body "Emergency rollback of degraded commit <bad-commit-sha>"
   ```
4. Wait for CI to turn green and team approval, then merge the PR. Render will detect the new commit on `main` and automatically build and deploy the reverted version.


#### Option C: Environment Variable Rollback
If the deploy failure was caused by a malformed environment variable (e.g. invalid `LOG_LEVEL` or non-float `TRIAGE_THRESHOLD`):
1. In Render Dashboard, open **Environment** settings.
2. Revert the variable to its safe baseline (e.g. `TRIAGE_THRESHOLD=0.35`, `LOG_LEVEL=INFO`).
3. Click **Save Changes** to trigger an automatic rolling redeployment.

#### Post-Rollback Health Verification
```bash
curl -i https://supportpilot-ingestion.onrender.com/health
curl -s -H "X-API-Key: $TRIAGE_API_KEY" https://supportpilot-ingestion.onrender.com/alerts | jq .
```

### Notify
- **Dev** (Infrastructure Lead): Confirm Render container rollback stability and review failure logs.
- **Sana** (Tech Lead): Convene incident review on why the regression passed CI checks or how branch protection was bypassed.
- **Neha** (Support Lead): Confirm service restoration to support agents and client teams.
