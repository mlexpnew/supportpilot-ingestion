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
| **Live Usage Endpoint** | `https://supportpilot-ingestion.onrender.com/usage` | Per-key usage and auth diagnostics (Admin `X-API-Key` protected) |
| **Live Quality Endpoint** | `https://supportpilot-ingestion.onrender.com/metrics/quality` | Accuracy, confusion matrix, precision/recall, and sample sizes (Admin `X-API-Key` protected) |

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

---

## Incident 6: Client Sends No Traffic

### Symptoms
- Active warning alert returned by `GET /alerts`:
  * Stale client traffic (>24 h since last request): `{"name": "no_client_traffic", "value": <hours>, "threshold": 24.0, "severity": "warning"}`.
  * Inactive client key (never sent traffic since deployment, service up >= 24 h): `{"name": "client_never_seen", "value": <hours>, "threshold": 24.0, "severity": "warning"}`.
- Telemetry shows zero requests to `/triage` and `/feedback` from known client API keys for over 24 hours.
- Expected ticket ingestion pipeline appears silent or dormant.

### Check
```bash
# 1. Query active alerts to identify inactive duration
curl -s -H "X-API-Key: $TRIAGE_API_KEY" https://supportpilot-ingestion.onrender.com/alerts | jq .

# 2. Query usage telemetry endpoint using an Admin API key
curl -s -H "X-API-Key: $ADMIN_API_KEY" https://supportpilot-ingestion.onrender.com/usage | jq .

# 3. Analyze per-key usage statistics:
# - Look for the client key hash:
#   * If last_seen is > 24 hours ago and auth_fail_count == 0: Client integration stopped sending traffic completely.
#   * If shared "unknown" row auth_fail_count is high: Client is attempting requests with an invalid/expired key (getting 401s).
#   * If client key_hash is completely missing from /usage: Client has never initiated a single request to this environment (triggers "client_never_seen" alert after 24h).
```

### Fix
1. **If authentication failures are high (`auth_fail_count > 0`)**:
   - The client may have misconfigured their API key header or is using an expired credential.
   - Verify active keys in Render Web Service Environment under `TRIAGE_API_KEYS`.
   - Re-issue or confirm the correct client API key with Neha.
2. **If no requests are arriving (`last_seen > 24h` or key missing)**:
   - Verify whether the client experienced an outage in their upstream ticket webhook or dispatch runner.
   - Send the client integration team the onboarding diagnostic tool:
     ```bash
     export SUPPORTPILOT_API_KEY="client-key-here"
     python examples/client_check.py
     ```
   - Review [Client Integration Guide](client-integration.md) with client engineers to resolve schema, network proxy, or payload size limits (10 KB).

### Notify
- **Neha** (Client & Support Lead): Contact client integration lead with diagnostic details from `/usage`.
- **Sana** (API Tech Lead): Monitor `/usage` and `/alerts` once client confirms traffic resumption.

---

## Incident 7: Client 401 Unauthorized (Missing Environment Key & Curly Quote Corruption)

### Symptoms
- `/triage` returned HTTP 401 (`401 Unauthorized`).
- Running `examples/client_check.py` fails on step 2:
  ```text
  ✅ /health
  ❌ /triage (401 Unauthorized)
     Fix: Check that $SUPPORTPILOT_API_KEY matches your assigned key with no trailing spaces.
  ```
- Client reports inability to ingest tickets or send backlogs (~5,000 tickets blocked).
- Render service logs record: `api_key_hash=unknown auth failed path=/triage status=401`.

### Cause
- The new client API key was missing from Render's `TRIAGE_API_KEYS` environment variable. The client had generated/received a key, but the production web service had not been provisioned with it, rejecting incoming requests.

### Check
```bash
# 1. Run integration diagnostic script to verify failure mode:
export SUPPORTPILOT_API_KEY="client-key-to-verify"
python examples/client_check.py

# 2. Check key formatting, length, and non-ASCII character corruption:
pbpaste | python3 -c "import sys; s = sys.stdin.read(); print(f'len={len(s)} ascii_only={s.isascii()} repr={repr(s[:6])}...{repr(s[-6:])}')"

# 3. Compute 8-char SHA-256 hash prefix for safe comparison (never expose raw keys):
pbpaste | python3 -c "import sys,hashlib; print(hashlib.sha256(sys.stdin.read().strip().encode('utf-8')).hexdigest()[:8])"

# 4. Compare computed hash against active key hashes via GET /usage:
curl -s -H "X-API-Key: $ADMIN_API_KEY" https://supportpilot-ingestion.onrender.com/usage | jq .
```

### Fix
1. **Compare key hashes, never raw keys**:
   - Compare the 8-character cryptographic SHA-256 hash prefix of the client key against the configured keys in Render.
   - **Never** paste or print raw API keys in terminal stdout, shell commands, Slack channels, or GitHub tickets.
2. **Provision missing key in Render Environment**:
   - Open [Render Dashboard](https://dashboard.render.com) -> SupportPilot Web Service -> **Environment**.
   - Inspect `TRIAGE_API_KEYS` and append the new client key (or comma-separated list of active keys).
   - If the old/leaked key (e.g. hash `3dfcb467`) is present, remove it.
   - Click **Save Changes** and wait for the rolling restart to complete.
3. **Verify with Diagnostic Tooling**:
   - Re-run `client_check.py` with the provisioned key:
     ```bash
     python examples/client_check.py
     python examples/client_check.py --batch
     ```
   - Both commands must succeed with `✅`.

### Notify
- **Neha** (Client & Support Lead): Confirm client key provisioning and successful `/triage` validation.
- **Sana** (API Tech Lead): Check `/usage` to confirm `auth_fail_count` ceases and legitimate traffic increments `ok_count`.
- **Dev** (Infrastructure Lead): Audit Render deployment logs.

### Lessons Learned & Takeaways
1. **Chat Copy-Paste Adds Curly Quotes**:
   Copying text from chat apps (Slack, Teams, Apple Notes) frequently transforms standard straight quotes (`'`, `"`) into Unicode "smart" or curly quotes (`‘`, `’`, `“`, `”`) or appends trailing whitespace/newlines. This corrupts the API key token byte sequence and causes silent 401s. Always run an ASCII sanity and length check on clipboard contents (`s.isascii() and len(s) == expected`).
2. **Hash-Compare Method Prevents Secret Exposure**:
   Always compare keys via their 8-character SHA-256 hash prefix (`hashlib.sha256(key.encode()).hexdigest()[:8]`). This enables operators and developers to verify credential identity without ever writing or logging the plaintext secret.
3. **Dual Verification Before Handoff**:
   Always execute `client_check.py` against production before notifying external clients that their credentials are active.

---

## Incident 8: Latency SLO Breach (p95 > 500 ms)

### Symptoms
- Active alert `latency_slo_breach` appears on `GET /alerts`.
- Automated monitor opens a GitHub issue titled `🔴 Alert: latency_slo_breach`.
- Client experiences slow triage response times; p95 latency exceeds the 500 ms SLO target over 20+ samples.

### Check
```bash
# 1. Query live metrics to check p50, p95, and sample count per endpoint
curl -s -H "X-API-Key: $ADMIN_API_KEY" https://supportpilot-ingestion.onrender.com/metrics | jq '.latencies'

# 2. Check active alerts payload
curl -s -H "X-API-Key: $ADMIN_API_KEY" https://supportpilot-ingestion.onrender.com/alerts | jq .

# 3. Check Render metrics dashboard for CPU/memory throttling
# Navigate to: Render Dashboard -> SupportPilot Web Service -> Metrics -> CPU / Memory Usage

# 4. Check Postgres active queries for lock contention or slow queries
# Connect to external database URL and run:
# SELECT pid, now() - pg_stat_activity.query_start AS duration, query, state FROM pg_stat_activity WHERE state != 'idle' ORDER BY duration DESC;
```

### Fix
1. **CPU/Memory Starvation**:
   - Check if Render free instance is CPU-throttled or experiencing memory pressure.
   - If memory exceeds threshold, trigger a service restart to clear memory fragmentation or upgrade instance tier.
2. **Database Lock Contention**:
   - If prediction logging or key usage writes are blocking triage requests, terminate stuck locks:
     ```sql
     SELECT pg_terminate_backend(pid) FROM pg_stat_activity WHERE state != 'idle' AND duration > interval '10 seconds';
     ```
3. **Rollback if Caused by Deploy**:
   - If the breach followed a recent deployment, trigger instant rollback:
     - Navigate to **Render Dashboard** -> **SupportPilot Web Service** -> **Deploys**.
     - Select previous deploy -> Click **Rollback to this deploy**.

### Notify
- **Sana** (API Tech Lead): Investigate endpoint latency bottlenecks.
- **Rohan** (Performance Lead): Review benchmark results and query execution plan.
- **Dev** (Infrastructure Lead): Monitor Render CPU/memory resource utilization.

---

## Incident 9: Model Quality Drop / Drift Alert

### Symptoms
- Active alert `low_live_accuracy` (accuracy < 0.75 across $\ge 30$ feedback rows) or `label_drift` (PSI > 0.20 across $\ge 50$ predictions) appears on `GET /alerts`.
- Automated GitHub Actions monitor opens an issue titled `🔴 Alert: low_live_accuracy` or `🔴 Alert: label_drift`.
- Support agents flag recurring misclassifications in Tier-1 triage queues.

### Check
```bash
# 1. Query live quality endpoint to inspect live accuracy, confusion matrix, and sample counts
curl -s -H "X-API-Key: $ADMIN_API_KEY" https://supportpilot-ingestion.onrender.com/metrics/quality | jq .

# 2. Inspect active alerts payload
curl -s -H "X-API-Key: $ADMIN_API_KEY" https://supportpilot-ingestion.onrender.com/alerts | jq .

# 3. Identify the confused labels
# Look at the off-diagonal counts in the confusion_matrix (e.g. matrix["billing"]["refund"])
# Check per-label precision and recall to find which categories experienced degradation.

# 4. Export feedback to inspect concrete misclassification examples
export DATABASE_URL="<render-postgres-external-url>"
python -m supportpilot.feedback export --out data/feedback_export.jsonl
```

### Fix
1. **Find Confused Labels & Review Examples**:
   - Inspect the exported feedback (`data/feedback_export.jsonl`) to identify ambiguous customer phrasing, shifting inquiry types, or emerging vocabulary not present in the original training data.
2. **Execute Retraining Pipeline (Champion vs. Challenger)**:
   - Run the automated retraining pipeline joining feedback with labeled historical tickets:
     ```bash
     python -m supportpilot.triage.retrain \
       --data data/triage_labeled.jsonl \
       --feedback data/feedback_export.jsonl \
       --eval-dataset data/eval_frozen.jsonl \
       --out-dir models/ \
       --expected-champion-version "$(curl -s https://supportpilot-ingestion.onrender.com/version | jq -r .model_version)"
     ```
   - Review comparison report (`reports/compare_<date>.json`).
   - If the challenger candidate beats the production champion on the sacred frozen evaluation set (`data/eval_frozen.jsonl`), tag and publish the release asset, update Render's `TRIAGE_MODEL_PATH`, and redeploy.
3. **Roll Back if Caused by a Recent Model Deployment**:
   - If the drop in accuracy or distribution drift occurred immediately following a model version release or deploy:
     - Navigate to **Render Dashboard** -> **SupportPilot Web Service** -> **Deploys**.
     - Locate the last known healthy deployment and click **Rollback to this deploy**.
     - Verify rollback success:
       ```bash
       curl -s https://supportpilot-ingestion.onrender.com/version | jq .
       curl -s -H "X-API-Key: $ADMIN_API_KEY" https://supportpilot-ingestion.onrender.com/metrics/quality | jq .
       ```

### Notify
- **Rohan** (ML & Evaluation Lead): Inspect confusion matrix, analyze per-label error modes, and supervise retraining.
- **Meera** (Data & Operations Lead): Audit incoming ticket distribution drift against baseline datasets.
- **Neha** (Support & Client Lead): Brief support agents on known confusion patterns (e.g., billing vs refund) and advise agents to verify predictions in affected categories until the retrained model is deployed.



