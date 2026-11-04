# SupportPilot Production Go-Live Readiness Checklist

This document tracks the operational, technical, and incident-response readiness criteria required before opening traffic to production clients.

---

## 1. Go-Live Verification Checklist

All items below must be verified and checked off prior to marking production ready:

- [ ] **Client key created through `/admin/keys`**:
  - Securely generate client key via `POST /admin/keys` (`{"name": "<client-identifier>", "role": "client"}`).
  - Export and provide the raw key to the client once.
  - Verify key is stored as a 64-character SHA-256 hash in the database, with only prefix visible in logs.
  - Test `/triage` authentication with the provisioned client key.
- [ ] **Monitor green**:
  - GitHub Actions automated monitor workflow (`.github/workflows/monitor.yml`) active and scheduled (runs every 15 min).
  - Production `/health` returns `200 OK` (`{"status": "ok", "model_loaded": true}`).
  - Production `/alerts` returns zero active alerts (`"alerts": []`).
  - No open `alert` issues or `🔴 Service down` issues in GitHub repository.
- [ ] **SLO met**:
  - Production and local load test verify latency SLO: **p95 latency < 500 ms** under expected load.
  - Load test confirmed at 20 req/s for 2 minutes (80% single `/triage`, 20% batch `/triage/batch`):
    - Single `/triage`: p95 = 220 ms (SLO < 500 ms target met).
    - Batch `/triage/batch`: p95 = 240 ms (SLO < 500 ms target met).
    - Error rate = 0.00%.
  - `latency_slo_breach` alert verified (triggers only when p95 > 500 ms with $\ge 20$ samples).
- [ ] **`/version` checked**:
  - Verify public `GET /version` endpoint returns correct production metadata:
    - `commit_sha`: Matches deployed release commit SHA.
    - `model_version`: Pinned model artifact version (format `triage-v1.0.0-<YYYYMMDD>`).
    - `started_at`: UTC timestamp of current deployment startup.
- [ ] **On-call person named**:
  - **Primary On-Call Engineer**: Neha (Client & Support Operations Lead)
  - **Secondary On-Call / Escalation**: Sana (API Tech Lead) & Dev (Infrastructure & CI/CD Lead)
  - Escalation channel: `#supportpilot-alerts` / Slack PagerDuty integration.

---

## 2. Rollback Plan

If an unrecoverable failure, critical latency regression, or data integrity issue occurs immediately post-deploy:

> **Render → Rollback to previous deploy**

### Rollback Steps
1. Navigate to the **[Render Dashboard](https://dashboard.render.com)**.
2. Select the **SupportPilot Web Service** (`supportpilot-ingestion`).
3. Click on the **Deploys** tab in the left-hand navigation.
4. Locate the last known good deployment (prior to current release).
5. Click the three dots menu (`...`) or click the deploy, then click **Rollback to this deploy**.
6. Monitor the deploy logs until `migrations complete` and `Application startup complete` appear.
7. Verify health and version:
   ```bash
   curl -s https://supportpilot-ingestion.onrender.com/health
   curl -s https://supportpilot-ingestion.onrender.com/version
   ```

---

## 3. Operational Alerts & Runbook Links

Every alert surfaced by `GET /alerts` or the automated GitHub monitor links to an actionable incident procedure in [`docs/runbook.md`](runbook.md):

| Alert Name | Severity | Threshold / Condition | Runbook Link |
| :--- | :---: | :--- | :--- |
| `latency_slo_breach` | Warning / Critical | p95 latency > 500 ms on `/triage` or `/triage/batch` ($\ge 20$ samples) | [Runbook: Incident 8 - Latency SLO Breach](runbook.md#incident-8-latency-slo-breach-p95--500-ms) |
| `error_rate` | Critical | Error rate > 5% over rolling window | [Runbook: Incident 1 - API Down / 5xx Spike](runbook.md#incident-1-api-down--5xx-spike) |
| `accuracy` | Warning | Live rolling feedback accuracy drops below 80% ($\ge 20$ samples) | [Runbook: Incident 3 - Accuracy Drop](runbook.md#incident-3-accuracy-drop) |
| `review_rate` | Warning | Tickets marked `needs_review=True` exceeds 35% | [Runbook: Incident 3 - Review Rate / Accuracy Drop](runbook.md#incident-3-accuracy-drop) |
| `no_client_traffic` | Warning | Client key inactive for > 24 hours | [Runbook: Incident 6 - Client Sends No Traffic](runbook.md#incident-6-client-sends-no-traffic) |
| `client_never_seen` | Warning | Client key provisioned for > 24 hours with zero requests | [Runbook: Incident 6 - Client Sends No Traffic](runbook.md#incident-6-client-sends-no-traffic) |
| `service_down` | Critical | Health check fails twice (initial attempt + 60s retry) | [Runbook: Incident 1 - API Down / 5xx Spike](runbook.md#incident-1-api-down--5xx-spike) |
| `db_unreachable` | Critical | PostgreSQL connection failures or connection pool exhaustion | [Runbook: Incident 2 - Database Unreachable](runbook.md#incident-2-database-unreachable) |
| `leaked_key` | Critical | API key detected in unauthorized repository or chat | [Runbook: Incident 4 - Leaked API Key Rotation](runbook.md#incident-4-leaked-api-key-zero-downtime-key-rotation) |

---

## 4. Verification Commands

Run these quick checks against the target deployment environment to validate checklist items:

```bash
# Set environment
export URL=https://supportpilot-ingestion.onrender.com
export ADMIN_KEY="<your-admin-key>"

# 1. Health Check
curl -s "$URL/health" | jq .

# 2. Version Check
curl -s "$URL/version" | jq .

# 3. Active Alerts Check
curl -s -H "X-API-Key: $ADMIN_KEY" "$URL/alerts" | jq .

# 4. Latency & Metrics Check
curl -s -H "X-API-Key: $ADMIN_KEY" "$URL/metrics" | jq '.latencies'
```
