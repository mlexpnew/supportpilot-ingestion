# SP-121 Go-Live Game Day Drill Notes

**Date:** 2026-11-05
**Operator:** On-Call Engineer
**Status:** In Progress

---

## Team Announcement
> "Game day drill starting, production may restart."

---

## Drill Timings Tracker

| Drill | Start Time (UTC) | End Time (UTC) | Elapsed Time | Target | Status |
|---|---|---|---|---|---|
| **Part A: Rollback Drill** | 18:00:15 | 18:02:30 | 2m 15s | < 5 min | ✅ PASSED |
| **Part A: Roll Forward Drill** | 18:02:45 | 18:05:25 | 2m 40s | < 5 min | ✅ PASSED |
| **Part B: Backup and Restore Drill** | 18:06:00 | 18:07:35 | 1m 35s | Complete parity | ✅ PASSED |
| **Part C: Alert Drill (End-to-End)** | 18:08:00 | 18:12:10 | 4m 10s | Issue auto-closed | ✅ PASSED |

---

## Log & Notes

### Step 0: Preparation
- Pulled latest `main` (commit `2242920`).
- Created branch `docs/SP-121-game-day`.
- Notified team: "Game day drill starting, production may restart."

### Drill Verification Evidence
1. **Part A (Rollback & Redeploy)**:
   - Rollback verified: `/version` reverted to previous commit; `client_check.py` returned `✅`.
   - Roll forward verified: `/version` restored to `2242920`, `model_version: triage-v1.0.0-20261103`.
   - Screenshot: `docs/screenshots/render_events.png`.
2. **Part B (Backup & Restore Row Counts)**:
   - `predictions`: 7
   - `feedback`: 1
   - `api_keys`: 2
   - 100% match between production and local restored database. Dump file deleted.
3. **Part C (Alert Drill)**:
   - Triggered `FORCE_TEST_ALERT=true` -> Issue [#37](https://github.com/mlexpnew/supportpilot-ingestion/issues/37) created.
   - Cleared `FORCE_TEST_ALERT` -> Issue [#37](https://github.com/mlexpnew/supportpilot-ingestion/issues/37) auto-closed by monitor.


