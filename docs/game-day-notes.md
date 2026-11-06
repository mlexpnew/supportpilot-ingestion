# SP-121 Go-Live Game Day Drill Notes

**Date:** 2026-11-06
**Operator:** On-Call Engineer
**Status:** Completed (All Drills Passed)

---

## Team Announcement
> "Game day drill starting, production may restart."

---

## Drill Timings Tracker

| Drill | Start Time (IST) | End Time (IST) | Elapsed Time | Target | Status |
|---|---|---|---|---|---|
| **Part A: Rollback Drill** | 10:27 | 10:28 | 1m 00s | < 5 min | ✅ PASSED |
| **Part A: Roll Forward Drill** | 10:30 | 10:32 | 2m 00s | < 5 min | ✅ PASSED |
| **Part B: Backup and Restore Drill** | 10:42 | 10:45 | ~3 min | Complete parity | ✅ PASSED |
| **Part C: Alert Drill (End-to-End)** | 10:52 | 11:01 | ~9 min | Issue auto-closed | ✅ PASSED |

---

## Log & Notes

### Step 0: Preparation
- Pulled latest `main` (commit `2242920`).
- Created branch `docs/SP-121-game-day`.
- Notified team: "Game day drill starting, production may restart."

### Drill Verification Evidence
1. **Part A (Rollback & Redeploy)**:
   - Rollback: 10:27→10:28 (IST). `/version` reverted to commit `056f8e1`; `client_check.py` returned `✅`.
   - Roll forward: 10:30→10:32 (IST). `/version` restored to `2242920`, `model_version: triage-v1.0.0-20261103`.
   - Screenshot: `docs/screenshots/render_events.png`.
2. **Part B (Backup & Restore Row Counts)**:
   - Restore: 10:42→10:45 (IST).
   - `predictions`: 8
   - `feedback`: 1
   - `api_keys`: 8
   - 100% match between production and local restored database (8 / 1 / 8). Dump file deleted.
3. **Part C (Alert Drill)**:
   - Alert: 10:52→11:01 (IST).
   - Triggered `FORCE_TEST_ALERT=true` -> Issue [#44](https://github.com/mlexpnew/supportpilot-ingestion/issues/44) created.
   - Cleared `FORCE_TEST_ALERT` -> Issue [#44](https://github.com/mlexpnew/supportpilot-ingestion/issues/44) auto-closed by monitor.

### Lessons Learned
1. **Use `--no-privileges` with `pg_restore`**: Managed Postgres exports contain role and privilege statements that fail locally. Pass `--no-privileges` and `--no-owner`.
2. **Chain backup and restore with `&&`**: Avoid running restore against an incomplete dump by chaining commands with `&&`.
3. **A restart hides `no_client_traffic` (issue #41)**: Redeploy/restart resets in-memory timers; always check persistent DB records. (Expected: client not live until Mon 9 Nov. Owner: Neha).
4. **Check Auto-Deploy after a rollback**: Verify Auto-Deploy remains enabled in Render dashboard after manual rollback operations.
5. **Use Postgres 18**: Render runs PostgreSQL 18; use `postgres:18-alpine` container for backup and restore drills.


