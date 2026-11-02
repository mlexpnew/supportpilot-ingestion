# SupportPilot Weekly Pilot Report & Reporting System (SP-114)

This document describes the automated weekly pilot reporting system for SupportPilot, including how to run the CLI, what each metric means, sample size safety rules, automated GitHub Actions scheduling, and operational limitations.

---

## 1. Quick Overview

- **Purpose**: Generates an executive-ready weekly pilot report in Markdown accompanied by visual charts. It answers the fundamental client question: *"Is SupportPilot working?"* with complete honesty and statistical transparency.
- **Reporting Cadence**: Generated automatically every Monday at 09:00 IST (03:30 UTC) covering the preceding 7 days of live production traffic.
- **Artifacts Created**:
  - `reports/pilot_report_<from>_<to>.md`: Human-readable executive Markdown report.
  - `reports/daily_volume_<from>_<to>.png`: Daily ticket triage volume bar chart.
  - `reports/recall_per_label_<from>_<to>.png`: Per-category recall bar chart with sample thresholds.

---

## 2. Zero-PII & Privacy Guarantees

The reporting generator enforces strict zero-PII invariants:
1. **Aggregates Only**: The report queries counts, sums, and ratios grouped by date and predicted/actual category labels.
2. **No Free Text**: Ticket bodies, customer emails, card numbers, and raw customer queries are never fetched, rendered, or saved.
3. **No Ticket Identifiers**: Individual ticket IDs (`T-...`) are excluded from reports and chart annotations.

---

## 3. How to Run the Report CLI

### Prerequisites
Install reporting dependencies (including `matplotlib`):
```bash
pip install -r requirements-dev.txt
```

Set the database connection string:
```bash
# Recommended: Read-only database user for reporting
export REPORT_DATABASE_URL="postgresql://report_reader:password@host:5432/feedback"
```
*(If `REPORT_DATABASE_URL` is omitted, the tool falls back to `DATABASE_URL`, or local SQLite if `ALLOW_SQLITE_FALLBACK=1` is set).*

### Running with Explicit Date Range
```bash
python -m supportpilot.reports.weekly \
  --from 2026-10-26 \
  --to 2026-11-01 \
  --out reports/
```

### Running for the Last 7 Days (Default)
When `--from` and `--to` are omitted, the CLI defaults to the last 7 days ending today (UTC):
```bash
python -m supportpilot.reports.weekly --out reports/
```

---

## 4. Report Sections & Metric Definitions

### Executive Summary (3 Bullets)
1. **Accuracy Verdict**:
   - Compares live accuracy against the production target (80.0%).
   - Displays *"Not enough data to judge"* if total human feedback corrections is below the statistical alert threshold (`ALERT_MIN_FEEDBACK`, default: 20).
2. **Operational Volume**:
   - Total tickets triaged and model version(s) active during the period.
   - Number of tickets routed to human review (`needs_review`) and the review rate.
3. **Model Health & Focus**:
   - Identifies the lowest-performing category among those with sufficient sample size ($n \ge 5$).
   - Recommends focus areas for agent review and model retraining.

### Section 1: Ticket Volume & Routing
- **Total Tickets Triaged**: Count of all prediction records in the date window.
- **Human Review Required (`needs_review`)**: Count of tickets where model confidence was below the category cutoff, necessitating agent review.
- **Tracking Disclosure**: Because `needs_review` tracking was introduced mid-pilot, pre-migration historical rows have `NULL` values. The review rate is calculated exclusively over tracked rows (`flagged ÷ tracked`). If `tracked < total`, the report explicitly notes:
  `Review rate based on X of Y tickets; tracking started <date>.`
- **Volume by Category**: Triage counts and share (%) across all canonical categories (`account`, `billing`, `cancellation`, `delivery`, `refund`, `other`).
- **Daily Triage Volume**: Daily triage counts. Labeled in UTC with an explicit note: *"Days are in UTC; IST is UTC+5:30."*

### Section 2: Accuracy & Feedback Analysis
- **Total Feedback Records**: Agent corrections and agreements recorded during the date window.
- **Live Accuracy**: `feedback_agreed ÷ feedback_total`.
- **Honesty Rule**: If `feedback_total < 20`, exact accuracy percentages are withheld with a warning: *"A minimum of 20 human corrections is required before accuracy metrics are statistically reliable. Presenting preliminary percentages would be misleading."*

### Section 3: Where the Model Struggles (Weak Spots)
- **Ground-Truth Recall**: `agreed ÷ total` ground-truth verified tickets for each category.
- **Small Sample Rule ($n < 5$)**:
  - In the recall table, categories with fewer than 5 verified feedback samples show `Not enough data (n=X)`.
  - Categories with $n \ge 5$ show the recall and sample size: e.g. `0.80 (n=10)`.
  - In the chart, small-sample categories ($n < 5$) are rendered as grey bars marked `n<5`.
  - Categories with $n < 5$ are never flagged as systematic weak spots in the narrative text.

### Section 4: Recommendations
Actionable next steps based on pilot metrics:
- Retraining triggers (SP-113) if live accuracy drops below 80.0%.
- Review queue monitoring if `needs_review_rate` exceeds 25.0%.
- Agent guidance to accelerate ambiguous ticket review if feedback sample size is thin.

---

## 5. Automated GitHub Actions Workflow

- **Workflow File**: `.github/workflows/weekly-report.yml`
- **Schedule**: Every Monday at 09:00 IST (`cron: "30 3 * * 1"`, UTC 03:30).
- **Manual Trigger**: Supports `workflow_dispatch` with optional `from_date` and `to_date` inputs in the GitHub Actions UI.
- **Artifact Upload**: The generated report and PNG charts are packaged and uploaded as a workflow artifact named `weekly-pilot-report`, downloadable directly from the run summary.
- **Database Credentials**: Injected securely via GitHub Repository Secret `REPORT_DATABASE_URL`. Never hardcoded or echoed into logs.

---

## 6. Limitations & Operational Caveats

1. **In-Memory Metrics Excluded (Persistent Storage Only)**: The report queries persistent PostgreSQL tables (`predictions` and `feedback`) directly. It deliberately excludes the volatile in-memory counters served by `GET /metrics`, which reset to zero on container restarts or during Render free-tier cold starts. This ensures consistent historical accounting across deployments.
2. **Feedback Lag**: Live accuracy reflects only tickets that human agents have inspected and submitted feedback for. If agents do not submit feedback for a batch of tickets, accuracy reflects only the verified subset.
3. **Cold-Start Bias**: During early pilot days or low volume periods, a single misclassification can skew raw ratios. Always rely on the sample size annotations (`n=X`) and minimum thresholds before deciding to retrain or adjust cutoffs.
4. **Time Zone Boundaries**: Production database timestamps are stored in UTC (ISO 8601). While daily aggregates are grouped by UTC calendar date, reports clearly annotate the IST conversion offset (+5:30) for India-based operational teams.
