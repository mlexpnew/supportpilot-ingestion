# SupportPilot Model Retraining & Champion/Challenger Pipeline (SP-113)

This runbook outlines when to retrain the triage classifier, how to execute the fair comparison pipeline on the sacred frozen evaluation set, how to interpret comparison reports, and how to safely promote or roll back models in production.

---

## 1. When to Retrain

- **Live Accuracy Drop (Rohan's Rule)**: Retrain whenever `live_accuracy` drops below **80.0%** with a minimum of 20 feedback samples (`feedback_total >= 20`), triggering a `warning` alert on `GET /alerts`.
- **Per-Label Recall Degradation**: Retrain if any single category recall drops below **75.0%** on `GET /metrics/labels`.
- **High Human Review Volume**: Retrain when `needs_review_rate` exceeds **25.0%** over a sustained period of regular triage traffic.
- **New Pilot Cohort Feedback**: Retrain after completing human review for a new pilot customer batch.

---

## 2. How to Run the Retraining Pipeline

The retraining CLI executes feedback export, join, challenger training, version bumping, and fair evaluation on the frozen evaluation set in a single command. The `--expected-champion-version` flag (obtained from production `GET /version`) is required to ensure the comparison uses the exact model that production is serving (safety checks that can be skipped get skipped).

```bash
# 1. Fetch active production model version from /version endpoint
CURRENT_VER=$(curl -s https://supportpilot.onrender.com/version | jq -r .model_version)

# 2. Run retraining pipeline with champion check
python -m supportpilot.triage.retrain \
  --base data/triage_labeled.jsonl \
  --feedback data/feedback_export.jsonl \
  --pilot data/sample_tickets.jsonl \
  --out models/ \
  --expected-champion-version "$CURRENT_VER"
```

### Pipeline Safety Invariants & Steps (Automated):
1. **Production Champion Verification**: Verifies `--expected-champion-version` matches the local champion model file. If there is a mismatch or the champion file is missing, it **fails fast** (never invents an unverified baseline).
2. **Metadata Integrity**: Validates that the champion has an explicit, valid semantic version (no magic fallbacks).
3. **Normalized Data Isolation Check**: Validates that the sacred frozen evaluation set (`data/eval_frozen.jsonl`) has **zero overlap** with training data. Text comparison normalizes lowercase and collapses whitespace so casing or space differences cannot slip through. Fails fast if either file is missing or any leakage is detected.
4. **Deterministic Join**: Relabels existing training tickets and appends newly reviewed pilot tickets using the SP-110 join utility.
5. **Version Bumping**: Automatically increments the minor semantic version (`triage-v1.0.0-...` -> `triage-v1.1.0-YYYYMMDD`).
6. **Challenger Training**: Fits the TF-IDF + Logistic Regression pipeline on the augmented dataset without overwriting the champion file.
7. **Startup Pre-Validation**: Verifies version regex and ensures trained classes match service `LABELS`.
8. **Fair Comparison**: Evaluates both Champion and Challenger on `data/eval_frozen.jsonl`.
9. **Report Generation**: Writes `reports/compare_<YYYYMMDD>.json` and outputs a human-readable summary table.

---

## 3. How to Read the Comparison Report

The promotion decision follows Rohan's strict fair test rule:

> **The Challenger wins if and only if:**
> 1. **Macro F1** improves by **$\ge 0.02$ (+2.0%)**.
> 2. **No individual label's recall** drops by more than **$0.05$ (5.0%)**.

### Example Comparison Table Output

```text
================================================================================
                 SUPPORTPILOT RETRAINING EVALUATION REPORT
================================================================================
 Frozen Evaluation Set: data/eval_frozen.jsonl (60 samples)
 Champion:   triage-v1.0.0-20261022
 Challenger: triage-v1.1.0-20261027
--------------------------------------------------------------------------------
 Metric                 | Champion       | Challenger     | Delta             
--------------------------------------------------------------------------------
 Macro F1               | 82.50%         | 85.10%         |  +2.60% (min +2.00%)
 Macro Recall           | 83.00%         | 85.50%         |  +2.50%
 Macro Precision        | 82.00%         | 84.80%         |  +2.80%
 Needs Review Rate      | 15.00%         | 11.67%         |  -3.33%
--------------------------------------------------------------------------------
 Per-Label Recall Comparison:
 Label           | Champion   | Challenger | Delta      | Status          
--------------------------------------------------------------------------------
 delivery        | 90.00%     | 90.00%     |   +0.00% | PASS            
 billing         | 80.00%     | 90.00%     |  +10.00% | PASS            
 refund          | 80.00%     | 80.00%     |   +0.00% | PASS            
 cancellation    | 80.00%     | 80.00%     |   +0.00% | PASS            
 account         | 80.00%     | 80.00%     |   +0.00% | PASS            
 other           | 80.00%     | 90.00%     |  +10.00% | PASS            
--------------------------------------------------------------------------------
 PROMOTION DECISION: PROMOTED (Challenger WON)
   • Macro F1 improved by +0.0260 (>= +0.0200)
   • No label recall dropped by more than 5.00%
================================================================================
```

### Decision Scenarios:
- **`PROMOTED`**: Both criteria satisfied. The challenger is proven superior and safe to deploy.
- **`REJECTED` (Sub-threshold improvement)**: Macro F1 improved by less than +0.02 (or tie). The model change is within noise threshold. Champion is retained.
- **`REJECTED` (Per-label regression)**: Macro F1 improved overall, but one category suffered a $> 5.0\%$ recall drop. Champion is retained to prevent category-specific regressions.

> [!NOTE]
> **Identical Scores in Initial Runs**: Champion and challenger currently tie with identical metrics because no real human feedback or new pilot data has been incorporated yet, training both models on the same underlying baseline data. This is expected behavior and does not mean the pipeline is broken; evaluation scores will diverge once verified agent corrections are joined.

---

## 4. How to Promote and Roll Back (Safe Promotion)

Arjun's rule: *“Promotion should be boring: one config change, one restart, one rollback path.”*

### Safe Promotion Procedure:
1. Ensure the comparison report confirms `PROMOTED`.
2. Locate the generated challenger file in `models/` (e.g. `models/triage-v1.1.0-20261027.joblib`).
3. Open [Render Dashboard](https://dashboard.render.com) -> SupportPilot Web Service -> **Environment**.
4. Update the environment variable `TRIAGE_MODEL_PATH` to point to the challenger file:
   ```text
   TRIAGE_MODEL_PATH=models/triage-v1.1.0-20261027.joblib
   ```
5. Click **Save Changes**. Render performs a rolling restart without any code changes or rebuild.
6. Verify promotion via the `/version` endpoint:
   ```bash
   curl -s https://supportpilot-ingestion.onrender.com/version | jq .
   # Expected: {"git_sha": "...", "model_version": "triage-v1.1.0-20261027", "started_at": "..."}
   ```

### Safe Rollback Procedure:
If the promoted model exhibits unexpected behavior or regression in live production:
1. Open [Render Dashboard](https://dashboard.render.com) -> SupportPilot Web Service -> **Environment**.
2. Point `TRIAGE_MODEL_PATH` back to the original champion:
   ```text
   TRIAGE_MODEL_PATH=models/triage_model.joblib
   ```
3. Click **Save Changes**. Render restarts the service with the champion model immediately restored.
4. Verify rollback:
   ```bash
   curl -s https://supportpilot-ingestion.onrender.com/version | jq .
   # Expected: model_version returns to champion version
   ```
