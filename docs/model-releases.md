# SupportPilot Model Release & Promotion Guide (SP-116)

This guide documents the operational lifecycle for training, validating, releasing, and promoting triage model artifacts in SupportPilot.

Model binaries are **never** trained inside Docker builds or committed directly to git repositories. Instead, model artifacts are published to GitHub Releases as versioned assets and pinned in the `Dockerfile` with cryptographic SHA-256 integrity verification.

---

## Model Release Lifecycle Overview

```text
+---------------------+     +--------------------+     +---------------------+
| 1. Retrain Locally  | --> | 2. Fair Evaluation | --> | 3. Publish Release  |
| (data + feedback)   |     | (frozen eval set)  |     | (GitHub Assets)     |
+---------------------+     +--------------------+     +---------------------+
                                                                  |
                                                                  v
+---------------------+     +--------------------+     +---------------------+
| 6. Production Deploy| <-- | 5. PR & CI Check   | <-- | 4. Update Config    |
| (Render /version)   |     | (Docker + SHA test)|     | (MODEL_URL & SHA)   |
+---------------------+     +--------------------+     +---------------------+
```

---

## Step 1: Retrain Locally (SP-113 Pipeline)

Run the retraining pipeline using the active production champion version (fetched from `/version`):

```bash
# 1. Fetch current production model version
CURRENT_VER=$(curl -s https://supportpilot-ingestion.onrender.com/version | python3 -c "import sys, json; print(json.load(sys.stdin)['model_version'])")

# 2. Run the retraining and comparison pipeline
python -m supportpilot.triage.retrain \
  --base data/triage_labeled.jsonl \
  --feedback data/feedback_export.jsonl \
  --pilot data/sample_tickets.jsonl \
  --out models/ \
  --expected-champion-version "$CURRENT_VER"
```

This outputs:
- A new candidate model file: `models/triage-v<NEW_SEMVER>-<YYYYMMDD>.joblib`
- Companion metadata: `models/triage-v<NEW_SEMVER>-<YYYYMMDD>.json`
- Evaluation report: `reports/compare_<YYYYMMDD>.json`

---

## Step 2: Compare & Fair Evaluation

Inspect the comparison summary printed by the retraining pipeline against Rohan's fair evaluation rules:

1. **Macro F1 Improvement**: Challenger Macro F1 must improve over Champion by at least **+0.02 (+2.0%)**.
2. **No Per-Label Regression**: No individual category recall may decrease by more than **0.05 (5.0%)**.

> [!IMPORTANT]
> If the comparison report outputs `REJECTED`, the challenger must **not** be published or promoted. The champion remains active.

---

## Step 3: Publish GitHub Release

When the challenger wins (`PROMOTED`), publish the new release using the GitHub CLI:

```bash
# Define release tag matching the semantic version (e.g. model-triage-v1.1.0)
TAG="model-triage-v1.1.0"
MODEL_FILE="models/triage-v1.1.0-20261027.joblib"
META_FILE="models/triage-v1.1.0-20261027.json"

# Calculate SHA-256 checksums
SHA256=$(shasum -a 256 "$MODEL_FILE" | awk '{print $1}')
META_SHA256=$(shasum -a 256 "$META_FILE" | awk '{print $1}')
echo "Model SHA-256: $SHA256"
echo "Metadata SHA-256: $META_SHA256"

# Create the GitHub Release and upload both assets
gh release create "$TAG" "$MODEL_FILE" "$META_FILE" \
  --title "$TAG" \
  --notes "Pinned triage model artifact $TAG with metadata. Model SHA-256: $SHA256 | Meta SHA-256: $META_SHA256" \
  --target main
```

---

## Step 4: Promote via Dockerfile Config

Promotion is purely declarative configuration. Edit `Dockerfile` to point default build arguments to the new release assets with explicit URLs and hashes:

```dockerfile
# Pinned model build args (defaults to published GitHub release)
ARG MODEL_URL=https://github.com/mlexpnew/supportpilot-ingestion/releases/download/model-triage-v1.1.0/triage-v1.1.0-20261027.joblib
ARG MODEL_SHA256=<NEW_MODEL_SHA256>
ARG MODEL_META_URL=https://github.com/mlexpnew/supportpilot-ingestion/releases/download/model-triage-v1.1.0/triage-v1.1.0-20261027.json
ARG MODEL_META_SHA256=<NEW_META_SHA256>
```

---

## Step 5: Pull Request & CI Verification

1. Create a dedicated promotion branch:
   ```bash
   git checkout -b promote/model-triage-v1.1.0
   git add Dockerfile docs/model_card.md
   git commit -m "feat(model): promote model-triage-v1.1.0"
   git push origin promote/model-triage-v1.1.0
   ```
2. Open a Pull Request requesting reviews from **Rohan (ML)** and **Dev (Infra)**.
3. CI automatically verifies:
   - Image builds cleanly and downloads the pinned artifact.
   - Checksum matches the declared `MODEL_SHA256`.
   - Mismatched checksum deliberately fails the build (`! docker build ...`).
   - Consecutive builds produce the exact same `model_version` on `GET /version`.

---

## Step 6: Deploy & Verify Production

Once merged to `main`:
1. Render automatically triggers a deployment build.
2. Confirm the active version on production:
   ```bash
   curl -s https://supportpilot-ingestion.onrender.com/version | python3 -m json.tool
   ```
3. Run the onboarding diagnostic kit to ensure end-to-end classification:
   ```bash
   python3 examples/client_check.py
   ```

---

## Safe Rollback Procedure

If a newly promoted model displays anomalies or degraded real-world performance:

1. Open a one-line PR restoring `MODEL_URL`, `MODEL_SHA256`, `MODEL_META_URL`, and `MODEL_META_SHA256` in `Dockerfile` to the previous release.
2. Merge the PR.
3. Render rebuilds and redeploys the proven previous release within minutes.

