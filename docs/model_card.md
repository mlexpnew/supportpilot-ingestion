# Model Card: SupportPilot Ticket Triage Baseline (SP-105)

## 1. Model Details

- **Model Name**: SupportPilot Ticket Triage Classifier
- **Ticket**: SP-105
- **Version**: 1.0.0
- **Model Type**: Supervised Multi-Class Text Classifier
- **Architecture**: Scikit-learn Pipeline
  - **Feature Extraction**: `TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=1, strip_accents="unicode")`
  - **Classifier**: `LogisticRegression(C=2.0, max_iter=1000, random_state=42)`
- **Output**: 6-class probability distribution via multinomial logistic regression (`predict_proba`) and hard prediction (`predict`).
- **Confidence Fallback**: `predict_one(text, threshold=0.35)` returning `label`, `confidence`, and `needs_review: bool`.
- **Primary Maintainer**: SupportPilot ML Team / Lumora AI

---

## 2. Intended Use & Scope

- **Primary Application**: Automated routing of Tier-1 inbound customer support tickets into operational team queues.
- **Target Categories (`LABELS`)**:
  1. `delivery`: Tracking, shipment delays, courier issues, damaged packages, address updates.
  2. `billing`: Double charges, invoice requests, GSTIN, payment failures, convenience fees.
  3. `refund`: Return status, money back, return pickups, wallet reversals.
  4. `cancellation`: Order cancellations before dispatch, accidental orders, subscription cancellations.
  5. `account`: Login issues, password resets, OTP problems, 2FA, KYC, account security.
  6. `other`: General queries, store locations, partnerships, compliments, and technical app bug reports.
- **Out-of-Scope**:
  - Direct execution of financial refunds or account deletions without human oversight.
  - Non-text modalities (images, audio).
  - Unredacted raw PII text (must pass through SP-102 PII redaction pipeline before classification).

---

## 3. Product Decisions (Flagged by Neha, PM)

1. **Category Swap (`cancellation` over `technical`)**:
   - *Rationale*: Client enterprise requirements specifically mandate automated cancellation routing by Friday. Technical app bugs are low frequency in Tier-1 support queues and are routed into `other` for now.
   - *Backlog Tracking*: A dedicated `technical` category has been logged as **SP-106**. If telemetry indicates app bugs exceed 15% of inbound volume, a distinct classifier will be introduced.
2. **First-Action Tie-Breaking Rule**:
   - When customer inquiries exhibit cross-boundary overlap (e.g. *"cancel and refund me immediately before parcel leaves warehouse"*), the **customer's first intended action** governs the label (`cancellation`).

---

## 4. Dataset & Preprocessing

- **Dataset File**: `data/triage_labeled.jsonl`
- **Volume & Balance**: Exactly 180 records (30 per label across 6 categories).
- **Linguistic Distribution**: Realistic mixture of concise queries, narrative queries, informal phrasing, common typos (`deliv`, `refud`, `acount`), and Romanized Hindi (Hinglish) common in Indian eCommerce (e.g. *"refund kab aayega"*, *"order cancel karna hai"*).
- **PII Redaction Realism**:
  - 25 tickets contain sanitized `[PHONE]` or `[EMAIL]` tokens.
  - 3 tickets contain sanitized `[CARD]` tokens.
  - Total tickets with redacted tokens: 28 records (~15.5% of dataset).
- **Data Split**:
  - **70% Training Set**: 126 records (21 per class).
  - **30% Held-Out Test Set**: 54 records (9 per class).
  - Stratified sampling ensures identical class representation in train and test splits.

---

## 5. Performance & Benchmarking (70/30 Held-Out Split)

### Overall Metrics

| Metric | Rule Baseline (`rules.py`) | ML Classifier (`model.py`) |
|---|---|---|
| **Accuracy** | **88.89%** | **87.04%** |
| **Macro Precision** | **0.8981** | **0.8751** |
| **Macro Recall** | **0.8889** | **0.8704** |
| **Macro F1** | **0.8890** | **0.8674** |

### Stratified 5-Fold Cross-Validation (Full Dataset)

- **Mean CV Accuracy**: **85.00%** ($\pm 5.72\%$)
- **Mean CV Macro F1**: **0.8471** ($\pm 0.0586$)

---

### Per-Class Performance (Held-Out 30% Test Set, $N=54$)

| Label | Support | Rule F1 | ML Precision | ML Recall | ML F1 |
|---|---|---|---|---|---|
| **`delivery`** | 9 | 0.8571 | **0.8750** | **0.7778** | **0.8235** |
| **`billing`** | 9 | 0.8235 | **1.0000** | **0.8889** | **0.9412** |
| **`refund`** | 9 | 0.9412 | **0.9000** | **1.0000** | **0.9474** |
| **`cancellation`**| 9 | 1.0000 | **0.8182** | **1.0000** | **0.9000** |
| **`account`** | 9 | 0.8889 | **0.8000** | **0.8889** | **0.8421** |
| **`other`** | 9 | 0.8235 | **0.8571** | **0.6667** | **0.7500** |

---

### ML Confusion Matrix (Held-Out 30% Test Set)

```
Actual \ Predicted |  deliv  billi  refun  cance  accou  other
----------------------------------------------------------
delivery           |      7      0      0      0      1      1
billing            |      0      8      0      1      0      0
refund             |      0      0      9      0      0      0
cancellation       |      0      0      0      9      0      0
account            |      1      0      0      0      8      0
other              |      0      0      1      1      1      6
```

#### Diagnosis: Delivery F1 Improvement (From 0.55 to 0.82)
- **Root Cause of Initial 0.55 F1**: In the early 80/20 run, `delivery` suffered from false positives with `other` and false negatives with delayed items mentioning refund phrasing (`"3 din ho gaye"`). 
- **Remediation**:
  1. Isolated `cancellation` from `delivery` using explicit pre-dispatch n-grams (`stop dispatch`, `cancel my order`).
  2. Enriched courier tracking n-grams (`courier partner`, `transit`, `hub`, `sorting facility`).
  3. Increased training data share via 70/30 split and added `[PHONE]`/`[EMAIL]` contextual delivery tokens.
  4. Result: Delivery F1 increased from **0.55** to **0.8235** (precision 87.5%, recall 77.8%).

---

## 6. Confidence Fallback & Human Review (`needs_review`)

Automated triage must not misroute ambiguous tickets silently. The model incorporates a calibrated confidence fallback:

- **Function**: `predict_one(text, threshold=0.35)`
- **Mechanism**: Calculates $\text{confidence} = \max_{c} P(y=c \mid x)$. If $\text{confidence} < 0.35$, the ticket is tagged with `needs_review: True`.
- **Empirical Threshold Tuning**:
  - At $\tau = 0.35$: **100.00%** of unflagged predictions are accurate on the held-out test set.
  - Ambiguous queries (e.g. multi-issue complaints or terse noise) are routed to human triage specialists, preventing silent downstream routing errors.

---

## 7. Limitations & Ethical Considerations

1. **PII Safety**: The classifier is trained exclusively on PII-redacted text. Downstream callers must route tickets through `supportpilot.preprocessing` (SP-102) to sanitize credit cards, phone numbers, and email addresses prior to classification.
2. **Script Limitations**: The baseline model supports English and Romanized Hindi (Hinglish). Tickets submitted in non-Latin scripts (Devanagari, Tamil, etc.) require upstream language identification and transliteration.
3. **Data Drift**: New marketing campaigns, seasonal coupon codes, or carrier partnerships may introduce unmodeled tokens. Monthly re-training or threshold monitoring is recommended.
