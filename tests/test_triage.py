"""Unit and integration tests for SupportPilot ticket triage (SP-105)."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

from supportpilot.triage import LABELS
from supportpilot.triage.__main__ import main as cli_main, process_triage_file
from supportpilot.triage.evaluate import (
    compare_classifiers,
    evaluate_predictions,
    format_comparison_table,
)
from supportpilot.triage.model import (
    TriageClassifier,
    load_dataset,
    predict_one,
    train_model,
)
from supportpilot.triage.rules import predict_rule

DATASET_PATH = Path("data/triage_labeled.jsonl")


# =====================================================================
# 1. Dataset Integrity Tests
# =====================================================================


def test_dataset_exists_and_has_180_balanced_records():
    assert DATASET_PATH.exists(), f"Dataset not found at {DATASET_PATH}"

    ticket_ids, texts, labels = load_dataset(DATASET_PATH)
    assert len(ticket_ids) == 180
    assert len(texts) == 180
    assert len(labels) == 180

    for label in labels:
        assert label in LABELS

    # Ensure exactly 30 records per label (Rohan's checklist)
    from collections import Counter

    counts = Counter(labels)
    for label in LABELS:
        assert (
            counts[label] == 30
        ), f"Expected 30 records for {label}, got {counts[label]}"


def test_dataset_schema_and_unique_ids():
    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]

    assert len(lines) == 180
    seen_ids = set()
    for idx, line in enumerate(lines, 1):
        item = json.loads(line)
        assert "ticket_id" in item, f"Line {idx} missing ticket_id"
        assert "body" in item, f"Line {idx} missing body"
        assert "label" in item, f"Line {idx} missing label"
        assert (
            item["ticket_id"] not in seen_ids
        ), f"Duplicate ticket_id: {item['ticket_id']}"
        seen_ids.add(item["ticket_id"])


def test_dataset_contains_at_least_20_redacted_pii_tokens():
    """Reviewer requirement: real redacted data has many more; aim for about 20."""
    _, texts, _ = load_dataset(DATASET_PATH)
    pii_count = sum(1 for t in texts if "[PHONE]" in t or "[EMAIL]" in t)
    assert (
        pii_count >= 20
    ), f"Expected >= 20 tickets with [PHONE] or [EMAIL], got {pii_count}"


# =====================================================================
# 2. Rule-Based Classifier Tests
# =====================================================================


@pytest.mark.parametrize(
    "text,expected_label",
    [
        ("Where is my delivery package?", "delivery"),
        ("I was charged twice on my invoice", "billing"),
        ("When will my refund arrive in my bank account?", "refund"),
        ("Cancel my order before it ships from warehouse", "cancellation"),
        ("Cannot login, forgot password and OTP not coming", "account"),
        ("Do you sell gift cards?", "other"),
    ],
)
def test_rules_canonical_keywords(text, expected_label):
    assert predict_rule(text) == expected_label


@pytest.mark.parametrize(
    "text,expected_label",
    [
        ("item never arrived, refund my money to my account", "refund"),
        (
            "Package damaged during transit, please send a replacement shipment right away.",
            "delivery",
        ),
        (
            "delivery agent demanded 200 extra cash on prepaid order, why was I charged?",
            "billing",
        ),
        ("you overcharged me 350 rs on invoice, credit it back to my card", "billing"),
        ("app crashes every time I open the login page", "other"),
        (
            "entered OTP 5 times but says session expired, now my account is locked",
            "account",
        ),
        (
            "Money debited from [CARD] but checkout screen froze and order is not confirmed.",
            "billing",
        ),
        ("do you accept cash on delivery or only credit cards and UPI?", "billing"),
        (
            "cancel my active subscription and permanently delete my profile from your database",
            "account",
        ),
        (
            "I returned the wrong item by accident, can I get my original shoes back?",
            "other",
        ),
        # Neha's rule: First customer action decides label ("cancel and refund me" -> cancellation)
        (
            "cancel and refund me immediately before parcel leaves warehouse",
            "cancellation",
        ),
    ],
)
def test_rules_tricky_cross_boundary_cases(text, expected_label):
    """Rohan & Neha's checklist: tricky boundary tickets must resolve correctly."""
    assert predict_rule(text) == expected_label


def test_rules_empty_and_unknown_fallback():
    assert predict_rule("") == "other"
    assert predict_rule("   ") == "other"
    assert predict_rule("xyz random gibberish 12345") == "other"


# =====================================================================
# 3. ML Model Classifier Tests
# =====================================================================


def test_model_training_and_predictions(tmp_path):
    texts = [
        "Where is my package parcel tracking",
        "Charged twice on my credit card statement invoice",
        "Refund returned item wapas paisa",
        "Cancel order before dispatch stop shipment",
        "Login password otp account verification",
        "Gift card wholesale partner retail store",
    ]
    labels = list(LABELS)

    clf = TriageClassifier()
    clf.fit(texts, labels)

    # Test single prediction
    assert clf.predict("Where is my parcel?") == "delivery"
    assert clf.predict("My card was charged twice") == "billing"
    assert clf.predict("Cancel my order immediately") == "cancellation"

    # Test predict_proba
    probs = clf.predict_proba("Where is my parcel?")
    assert set(probs.keys()) == set(LABELS)
    assert abs(sum(probs.values()) - 1.0) < 1e-4

    # Test empty input handling
    assert clf.predict("") == "other"
    assert clf.predict_proba("")["other"] == 1.0

    # Test model persistence and metadata
    assert clf.metadata["macro_f1"] is None
    model_file = tmp_path / "model.joblib"
    clf.save(model_file)
    assert model_file.exists()

    reloaded = TriageClassifier.load(model_file)
    assert reloaded.metadata["macro_f1"] is None
    assert reloaded.predict("Where is my parcel?") == "delivery"


def test_train_model_helper(tmp_path):
    model_path = tmp_path / "saved_triage.joblib"
    clf = train_model(DATASET_PATH, save_path=model_path)
    assert model_path.exists()
    assert clf.predict("Where is my courier tracking?") == "delivery"


def test_predict_one_confidence_fallback():
    """Verify predict_one returns label, confidence, and needs_review flag."""
    clf = train_model(DATASET_PATH)

    # 1. High confidence prediction
    res_high = clf.predict_one("Where is my parcel courier tracking?")
    assert res_high["label"] == "delivery"
    assert isinstance(res_high["confidence"], float)
    assert res_high["needs_review"] is False

    # 2. Low confidence / ambiguous prediction with strict threshold
    res_low = clf.predict_one("Random gibberish words 12345", threshold=0.99)
    assert res_low["needs_review"] is True

    # 3. Empty input handling
    res_empty = clf.predict_one("")
    assert res_empty["label"] == "other"
    assert res_empty["confidence"] == 1.0
    assert res_empty["needs_review"] is False

    # 4. Top-level helper function
    top_res = predict_one("Cancel my order immediately")
    assert top_res["label"] == "cancellation"
    assert "needs_review" in top_res


# =====================================================================
# 4. Evaluation Module Tests
# =====================================================================


def test_evaluate_predictions_metrics():
    y_true = ["delivery", "billing", "refund", "cancellation", "account", "other"]
    y_pred = ["delivery", "billing", "refund", "cancellation", "account", "delivery"]

    res = evaluate_predictions(y_true, y_pred)
    assert res["total_samples"] == 6
    assert 0.0 <= res["accuracy"] <= 1.0
    assert 0.0 <= res["macro_precision"] <= 1.0
    assert 0.0 <= res["macro_recall"] <= 1.0
    assert 0.0 <= res["macro_f1"] <= 1.0
    assert set(res["per_class"].keys()) == set(LABELS)
    assert len(res["confusion_matrix"]) == 6


def test_compare_classifiers_70_30_split():
    """Verify 70/30 split and evaluation report output."""
    results = compare_classifiers(
        data_path=DATASET_PATH, test_size=0.3, random_state=42
    )
    assert results["train_samples"] == 126
    assert results["test_samples"] == 54
    assert "rule_baseline" in results
    assert "ml_model" in results
    assert "fallback_analysis" in results
    assert "ml_cross_validation" in results

    table = format_comparison_table(results)
    assert "SUPPORTPILOT TICKET TRIAGE EVALUATION REPORT" in table
    assert "Dataset Split: 70/30" in table
    assert "Rule Baseline" in table
    assert "ML Model (Held-Out)" in table
    assert "Confidence Fallback Threshold" in table
    assert "ML Confusion Matrix" in table


# =====================================================================
# 5. CLI Entry Point Tests
# =====================================================================


def test_cli_predict_single_ticket_model(capsys):
    test_args = ["prog", "--predict", "Refund kab aayega?", "--method", "model"]
    with patch.object(sys, "argv", test_args):
        code = cli_main()
        assert code == 0

    captured = capsys.readouterr()
    data = json.loads(captured.out)
    assert data["label"] == "refund"
    assert "confidence" in data
    assert data["method"] == "model"


def test_cli_predict_single_ticket_rules(capsys):
    test_args = [
        "prog",
        "--predict",
        "delivery boy cancelled without calling",
        "--method",
        "rules",
    ]
    with patch.object(sys, "argv", test_args):
        code = cli_main()
        assert code == 0

    captured = capsys.readouterr()
    data = json.loads(captured.out)
    assert data["label"] == "delivery"
    assert data["method"] == "rules"


def test_cli_batch_file_processing(tmp_path, capsys):
    input_file = tmp_path / "tickets.jsonl"
    output_dir = tmp_path / "out_triage"

    sample_tickets = [
        {"ticket_id": "T-1", "body": "Where is my parcel?"},
        {"ticket_id": "T-2", "body": "Cancel my order before it ships"},
    ]
    with open(input_file, "w", encoding="utf-8") as f:
        for t in sample_tickets:
            f.write(json.dumps(t) + "\n")

    test_args = [
        "prog",
        "--input",
        str(input_file),
        "--output-dir",
        str(output_dir),
        "--method",
        "rules",
    ]
    with patch.object(sys, "argv", test_args):
        code = cli_main()
        assert code == 0

    triaged_file = output_dir / "triaged.jsonl"
    report_file = output_dir / "triage_report.json"
    assert triaged_file.exists()
    assert report_file.exists()

    with open(triaged_file, "r", encoding="utf-8") as f:
        triaged_records = [json.loads(line) for line in f if line.strip()]

    assert len(triaged_records) == 2
    assert triaged_records[0]["label"] == "delivery"
    assert triaged_records[1]["label"] == "cancellation"

    expected_keys = {"ticket_id", "label", "confidence", "needs_review"}
    for record in triaged_records:
        assert set(record.keys()) == expected_keys
        assert len(record) == 4
        assert "body" not in record

    with open(report_file, "r", encoding="utf-8") as f:
        report = json.load(f)
    assert report["total_records"] == 2
    assert report["counts"]["delivery"] == 1
    assert report["counts"]["cancellation"] == 1

    captured = capsys.readouterr()
    assert captured.out == ""
    assert "Ticket triage completed successfully." in captured.err
    assert "Total records: 2" in captured.err
    assert "Output directory:" in captured.err


def test_triage_output_contract_exact_keys(tmp_path):
    """Assert output contract strictly produces ticket_id, label, confidence, needs_review."""
    input_file = tmp_path / "tickets.jsonl"
    output_dir = tmp_path / "out_triage"

    sample_tickets = [
        {
            "ticket_id": "T-101",
            "subject": "Delay inquiry",
            "body": "Where is my package? It is 3 days late.",
            "customer_id": "C-999",
            "extra_field": "do_not_leak",
        },
        {
            "ticket_id": "T-102",
            "subject": "Cancellation",
            "body": "Cancel my order immediately before dispatch",
            "channel": "email",
        },
    ]
    with open(input_file, "w", encoding="utf-8") as f:
        for t in sample_tickets:
            f.write(json.dumps(t) + "\n")

    report = process_triage_file(
        input_path=input_file,
        output_dir=output_dir,
        method="model",
    )
    assert report["total_records"] == 2

    triaged_file = output_dir / "triaged.jsonl"
    assert triaged_file.exists()

    with open(triaged_file, "r", encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]

    assert len(records) == 2
    expected_keys = {"ticket_id", "label", "confidence", "needs_review"}

    for r in records:
        assert set(r.keys()) == expected_keys, f"Keys mismatch: {set(r.keys())}"
        assert len(r) == 4
        # Explicitly verify sensitive/unneeded fields from original ticket are omitted
        assert "body" not in r
        assert "subject" not in r
        assert "customer_id" not in r
        assert "extra_field" not in r
        assert "channel" not in r
        assert "predicted_label" not in r

    assert records[0]["ticket_id"] == "T-101"
    assert records[0]["label"] == "delivery"
    assert isinstance(records[0]["confidence"], float)
    assert isinstance(records[0]["needs_review"], bool)

    assert records[1]["ticket_id"] == "T-102"
    assert records[1]["label"] == "cancellation"
    assert isinstance(records[1]["confidence"], float)
    assert isinstance(records[1]["needs_review"], bool)


def test_process_triage_file_skips_invalid_json_and_counts(tmp_path, capsys):
    """Verify that bad JSON lines do not crash the run, are skipped and counted."""
    input_file = tmp_path / "mixed_tickets.jsonl"
    output_dir = tmp_path / "out_triage"

    lines = [
        json.dumps({"ticket_id": "T-1", "body": "Where is my courier package?"}),
        "{ bad json line without closing brackets",
        json.dumps(
            {"ticket_id": "T-2", "body": "Please refund my money to original card"}
        ),
        json.dumps(["not_a_dict_record"]),
        "",  # blank line
        json.dumps({"ticket_id": "T-3", "body": "Cancel my order before it ships"}),
    ]
    with open(input_file, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    report = process_triage_file(
        input_path=input_file,
        output_dir=output_dir,
        method="rules",
    )

    assert report["total_records"] == 3
    assert report["skipped_records"] == 2

    triaged_file = output_dir / "triaged.jsonl"
    with open(triaged_file, "r", encoding="utf-8") as f:
        triaged_records = [json.loads(line) for line in f if line.strip()]

    assert len(triaged_records) == 3
    expected_keys = {"ticket_id", "label", "confidence", "needs_review"}
    for record in triaged_records:
        assert set(record.keys()) == expected_keys
        assert len(record) == 4

    assert [r["ticket_id"] for r in triaged_records] == ["T-1", "T-2", "T-3"]

    captured = capsys.readouterr()
    assert "Skipping invalid JSON" in captured.err
    assert "Skipping non-object record" in captured.err


def test_cli_file_not_found(capsys):
    test_args = [
        "prog",
        "--input",
        "nonexistent_file.jsonl",
        "--output-dir",
        "some_out",
    ]
    with patch.object(sys, "argv", test_args):
        code = cli_main()
        assert code == 1

    captured = capsys.readouterr()
    assert "Error: input file not found" in captured.err


def test_cli_evaluate_mode(capsys):
    test_args = ["prog", "--evaluate"]
    with patch.object(sys, "argv", test_args):
        code = cli_main()
        assert code == 0

    captured = capsys.readouterr()
    assert "SUPPORTPILOT TICKET TRIAGE EVALUATION REPORT" in captured.out


def test_cli_batch_missing_ticket_id_is_skipped_and_counted(tmp_path, capsys):
    """Verify that records with missing, null, or empty ticket_id are skipped and counted."""
    input_file = tmp_path / "missing_id_tickets.jsonl"
    output_dir = tmp_path / "out_triage"

    sample_tickets = [
        {"ticket_id": "T-VALID-1", "body": "Where is my parcel?"},
        {"body": "Missing ticket_id entirely"},
        {"ticket_id": None, "body": "None ticket_id"},
        {"ticket_id": "", "body": "Empty string ticket_id"},
        {"ticket_id": "   ", "body": "Whitespace ticket_id"},
        {"ticket_id": "T-VALID-2", "body": "Cancel my order before it ships"},
    ]
    with open(input_file, "w", encoding="utf-8") as f:
        for t in sample_tickets:
            f.write(json.dumps(t) + "\n")

    test_args = [
        "prog",
        "--input",
        str(input_file),
        "--output-dir",
        str(output_dir),
        "--method",
        "rules",
    ]
    with patch.object(sys, "argv", test_args):
        code = cli_main()
        assert code == 0

    triaged_file = output_dir / "triaged.jsonl"
    with open(triaged_file, "r", encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]

    assert len(records) == 2
    assert [r["ticket_id"] for r in records] == ["T-VALID-1", "T-VALID-2"]
    for r in records:
        assert r["ticket_id"] is not None
        assert r["ticket_id"] != ""

    report_file = output_dir / "triage_report.json"
    with open(report_file, "r", encoding="utf-8") as f:
        report = json.load(f)
    assert report["total_records"] == 2
    assert report["skipped_records"] == 4

    captured = capsys.readouterr()
    assert "Skipping record with missing ticket_id" in captured.err


def test_cli_stdout_is_empty_or_one_json_line(tmp_path, capsys):
    """Verify stdout contract: batch mode has empty stdout (human output in stderr); predict mode has exactly one JSON line."""
    input_file = tmp_path / "tickets.jsonl"
    output_dir = tmp_path / "out_triage"

    sample_tickets = [
        {"ticket_id": "T-1", "body": "Where is my parcel?"},
        {"ticket_id": "T-2", "body": "Please refund my payment"},
    ]
    with open(input_file, "w", encoding="utf-8") as f:
        for t in sample_tickets:
            f.write(json.dumps(t) + "\n")

    # 1. Batch mode: stdout must be empty (or at most one JSON line; strictly empty here)
    batch_args = [
        "prog",
        "--input",
        str(input_file),
        "--output-dir",
        str(output_dir),
        "--method",
        "rules",
    ]
    with patch.object(sys, "argv", batch_args):
        code = cli_main()
        assert code == 0

    captured_batch = capsys.readouterr()
    batch_stdout_lines = [
        line for line in captured_batch.out.strip().splitlines() if line.strip()
    ]
    assert len(batch_stdout_lines) in (0, 1)
    assert captured_batch.out == ""
    # Verify human status lines were redirected to stderr
    assert "Ticket triage completed successfully." in captured_batch.err
    assert "Total records: 2" in captured_batch.err
    assert "Output directory:" in captured_batch.err

    # 2. Predict mode: stdout must be strictly one valid JSON line
    predict_args = [
        "prog",
        "--predict",
        "Where is my parcel?",
        "--method",
        "rules",
    ]
    with patch.object(sys, "argv", predict_args):
        code = cli_main()
        assert code == 0

    captured_predict = capsys.readouterr()
    predict_stdout_lines = [
        line for line in captured_predict.out.strip().splitlines() if line.strip()
    ]
    assert len(predict_stdout_lines) == 1
    predict_data = json.loads(predict_stdout_lines[0])
    assert predict_data["label"] == "delivery"


def test_model_metadata_version_date_size_macro_f1(tmp_path):
    """Verify metadata is saved and loaded: version, date, size, F1, labels."""
    import re
    from datetime import datetime, timezone
    from supportpilot.triage.model import (
        DEFAULT_MODEL_VERSION,
        TriageClassifier,
        train_model,
        is_valid_model_version,
    )

    model_file = tmp_path / "versioned_model.joblib"
    sidecar_json = tmp_path / "versioned_model.json"

    clf = train_model(DATASET_PATH, save_path=model_file, version="2.1.0")
    assert model_file.exists()
    assert sidecar_json.exists()

    today_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    today_compact = today_str.replace("-", "")
    expected_version = f"triage-v2.1.0-{today_compact}"

    # 1. In-memory metadata
    assert clf.version == expected_version
    assert is_valid_model_version(clf.version)
    meta = clf.metadata
    assert meta["version"] == expected_version
    assert meta["training_date"] == today_str
    assert meta["dataset_size"] == 180
    assert isinstance(meta["macro_f1"], float)
    # Cross-validated macro F1 is around 0.8471, not inflated ~1.0 memorization
    assert 0.80 <= meta["macro_f1"] <= 0.90
    assert "labels" in meta
    assert isinstance(meta["labels"], list)
    for lbl in LABELS:
        assert lbl in meta["labels"]

    # 2. Sidecar JSON content
    with open(sidecar_json, "r", encoding="utf-8") as f:
        json_meta = json.load(f)
    assert json_meta == meta

    # 3. Reloaded model from .joblib
    reloaded = TriageClassifier.load(model_file)
    assert reloaded.version == expected_version
    assert reloaded.metadata == meta
    assert reloaded.predict("Where is my parcel?") == "delivery"

    # 4. Fail-fast tests: missing metadata, missing version, broken format
    import joblib

    # Raw pipeline without metadata dictionary must fail fast
    raw_pipeline_file = tmp_path / "raw_pipeline.joblib"
    joblib.dump(clf.pipeline, raw_pipeline_file)
    with pytest.raises(ValueError, match="expected serialized dictionary"):
        TriageClassifier.load(raw_pipeline_file)

    # Missing metadata dict must fail fast
    no_meta_file = tmp_path / "no_meta.joblib"
    joblib.dump({"pipeline": clf.pipeline}, no_meta_file)
    with pytest.raises(ValueError, match="Missing or invalid 'metadata'"):
        TriageClassifier.load(no_meta_file)

    # Missing version in metadata must fail fast
    no_version_file = tmp_path / "no_version.joblib"
    joblib.dump(
        {
            "pipeline": clf.pipeline,
            "metadata": {
                "training_date": today_str,
                "dataset_size": 180,
                "macro_f1": 0.85,
                "labels": list(LABELS),
            },
        },
        no_version_file,
    )
    with pytest.raises(ValueError, match="Missing 'version'"):
        TriageClassifier.load(no_version_file)

    # Broken version format (e.g. legacy '1.0.0' without triage-v prefix and date)
    broken_ver_file = tmp_path / "broken_ver.joblib"
    joblib.dump(
        {
            "pipeline": clf.pipeline,
            "version": "1.0.0",
            "metadata": {
                "version": "1.0.0",
                "training_date": today_str,
                "dataset_size": 180,
                "macro_f1": 0.85,
                "labels": list(LABELS),
            },
        },
        broken_ver_file,
    )
    with pytest.raises(ValueError, match="Broken or invalid model version"):
        TriageClassifier.load(broken_ver_file)

    # Missing labels in metadata must fail fast
    no_labels_file = tmp_path / "no_labels.joblib"
    joblib.dump(
        {
            "pipeline": clf.pipeline,
            "version": expected_version,
            "metadata": {
                "version": expected_version,
                "training_date": today_str,
                "dataset_size": 180,
                "macro_f1": 0.85,
            },
        },
        no_labels_file,
    )
    with pytest.raises(ValueError, match="Missing or invalid 'labels'"):
        TriageClassifier.load(no_labels_file)


def test_tiny_dataset_macro_f1_returns_none_with_warning(caplog) -> None:
    """Tiny dataset (<2 examples/label) sets macro_f1 to None and logs warning."""
    import logging

    texts = [
        "Where is my package parcel tracking",
        "Charged twice on my credit card statement invoice",
        "Refund returned item wapas paisa",
        "Cancel order before dispatch stop shipment",
        "Login password otp account verification",
        "Gift card wholesale partner retail store",
    ]
    labels = list(LABELS)

    clf = TriageClassifier()
    with caplog.at_level(logging.WARNING):
        clf.fit(texts, labels)

    assert clf.metadata["macro_f1"] is None
    assert (
        "Dataset too small for cross-validation (<2 examples per label); "
        "macro_f1 set to None."
    ) in caplog.text
