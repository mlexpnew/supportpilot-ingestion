"""Unit and integration tests for SupportPilot ticket triage (SP-105)."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

from supportpilot.triage import LABELS
from supportpilot.triage.__main__ import main as cli_main
from supportpilot.triage.evaluate import (
    compare_classifiers,
    evaluate_predictions,
    format_comparison_table,
)
from supportpilot.triage.model import (
    TriageClassifier,
    load_dataset,
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

    # Ensure all labels in dataset are valid
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


# =====================================================================
# 2. Rule-Based Classifier Tests
# =====================================================================


@pytest.mark.parametrize(
    "text,expected_label",
    [
        ("Where is my delivery package?", "delivery"),
        ("I was charged twice on my invoice", "billing"),
        ("When will my refund arrive in my bank account?", "refund"),
        ("App crashes on launch with 500 internal server error", "technical"),
        ("Cannot login, forgot password and OTP not coming", "account"),
        ("Do you sell gift cards?", "other"),
    ],
)
def test_rules_canonical_keywords(text, expected_label):
    assert predict_rule(text) == expected_label


@pytest.mark.parametrize(
    "text,expected_label",
    [
        ("item never arrived, refund my money", "refund"),
        (
            "Package damaged during transit, please send a replacement unit right away",
            "delivery",
        ),
        (
            "delivery agent demanded 200 extra cash on prepaid order, why was I charged?",
            "billing",
        ),
        ("you overcharged me 350 rs on invoice, credit it back to my card", "billing"),
        ("app crashes every time I open the login page", "technical"),
        (
            "entered OTP 5 times but says session expired, now my account is locked",
            "account",
        ),
        (
            "Money debited from [CARD] but checkout screen froze and order is not confirmed",
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
    ],
)
def test_rules_10_tricky_cross_boundary_cases(text, expected_label):
    """Rohan's checklist: all 10 tricky boundary tickets must resolve correctly."""
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
        "App crash freeze 500 error bug",
        "Login password otp account verification",
        "Gift card wholesale partner retail store",
    ]
    labels = list(LABELS)

    clf = TriageClassifier()
    clf.fit(texts, labels)

    # Test single prediction
    assert clf.predict("Where is my parcel?") == "delivery"
    assert clf.predict("My card was charged twice") == "billing"

    # Test predict_proba
    probs = clf.predict_proba("Where is my parcel?")
    assert set(probs.keys()) == set(LABELS)
    assert abs(sum(probs.values()) - 1.0) < 1e-4

    # Test empty input handling
    assert clf.predict("") == "other"
    assert clf.predict_proba("")["other"] == 1.0

    # Test model persistence
    model_file = tmp_path / "model.joblib"
    clf.save(model_file)
    assert model_file.exists()

    reloaded = TriageClassifier.load(model_file)
    assert reloaded.predict("Where is my parcel?") == "delivery"


def test_train_model_helper(tmp_path):
    model_path = tmp_path / "saved_triage.joblib"
    clf = train_model(DATASET_PATH, save_path=model_path)
    assert model_path.exists()
    assert clf.predict("Where is my courier tracking?") == "delivery"


# =====================================================================
# 4. Evaluation Module Tests
# =====================================================================


def test_evaluate_predictions_metrics():
    y_true = ["delivery", "billing", "refund", "technical", "account", "other"]
    y_pred = ["delivery", "billing", "refund", "technical", "account", "delivery"]

    res = evaluate_predictions(y_true, y_pred)
    assert res["total_samples"] == 6
    assert 0.0 <= res["accuracy"] <= 1.0
    assert 0.0 <= res["macro_precision"] <= 1.0
    assert 0.0 <= res["macro_recall"] <= 1.0
    assert 0.0 <= res["macro_f1"] <= 1.0
    assert set(res["per_class"].keys()) == set(LABELS)


def test_compare_classifiers_execution():
    results = compare_classifiers(
        data_path=DATASET_PATH, test_size=0.2, random_state=42
    )
    assert results["train_samples"] == 144
    assert results["test_samples"] == 36
    assert "rule_baseline" in results
    assert "ml_model" in results
    assert "ml_cross_validation" in results

    table = format_comparison_table(results)
    assert "SUPPORTPILOT TICKET TRIAGE EVALUATION REPORT" in table
    assert "Rule Baseline" in table
    assert "ML Model (Held-Out)" in table


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
        {"ticket_id": "T-2", "body": "Charged twice on card"},
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
    assert triaged_records[0]["predicted_label"] == "delivery"
    assert triaged_records[1]["predicted_label"] == "billing"

    with open(report_file, "r", encoding="utf-8") as f:
        report = json.load(f)
    assert report["total_records"] == 2
    assert report["counts"]["delivery"] == 1
    assert report["counts"]["billing"] == 1


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
