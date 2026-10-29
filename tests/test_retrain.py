"""Unit and integration tests for SupportPilot retraining pipeline (SP-113).

Covers:
- Semantic version bumping (triage-v1.1.0-YYYYMMDD)
- Promotion rule in all 4 required scenarios:
    1. Win: Macro F1 improves by >= 0.02 and no label recall drops by > 0.05
    2. Lose: Macro F1 decreases (< 0.00)
    3. Tie / Sub-threshold: Macro F1 delta is 0 or < 0.02
    4. One label regresses: Macro F1 improves by >= 0.02 but one label recall drops by > 0.05
- Frozen evaluation set is sacred (never trained on, zero data leak)
- Champion model is never overwritten
- End-to-end pipeline execution and JSON report contract
"""

from __future__ import annotations

import collections
import json
from pathlib import Path
import re
from typing import Any
import pytest

from supportpilot.triage import LABELS
from supportpilot.triage.retrain import (
    bump_model_version,
    evaluate_promotion_rule,
    retrain_pipeline,
    verify_eval_frozen_isolation,
)


# =====================================================================
# 1. Version Bumping Tests
# =====================================================================


def test_bump_model_version_minor() -> None:
    """Verify champion version v1.0.0 bumps to v1.1.0-YYYYMMDD."""
    bumped = bump_model_version("triage-v1.0.0-20261022", "20261027")
    assert bumped == "triage-v1.1.0-20261027"


def test_bump_model_version_when_champion_none_raises_value_error() -> None:
    """Verify bump_model_version fails fast when champion version is missing or empty."""
    with pytest.raises(ValueError, match="Champion version is required"):
        bump_model_version(None, "20261027")

    with pytest.raises(ValueError, match="Champion version is required"):
        bump_model_version("   ", "20261027")


def test_bump_model_version_subsequent_retrain() -> None:
    """Verify second retraining bumps v1.1.0 to v1.2.0-YYYYMMDD."""
    bumped = bump_model_version("triage-v1.1.0-20261027", "20261029")
    assert bumped == "triage-v1.2.0-20261029"


# =====================================================================
# 2. Promotion Rule Tests (Win / Lose / Tie / One Label Regresses)
# =====================================================================


def test_promotion_rule_win() -> None:
    """Scenario 1: Challenger wins when Macro F1 improves by >= 0.02 and no label drops by > 0.05."""
    champ = {
        "macro_f1": 0.82,
        "per_label_recall": {label: 0.80 for label in LABELS},
    }
    chall = {
        "macro_f1": 0.85,  # +0.03 improvement (>= +0.02)
        "per_label_recall": {label: 0.82 for label in LABELS},  # all improved
    }
    decision = evaluate_promotion_rule(champ, chall)
    assert decision.promoted is True
    assert decision.winner == "challenger"
    assert decision.f1_improved is True
    assert decision.no_recall_regressed is True
    assert decision.f1_diff == 0.03
    assert len(decision.regressed_labels) == 0


def test_promotion_rule_lose() -> None:
    """Scenario 2: Challenger loses when Macro F1 is worse than champion."""
    champ = {
        "macro_f1": 0.85,
        "per_label_recall": {label: 0.85 for label in LABELS},
    }
    chall = {
        "macro_f1": 0.81,  # -0.04 drop
        "per_label_recall": {label: 0.80 for label in LABELS},
    }
    decision = evaluate_promotion_rule(champ, chall)
    assert decision.promoted is False
    assert decision.winner == "champion"
    assert decision.f1_improved is False


def test_promotion_rule_tie_or_subthreshold() -> None:
    """Scenario 3: Challenger loses on tie or sub-threshold improvement (< 0.02)."""
    # Exactly 0 delta (tie)
    champ = {
        "macro_f1": 0.85,
        "per_label_recall": {label: 0.85 for label in LABELS},
    }
    chall_tie = {
        "macro_f1": 0.85,
        "per_label_recall": {label: 0.85 for label in LABELS},
    }
    decision_tie = evaluate_promotion_rule(champ, chall_tie)
    assert decision_tie.promoted is False
    assert decision_tie.winner == "champion"

    # Sub-threshold improvement (+0.015 < 0.02)
    chall_sub = {
        "macro_f1": 0.865,
        "per_label_recall": {label: 0.86 for label in LABELS},
    }
    decision_sub = evaluate_promotion_rule(champ, chall_sub)
    assert decision_sub.promoted is False
    assert decision_sub.winner == "champion"


def test_promotion_rule_one_label_regresses() -> None:
    """Scenario 4: Challenger loses when F1 improves by >= 0.02, but ONE label's recall drops by > 0.05."""
    champ = {
        "macro_f1": 0.80,
        "per_label_recall": {
            "delivery": 0.85,
            "billing": 0.85,
            "refund": 0.85,
            "cancellation": 0.85,
            "account": 0.85,
            "other": 0.85,
        },
    }
    chall = {
        "macro_f1": 0.84,  # +0.04 improvement (>= 0.02)
        "per_label_recall": {
            "delivery": 0.95,
            "billing": 0.95,
            "refund": 0.95,
            "cancellation": 0.95,
            "account": 0.95,
            "other": 0.78,  # Dropped from 0.85 to 0.78 (-0.07 drop, which is > 0.05!)
        },
    }
    decision = evaluate_promotion_rule(champ, chall)
    assert decision.promoted is False
    assert decision.winner == "champion"
    assert decision.f1_improved is True
    assert decision.no_recall_regressed is False
    assert "other" in decision.regressed_labels


# =====================================================================
# 3. Sacred Frozen Evaluation Set Isolation Tests
# =====================================================================


def test_eval_frozen_set_structure_and_zero_overlap() -> None:
    """Verify data/eval_frozen.jsonl contains 60 tickets (10 per label) with zero training overlap."""
    eval_path = Path("data/eval_frozen.jsonl")
    train_path = Path("data/triage_labeled.jsonl")

    assert eval_path.exists(), "data/eval_frozen.jsonl must exist"
    assert train_path.exists(), "data/triage_labeled.jsonl must exist"

    eval_ids: set[str] = set()
    eval_texts: set[str] = set()
    eval_counts: collections.Counter[str] = collections.Counter()

    with open(eval_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                item = json.loads(line.strip())
                eval_ids.add(item["ticket_id"])
                eval_texts.add(item["body"])
                eval_counts[item["label"]] += 1

    train_ids: set[str] = set()
    train_texts: set[str] = set()
    with open(train_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                item = json.loads(line.strip())
                train_ids.add(item["ticket_id"])
                train_texts.add(item["body"])

    # Must contain 60 samples, exactly 10 per label
    assert len(eval_ids) == 60
    assert set(eval_counts.keys()) == set(LABELS)
    assert all(count == 10 for count in eval_counts.values())

    # Sacred isolation: Zero intersection in IDs or body text
    overlap_ids = eval_ids.intersection(train_ids)
    overlap_texts = eval_texts.intersection(train_texts)

    assert len(overlap_ids) == 0, f"Found overlapping ticket IDs: {overlap_ids}"
    assert len(overlap_texts) == 0, f"Found overlapping ticket texts: {overlap_texts}"


def test_verify_eval_frozen_isolation_detects_data_leak(tmp_path: Path) -> None:
    """Verify verify_eval_frozen_isolation raises RuntimeError on leaked ticket ID or body."""
    eval_file = tmp_path / "eval.jsonl"
    eval_file.write_text(
        json.dumps(
            {
                "ticket_id": "EV-LEAK-1",
                "body": "Where is my parcel?",
                "label": "delivery",
            }
        )
        + "\n"
    )

    # 1. Leaked ticket ID
    train_leak_id = tmp_path / "train_leak_id.jsonl"
    train_leak_id.write_text(
        json.dumps(
            {
                "ticket_id": "EV-LEAK-1",
                "body": "Completely different text",
                "label": "delivery",
            }
        )
        + "\n"
    )
    with pytest.raises(RuntimeError, match="Data leakage detected.*EV-LEAK-1"):
        verify_eval_frozen_isolation(train_leak_id, eval_file)

    # 2. Leaked body text (exact match)
    train_leak_text = tmp_path / "train_leak_text.jsonl"
    train_leak_text.write_text(
        json.dumps(
            {"ticket_id": "TR-999", "body": "Where is my parcel?", "label": "delivery"}
        )
        + "\n"
    )
    with pytest.raises(RuntimeError, match="Data leakage detected.*ticket text"):
        verify_eval_frozen_isolation(train_leak_text, eval_file)

    # 3. Leaked body text with different casing and collapsed whitespace
    train_leak_norm = tmp_path / "train_leak_norm.jsonl"
    train_leak_norm.write_text(
        json.dumps(
            {
                "ticket_id": "TR-1000",
                "body": "   where   is   MY   parcel?  \n",
                "label": "delivery",
            }
        )
        + "\n"
    )
    with pytest.raises(RuntimeError, match="Data leakage detected.*ticket text"):
        verify_eval_frozen_isolation(train_leak_norm, eval_file)


def test_verify_eval_frozen_isolation_missing_files_fails_fast(tmp_path: Path) -> None:
    """Verify verify_eval_frozen_isolation fails fast on missing datasets."""
    missing_train = tmp_path / "nonexistent_train.jsonl"
    missing_eval = tmp_path / "nonexistent_eval.jsonl"
    real_eval = Path("data/eval_frozen.jsonl")

    with pytest.raises(FileNotFoundError, match="Training dataset not found"):
        verify_eval_frozen_isolation(missing_train, real_eval)

    with pytest.raises(FileNotFoundError, match="Frozen evaluation dataset not found"):
        verify_eval_frozen_isolation("data/triage_labeled.jsonl", missing_eval)


# =====================================================================
# 4. End-to-End Retraining, Champion Invariants & Expected Version Tests
# =====================================================================


def test_retrain_pipeline_never_overwrites_champion(tmp_path: Path) -> None:
    """Verify champion file is preserved and challenger is saved as a separate artifact."""
    champ_file = tmp_path / "triage_model.joblib"
    out_dir = tmp_path / "models"
    reports_dir = tmp_path / "reports"

    # Train initial baseline champion
    from supportpilot.triage.model import train_model

    train_model("data/triage_labeled.jsonl", save_path=champ_file, version="1.0.0")
    champ_mtime = champ_file.stat().st_mtime
    champ_bytes = champ_file.read_bytes()

    # Run retraining pipeline
    results = retrain_pipeline(
        base_dataset="data/triage_labeled.jsonl",
        output_dir=out_dir,
        eval_dataset="data/eval_frozen.jsonl",
        champion_path=champ_file,
        reports_dir=reports_dir,
    )

    # Champion must be completely untouched
    assert champ_file.stat().st_mtime == champ_mtime
    assert champ_file.read_bytes() == champ_bytes

    # Challenger must be saved under a distinct filename
    challenger_path = Path(results["challenger_path"])
    assert challenger_path.exists()
    assert challenger_path.resolve() != champ_file.resolve()
    assert re.search(r"triage-v\d+\.\d+\.\d+-\d{8}\.joblib", challenger_path.name)

    # Report must be written and readable
    report_file = Path(results["report_file"])
    assert report_file.exists()
    report_data = json.loads(report_file.read_text(encoding="utf-8"))
    assert "champion" in report_data
    assert "challenger" in report_data
    assert "decision" in report_data
    assert "eval_dataset" in report_data


def test_retrain_missing_champion_fails_fast(tmp_path: Path) -> None:
    """Verify missing champion file raises FileNotFoundError instead of inventing a baseline."""
    missing_champ = tmp_path / "nonexistent_champion.joblib"
    out_dir = tmp_path / "models"

    with pytest.raises(
        FileNotFoundError, match="Refusing to invent an unverified baseline champion"
    ):
        retrain_pipeline(
            base_dataset="data/triage_labeled.jsonl",
            output_dir=out_dir,
            eval_dataset="data/eval_frozen.jsonl",
            champion_path=missing_champ,
        )


def test_retrain_unversioned_champion_fails_fast(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verify champion with missing version metadata fails fast."""
    import joblib
    from supportpilot.triage.model import train_model, TriageClassifier

    champ_file = tmp_path / "broken_champion.joblib"
    train_model("data/triage_labeled.jsonl", save_path=champ_file, version="1.0.0")
    classifier = TriageClassifier.load(champ_file)

    # 1. Missing version in serialized file raises ValueError on load
    joblib.dump(
        {"pipeline": classifier.pipeline, "metadata": {"labels": list(LABELS)}},
        champ_file,
    )
    out_dir = tmp_path / "models"
    with pytest.raises(ValueError, match="Missing 'version' in model file"):
        retrain_pipeline(
            base_dataset="data/triage_labeled.jsonl",
            output_dir=out_dir,
            eval_dataset="data/eval_frozen.jsonl",
            champion_path=champ_file,
        )

    # 2. Champion object with empty version string fails fast in retrain_pipeline
    def mock_load(path: Any) -> Any:
        classifier.version = ""
        return classifier

    monkeypatch.setattr(TriageClassifier, "load", mock_load)
    with pytest.raises(ValueError, match="missing a version in its metadata"):
        retrain_pipeline(
            base_dataset="data/triage_labeled.jsonl",
            output_dir=out_dir,
            eval_dataset="data/eval_frozen.jsonl",
            champion_path=champ_file,
        )


def test_retrain_expected_champion_version_check(tmp_path: Path) -> None:
    """Verify --expected-champion-version succeeds on match and aborts on mismatch."""
    champ_file = tmp_path / "triage_model.joblib"
    out_dir = tmp_path / "models"
    reports_dir = tmp_path / "reports"

    from supportpilot.triage.model import train_model, TriageClassifier

    train_model("data/triage_labeled.jsonl", save_path=champ_file, version="1.0.0")
    loaded = TriageClassifier.load(champ_file)
    actual_version = loaded.version
    assert actual_version is not None

    # Mismatch must fail fast
    with pytest.raises(
        ValueError, match="Champion version mismatch: expected 'triage-v1.0.0-20261023'"
    ):
        retrain_pipeline(
            base_dataset="data/triage_labeled.jsonl",
            output_dir=out_dir,
            eval_dataset="data/eval_frozen.jsonl",
            champion_path=champ_file,
            reports_dir=reports_dir,
            expected_champion_version="triage-v1.0.0-20261023",
        )

    # Match must succeed
    results = retrain_pipeline(
        base_dataset="data/triage_labeled.jsonl",
        output_dir=out_dir,
        eval_dataset="data/eval_frozen.jsonl",
        champion_path=champ_file,
        reports_dir=reports_dir,
        expected_champion_version=actual_version,
    )
    assert results["champion_version"] == actual_version
