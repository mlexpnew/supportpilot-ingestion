"""Unit and integration tests for SupportPilot weekly pilot report generator (SP-114 Part B)."""

from __future__ import annotations

from pathlib import Path
import pytest

from supportpilot.feedback.models import DateRangeStats
from supportpilot.feedback.store import FeedbackStore, init_db
from supportpilot.reports.weekly import (
    generate_weekly_report,
    main,
    render_markdown,
)
from supportpilot.triage import LABELS


def test_render_markdown_thin_data_honest_wording() -> None:
    """Verify that when feedback < 20 or triage < 20, report displays 'not enough data to judge'."""
    stats = DateRangeStats(
        start="2026-10-26",
        end="2026-11-01",
        total_triaged=5,
        predictions_by_label={
            lbl: (1 if lbl in ("delivery", "billing") else 0) for lbl in LABELS
        },
        needs_review_count=1,
        needs_review_rate=0.20,
        feedback_total=3,
        feedback_agreed=3,
        live_accuracy=1.0,  # 3 of 3 is 100%, but must NOT look final!
        per_label_recall={lbl: 1.0 if lbl == "delivery" else None for lbl in LABELS},
        per_label_feedback={
            lbl: (
                {"total": 3, "agreed": 3, "recall": 1.0}
                if lbl == "delivery"
                else {"total": 0, "agreed": 0, "recall": None}
            )
            for lbl in LABELS
        },
        model_versions=["triage-v1.0.0-20261022"],
        daily_triage_counts={"2026-10-26": 2, "2026-10-27": 3},
    )

    md = render_markdown(stats)

    # 1. Honest wording required
    assert "not enough data to judge" in md.lower()

    # 2. Never show 100.0% as final accuracy
    assert "Live accuracy is **100.0%**" not in md
    assert "Accuracy Verdict**: Not enough data to judge" in md

    # 3. Volume below 20 has not enough data
    assert "Operational Volume**: Not enough data to judge" in md

    # 4. Required sections exist
    assert "## Executive Summary" in md
    assert "## 1. Ticket Volume & Routing" in md
    assert "## 2. Accuracy & Feedback Analysis" in md
    assert "## 3. Where the Model Struggles (Weak Spots)" in md
    assert "## 4. Recommendations" in md

    # 5. Model version included
    assert "triage-v1.0.0-20261022" in md


def test_render_markdown_sufficient_data_shows_exact_metrics() -> None:
    """Verify that with >= 20 feedback and >= 20 triage, exact metrics and weak spots are displayed."""
    stats = DateRangeStats(
        start="2026-10-26",
        end="2026-11-01",
        total_triaged=100,
        predictions_by_label={
            "delivery": 30,
            "billing": 25,
            "refund": 20,
            "cancellation": 10,
            "account": 10,
            "other": 5,
        },
        needs_review_count=15,
        needs_review_rate=0.15,
        feedback_total=40,
        feedback_agreed=34,
        live_accuracy=0.85,
        per_label_recall={
            "delivery": 0.90,
            "billing": 0.88,
            "refund": 0.80,
            "cancellation": 0.90,
            "account": 0.85,
            "other": 0.60,  # Weak spot!
        },
        per_label_feedback={
            "delivery": {"total": 10, "agreed": 9, "recall": 0.90},
            "billing": {"total": 8, "agreed": 7, "recall": 0.875},
            "refund": {"total": 10, "agreed": 8, "recall": 0.80},
            "cancellation": {"total": 5, "agreed": 5, "recall": 1.0},
            "account": {"total": 2, "agreed": 2, "recall": 1.0},
            "other": {"total": 5, "agreed": 3, "recall": 0.60},
        },
        model_versions=["triage-v1.0.0-20261022", "triage-v1.1.0-20261027"],
        daily_triage_counts={"2026-10-26": 20, "2026-10-27": 25},
    )

    md = render_markdown(stats)

    # 1. Exact numbers rendered
    assert "85.0%" in md
    assert "100" in md
    assert "15.0%" in md

    # 2. Executive summary bullets
    assert "Accuracy Verdict**: Working well — Live accuracy is **85.0%**" in md
    assert "Operational Volume**: **100** tickets automatically triaged" in md

    # 3. Identifies struggling category (other at 60.0%)
    assert "`other`" in md
    assert "60.0%" in md

    # 4. Both model versions listed
    assert "`triage-v1.0.0-20261022`" in md
    assert "`triage-v1.1.0-20261027`" in md

    # 5. Fix 2: Per-label sample sizes and threshold
    # account has n=2 (< 5) -> "Not enough data (n=2)"
    assert "Not enough data (n=2)" in md
    # delivery has n=10 (>= 5) -> "0.90 (n=10)"
    assert "0.90 (n=10)" in md
    # other has n=5 (>= 5) -> "0.60 (n=5)"
    assert "0.60 (n=5)" in md

    # 6. Time zone UTC notice
    assert "Days are in UTC; IST is UTC+5:30." in md
    assert "Date (UTC)" in md


def test_render_markdown_tracked_review_rate_notice() -> None:
    """Verify that when tracked < total, the report explicitly notes partial tracking."""
    stats = DateRangeStats(
        start="2026-10-26",
        end="2026-11-01",
        total_triaged=100,
        predictions_by_label={"delivery": 100},
        needs_review_count=18,
        needs_review_rate=0.36,  # 18 / 50 = 36%
        feedback_total=30,
        feedback_agreed=27,
        live_accuracy=0.90,
        per_label_recall={"delivery": 0.90},
        per_label_feedback={"delivery": {"total": 30, "agreed": 27, "recall": 0.90}},
        model_versions=["v1"],
        daily_triage_counts={"2026-10-26": 50, "2026-10-29": 50},
        needs_review_tracked=50,
        first_tracked_date="2026-10-29",
    )

    md = render_markdown(stats)
    assert "Review rate based on 50 of 100 tickets; tracking started 2026-10-29." in md

    # When tracked == total, the notice should not appear
    stats_full = DateRangeStats(
        start="2026-10-26",
        end="2026-11-01",
        total_triaged=100,
        predictions_by_label={"delivery": 100},
        needs_review_count=36,
        needs_review_rate=0.36,
        feedback_total=30,
        feedback_agreed=27,
        live_accuracy=0.90,
        per_label_recall={"delivery": 0.90},
        per_label_feedback={"delivery": {"total": 30, "agreed": 27, "recall": 0.90}},
        model_versions=["v1"],
        daily_triage_counts={"2026-10-26": 100},
        needs_review_tracked=100,
        first_tracked_date="2026-10-26",
    )
    md_full = render_markdown(stats_full)
    assert "Review rate based on" not in md_full


def test_render_markdown_label_recall_sample_sizes() -> None:
    """Verify that a label with n=1 shows 'Not enough data' and n=5 shows the number."""
    stats = DateRangeStats(
        start="2026-10-26",
        end="2026-11-01",
        total_triaged=50,
        predictions_by_label={
            lbl: 10 if lbl in ("delivery", "refund") else 0 for lbl in LABELS
        },
        needs_review_count=5,
        needs_review_rate=0.10,
        feedback_total=25,
        feedback_agreed=20,
        live_accuracy=0.80,
        per_label_recall={
            "delivery": 0.80,
            "refund": 0.0,
            "billing": None,
            "cancellation": None,
            "account": None,
            "other": None,
        },
        per_label_feedback={
            "delivery": {"total": 24, "agreed": 20, "recall": 0.8333},
            "refund": {"total": 1, "agreed": 0, "recall": 0.0},
            "billing": {"total": 0, "agreed": 0, "recall": None},
            "cancellation": {"total": 0, "agreed": 0, "recall": None},
            "account": {"total": 0, "agreed": 0, "recall": None},
            "other": {"total": 0, "agreed": 0, "recall": None},
        },
        model_versions=["v1"],
        daily_triage_counts={"2026-10-26": 50},
        needs_review_tracked=50,
        first_tracked_date="2026-10-26",
    )

    md = render_markdown(stats)

    # refund has n=1 (< 5) -> "Not enough data (n=1)", NOT "0.0%"
    assert "Not enough data (n=1)" in md
    assert "0.0% (n=1)" not in md

    # refund with n=1 must NOT be listed in "Where the model struggles"
    assert "The model struggles most with `refund`" not in md
    assert "Lowest recall observed in `refund`" not in md


def test_generate_weekly_report_writes_file_and_cli(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verify generate_weekly_report writes the report file and CLI generates reports/pilot_report_<from>_<to>.md."""
    db_file = tmp_path / "weekly_test.db"
    init_db(db_file)
    store = FeedbackStore(db_file)

    # Seed 25 tickets (meets 20 threshold)
    for i in range(25):
        store.save_prediction(
            ticket_id=f"T-WEEKLY-{i}",
            predicted_label="delivery",
            confidence=0.90,
            model_version="triage-v1.0.0-20261022",
            created_at="2026-10-27T10:00:00+00:00",
            needs_review=(i < 3),
        )

    # Seed 22 feedback entries (meets 20 threshold)
    for i in range(22):
        store.save_feedback(
            ticket_id=f"T-WEEKLY-{i}",
            correct_label="delivery",
            agent_id="AG-101",
            created_at="2026-10-28T10:00:00+00:00",
        )

    out_dir = tmp_path / "reports"
    report_file, content = generate_weekly_report(
        from_date="2026-10-26",
        to_date="2026-11-01",
        output_path=out_dir,
        store=store,
    )

    assert report_file.exists()
    assert report_file.name == "pilot_report_2026-10-26_2026-11-01.md"
    assert report_file.read_text(encoding="utf-8") == content
    assert "100.0%" in content
    assert "**25** tickets automatically triaged" in content

    # Verify PNG charts were generated next to the report
    vol_png = out_dir / "daily_volume_2026-10-26_2026-11-01.png"
    rec_png = out_dir / "recall_per_label_2026-10-26_2026-11-01.png"
    assert vol_png.exists() and vol_png.stat().st_size > 0
    assert rec_png.exists() and rec_png.stat().st_size > 0

    # Verify PNG header
    assert vol_png.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    assert rec_png.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")

    # Verify Markdown links to charts
    assert f"![Tickets Triaged Per Day]({vol_png.name})" in content
    assert f"![Recall Per Label]({rec_png.name})" in content

    # Test CLI execution
    import sys

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "weekly.py",
            "--from",
            "2026-10-26",
            "--to",
            "2026-11-01",
            "--out",
            str(out_dir),
        ],
    )
    # Monkeypatch store in weekly module
    monkeypatch.setattr("supportpilot.reports.weekly.get_feedback_store", lambda: store)

    code = main()
    assert code == 0

    store.close()


def test_zero_pii_in_rendered_markdown() -> None:
    """Verify that no ticket text, customer email, credit card, or ticket ID leaks into the report."""
    stats = DateRangeStats(
        start="2026-10-26",
        end="2026-11-01",
        total_triaged=50,
        predictions_by_label={lbl: 10 if lbl != "other" else 0 for lbl in LABELS},
        needs_review_count=5,
        needs_review_rate=0.10,
        feedback_total=25,
        feedback_agreed=20,
        live_accuracy=0.80,
        per_label_recall={lbl: 0.80 for lbl in LABELS},
        per_label_feedback={
            lbl: {"total": 5, "agreed": 4, "recall": 0.80} for lbl in LABELS
        },
        model_versions=["triage-v1.0.0-20261022"],
        daily_triage_counts={"2026-10-26": 25, "2026-10-27": 25},
    )

    md = render_markdown(stats)

    # Check for raw PII markers or common sensitive data
    assert "@" not in md  # No email addresses
    assert "4111" not in md  # No card numbers
    assert "T-" not in md  # No ticket IDs
    assert "body" not in md.lower()
    assert "customer" not in md.lower() or "customer phrasing" in md.lower()


def test_generate_charts_output(tmp_path: Path) -> None:
    """Verify generate_charts produces both daily_volume and recall_per_label PNGs with null as 'No data'."""
    from supportpilot.reports.charts import (
        generate_charts,
        generate_daily_volume_chart,
        generate_label_recall_chart,
    )

    stats = DateRangeStats(
        start="2026-10-26",
        end="2026-11-01",
        total_triaged=10,
        predictions_by_label={
            lbl: 2 if lbl in ("delivery", "billing") else 0 for lbl in LABELS
        },
        needs_review_count=2,
        needs_review_rate=0.2,
        feedback_total=6,
        feedback_agreed=5,
        live_accuracy=0.8333,
        per_label_recall={
            "delivery": 1.0,
            "billing": 0.5,
            "refund": None,  # Null -> 'No data'
            "cancellation": None,
            "account": None,
            "other": None,
        },
        per_label_feedback={
            "delivery": {"total": 4, "agreed": 4, "recall": 1.0},
            "billing": {"total": 2, "agreed": 1, "recall": 0.5},
            "refund": {"total": 0, "agreed": 0, "recall": None},
            "cancellation": {"total": 0, "agreed": 0, "recall": None},
            "account": {"total": 0, "agreed": 0, "recall": None},
            "other": {"total": 0, "agreed": 0, "recall": None},
        },
        model_versions=["triage-v1.0.0-20261022"],
        daily_triage_counts={"2026-10-26": 5, "2026-10-27": 5},
    )

    # 1. Direct chart generation
    vol_file = generate_daily_volume_chart(stats, tmp_path / "custom_vol.png")
    assert vol_file.exists() and vol_file.stat().st_size > 0
    assert vol_file.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")

    rec_file = generate_label_recall_chart(stats, tmp_path / "custom_rec.png")
    assert rec_file.exists() and rec_file.stat().st_size > 0
    assert rec_file.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")

    # 2. Bundle generation via generate_charts
    chart_dict = generate_charts(stats, output_dir=tmp_path / "bundled")
    assert "daily_volume" in chart_dict
    assert "per_label_recall" in chart_dict
    assert (tmp_path / "bundled" / chart_dict["daily_volume"]).exists()
    assert (tmp_path / "bundled" / chart_dict["per_label_recall"]).exists()
