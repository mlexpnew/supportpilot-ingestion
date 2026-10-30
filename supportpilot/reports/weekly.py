"""Weekly pilot report generator for SupportPilot (SP-114 Part B).

Usage:
    python -m supportpilot.reports.weekly --from 2026-10-26 --to 2026-11-01 --out reports/
"""

from __future__ import annotations

import argparse
from datetime import UTC, datetime, timedelta
import logging
from pathlib import Path

from supportpilot.api.alerts import get_alert_thresholds
from supportpilot.feedback.models import DateRangeStats
from supportpilot.feedback.store import FeedbackStore, get_feedback_store
from supportpilot.triage import LABELS

logger = logging.getLogger(__name__)

MIN_LABEL_SAMPLES: int = 5


def render_markdown(
    stats: DateRangeStats,
    chart_paths: dict[str, str] | None = None,
) -> str:
    """Render a clean, human-readable weekly pilot report in Markdown.

    Pure function with no I/O, designed for testability (Arjun's rule).
    Executive summary leads with 3 bullets answering 'is it working?' (Meera's rule).
    Honest wording when data is thin (< 20 samples), never overselling (Neha's rule).
    Model versions clearly attributed (Rohan's rule).
    Zero PII (aggregates only).
    """
    charts = chart_paths or {}
    thresholds = get_alert_thresholds()
    min_feedback = thresholds.min_feedback
    min_triage = thresholds.min_triage

    versions_str = (
        ", ".join(f"`{v}`" for v in stats.model_versions)
        if stats.model_versions
        else "None recorded"
    )

    # 1. Summary: 3 bullets answering "is it working?"
    # Bullet 1: Accuracy verdict
    if stats.feedback_total < min_feedback:
        bullet_accuracy = (
            f"**Accuracy Verdict**: Not enough data to judge (only {stats.feedback_total} "
            f"feedback correction{'s' if stats.feedback_total != 1 else ''} collected; "
            f"minimum {min_feedback} required to judge accuracy)."
        )
    else:
        acc_pct = (
            f"{stats.live_accuracy:.1%}" if stats.live_accuracy is not None else "N/A"
        )
        health_str = (
            "Working well" if (stats.live_accuracy or 0) >= 0.80 else "Needs attention"
        )
        bullet_accuracy = (
            f"**Accuracy Verdict**: {health_str} — Live accuracy is **{acc_pct}** "
            f"({stats.feedback_agreed}/{stats.feedback_total} verified corrections) "
            f"against the 80.0% production target."
        )

    # Bullet 2: Volume & Automation
    if stats.total_triaged < min_triage:
        bullet_volume = (
            f"**Operational Volume**: Not enough data to judge (only {stats.total_triaged} "
            f"ticket{'s' if stats.total_triaged != 1 else ''} triaged; "
            f"minimum {min_triage} required for operational assessment)."
        )
    else:
        bullet_volume = (
            f"**Operational Volume**: **{stats.total_triaged}** tickets automatically triaged "
            f"across {versions_str}; **{stats.needs_review_count}** flagged for manual human review "
            f"({stats.needs_review_rate:.1%} review rate)."
        )

    # Bullet 3: Weak spots & readiness
    # Identify lowest recall label among those with at least MIN_LABEL_SAMPLES feedback
    evaluated_labels = [
        (
            lbl,
            stats.per_label_feedback[lbl]["recall"],
            stats.per_label_feedback[lbl]["total"],
        )
        for lbl in LABELS
        if stats.per_label_feedback.get(lbl, {}).get("total", 0) >= MIN_LABEL_SAMPLES
    ]

    if stats.feedback_total < min_feedback:
        bullet_weakspots = (
            "**Model Health & Focus**: Feedback volume is too thin to confirm category regressions; "
            "continue gathering verified agent corrections during pilot operations before drawing conclusions."
        )
    elif evaluated_labels:
        # Sort by recall ascending, then total descending
        evaluated_labels.sort(key=lambda x: (x[1] if x[1] is not None else -1.0, -x[2]))
        weakest_label, weakest_rec, weakest_tot = evaluated_labels[0]
        rec_str = f"{weakest_rec:.1%}" if weakest_rec is not None else "0.0%"
        bullet_weakspots = (
            f"**Model Health & Focus**: Lowest recall observed in `{weakest_label}` "
            f"at **{rec_str}** ({weakest_tot} sample{'s' if weakest_tot != 1 else ''}); "
            f"priority candidate for targeted review and retraining."
        )
    else:
        bullet_weakspots = (
            f"**Model Health & Focus**: No category has reached the minimum sample size (n >= {MIN_LABEL_SAMPLES}) "
            "to identify specific weak spots; continue gathering verified feedback across categories."
        )

    lines: list[str] = [
        f"# SupportPilot Pilot Report: {stats.start} to {stats.end}",
        "",
        f"- **Reporting Period**: {stats.start} to {stats.end} (Days are in UTC; IST is UTC+5:30)",
        f"- **Model Version(s)**: {versions_str}",
        "",
        "---",
        "",
        "## Executive Summary",
        "",
        f"- {bullet_accuracy}",
        f"- {bullet_volume}",
        f"- {bullet_weakspots}",
        "",
        "---",
        "",
        "## 1. Ticket Volume & Routing",
        "",
        f"- **Total Tickets Triaged**: {stats.total_triaged}",
        f"- **Active Model Version(s)**: {versions_str}",
    ]

    if stats.total_triaged < min_triage:
        lines.extend(
            [
                f"- **Human Review Required (`needs_review`)**: {stats.needs_review_count} tickets (Not enough data to judge review rate; minimum {min_triage} tickets required)",
                "",
                "> [!NOTE]",
                f"> **Low Sample Size**: Only {stats.total_triaged} tickets triaged during this window (below the {min_triage}-ticket threshold). "
                "Traffic metrics represent early pilot activity rather than steady-state distribution.",
                "",
            ]
        )
    else:
        lines.extend(
            [
                f"- **Human Review Required (`needs_review`)**: {stats.needs_review_count} tickets ({stats.needs_review_rate:.1%})",
                "",
            ]
        )

    if stats.needs_review_tracked < stats.total_triaged:
        first_date_str = stats.first_tracked_date if stats.first_tracked_date else "N/A"
        lines.extend(
            [
                f"Review rate based on {stats.needs_review_tracked} of {stats.total_triaged} tickets; tracking started {first_date_str}.",
                "",
            ]
        )

    lines.extend(
        [
            "### Volume by Category",
            "",
            "| Category | Tickets Triaged | Share |",
            "| :--- | :--- | :--- |",
        ]
    )

    for lbl in LABELS:
        cnt = stats.predictions_by_label.get(lbl, 0)
        share = (
            f"{(cnt / stats.total_triaged):.1%}" if stats.total_triaged > 0 else "0.0%"
        )
        lines.append(f"| `{lbl}` | {cnt} | {share} |")

    if stats.daily_triage_counts:
        lines.extend(
            [
                "",
                "### Daily Triage Volume",
                "",
                "Days are in UTC; IST is UTC+5:30.",
                "",
                "| Date (UTC) | Tickets Triaged |",
                "| :--- | :--- |",
            ]
        )
        for day, cnt in stats.daily_triage_counts.items():
            lines.append(f"| {day} | {cnt} |")

    if "daily_volume" in charts:
        lines.extend(
            [
                "",
                f"![Tickets Triaged Per Day]({charts['daily_volume']})",
            ]
        )

    lines.extend(
        [
            "",
            "---",
            "",
            "## 2. Accuracy & Feedback Analysis",
            "",
            f"- **Total Human Feedback Records**: {stats.feedback_total}",
            f"- **Agent Agreed**: {stats.feedback_agreed}",
        ]
    )

    if stats.feedback_total < min_feedback:
        lines.extend(
            [
                f"- **Live Accuracy**: Not enough data to judge (only {stats.feedback_total} feedback corrections; minimum {min_feedback} required)",
                "",
                "> [!WARNING]",
                f"> **Not Enough Data to Judge Accuracy**: Only {stats.feedback_total} feedback corrections recorded in this period. "
                f"A minimum of {min_feedback} human corrections is required before accuracy metrics are statistically reliable. "
                "Presenting preliminary percentages would be misleading.",
                "",
            ]
        )
    else:
        acc_str = (
            f"{stats.live_accuracy:.1%}" if stats.live_accuracy is not None else "N/A"
        )
        lines.extend(
            [
                f"- **Live Accuracy**: **{acc_str}**",
                "",
                f"Live accuracy is **{acc_str}** based on {stats.feedback_total} agent-verified corrections.",
                "",
            ]
        )

    lines.extend(
        [
            "---",
            "",
            "## 3. Where the Model Struggles (Weak Spots)",
            "",
            "| Category | Ground-Truth Samples | Correctly Predicted | Recall | Assessment |",
            "| :--- | :--- | :--- | :--- | :--- |",
        ]
    )

    for lbl in LABELS:
        info = stats.per_label_feedback.get(
            lbl, {"total": 0, "agreed": 0, "recall": None}
        )
        tot = info["total"]
        agr = info["agreed"]
        rec = info["recall"]

        if tot < MIN_LABEL_SAMPLES:
            rec_display = f"Not enough data (n={tot})"
            assessment = f"Thin data (n={tot})"
        else:
            rec_display = (
                f"{rec:.2f} (n={tot})" if rec is not None else f"0.00 (n={tot})"
            )
            assessment = "Pass (>= 75%)" if (rec or 0) >= 0.75 else "Degraded (< 75%)"

        lines.append(f"| `{lbl}` | {tot} | {agr} | {rec_display} | {assessment} |")

    lines.append("")

    if stats.feedback_total < min_feedback:
        lines.extend(
            [
                "> [!NOTE]",
                f"> Category-level recall numbers are withheld because overall feedback volume ({stats.feedback_total}) "
                f"is below the statistical minimum of {min_feedback}. Individual misclassifications at this stage "
                "represent anecdotal samples rather than systematic failure modes.",
                "",
            ]
        )
    elif evaluated_labels:
        weakest_label, weakest_rec, weakest_tot = evaluated_labels[0]
        rec_str = f"{weakest_rec:.1%}" if weakest_rec is not None else "0.0%"
        lines.extend(
            [
                f"The model struggles most with `{weakest_label}` where recall is **{rec_str}** "
                f"across {weakest_tot} verified tickets. Tickets in this category often contain cross-boundary overlap "
                "or ambiguous customer phrasing.",
                "",
            ]
        )
    else:
        lines.extend(
            [
                "> [!NOTE]",
                f"> No category currently has >= {MIN_LABEL_SAMPLES} ground-truth samples to reliably identify systematic weak spots.",
                "",
            ]
        )

    if "per_label_recall" in charts:
        lines.extend(
            [
                f"![Recall Per Label]({charts['per_label_recall']})",
                "",
            ]
        )

    lines.extend(
        [
            "---",
            "",
            "## 4. Recommendations",
            "",
        ]
    )

    if stats.feedback_total < min_feedback:
        lines.append(
            f"1. **Accelerate Feedback Collection**: We have collected {stats.feedback_total}/{min_feedback} "
            "corrections. Instruct pilot agents to continue reviewing ambiguous tickets so the feedback sample reaches "
            f"the {min_feedback}-sample mark for statistically valid evaluation."
        )
    else:
        if (stats.live_accuracy or 0) < 0.80:
            lines.append(
                f"1. **Trigger Model Retraining (SP-113)**: Live accuracy ({stats.live_accuracy:.1%}) is below the "
                "80.0% alert threshold. Execute the Champion/Challenger retraining pipeline to incorporate agent corrections."
            )
        else:
            lines.append(
                f"1. **Maintain Quality Monitoring**: Live accuracy ({stats.live_accuracy:.1%}) meets client expectations. "
                "Continue weekly tracking to detect category drift."
            )

    if stats.total_triaged >= min_triage and stats.needs_review_rate > 0.25:
        lines.append(
            f"2. **Calibrate Confidence Cutoff**: The human review rate is currently {stats.needs_review_rate:.1%}, "
            "which exceeds the 25.0% threshold. Consider tuning the confidence cutoff once more verified tickets are collected."
        )
    else:
        lines.append(
            "2. **Monitor Human Review Queue**: Ensure triage agents process low-confidence items daily to maintain healthy latency."
        )

    lines.append(
        "3. **Zero-PII Compliance**: Confirm all ingested tickets pass through the SP-102 redaction pipeline before model inference."
    )
    lines.append("")

    return "\n".join(lines)


def generate_weekly_report(
    from_date: str,
    to_date: str,
    output_path: Path | str = "reports",
    store: FeedbackStore | None = None,
    chart_paths: dict[str, str] | None = None,
) -> tuple[Path, str]:
    """Fetch persistent date range statistics, generate charts, and write report file.

    Returns:
        (report_file_path, markdown_content)
    """
    if store is None:
        store = get_feedback_store()

    stats = store.get_stats(from_date, to_date)

    out = Path(output_path)
    if out.suffix == ".md":
        report_file = out
    else:
        out.mkdir(parents=True, exist_ok=True)
        clean_from = from_date.replace(":", "-")
        clean_to = to_date.replace(":", "-")
        report_file = out / f"pilot_report_{clean_from}_{clean_to}.md"

    report_file.parent.mkdir(parents=True, exist_ok=True)

    if chart_paths is None:
        try:
            from .charts import generate_charts

            chart_paths = generate_charts(stats, output_dir=report_file.parent)
        except Exception:
            logger.exception(
                "Failed to generate report charts; proceeding without charts."
            )
            chart_paths = {}

    content = render_markdown(stats, chart_paths=chart_paths)
    report_file.write_text(content, encoding="utf-8")
    logger.info("Weekly report written to %s", report_file)

    return report_file, content


def main() -> int:
    """CLI entrypoint for weekly pilot report generation."""
    now = datetime.now(UTC)
    default_to = now.strftime("%Y-%m-%d")
    default_from = (now - timedelta(days=7)).strftime("%Y-%m-%d")

    parser = argparse.ArgumentParser(
        description="Generate SupportPilot weekly pilot report (SP-114 Part B)."
    )
    parser.add_argument(
        "--from",
        dest="from_date",
        type=str,
        default=default_from,
        help=f"Start date in YYYY-MM-DD format (default: 7 days ago, {default_from})",
    )
    parser.add_argument(
        "--to",
        dest="to_date",
        type=str,
        default=default_to,
        help=f"End date in YYYY-MM-DD format (default: today, {default_to})",
    )
    parser.add_argument(
        "--out",
        dest="out_dir",
        type=Path,
        default=Path("reports"),
        help="Destination directory or filepath for the report markdown (default: reports/)",
    )

    args = parser.parse_args()

    report_file, _ = generate_weekly_report(
        from_date=args.from_date,
        to_date=args.to_date,
        output_path=args.out_dir,
    )

    print(f"Weekly pilot report generated successfully: {report_file}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
