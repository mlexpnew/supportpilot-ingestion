"""Chart generation for SupportPilot weekly pilot report (SP-114 Part C).

Generates:
1. Daily ticket triage volume bar chart.
2. Per-label recall bar chart (with nulls explicitly labeled 'No data').

Zero-PII guarantee: Only aggregates, dates, and category names are visualized.
"""

from __future__ import annotations

from datetime import datetime, timedelta
import logging
from pathlib import Path

import matplotlib

# Headless backend to prevent GUI display dependencies
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from supportpilot.feedback.models import DateRangeStats
from supportpilot.triage import LABELS

logger = logging.getLogger(__name__)


def generate_daily_volume_chart(
    stats: DateRangeStats,
    output_path: Path | str,
) -> Path:
    """Generate tickets per day bar chart (SP-114 Part C Chart 1)."""
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    # Determine date range for x-axis
    dates: list[str] = []
    counts: list[int] = []

    try:
        s_date = datetime.strptime(stats.start[:10], "%Y-%m-%d").date()
        e_date = datetime.strptime(stats.end[:10], "%Y-%m-%d").date()
        if 0 <= (e_date - s_date).days <= 31:
            curr = s_date
            while curr <= e_date:
                d_str = curr.isoformat()
                dates.append(d_str)
                counts.append(stats.daily_triage_counts.get(d_str, 0))
                curr += timedelta(days=1)
        else:
            dates = sorted(stats.daily_triage_counts.keys())
            counts = [stats.daily_triage_counts[d] for d in dates]
    except Exception:
        dates = sorted(stats.daily_triage_counts.keys())
        counts = [stats.daily_triage_counts[d] for d in dates]

    if not dates:
        dates = [stats.start[:10], stats.end[:10]]
        counts = [0, 0]

    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=150)

    # Format x labels: short date (MM-DD or Day)
    x_indices = list(range(len(dates)))
    ax.bar(x_indices, counts, color="#2563eb", width=0.6, edgecolor="#1d4ed8")

    # Add count labels above bars
    max_count = max(counts) if counts else 10
    for idx, count in enumerate(counts):
        ax.text(
            idx,
            count + (max_count * 0.02 + 0.2),
            str(count),
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
            color="#1e293b",
        )

    # Styling
    ax.set_title(
        f"Daily Ticket Volume (UTC) ({stats.start} to {stats.end})",
        fontsize=13,
        fontweight="bold",
        pad=15,
        color="#0f172a",
    )
    ax.set_xlabel("Date (UTC)", fontsize=10, labelpad=8, color="#334155")
    ax.set_ylabel("Tickets Triaged", fontsize=10, labelpad=8, color="#334155")
    ax.set_xticks(x_indices)
    ax.set_xticklabels(dates, rotation=35, ha="right", fontsize=9)
    ax.set_ylim(0, max(max_count * 1.18, 5))

    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    fig.savefig(out_file, dpi=150)
    plt.close(fig)

    logger.info("Daily volume chart written to %s", out_file)
    return out_file


def generate_label_recall_chart(
    stats: DateRangeStats,
    output_path: Path | str,
) -> Path:
    """Generate recall per label bar chart (SP-114 Part C Chart 2).

    Null recalls are explicitly displayed as 'No data'.
    """
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    labels = list(LABELS)
    x_indices = list(range(len(labels)))
    values: list[float] = []
    colors: list[str] = []
    text_labels: list[str] = []

    for lbl in labels:
        rec = stats.per_label_recall.get(lbl)
        info = stats.per_label_feedback.get(lbl, {})
        tot = info.get("total", 0) if isinstance(info, dict) else 0

        if tot < 5 or rec is None:
            values.append(0.0)
            colors.append("#cbd5e1")  # Light slate for missing or small sample data
            text_labels.append("n<5" if tot > 0 else "No data")
        else:
            pct = rec * 100.0
            values.append(pct)
            if pct >= 80.0:
                colors.append("#10b981")  # Green (meets 80% target)
            elif pct >= 75.0:
                colors.append("#f59e0b")  # Amber (75-80%)
            else:
                colors.append("#ef4444")  # Red (below 75% alert floor)
            text_labels.append(f"{pct:.1f}%")

    fig, ax = plt.subplots(figsize=(8.5, 4.5), dpi=150)

    ax.bar(
        x_indices,
        values,
        color=colors,
        width=0.55,
        edgecolor="#64748b",
        linewidth=0.8,
    )

    # Reference threshold lines
    ax.axhline(
        80.0,
        color="#059669",
        linestyle="--",
        linewidth=1.2,
        alpha=0.8,
        label="Target (80%)",
    )
    ax.axhline(
        75.0,
        color="#dc2626",
        linestyle=":",
        linewidth=1.2,
        alpha=0.8,
        label="Alert Floor (75%)",
    )

    # Text annotations on each bar
    for idx, (val, txt) in enumerate(zip(values, text_labels)):
        y_pos = max(val + 2.0, 4.0)
        ax.text(
            idx,
            y_pos,
            txt,
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
            color="#334155" if txt in ("No data", "n<5") else "#0f172a",
        )

    # Styling
    ax.set_title(
        f"Per-Category Ground-Truth Recall ({stats.start} to {stats.end})",
        fontsize=13,
        fontweight="bold",
        pad=15,
        color="#0f172a",
    )
    ax.set_xlabel("Ticket Category", fontsize=10, labelpad=8, color="#334155")
    ax.set_ylabel("Recall (%)", fontsize=10, labelpad=8, color="#334155")
    ax.set_xticks(x_indices)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylim(0, 115)

    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="upper right", framealpha=0.9, fontsize=9)

    plt.tight_layout()
    fig.savefig(out_file, dpi=150)
    plt.close(fig)

    logger.info("Recall per label chart written to %s", out_file)
    return out_file


def generate_charts(
    stats: DateRangeStats,
    output_dir: Path | str,
    prefix: str = "",
) -> dict[str, str]:
    """Generate both weekly report charts as PNG files in output_dir.

    Returns:
        dict mapping chart keys ('daily_volume', 'per_label_recall') to their relative filenames.
    """
    out_d = Path(output_dir)
    out_d.mkdir(parents=True, exist_ok=True)

    clean_from = stats.start[:10].replace(":", "-")
    clean_to = stats.end[:10].replace(":", "-")

    vol_name = (
        f"{prefix}daily_volume_{clean_from}_{clean_to}.png"
        if not prefix.endswith(".png")
        else prefix
    )
    rec_name = (
        f"{prefix}recall_per_label_{clean_from}_{clean_to}.png"
        if not prefix.endswith(".png")
        else prefix
    )

    generate_daily_volume_chart(stats, out_d / vol_name)
    generate_label_recall_chart(stats, out_d / rec_name)

    return {
        "daily_volume": vol_name,
        "per_label_recall": rec_name,
    }
