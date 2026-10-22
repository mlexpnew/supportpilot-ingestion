"""Retraining utilities for joining feedback exports with ticket datasets (SP-110 Part D)."""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path


class RetrainJoinReport(dict[str, int]):
    """Join report dictionary containing matched, appended, and unmatched counts."""

    def __init__(self, matched: int, appended: int, unmatched: int) -> None:
        total = matched + appended + unmatched
        super().__init__(
            matched=matched,
            appended=appended,
            unmatched=unmatched,
            updated=matched + appended,
            total_feedback=total,
        )

    @property
    def matched(self) -> int:
        return self["matched"]

    @property
    def appended(self) -> int:
        return self["appended"]

    @property
    def unmatched(self) -> int:
        return self["unmatched"]

    @property
    def updated(self) -> int:
        return self["updated"]

    @property
    def total_feedback(self) -> int:
        return self["total_feedback"]


def join_feedback_for_retraining(
    original_dataset: Path | str,
    feedback_export: Path | str,
    output_dataset: Path | str,
    redacted_tickets: Path | str | None = None,
) -> RetrainJoinReport:
    """Join feedback_export.jsonl with original and redacted pilot tickets on ticket_id.

    Args:
        original_dataset: Path to original labeled tickets JSONL (e.g. data/triage_labeled.jsonl).
        feedback_export: Path to feedback export JSONL ({ticket_id, correct_label}).
        output_dataset: Destination path for the updated dataset JSONL.
        redacted_tickets: Optional path to redacted pilot tickets JSONL (e.g. out_redacted/redacted.jsonl).
            If None and out_redacted/redacted.jsonl exists, defaults to that path.

    Returns:
        RetrainJoinReport with matched, appended, and unmatched counts.

    Raises:
        FileNotFoundError: If feedback_export or original_dataset does not exist,
            or if an explicitly specified redacted_tickets path does not exist.
    """
    fb_path = Path(feedback_export)
    if not fb_path.exists():
        raise FileNotFoundError(f"Feedback export file not found: {fb_path}")

    orig_path = Path(original_dataset)
    if not orig_path.exists():
        raise FileNotFoundError(f"Original dataset file not found: {orig_path}")

    # Resolve redacted pilot tickets path
    red_path: Path | None = None
    if redacted_tickets is not None:
        red_path = Path(redacted_tickets)
        if not red_path.exists():
            raise FileNotFoundError(
                f"Redacted pilot tickets file not found: {red_path}"
            )
    else:
        candidate = Path("out_redacted/redacted.jsonl")
        if candidate.is_file():
            red_path = candidate

    # 1. Load feedback corrections into mapping and tracker
    corrections: dict[str, str] = {}
    with open(fb_path, "r", encoding="utf-8") as f:
        for line in f:
            line_str = line.strip()
            if line_str:
                item = json.loads(line_str)
                corrections[item["ticket_id"]] = item["correct_label"]

    remaining_feedback = dict(corrections)
    matched = 0
    appended = 0

    out_path = Path(output_dataset)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    tmp_fd, tmp_file = tempfile.mkstemp(
        prefix=".retrain_join_", suffix=".tmp", dir=str(out_path.parent)
    )
    try:
        with (
            open(orig_path, "r", encoding="utf-8") as src,
            open(tmp_fd, "w", encoding="utf-8") as dst,
        ):
            # 2. Relabel tickets existing in original_dataset
            for line in src:
                line_str = line.strip()
                if not line_str:
                    continue
                ticket = json.loads(line_str)
                tid = ticket.get("ticket_id")
                if tid and tid in remaining_feedback:
                    ticket["label"] = remaining_feedback.pop(tid)
                    matched += 1
                dst.write(json.dumps(ticket, ensure_ascii=False) + "\n")

            # 3. Append pilot tickets from redacted_tickets that received feedback
            if red_path is not None and red_path.exists():
                with open(red_path, "r", encoding="utf-8") as rf:
                    for line in rf:
                        line_str = line.strip()
                        if not line_str:
                            continue
                        pilot_ticket = json.loads(line_str)
                        tid = pilot_ticket.get("ticket_id")
                        if tid and tid in remaining_feedback:
                            correct_label = remaining_feedback.pop(tid)
                            new_row = {
                                "ticket_id": tid,
                                "body": pilot_ticket.get("body", ""),
                                "label": correct_label,
                            }
                            dst.write(json.dumps(new_row, ensure_ascii=False) + "\n")
                            appended += 1

        os.replace(tmp_file, out_path)
    except Exception:
        if os.path.exists(tmp_file):
            os.unlink(tmp_file)
        raise

    # 4. Count unmatched feedback (feedback rows with no ticket text in either source)
    unmatched = len(remaining_feedback)

    # 5. Print summary report to stderr so operators know what was matched/lost
    print(
        f"Retraining join complete: {matched} matched, {appended} appended, "
        f"{unmatched} unmatched (no ticket text).",
        file=sys.stderr,
    )
    if unmatched > 0:
        print(
            f"Warning: {unmatched} feedback record(s) had no matching ticket text and were omitted.",
            file=sys.stderr,
        )

    return RetrainJoinReport(matched=matched, appended=appended, unmatched=unmatched)
