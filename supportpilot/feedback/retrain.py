"""Retraining utilities for joining feedback exports with ticket datasets (SP-110 Part D)."""

from __future__ import annotations

import json
import os
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path


@dataclass
class RetrainJoinReport:
    """Summary report of retraining join operation."""

    matched: int = 0
    appended: int = 0
    unmatched: int = 0

    @property
    def total_feedback(self) -> int:
        return self.matched + self.appended + self.unmatched

    @property
    def updated(self) -> int:
        return self.matched + self.appended

    def __getitem__(self, item: str) -> int:
        """Support dict-style subscripting for backwards compatibility."""
        if hasattr(self, item):
            val = getattr(self, item)
            if isinstance(val, int):
                return val
        raise KeyError(item)


def join_feedback_for_retraining(
    original_dataset: Path | str,
    feedback_export: Path | str,
    redacted_tickets: Path | str,
    output_dataset: Path | str,
) -> RetrainJoinReport:
    """Join feedback_export.jsonl with original dataset and redacted pilot tickets on ticket_id.

    Args:
        original_dataset: Path to original labeled tickets JSONL (e.g. data/triage_labeled.jsonl).
        feedback_export: Path to feedback export JSONL ({ticket_id, correct_label}).
        redacted_tickets: Path to redacted pilot tickets JSONL (e.g. out_redacted/redacted.jsonl).
            Must be explicitly provided (no hidden magic defaults).
        output_dataset: Destination path for the updated dataset JSONL.

    Returns:
        RetrainJoinReport with matched, appended, and unmatched counts.

    Raises:
        FileNotFoundError: If feedback_export, original_dataset, or redacted_tickets does not exist.
    """
    fb_path = Path(feedback_export)
    if not fb_path.exists():
        raise FileNotFoundError(f"Feedback export file not found: {fb_path}")

    orig_path = Path(original_dataset)
    if not orig_path.exists():
        raise FileNotFoundError(f"Original dataset file not found: {orig_path}")

    red_path = Path(redacted_tickets)
    if not red_path.exists():
        raise FileNotFoundError(f"Redacted pilot tickets file not found: {red_path}")

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
            with open(red_path, "r", encoding="utf-8") as rf:
                for line in rf:
                    line_str = line.strip()
                    if not line_str:
                        continue
                    pilot_ticket = json.loads(line_str)
                    tid = pilot_ticket.get("ticket_id")
                    if tid and tid in remaining_feedback:
                        correct_label = remaining_feedback.pop(tid)
                        subject = pilot_ticket.get("subject")
                        raw_body = pilot_ticket.get("body", "")
                        # /triage classifies "subject body" if subject present;
                        # train on the exact same text we predict on.
                        trained_body = (
                            f"{subject} {raw_body}".strip() if subject else raw_body
                        )
                        new_row = {
                            "ticket_id": tid,
                            "body": trained_body,
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
