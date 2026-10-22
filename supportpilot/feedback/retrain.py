"""Retraining utilities for joining feedback exports with ticket datasets (SP-110 Part D)."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path


def join_feedback_for_retraining(
    original_dataset: Path | str,
    feedback_export: Path | str,
    output_dataset: Path | str,
) -> int:
    """Join feedback_export.jsonl with original dataset on ticket_id.

    Args:
        original_dataset: Path to original labeled/redacted tickets JSONL.
        feedback_export: Path to feedback export JSONL ({ticket_id, correct_label}).
        output_dataset: Destination path for the updated dataset JSONL.

    Returns:
        Number of tickets whose labels were updated with human feedback.
    """
    fb_path = Path(feedback_export)
    corrections: dict[str, str] = {}
    if fb_path.exists():
        with open(fb_path, "r", encoding="utf-8") as f:
            for line in f:
                line_str = line.strip()
                if line_str:
                    item = json.loads(line_str)
                    corrections[item["ticket_id"]] = item["correct_label"]

    orig_path = Path(original_dataset)
    out_path = Path(output_dataset)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    updated_count = 0
    tmp_fd, tmp_file = tempfile.mkstemp(
        prefix=".retrain_join_", suffix=".tmp", dir=str(out_path.parent)
    )
    try:
        with (
            open(orig_path, "r", encoding="utf-8") as src,
            open(tmp_fd, "w", encoding="utf-8") as dst,
        ):
            for line in src:
                line_str = line.strip()
                if not line_str:
                    continue
                ticket = json.loads(line_str)
                tid = ticket.get("ticket_id")
                if tid and tid in corrections:
                    ticket["label"] = corrections[tid]
                    updated_count += 1
                dst.write(json.dumps(ticket, ensure_ascii=False) + "\n")
        os.replace(tmp_file, out_path)
    except Exception:
        if os.path.exists(tmp_file):
            os.unlink(tmp_file)
        raise

    return updated_count
