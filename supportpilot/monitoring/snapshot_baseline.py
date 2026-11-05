"""CLI script and utility to snapshot the predicted-label mix from production (SP-120).

Snapshots the predicted-label mix once there are at least 500 real predictions in production,
saving it inside the package at supportpilot/monitoring/drift_baseline.json so it is included in Docker.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from supportpilot.feedback.store import FeedbackStore
from supportpilot.monitoring.quality import DEFAULT_DRIFT_BASELINE_PATH

logger = logging.getLogger(__name__)

DEFAULT_MIN_SNAPSHOT_PREDICTIONS: int = 500


def snapshot_baseline(
    store: FeedbackStore | None = None,
    db_path: Path | str | None = None,
    out_path: Path | str | None = None,
    min_predictions: int = DEFAULT_MIN_SNAPSHOT_PREDICTIONS,
    force: bool = False,
) -> dict[str, Any]:
    """Snapshot the predicted-label distribution from the predictions store.

    Args:
        store: Optional initialized FeedbackStore. If omitted, initialized from db_path or DATABASE_URL.
        db_path: Optional SQLite database file path.
        out_path: Destination JSON file path (defaults to supportpilot/monitoring/drift_baseline.json).
        min_predictions: Minimum real predictions required (default 500).
        force: If True, bypass the minimum predictions guard (for testing or manual override).

    Returns:
        dict containing created_at timestamp, total_predictions, and counts.

    Raises:
        ValueError: If total predictions < min_predictions and force is False.
    """
    owns_store = False
    if store is None:
        store = FeedbackStore(db_path=db_path)
        owns_store = True

    try:
        counts = store.get_all_predicted_label_counts()
        total = sum(counts.values())

        if total < min_predictions and not force:
            msg = (
                f"Cannot snapshot baseline: found {total} real predictions, "
                f"but at least {min_predictions} are required."
            )
            logger.error(msg)
            raise ValueError(msg)

        payload: dict[str, Any] = {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "total_predictions": total,
            "counts": counts,
        }

        target = Path(out_path) if out_path is not None else DEFAULT_DRIFT_BASELINE_PATH
        target.parent.mkdir(parents=True, exist_ok=True)

        with open(target, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

        logger.info(
            "Saved drift baseline snapshot with %d predictions to %s", total, target
        )
        return payload
    finally:
        if owns_store:
            store.close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Snapshot predicted-label mix from production for drift monitoring (SP-120)."
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=DEFAULT_DRIFT_BASELINE_PATH,
        help="Destination path for drift baseline JSON (default: supportpilot/monitoring/drift_baseline.json).",
    )
    parser.add_argument(
        "--db-path",
        type=Path,
        default=None,
        help="Path to SQLite database file (falls back to DATABASE_URL or FEEDBACK_DB_PATH).",
    )
    parser.add_argument(
        "--min-predictions",
        type=int,
        default=DEFAULT_MIN_SNAPSHOT_PREDICTIONS,
        help="Minimum required predictions to snapshot (default: 500).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Snapshot even if fewer than min-predictions exist.",
    )

    args = parser.parse_args(argv)

    try:
        payload = snapshot_baseline(
            db_path=args.db_path,
            out_path=args.output,
            min_predictions=args.min_predictions,
            force=args.force,
        )
        print(
            f"✅ Drift baseline snapshot successfully saved to {args.output} "
            f"({payload['total_predictions']} real predictions)."
        )
        return 0
    except ValueError as exc:
        print(f"❌ {exc}", file=sys.stderr)
        return 1
    except Exception as exc:
        logger.exception("Failed to snapshot baseline: %s", exc)
        print(f"❌ Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
