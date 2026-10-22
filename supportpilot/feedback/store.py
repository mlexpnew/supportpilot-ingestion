"""SQLite storage layer for ticket triage predictions and agent feedback (SP-110).

Zero-PII / Zero-Leakage Guarantee:
Neither the 'predictions' table nor the 'feedback' table stores ticket bodies,
subjects, customer metadata, or raw text. Parameterized SQL queries are
enforced for all operations to prevent SQL injection.
"""

from __future__ import annotations

import json
import os
import sqlite3
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from supportpilot.triage import LABELS

from .models import normalize_agent_id, normalize_ticket_id

DEFAULT_DB_PATH = Path("data/feedback.db")


def get_feedback_db_path() -> Path:
    """Return configured database path from FEEDBACK_DB_PATH or default data/feedback.db."""
    env_path = os.getenv("FEEDBACK_DB_PATH")
    if env_path is not None and env_path.strip():
        return Path(env_path.strip())
    return DEFAULT_DB_PATH


def init_db(path: Path | str | None = None) -> Path:
    """Initialize the SQLite feedback database, creating both tables if missing.

    Tables:
        - predictions: ticket_id (PK), predicted_label, confidence,
                       model_version, created_at
        - feedback: ticket_id (PRIMARY KEY, exactly one row per ticket),
                    correct_label, agent_id, created_at

    Zero-PII guarantee: neither table stores ticket body or customer text.
    Uses parameterized DDL and statements.

    Args:
        path: Path to SQLite database file. If None, uses get_feedback_db_path().

    Returns:
        Path to the initialized database file.
    """
    db_path = Path(path) if path is not None else get_feedback_db_path()
    db_path.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(
        str(db_path),
        timeout=10.0,
        check_same_thread=False,
    )
    try:
        # WAL mode allows concurrent readers and a writer
        conn.execute("PRAGMA journal_mode = WAL;")
        conn.execute("PRAGMA synchronous = NORMAL;")
        conn.execute("""
            CREATE TABLE IF NOT EXISTS predictions (
                ticket_id TEXT PRIMARY KEY,
                predicted_label TEXT NOT NULL,
                confidence REAL NOT NULL,
                model_version TEXT NOT NULL,
                created_at TEXT NOT NULL
            );
            """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_predictions_predicted_label
            ON predictions(predicted_label);
            """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS feedback (
                ticket_id TEXT PRIMARY KEY,
                correct_label TEXT NOT NULL,
                agent_id TEXT NOT NULL,
                created_at TEXT NOT NULL,
                FOREIGN KEY (ticket_id) REFERENCES predictions(ticket_id)
            );
            """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_feedback_correct_label
            ON feedback(correct_label);
            """)
        conn.commit()
    finally:
        conn.close()

    return db_path


class FeedbackStore:
    """SQLite-backed store for predictions and agent feedback."""

    def __init__(self, db_path: Path | str | None = None) -> None:
        if db_path is None:
            self.db_path = get_feedback_db_path()
        else:
            self.db_path = Path(db_path)
        self._init_db()

    @contextmanager
    def _get_connection(self) -> Iterator[sqlite3.Connection]:
        """Context manager providing a SQLite connection with row factory."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(
            str(self.db_path),
            timeout=10.0,
            check_same_thread=False,
        )
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _init_db(self) -> None:
        """Initialize tables and indices if they do not exist."""
        init_db(self.db_path)

    def save_prediction(
        self,
        ticket_id: str,
        predicted_label: str,
        confidence: float,
        model_version: str,
        created_at: str | None = None,
    ) -> None:
        """Record a triage prediction.

        Zero-PII guarantee: ticket body and customer text are never stored.
        Uses parameterized query to prevent SQL injection.
        """
        ticket_id = normalize_ticket_id(ticket_id)
        ts = created_at or datetime.now(UTC).isoformat()
        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT INTO predictions (ticket_id, predicted_label, confidence, model_version, created_at)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(ticket_id) DO UPDATE SET
                    predicted_label = excluded.predicted_label,
                    confidence = excluded.confidence,
                    model_version = excluded.model_version,
                    created_at = excluded.created_at
                WHERE NOT EXISTS (
                    SELECT 1 FROM feedback WHERE feedback.ticket_id = predictions.ticket_id
                );
                """,
                (ticket_id, predicted_label, float(confidence), model_version, ts),
            )

    def get_prediction(self, ticket_id: str) -> dict[str, Any] | None:
        """Fetch prediction record by ticket_id."""
        ticket_id = normalize_ticket_id(ticket_id)
        with self._get_connection() as conn:
            cursor = conn.execute(
                """
                SELECT ticket_id, predicted_label, confidence, model_version, created_at
                FROM predictions
                WHERE ticket_id = ?;
                """,
                (ticket_id,),
            )
            row = cursor.fetchone()
            if row is None:
                return None
            return dict(row)

    def has_prediction(self, ticket_id: str) -> bool:
        """Check whether a prediction exists for ticket_id."""
        ticket_id = normalize_ticket_id(ticket_id)
        with self._get_connection() as conn:
            cursor = conn.execute(
                "SELECT 1 FROM predictions WHERE ticket_id = ? LIMIT 1;",
                (ticket_id,),
            )
            return cursor.fetchone() is not None

    def save_feedback(
        self,
        ticket_id: str,
        correct_label: str,
        agent_id: str,
        created_at: str | None = None,
    ) -> dict[str, Any]:
        """Record or update human agent feedback for a triaged ticket.

        If ticket_id was never triaged, raises KeyError (maps to HTTP 404).
        Duplicate feedback for the same ticket_id updates the existing row.
        """
        ticket_id = normalize_ticket_id(ticket_id)
        agent_id = normalize_agent_id(agent_id)

        if correct_label not in LABELS:
            raise ValueError(
                f"Invalid correct_label '{correct_label}'. Must be one of {LABELS}"
            )

        ts = created_at or datetime.now(UTC).isoformat()
        with self._get_connection() as conn:
            # Check existence of prediction first and fetch predicted_label
            cursor = conn.execute(
                "SELECT predicted_label FROM predictions WHERE ticket_id = ? LIMIT 1;",
                (ticket_id,),
            )
            row = cursor.fetchone()
            if row is None:
                raise KeyError(f"Ticket '{ticket_id}' has not been triaged.")

            predicted_label = str(row["predicted_label"])

            conn.execute(
                """
                INSERT INTO feedback (ticket_id, correct_label, agent_id, created_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(ticket_id) DO UPDATE SET
                    correct_label = excluded.correct_label,
                    agent_id = excluded.agent_id,
                    created_at = excluded.created_at;
                """,
                (ticket_id, correct_label, agent_id, ts),
            )

        return {
            "ticket_id": ticket_id,
            "correct_label": correct_label,
            "predicted_label": predicted_label,
            "agreed": predicted_label == correct_label,
        }

    def get_feedback(self, ticket_id: str) -> dict[str, Any] | None:
        """Fetch feedback record by ticket_id."""
        ticket_id = normalize_ticket_id(ticket_id)
        with self._get_connection() as conn:
            cursor = conn.execute(
                """
                SELECT ticket_id, correct_label, agent_id, created_at
                FROM feedback
                WHERE ticket_id = ?;
                """,
                (ticket_id,),
            )
            row = cursor.fetchone()
            if row is None:
                return None
            return dict(row)

    def get_feedback_with_predictions(
        self, ticket_id: str | None = None
    ) -> list[dict[str, Any]]:
        """Fetch feedback records joined with predictions on ticket_id.

        Returns list of records containing:
            ticket_id: Ticket identifier
            correct_label: Agent verified label
            agent_id: Identifier of the agent
            feedback_created_at: Feedback submission timestamp
            predicted_label: Model prediction label
            confidence: Prediction confidence score
            model_version: Version of the triage model
            prediction_created_at: Prediction creation timestamp
            agreed: True if predicted_label == correct_label, else False
        """
        query = """
            SELECT
                f.ticket_id,
                f.correct_label,
                f.agent_id,
                f.created_at AS feedback_created_at,
                p.predicted_label,
                p.confidence,
                p.model_version,
                p.created_at AS prediction_created_at,
                (p.predicted_label = f.correct_label) AS agreed
            FROM feedback f
            JOIN predictions p ON f.ticket_id = p.ticket_id
        """
        params: tuple[Any, ...] = ()
        if ticket_id is not None:
            query += " WHERE f.ticket_id = ?"
            params = (normalize_ticket_id(ticket_id),)
        query += " ORDER BY f.created_at ASC;"

        with self._get_connection() as conn:
            cursor = conn.execute(query, params)
            return [
                {
                    "ticket_id": row["ticket_id"],
                    "correct_label": row["correct_label"],
                    "agent_id": row["agent_id"],
                    "feedback_created_at": row["feedback_created_at"],
                    "predicted_label": row["predicted_label"],
                    "confidence": float(row["confidence"]),
                    "model_version": row["model_version"],
                    "prediction_created_at": row["prediction_created_at"],
                    "agreed": bool(row["agreed"]),
                }
                for row in cursor.fetchall()
            ]

    def get_live_accuracy(self) -> tuple[int, float | None]:
        """Compute live accuracy across all agent feedback.

        Returns:
            (feedback_total, live_accuracy)
            where live_accuracy is agreed predictions / total feedback,
            or None if feedback_total == 0.
        """
        with self._get_connection() as conn:
            cursor = conn.execute("""
                SELECT
                    COUNT(f.ticket_id) AS total,
                    SUM(CASE WHEN p.predicted_label = f.correct_label THEN 1 ELSE 0 END) AS agreed
                FROM feedback f
                JOIN predictions p ON f.ticket_id = p.ticket_id;
                """)
            row = cursor.fetchone()
            total = int(row["total"]) if row and row["total"] is not None else 0
            agreed = int(row["agreed"]) if row and row["agreed"] is not None else 0

            if total == 0:
                return (0, None)
            return (total, round(agreed / total, 4))

    def get_label_metrics(self) -> dict[str, dict[str, Any]]:
        """Compute per-label recall/accuracy metrics based on agent corrections.

        Since rows are grouped by ground-truth `correct_label`, this metric
        represents per-class Recall: of all true tickets for this class,
        what fraction did the model predict correctly?

        Returns dictionary mapping label -> {total, agreed, recall, accuracy}.
        """
        metrics: dict[str, dict[str, Any]] = {
            lbl: {"total": 0, "agreed": 0, "recall": None, "accuracy": None}
            for lbl in LABELS
        }
        with self._get_connection() as conn:
            cursor = conn.execute("""
                SELECT
                    f.correct_label,
                    COUNT(f.ticket_id) AS total,
                    SUM(CASE WHEN p.predicted_label = f.correct_label THEN 1 ELSE 0 END) AS agreed
                FROM feedback f
                JOIN predictions p ON f.ticket_id = p.ticket_id
                GROUP BY f.correct_label;
                """)
            for row in cursor.fetchall():
                lbl = row["correct_label"]
                total = int(row["total"])
                agreed = int(row["agreed"]) if row["agreed"] is not None else 0
                score = round(agreed / total, 4) if total > 0 else None
                metrics[lbl] = {
                    "total": total,
                    "agreed": agreed,
                    "recall": score,
                    "accuracy": score,
                }
        return metrics

    def export_feedback(self, output_path: Path | str) -> int:
        """Export feedback to JSONL file for retraining (SP-110 Part D).

        Exports ticket_id + correct_label only.
        Atomic write guarantees crash safety.
        """
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)

        count = 0
        tmp_fd, tmp_file = tempfile.mkstemp(
            prefix=".feedback_export_", suffix=".tmp", dir=str(out.parent)
        )
        try:
            with (
                open(tmp_fd, "w", encoding="utf-8") as f,
                self._get_connection() as conn,
            ):
                cursor = conn.execute("""
                    SELECT ticket_id, correct_label
                    FROM feedback
                    ORDER BY created_at ASC;
                    """)
                for row in cursor.fetchall():
                    record = {
                        "ticket_id": row["ticket_id"],
                        "correct_label": row["correct_label"],
                    }
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
                    count += 1
            os.replace(tmp_file, out)
        except Exception:
            if os.path.exists(tmp_file):
                os.unlink(tmp_file)
            raise
        return count


_global_store: FeedbackStore | None = None


def get_feedback_store() -> FeedbackStore:
    """Return process-wide singleton FeedbackStore instance."""
    global _global_store
    if _global_store is None:
        _global_store = FeedbackStore()
    return _global_store
