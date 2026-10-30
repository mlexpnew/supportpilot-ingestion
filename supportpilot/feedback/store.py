"""SQLAlchemy storage layer for ticket triage predictions and agent feedback (SP-111).

Zero-PII / Zero-Leakage Guarantee:
Neither the 'predictions' table nor the 'feedback' table stores ticket bodies,
subjects, customer metadata, or raw text. Parameterized queries are
enforced via SQLAlchemy statements for all operations to prevent SQL injection.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from datetime import UTC, date as date_type, datetime, timedelta
from pathlib import Path
from typing import Any

from sqlalchemy import (
    Boolean,
    Column,
    Float,
    ForeignKey,
    Index,
    MetaData,
    String,
    Table,
    case,
    create_engine,
    event,
    func,
    select,
)
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.engine import Engine

from supportpilot.triage import LABELS

from .models import DateRangeStats, normalize_agent_id, normalize_ticket_id

DEFAULT_DB_PATH = Path("data/feedback.db")
logger = logging.getLogger(__name__)

metadata = MetaData()

predictions = Table(
    "predictions",
    metadata,
    Column("ticket_id", String(64), primary_key=True),
    Column("predicted_label", String(64), nullable=False),
    Column("confidence", Float, nullable=False),
    Column("model_version", String(64), nullable=False),
    Column("created_at", String(64), nullable=False),
    Column("needs_review", Boolean, nullable=False, server_default=func.false()),
    Index("idx_predictions_predicted_label", "predicted_label"),
)

feedback = Table(
    "feedback",
    metadata,
    Column(
        "ticket_id",
        String(64),
        ForeignKey("predictions.ticket_id"),
        primary_key=True,
    ),
    Column("correct_label", String(64), nullable=False),
    Column("agent_id", String(64), nullable=False),
    Column("created_at", String(64), nullable=False),
    Index("idx_feedback_correct_label", "correct_label"),
)


def _sqlite_engine(path: Path | str) -> Engine:
    """Create a SQLite engine with WAL mode and connection pooling."""
    target_path = Path(path)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    sqlite_url = f"sqlite:///{target_path.resolve()}"
    engine = create_engine(sqlite_url, pool_size=5, pool_pre_ping=True)

    @event.listens_for(engine, "connect")
    def set_sqlite_pragma(dbapi_connection: Any, connection_record: Any) -> None:
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA journal_mode = WAL;")
        cursor.execute("PRAGMA synchronous = NORMAL;")
        cursor.close()

    logger.info("Database initialized: sqlite")
    return engine


def get_feedback_db_path() -> Path:
    """Return configured database path from FEEDBACK_DB_PATH or default data/feedback.db."""
    env_path = os.getenv("FEEDBACK_DB_PATH")
    if env_path is not None and env_path.strip():
        return Path(env_path.strip())
    return DEFAULT_DB_PATH


def normalize_database_url(url: str) -> str:
    """Normalize database URL for SQLAlchemy with psycopg driver."""
    clean_url = url.strip()
    if clean_url.startswith("postgres://"):
        return "postgresql+psycopg://" + clean_url[len("postgres://") :]
    if clean_url.startswith("postgresql://") and not clean_url.startswith(
        "postgresql+psycopg://"
    ):
        return "postgresql+psycopg://" + clean_url[len("postgresql://") :]
    return clean_url


def create_feedback_engine(
    database_url: str | None = None,
    db_path: Path | str | None = None,
) -> tuple[Engine, str]:
    """Create a SQLAlchemy Engine with connection pooling.

    Uses DATABASE_URL if present. If missing, falls back to SQLite only if
    ALLOW_SQLITE_FALLBACK=1 or if db_path is explicitly provided.
    Otherwise raises RuntimeError to prevent silent data loss in production.
    Logs database type only (postgres / sqlite), never the URL.
    Pool settings: pool_size=5, pool_pre_ping=True.
    """
    if database_url is not None and database_url.strip():
        norm_url = normalize_database_url(database_url)
        engine = create_engine(norm_url, pool_size=5, pool_pre_ping=True)
        db_type = (
            "postgres" if engine.dialect.name == "postgresql" else engine.dialect.name
        )
        logger.info("Database initialized: %s", db_type)
        return engine, db_type

    if db_path is not None:
        return _sqlite_engine(db_path), "sqlite"

    env_url = os.getenv("DATABASE_URL")
    if env_url is not None and env_url.strip():
        norm_url = normalize_database_url(env_url)
        engine = create_engine(norm_url, pool_size=5, pool_pre_ping=True)
        db_type = (
            "postgres" if engine.dialect.name == "postgresql" else engine.dialect.name
        )
        logger.info("Database initialized: %s", db_type)
        return engine, db_type

    allow_fallback = os.getenv("ALLOW_SQLITE_FALLBACK", "").strip().lower() in (
        "1",
        "true",
        "yes",
    )
    if not allow_fallback:
        raise RuntimeError(
            "DATABASE_URL is not set and SQLite fallback is disabled. "
            "Set DATABASE_URL to a valid database connection string, "
            "or set ALLOW_SQLITE_FALLBACK=1 for local development."
        )

    logger.warning("DATABASE_URL not set; falling back to SQLite storage.")
    return _sqlite_engine(get_feedback_db_path()), "sqlite"


def init_db(
    path: Path | str | None = None,
    engine: Engine | None = None,
) -> str:
    """Initialize the feedback database schema via Alembic migrations.

    Zero-PII guarantee: neither table stores ticket body or customer text.
    Alembic owns the schema.

    Args:
        path: Optional path to SQLite database file.
        engine: Existing SQLAlchemy engine to initialize schema on.

    Returns:
        The database type string ("postgres" or "sqlite").
    """
    from alembic import command
    from alembic.config import Config

    alembic_ini_path = Path(__file__).parents[2] / "alembic.ini"
    alembic_cfg = Config(str(alembic_ini_path))

    if engine is not None:
        with engine.begin() as conn:
            alembic_cfg.attributes["connection"] = conn
            command.upgrade(alembic_cfg, "head")
        return (
            "postgres" if engine.dialect.name == "postgresql" else engine.dialect.name
        )

    if path is not None:
        target_path = Path(path)
        target_path.parent.mkdir(parents=True, exist_ok=True)
        eng = _sqlite_engine(target_path)
        try:
            with eng.begin() as conn:
                alembic_cfg.attributes["connection"] = conn
                command.upgrade(alembic_cfg, "head")
        finally:
            eng.dispose()
        return "sqlite"

    eng, db_type = create_feedback_engine(db_path=path)
    try:
        with eng.begin() as conn:
            alembic_cfg.attributes["connection"] = conn
            command.upgrade(alembic_cfg, "head")
    finally:
        eng.dispose()

    return db_type


class FeedbackStore:
    """SQLAlchemy-backed store for predictions and agent feedback."""

    def __init__(
        self,
        db_path: Path | str | None = None,
        database_url: str | None = None,
        engine: Engine | None = None,
    ) -> None:
        if engine is not None:
            self.engine = engine
            self.db_type = (
                "postgres"
                if self.engine.dialect.name == "postgresql"
                else self.engine.dialect.name
            )
            self.db_path = Path(db_path) if db_path is not None else None
        else:
            self.engine, self.db_type = create_feedback_engine(
                database_url=database_url,
                db_path=db_path,
            )
            if self.db_type == "sqlite":
                self.db_path = (
                    Path(db_path) if db_path is not None else get_feedback_db_path()
                )
            else:
                self.db_path = None

        self.predictions = predictions
        self.feedback = feedback

    def _insert(self, table: Table) -> Any:
        if self.engine.dialect.name == "sqlite":
            return sqlite_insert(table)
        return pg_insert(table)

    def save_prediction(
        self,
        ticket_id: str,
        predicted_label: str,
        confidence: float,
        model_version: str,
        created_at: str | None = None,
        needs_review: bool = False,
    ) -> None:
        """Record a triage prediction.

        Zero-PII guarantee: ticket body and customer text are never stored.
        Uses parameterized query via SQLAlchemy statement to prevent SQL injection.
        """
        ticket_id = normalize_ticket_id(ticket_id)
        ts = created_at or datetime.now(UTC).isoformat()
        stmt = self._insert(self.predictions).values(
            ticket_id=ticket_id,
            predicted_label=predicted_label,
            confidence=float(confidence),
            model_version=model_version,
            created_at=ts,
            needs_review=bool(needs_review),
        )
        stmt = stmt.on_conflict_do_update(
            index_elements=[self.predictions.c.ticket_id],
            set_={
                "predicted_label": stmt.excluded.predicted_label,
                "confidence": stmt.excluded.confidence,
                "model_version": stmt.excluded.model_version,
                "created_at": stmt.excluded.created_at,
                "needs_review": stmt.excluded.needs_review,
            },
            where=self.predictions.c.ticket_id.not_in(
                select(self.feedback.c.ticket_id)
            ),
        )
        with self.engine.begin() as conn:
            conn.execute(stmt)

    def get_prediction(self, ticket_id: str) -> dict[str, Any] | None:
        """Fetch prediction record by ticket_id."""
        ticket_id = normalize_ticket_id(ticket_id)
        stmt = select(
            self.predictions.c.ticket_id,
            self.predictions.c.predicted_label,
            self.predictions.c.confidence,
            self.predictions.c.model_version,
            self.predictions.c.created_at,
            self.predictions.c.needs_review,
        ).where(self.predictions.c.ticket_id == ticket_id)
        with self.engine.connect() as conn:
            row = conn.execute(stmt).mappings().first()
            if row is None:
                return None
            res = dict(row)
            if "needs_review" in res:
                res["needs_review"] = bool(res["needs_review"])
            return res

    def has_prediction(self, ticket_id: str) -> bool:
        """Check whether a prediction exists for ticket_id."""
        ticket_id = normalize_ticket_id(ticket_id)
        stmt = select(1).where(self.predictions.c.ticket_id == ticket_id).limit(1)
        with self.engine.connect() as conn:
            return conn.execute(stmt).first() is not None

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
        with self.engine.begin() as conn:
            pred_stmt = (
                select(self.predictions.c.predicted_label)
                .where(self.predictions.c.ticket_id == ticket_id)
                .limit(1)
            )
            row = conn.execute(pred_stmt).mappings().first()
            if row is None:
                raise KeyError(f"Ticket '{ticket_id}' has not been triaged.")

            predicted_label = str(row["predicted_label"])

            stmt = self._insert(self.feedback).values(
                ticket_id=ticket_id,
                correct_label=correct_label,
                agent_id=agent_id,
                created_at=ts,
            )
            stmt = stmt.on_conflict_do_update(
                index_elements=[self.feedback.c.ticket_id],
                set_={
                    "correct_label": stmt.excluded.correct_label,
                    "agent_id": stmt.excluded.agent_id,
                    "created_at": stmt.excluded.created_at,
                },
            )
            conn.execute(stmt)

        return {
            "ticket_id": ticket_id,
            "correct_label": correct_label,
            "predicted_label": predicted_label,
            "agreed": predicted_label == correct_label,
        }

    def get_feedback(self, ticket_id: str) -> dict[str, Any] | None:
        """Fetch feedback record by ticket_id."""
        ticket_id = normalize_ticket_id(ticket_id)
        stmt = select(
            self.feedback.c.ticket_id,
            self.feedback.c.correct_label,
            self.feedback.c.agent_id,
            self.feedback.c.created_at,
        ).where(self.feedback.c.ticket_id == ticket_id)
        with self.engine.connect() as conn:
            row = conn.execute(stmt).mappings().first()
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
        stmt = (
            select(
                self.feedback.c.ticket_id,
                self.feedback.c.correct_label,
                self.feedback.c.agent_id,
                self.feedback.c.created_at.label("feedback_created_at"),
                self.predictions.c.predicted_label,
                self.predictions.c.confidence,
                self.predictions.c.model_version,
                self.predictions.c.created_at.label("prediction_created_at"),
                case(
                    (
                        self.predictions.c.predicted_label
                        == self.feedback.c.correct_label,
                        True,
                    ),
                    else_=False,
                ).label("agreed"),
            )
            .select_from(
                self.feedback.join(
                    self.predictions,
                    self.feedback.c.ticket_id == self.predictions.c.ticket_id,
                )
            )
            .order_by(self.feedback.c.created_at.asc())
        )
        if ticket_id is not None:
            stmt = stmt.where(
                self.feedback.c.ticket_id == normalize_ticket_id(ticket_id)
            )

        with self.engine.connect() as conn:
            rows = conn.execute(stmt).mappings().all()
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
                for row in rows
            ]

    def get_live_accuracy(self) -> tuple[int, float | None]:
        """Compute live accuracy across all agent feedback.

        Returns:
            (feedback_total, live_accuracy)
            where live_accuracy is agreed predictions / total feedback,
            or None if feedback_total == 0.
        """
        stmt = select(
            func.count(self.feedback.c.ticket_id).label("total"),
            func.sum(
                case(
                    (
                        self.predictions.c.predicted_label
                        == self.feedback.c.correct_label,
                        1,
                    ),
                    else_=0,
                )
            ).label("agreed"),
        ).select_from(
            self.feedback.join(
                self.predictions,
                self.feedback.c.ticket_id == self.predictions.c.ticket_id,
            )
        )
        with self.engine.connect() as conn:
            row = conn.execute(stmt).mappings().first()
            total = int(row["total"]) if row and row["total"] is not None else 0
            agreed = int(row["agreed"]) if row and row["agreed"] is not None else 0

            if total == 0:
                return (0, None)
            return (total, round(agreed / total, 4))

    def get_label_metrics(self) -> dict[str, dict[str, Any]]:
        """Compute per-label recall metrics based on agent corrections.

        Since rows are grouped by ground-truth `correct_label`, this metric
        represents per-class Recall: of all true tickets for this class,
        what fraction did the model predict correctly?

        Returns dictionary mapping label -> {total, agreed, recall}.
        """
        metrics: dict[str, dict[str, Any]] = {
            lbl: {"total": 0, "agreed": 0, "recall": None} for lbl in LABELS
        }
        stmt = (
            select(
                self.feedback.c.correct_label,
                func.count(self.feedback.c.ticket_id).label("total"),
                func.sum(
                    case(
                        (
                            self.predictions.c.predicted_label
                            == self.feedback.c.correct_label,
                            1,
                        ),
                        else_=0,
                    )
                ).label("agreed"),
            )
            .select_from(
                self.feedback.join(
                    self.predictions,
                    self.feedback.c.ticket_id == self.predictions.c.ticket_id,
                )
            )
            .group_by(self.feedback.c.correct_label)
        )
        with self.engine.connect() as conn:
            for row in conn.execute(stmt).mappings():
                lbl = row["correct_label"]
                total = int(row["total"])
                agreed = int(row["agreed"]) if row["agreed"] is not None else 0
                rec = round(agreed / total, 4) if total > 0 else None
                metrics[lbl] = {
                    "total": total,
                    "agreed": agreed,
                    "recall": rec,
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
        stmt = select(
            self.feedback.c.ticket_id,
            self.feedback.c.correct_label,
        ).order_by(self.feedback.c.created_at.asc())
        try:
            with (
                open(tmp_fd, "w", encoding="utf-8") as f,
                self.engine.connect() as conn,
            ):
                for row in conn.execute(stmt).mappings():
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

    def get_stats(
        self,
        start: datetime | date_type | str,
        end: datetime | date_type | str,
    ) -> DateRangeStats:
        """Compute aggregated statistics for a date range from persistent storage (SP-114 Part A).

        Zero-PII guarantee: only aggregates and metadata are queried and returned.
        Meera's specification:
        - Tickets triaged count, per-label counts, needs_review_rate
        - Feedback count, live_accuracy, per-label recall
        - Distinct model version(s) used
        - Daily triage counts for charting
        """
        # 1. Normalize start and end boundaries
        if isinstance(start, (datetime, date_type)):
            start_str = start.isoformat()
        else:
            start_str = str(start).strip()

        end_is_exclusive = False
        if isinstance(end, datetime):
            end_str = end.isoformat()
        elif isinstance(end, date_type):
            next_day = end + timedelta(days=1)
            end_str = next_day.isoformat()
            end_is_exclusive = True
        else:
            end_clean = str(end).strip()
            if len(end_clean) == 10:  # YYYY-MM-DD
                parsed_d = datetime.strptime(end_clean, "%Y-%m-%d").date()
                next_day = parsed_d + timedelta(days=1)
                end_str = next_day.isoformat()
                end_is_exclusive = True
            else:
                end_str = end_clean

        # Predictions filter
        pred_where = [self.predictions.c.created_at >= start_str]
        if end_is_exclusive:
            pred_where.append(self.predictions.c.created_at < end_str)
        else:
            pred_where.append(self.predictions.c.created_at <= end_str)

        # Feedback filter
        fb_where = [self.feedback.c.created_at >= start_str]
        if end_is_exclusive:
            fb_where.append(self.feedback.c.created_at < end_str)
        else:
            fb_where.append(self.feedback.c.created_at <= end_str)

        with self.engine.connect() as conn:
            # 2. Prediction aggregates: per-label count and needs_review
            pred_agg_stmt = (
                select(
                    self.predictions.c.predicted_label,
                    func.count(self.predictions.c.ticket_id).label("label_count"),
                    func.sum(
                        case(
                            (self.predictions.c.needs_review.is_(True), 1),
                            (self.predictions.c.needs_review == 1, 1),
                            else_=0,
                        )
                    ).label("label_needs_review"),
                )
                .where(*pred_where)
                .group_by(self.predictions.c.predicted_label)
            )

            predictions_by_label: dict[str, int] = {lbl: 0 for lbl in LABELS}
            needs_review_count = 0
            total_triaged = 0

            for row in conn.execute(pred_agg_stmt).mappings():
                lbl = str(row["predicted_label"])
                cnt = int(row["label_count"])
                nr = (
                    int(row["label_needs_review"])
                    if row["label_needs_review"] is not None
                    else 0
                )
                predictions_by_label[lbl] = cnt
                total_triaged += cnt
                needs_review_count += nr

            needs_review_rate = (
                round(needs_review_count / total_triaged, 4)
                if total_triaged > 0
                else 0.0
            )

            # 3. Distinct model versions in range
            versions_stmt = (
                select(self.predictions.c.model_version)
                .where(*pred_where)
                .distinct()
                .order_by(self.predictions.c.model_version.asc())
            )
            model_versions = [
                str(r[0]) for r in conn.execute(versions_stmt).all() if r[0] is not None
            ]

            # 4. Daily triage counts (tickets per day)
            daily_stmt = (
                select(
                    func.substr(self.predictions.c.created_at, 1, 10).label("day"),
                    func.count(self.predictions.c.ticket_id).label("count"),
                )
                .where(*pred_where)
                .group_by(func.substr(self.predictions.c.created_at, 1, 10))
                .order_by(func.substr(self.predictions.c.created_at, 1, 10).asc())
            )
            daily_triage_counts = {
                str(r["day"]): int(r["count"])
                for r in conn.execute(daily_stmt).mappings()
            }

            # 5. Feedback in range: total, agreed, live_accuracy, per-label recall
            fb_stmt = (
                select(
                    self.feedback.c.correct_label,
                    func.count(self.feedback.c.ticket_id).label("total"),
                    func.sum(
                        case(
                            (
                                self.predictions.c.predicted_label
                                == self.feedback.c.correct_label,
                                1,
                            ),
                            else_=0,
                        )
                    ).label("agreed"),
                )
                .select_from(
                    self.feedback.join(
                        self.predictions,
                        self.feedback.c.ticket_id == self.predictions.c.ticket_id,
                    )
                )
                .where(*fb_where)
                .group_by(self.feedback.c.correct_label)
            )

            per_label_feedback: dict[str, dict[str, Any]] = {
                lbl: {"total": 0, "agreed": 0, "recall": None} for lbl in LABELS
            }
            per_label_recall: dict[str, float | None] = {lbl: None for lbl in LABELS}
            feedback_total = 0
            feedback_agreed = 0

            for row in conn.execute(fb_stmt).mappings():
                lbl = str(row["correct_label"])
                tot = int(row["total"])
                agr = int(row["agreed"]) if row["agreed"] is not None else 0
                rec = round(agr / tot, 4) if tot > 0 else None
                feedback_total += tot
                feedback_agreed += agr
                per_label_feedback[lbl] = {
                    "total": tot,
                    "agreed": agr,
                    "recall": rec,
                }
                per_label_recall[lbl] = rec

            live_accuracy = (
                round(feedback_agreed / feedback_total, 4)
                if feedback_total > 0
                else None
            )

        return DateRangeStats(
            start=str(start),
            end=str(end),
            total_triaged=total_triaged,
            predictions_by_label=predictions_by_label,
            needs_review_count=needs_review_count,
            needs_review_rate=needs_review_rate,
            feedback_total=feedback_total,
            feedback_agreed=feedback_agreed,
            live_accuracy=live_accuracy,
            per_label_recall=per_label_recall,
            per_label_feedback=per_label_feedback,
            model_versions=model_versions,
            daily_triage_counts=daily_triage_counts,
        )

    def close(self) -> None:
        """Dispose of the SQLAlchemy engine and connection pool."""
        if hasattr(self, "engine") and self.engine is not None:
            self.engine.dispose()


_global_store: FeedbackStore | None = None


def get_feedback_store() -> FeedbackStore:
    """Return process-wide singleton FeedbackStore instance."""
    global _global_store
    if _global_store is None:
        _global_store = FeedbackStore()
    return _global_store
