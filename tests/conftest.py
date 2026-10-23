from __future__ import annotations

from pathlib import Path

import pytest

from supportpilot.triage.model import train_model

TEST_API_KEYS: str = "test-key-1,test-key-2"


@pytest.fixture(scope="session")
def model_file(tmp_path_factory: pytest.TempPathFactory) -> Path:
    path = tmp_path_factory.mktemp("models") / "triage.joblib"
    train_model("data/triage_labeled.jsonl", save_path=path)
    return path


@pytest.fixture(autouse=True)
def test_keys_env(monkeypatch: pytest.MonkeyPatch) -> str:
    """Set test API keys via TRIAGE_API_KEYS environment variable (Part D)."""
    monkeypatch.setenv("TRIAGE_API_KEYS", TEST_API_KEYS)
    return TEST_API_KEYS


@pytest.fixture(autouse=True)
def _model_env(model_file: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TRIAGE_MODEL_PATH", str(model_file))
    monkeypatch.setenv("TRIAGE_API_KEYS", TEST_API_KEYS)


@pytest.fixture(autouse=True)
def feedback_db_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path | str:
    """Set temporary SQLite database path or use DATABASE_URL and run alembic migrations (Part C & D)."""
    import os
    from sqlalchemy import text
    from supportpilot.feedback.store import create_feedback_engine, init_db

    db_url = os.getenv("DATABASE_URL")
    if db_url is not None and db_url.strip():
        init_db()
        # Test isolation: clean shared database tables before each test
        engine, _ = create_feedback_engine()
        try:
            with engine.begin() as conn:
                if engine.dialect.name == "postgresql":
                    conn.execute(text("TRUNCATE feedback, predictions CASCADE;"))
                else:
                    conn.execute(text("DELETE FROM feedback;"))
                    conn.execute(text("DELETE FROM predictions;"))
        finally:
            engine.dispose()
        return db_url.strip()

    db_path = tmp_path / "test_feedback.db"
    monkeypatch.setenv("FEEDBACK_DB_PATH", str(db_path))
    monkeypatch.setenv("ALLOW_SQLITE_FALLBACK", "1")

    init_db(path=db_path)
    return db_path


@pytest.fixture
def test_keys() -> list[str]:
    """Provide list of configured test API keys."""
    return [k.strip() for k in TEST_API_KEYS.split(",") if k.strip()]


@pytest.fixture
def auth_headers() -> dict[str, str]:
    """Provide default authentication headers with a valid test key."""
    return {"X-API-Key": "test-key-1"}
