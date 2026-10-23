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
def feedback_db_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Set temporary SQLite database path via FEEDBACK_DB_PATH and run alembic migrations (Part C)."""
    db_path = tmp_path / "test_feedback.db"
    monkeypatch.setenv("FEEDBACK_DB_PATH", str(db_path))
    monkeypatch.setenv("ALLOW_SQLITE_FALLBACK", "1")

    from alembic import command
    from alembic.config import Config

    cfg = Config("alembic.ini")
    cfg.set_main_option("sqlalchemy.url", f"sqlite:///{db_path.resolve()}")
    command.upgrade(cfg, "head")

    return db_path


@pytest.fixture
def test_keys() -> list[str]:
    """Provide list of configured test API keys."""
    return [k.strip() for k in TEST_API_KEYS.split(",") if k.strip()]


@pytest.fixture
def auth_headers() -> dict[str, str]:
    """Provide default authentication headers with a valid test key."""
    return {"X-API-Key": "test-key-1"}
