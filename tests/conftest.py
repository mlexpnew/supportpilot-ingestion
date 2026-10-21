from __future__ import annotations

from pathlib import Path

import pytest

from supportpilot.triage.model import train_model


@pytest.fixture(scope="session")
def model_file(tmp_path_factory: pytest.TempPathFactory) -> Path:
    path = tmp_path_factory.mktemp("models") / "triage.joblib"
    train_model("data/triage_labeled.jsonl", save_path=path)
    return path


@pytest.fixture(autouse=True)
def _model_env(model_file: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TRIAGE_MODEL_PATH", str(model_file))
