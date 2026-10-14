"""Machine learning model for customer ticket triage classification."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from . import LABELS

DEFAULT_MODEL_PATH = Path("models/triage_model.joblib")


def load_dataset(path: Path | str) -> tuple[list[str], list[str], list[str]]:
    """Load ticket triage dataset from a JSONL file.

    Args:
        path: Path to the triage_labeled.jsonl file.

    Returns:
        Tuple of (ticket_ids, texts, labels).
    """
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {file_path}")

    ids: list[str] = []
    texts: list[str] = []
    labels: list[str] = []

    with open(file_path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f, 1):
            line_str = line.strip()
            if not line_str:
                continue
            item = json.loads(line_str)
            label = item.get("label")
            if label not in LABELS:
                raise ValueError(
                    f"Line {idx}: invalid label '{label}', expected one of {LABELS}"
                )
            ids.append(item.get("ticket_id", f"TR-{idx:03d}"))
            texts.append(item.get("body", ""))
            labels.append(label)

    return ids, texts, labels


class TriageClassifier:
    """Scikit-learn pipeline wrapper for ticket triage classification."""

    def __init__(self, pipeline: Pipeline | None = None) -> None:
        """Initialize classifier with an optional pre-configured pipeline."""
        if pipeline is not None:
            self.pipeline: Pipeline = pipeline
        else:
            self.pipeline = self._build_pipeline()
        self._is_fitted: bool = False

    @staticmethod
    def _build_pipeline() -> Pipeline:
        """Construct the default TF-IDF + LogisticRegression pipeline."""
        return Pipeline(
            [
                (
                    "tfidf",
                    TfidfVectorizer(
                        ngram_range=(1, 2),
                        sublinear_tf=True,
                        min_df=1,
                        strip_accents="unicode",
                    ),
                ),
                (
                    "clf",
                    LogisticRegression(
                        C=2.0,
                        max_iter=1000,
                        random_state=42,
                    ),
                ),
            ]
        )

    def fit(self, texts: Sequence[str], labels: Sequence[str]) -> TriageClassifier:
        """Fit the pipeline on training texts and labels."""
        if not texts or not labels:
            raise ValueError("Training data cannot be empty.")
        self.pipeline.fit(texts, labels)
        self._is_fitted = True
        return self

    def predict(self, text: str) -> str:
        """Predict the triage label for a single ticket body.

        Args:
            text: Customer ticket body text.

        Returns:
            Predicted label from LABELS ('delivery', 'billing', 'refund',
            'technical', 'account', 'other'). Defaults to 'other' for empty text.
        """
        if not text or not text.strip():
            return "other"
        if not self._is_fitted:
            raise RuntimeError(
                "Model is not fitted. Call fit() or load a trained model first."
            )

        pred = self.pipeline.predict([text])[0]
        return str(pred)

    def predict_batch(self, texts: Sequence[str]) -> list[str]:
        """Predict labels for a batch of ticket bodies."""
        if not self._is_fitted:
            raise RuntimeError("Model is not fitted.")
        return [self.predict(t) for t in texts]

    def predict_proba(self, text: str) -> dict[str, float]:
        """Predict probability distribution over labels for a single ticket.

        Args:
            text: Customer ticket body text.

        Returns:
            Mapping from label name to predicted probability.
        """
        if not text or not text.strip():
            return {label: (1.0 if label == "other" else 0.0) for label in LABELS}
        if not self._is_fitted:
            raise RuntimeError("Model is not fitted.")

        probabilities = self.pipeline.predict_proba([text])[0]
        classes = self.pipeline.classes_
        return {str(cls): float(prob) for cls, prob in zip(classes, probabilities)}

    def save(self, path: Path | str) -> Path:
        """Serialize and save the trained pipeline to disk."""
        target_path = Path(path)
        target_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.pipeline, target_path)
        return target_path

    @classmethod
    def load(cls, path: Path | str) -> TriageClassifier:
        """Load a trained pipeline from disk."""
        source_path = Path(path)
        if not source_path.exists():
            raise FileNotFoundError(f"Model file not found: {source_path}")
        pipeline = joblib.load(source_path)
        classifier = cls(pipeline=pipeline)
        classifier._is_fitted = True
        return classifier


def train_model(
    data_path: Path | str = "data/triage_labeled.jsonl",
    save_path: Path | str | None = None,
) -> TriageClassifier:
    """Helper function to load dataset, train classifier, and optionally save.

    Args:
        data_path: Path to training data JSONL.
        save_path: Optional path to save serialized model.

    Returns:
        Trained TriageClassifier instance.
    """
    _, texts, labels = load_dataset(data_path)
    classifier = TriageClassifier()
    classifier.fit(texts, labels)

    if save_path:
        classifier.save(save_path)

    return classifier
