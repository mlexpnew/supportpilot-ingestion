"""Machine learning model for customer ticket triage classification."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.pipeline import Pipeline

from . import LABELS

DEFAULT_MODEL_PATH = Path("models/triage_model.joblib")
DEFAULT_CONFIDENCE_THRESHOLD: float = 0.35
DEFAULT_MODEL_VERSION: str = "1.0.0"


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

    def __init__(
        self,
        pipeline: Pipeline | None = None,
        confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
        version: str = DEFAULT_MODEL_VERSION,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Initialize classifier with an optional pre-configured pipeline."""
        if pipeline is not None:
            self.pipeline: Pipeline = pipeline
        else:
            self.pipeline = self._build_pipeline()
        self.confidence_threshold: float = confidence_threshold
        self.version: str = version
        self.metadata: dict[str, Any] = (
            dict(metadata)
            if metadata is not None
            else {
                "version": version,
                "training_date": None,
                "dataset_size": 0,
                "macro_f1": 0.0,
            }
        )
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

    def fit(
        self,
        texts: Sequence[str],
        labels: Sequence[str],
        version: str | None = None,
    ) -> TriageClassifier:
        """Fit the pipeline on training texts and labels, computing metadata."""
        if not texts or not labels:
            raise ValueError("Training data cannot be empty.")
        if version is not None:
            self.version = version

        self.pipeline.fit(texts, labels)
        self._is_fitted = True

        preds = self.predict_batch(texts)
        macro_f1 = round(
            float(f1_score(labels, preds, average="macro", zero_division=0)), 4
        )
        today_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")

        self.metadata = {
            "version": self.version,
            "training_date": today_str,
            "dataset_size": len(texts),
            "macro_f1": macro_f1,
        }
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

    def predict_one(self, text: str, threshold: float | None = None) -> dict[str, Any]:
        """Predict ticket category with confidence score and human review flag.

        Args:
            text: Customer ticket body text.
            threshold: Confidence threshold for human review. If confidence < threshold,
                needs_review is True. Defaults to self.confidence_threshold.

        Returns:
            Dictionary containing:
                - 'label': predicted category string from LABELS
                - 'confidence': predicted probability float (rounded to 4 decimal places)
                - 'needs_review': boolean flag indicating low confidence
        """
        th = self.confidence_threshold if threshold is None else threshold

        if not text or not text.strip():
            return {
                "label": "other",
                "confidence": 1.0,
                "needs_review": False,
            }

        pred_label = self.predict(text)
        probs = self.predict_proba(text)
        confidence = round(probs.get(pred_label, 0.0), 4)
        needs_review = confidence < th

        return {
            "label": pred_label,
            "confidence": confidence,
            "needs_review": needs_review,
        }

    def save(self, path: Path | str) -> Path:
        """Serialize and save the trained pipeline and metadata to disk."""
        target_path = Path(path)
        target_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "pipeline": self.pipeline,
            "metadata": self.metadata,
            "version": self.version,
        }
        joblib.dump(payload, target_path)

        # Also write sidecar JSON metadata
        meta_path = target_path.with_suffix(".json")
        try:
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(self.metadata, f, indent=2)
        except OSError:
            pass

        return target_path

    @classmethod
    def load(cls, path: Path | str) -> TriageClassifier:
        """Load a trained pipeline and metadata from disk."""
        source_path = Path(path)
        if not source_path.exists():
            raise FileNotFoundError(f"Model file not found: {source_path}")
        loaded = joblib.load(source_path)

        if isinstance(loaded, dict) and "pipeline" in loaded:
            pipeline = loaded["pipeline"]
            metadata = loaded.get("metadata", {})
            version = loaded.get("version") or metadata.get(
                "version", DEFAULT_MODEL_VERSION
            )
        else:
            pipeline = loaded
            meta_path = source_path.with_suffix(".json")
            if meta_path.exists():
                try:
                    with open(meta_path, "r", encoding="utf-8") as f:
                        metadata = json.load(f)
                except Exception:
                    metadata = {}
            else:
                metadata = {}
            version = metadata.get("version", DEFAULT_MODEL_VERSION)

        classifier = cls(pipeline=pipeline, version=version, metadata=metadata)
        classifier._is_fitted = True
        return classifier


def train_model(
    data_path: Path | str = "data/triage_labeled.jsonl",
    save_path: Path | str | None = None,
    version: str = DEFAULT_MODEL_VERSION,
) -> TriageClassifier:
    """Helper function to load dataset, train classifier, and optionally save.

    Args:
        data_path: Path to training data JSONL.
        save_path: Optional path to save serialized model.
        version: Version string for the model (default: 1.0.0).

    Returns:
        Trained TriageClassifier instance.
    """
    _, texts, labels = load_dataset(data_path)
    classifier = TriageClassifier(version=version)
    classifier.fit(texts, labels, version=version)

    if save_path:
        classifier.save(save_path)

    return classifier


def predict_one(
    text: str,
    model_path: Path | str | None = None,
    threshold: float | None = None,
) -> dict[str, Any]:
    """Convenience function to predict label with confidence fallback.

    Args:
        text: Customer ticket body text.
        model_path: Optional path to trained model.
        threshold: Optional confidence threshold.

    Returns:
        Dictionary with 'label', 'confidence', and 'needs_review'.
    """
    path = Path(model_path or DEFAULT_MODEL_PATH)
    if path.exists():
        classifier = TriageClassifier.load(path)
    else:
        classifier = train_model(save_path=path)
    return classifier.predict_one(text, threshold=threshold)
