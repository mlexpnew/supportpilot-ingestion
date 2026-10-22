"""Machine learning model for customer ticket triage classification."""

from __future__ import annotations

import collections
from datetime import datetime, timezone
import json
import logging
from pathlib import Path
import re
from typing import Any, Sequence

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline

from . import LABELS

logger = logging.getLogger(__name__)

DEFAULT_MODEL_PATH = Path("models/triage_model.joblib")
DEFAULT_CONFIDENCE_THRESHOLD: float = 0.35
DEFAULT_MODEL_VERSION: str = "1.0.0"
MODEL_VERSION_PATTERN = re.compile(r"^triage-v\d+\.\d+\.\d+-\d{8}$")


def is_valid_model_version(version: str | None) -> bool:
    """Validate model version string against format triage-v<semver>-<YYYYMMDD>."""
    if not version or not isinstance(version, str):
        return False
    return bool(MODEL_VERSION_PATTERN.fullmatch(version.strip()))


def format_model_version(version: str, training_date: str) -> str:
    """Build model version adhering to format triage-v<semver>-<YYYYMMDD>.

    Args:
        version: Semantic or version string (e.g. '1.0.0', 'v1.0.0').
        training_date: ISO date string 'YYYY-MM-DD' or compact 'YYYYMMDD'.

    Returns:
        Formatted version string 'triage-v<semver>-<YYYYMMDD>'.
    """
    cleaned_date = training_date.replace("-", "").strip()
    if is_valid_model_version(version):
        return version.strip()

    v_str = version.strip()
    if v_str.startswith("triage-"):
        v_str = v_str[len("triage-") :]
    if v_str.startswith("v"):
        v_str = v_str[1:]

    semver_match = re.match(r"^(\d+\.\d+\.\d+)", v_str)
    semver = semver_match.group(1) if semver_match else "1.0.0"

    return f"triage-v{semver}-{cleaned_date}"


def compute_cv_macro_f1(
    pipeline: Pipeline,
    texts: Sequence[str],
    labels: Sequence[str],
    val_texts: Sequence[str] | None = None,
    val_labels: Sequence[str] | None = None,
    macro_f1: float | None = None,
    n_splits: int = 5,
) -> float | None:
    """Compute held-out or cross-validated macro F1 score.

    Avoids evaluating on training data to prevent memorization score inflation.
    Returns None with a warning if dataset has fewer than 2 examples per label.
    """
    if macro_f1 is not None:
        return round(float(macro_f1), 4)

    if val_texts is not None and val_labels is not None:
        if len(val_texts) == 0:
            return 0.0
        preds = pipeline.predict(val_texts)
        return round(
            float(f1_score(val_labels, preds, average="macro", zero_division=0)),
            4,
        )

    counts = collections.Counter(labels)
    min_samples = min(counts.values()) if counts else 0
    splits = min(n_splits, min_samples)

    if splits >= 2 and len(counts) >= 2:
        cv = StratifiedKFold(n_splits=splits, shuffle=True, random_state=42)
        scores = cross_val_score(
            pipeline,
            texts,
            labels,
            cv=cv,
            scoring="f1_macro",
        )
        return round(float(scores.mean()), 4)

    logger.warning(
        "Dataset too small for cross-validation (<2 examples per label); "
        "macro_f1 set to None."
    )
    return None


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
        if metadata is not None:
            self.metadata: dict[str, Any] = dict(metadata)
            self.version: str = metadata.get("version", version)
        else:
            self.version = version
            self.metadata = {
                "version": version,
                "training_date": None,
                "dataset_size": 0,
                "macro_f1": None,
                "labels": [],
            }
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
        val_texts: Sequence[str] | None = None,
        val_labels: Sequence[str] | None = None,
        macro_f1: float | None = None,
    ) -> TriageClassifier:
        """Fit pipeline on training data and compute held-out/CV metadata."""
        if not texts or not labels:
            raise ValueError("Training data cannot be empty.")

        today_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        base_version = version if version is not None else self.version
        self.version = format_model_version(base_version, today_str)

        self.pipeline.fit(texts, labels)
        self._is_fitted = True

        computed_f1 = compute_cv_macro_f1(
            pipeline=self.pipeline,
            texts=texts,
            labels=labels,
            val_texts=val_texts,
            val_labels=val_labels,
            macro_f1=macro_f1,
        )

        trained_labels = [str(c) for c in self.pipeline.classes_]

        self.metadata = {
            "version": self.version,
            "training_date": today_str,
            "dataset_size": len(texts),
            "macro_f1": computed_f1,
            "labels": trained_labels,
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
                - 'confidence': predicted probability float (rounded to
                  4 decimal places)
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

        if not is_valid_model_version(self.version):
            today_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
            self.version = format_model_version(self.version, today_str)
            self.metadata["version"] = self.version

        if self._is_fitted and not self.metadata.get("labels"):
            self.metadata["labels"] = [str(c) for c in self.pipeline.classes_]

        payload = {
            "pipeline": self.pipeline,
            "metadata": self.metadata,
            "version": self.version,
        }
        joblib.dump(payload, target_path)

        # Also write sidecar JSON metadata (strict error propagation)
        meta_path = target_path.with_suffix(".json")
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(self.metadata, f, indent=2)

        return target_path

    @classmethod
    def load(cls, path: Path | str) -> TriageClassifier:
        """Load a trained pipeline and metadata from disk.

        Fails fast if file is missing, corrupted, or lacks metadata/version.
        """
        source_path = Path(path)
        if not source_path.exists():
            raise FileNotFoundError(f"Model file not found: {source_path}")

        loaded = joblib.load(source_path)

        if not isinstance(loaded, dict) or "pipeline" not in loaded:
            raise ValueError(
                f"Invalid model file format in '{source_path}': "
                "expected serialized dictionary containing 'pipeline' and 'metadata'."
            )

        metadata = loaded.get("metadata")
        if not isinstance(metadata, dict):
            raise ValueError(
                f"Missing or invalid 'metadata' in model file '{source_path}'."
            )

        version = loaded.get("version") or metadata.get("version")
        if not version or not isinstance(version, str):
            raise ValueError(
                f"Missing 'version' in model file '{source_path}'."
            )

        if not is_valid_model_version(version):
            raise ValueError(
                f"Broken or invalid model version '{version}' in '{source_path}'. "
                "Expected format 'triage-v<semver>-<YYYYMMDD>'."
            )

        if "labels" not in metadata or not isinstance(metadata["labels"], list):
            raise ValueError(
                f"Missing or invalid 'labels' in metadata of '{source_path}'."
            )

        classifier = cls(
            pipeline=loaded["pipeline"],
            version=version,
            metadata=metadata,
        )
        classifier._is_fitted = True
        return classifier


def train_model(
    data_path: Path | str = "data/triage_labeled.jsonl",
    save_path: Path | str | None = None,
    version: str = DEFAULT_MODEL_VERSION,
    val_texts: Sequence[str] | None = None,
    val_labels: Sequence[str] | None = None,
    macro_f1: float | None = None,
) -> TriageClassifier:
    """Helper function to load dataset, train classifier, and optionally save.

    Args:
        data_path: Path to training data JSONL.
        save_path: Optional path to save serialized model.
        version: Version string for the model (default: 1.0.0).
        val_texts: Optional held-out validation texts.
        val_labels: Optional held-out validation labels.
        macro_f1: Optional pre-computed macro F1 score.

    Returns:
        Trained TriageClassifier instance.
    """
    _, texts, labels = load_dataset(data_path)
    classifier = TriageClassifier(version=version)
    classifier.fit(
        texts,
        labels,
        version=version,
        val_texts=val_texts,
        val_labels=val_labels,
        macro_f1=macro_f1,
    )

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
