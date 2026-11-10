"""Sentence embeddings for the ReplyPilot knowledge base (SP-122).

Uses a pinned sentence-transformers model. Vectors are L2-normalised, so cosine
similarity equals a plain dot product.

The model is loaded lazily and cached once per process, because loading takes
seconds and hundreds of MB of RAM.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from functools import lru_cache
from typing import Any, Protocol

import numpy as np

logger = logging.getLogger(__name__)

MODEL_NAME: str = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIM: int = 384
DEFAULT_BATCH_SIZE: int = 32


class EncoderModel(Protocol):
    """The subset of SentenceTransformer that we rely on (easy to fake in tests)."""

    def encode(self, sentences: list[str], **kwargs: Any) -> Any: ...


class EmbeddingError(RuntimeError):
    """Raised when the model returns vectors of an unexpected shape."""


def _load_sentence_transformer(model_name: str) -> EncoderModel:
    from sentence_transformers import SentenceTransformer  # heavy import

    logger.info("loading embedding model %s", model_name)
    model: EncoderModel = SentenceTransformer(model_name, device="cpu")
    return model


class Embedder:
    """Turns text into normalised 384-dim vectors."""

    def __init__(
        self,
        model: EncoderModel | None = None,
        model_name: str = MODEL_NAME,
        dim: int = EMBEDDING_DIM,
    ) -> None:
        self._model = model
        self.model_name = model_name
        self.dim = dim

    @property
    def model(self) -> EncoderModel:
        if self._model is None:
            self._model = _load_sentence_transformer(self.model_name)
        return self._model

    def embed_texts(
        self, texts: Sequence[str], batch_size: int = DEFAULT_BATCH_SIZE
    ) -> np.ndarray:
        """Embed many texts. Returns an array of shape (len(texts), dim)."""
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        if any(not t or not t.strip() for t in texts):
            raise ValueError("cannot embed empty text")

        raw = self.model.encode(
            list(texts),
            batch_size=batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        vectors = np.asarray(raw, dtype=np.float32)

        if vectors.ndim != 2 or vectors.shape != (len(texts), self.dim):
            raise EmbeddingError(
                f"expected shape ({len(texts)}, {self.dim}), got {vectors.shape}"
            )
        # Normalise again defensively: cheap, and guarantees cosine == dot.
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        if np.any(norms == 0):
            raise EmbeddingError("model returned a zero vector")
        return vectors / norms

    def embed_query(self, query: str) -> np.ndarray:
        """Embed one search query. Returns a vector of shape (dim,)."""
        return self.embed_texts([query])[0]


@lru_cache(maxsize=1)
def get_embedder() -> Embedder:
    """Process-wide embedder (model loads on first use)."""
    return Embedder()
