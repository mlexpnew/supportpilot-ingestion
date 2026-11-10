"""Tests for the knowledge-base embedder (SP-122).

Most tests use a fake model so they are fast and offline. One test uses the
real pinned model and is skipped if sentence-transformers is not installed.
"""

from __future__ import annotations

import hashlib
from typing import Any

import numpy as np
import pytest

from supportpilot.rag.embedder import (
    EMBEDDING_DIM,
    Embedder,
    EmbeddingError,
)


class FakeModel:
    """Deterministic, un-normalised vectors derived from a hash of the text."""

    def __init__(self, dim: int = EMBEDDING_DIM) -> None:
        self.dim = dim
        self.calls = 0

    def encode(self, sentences: list[str], **kwargs: Any) -> np.ndarray:
        self.calls += 1
        rows = []
        for s in sentences:
            seed = int(hashlib.sha256(s.encode()).hexdigest()[:8], 16)
            rows.append(np.random.default_rng(seed).normal(size=self.dim) * 7.0)
        return np.array(rows)


def test_vectors_have_unit_length() -> None:
    vectors = Embedder(model=FakeModel()).embed_texts(["a refund", "my parcel"])
    assert vectors.shape == (2, EMBEDDING_DIM)
    np.testing.assert_allclose(np.linalg.norm(vectors, axis=1), 1.0, rtol=1e-5)


def test_same_text_gives_same_vector() -> None:
    embedder = Embedder(model=FakeModel())
    first = embedder.embed_query("where is my parcel")
    second = embedder.embed_query("where is my parcel")
    np.testing.assert_array_equal(first, second)


def test_order_is_preserved() -> None:
    embedder = Embedder(model=FakeModel())
    batch = embedder.embed_texts(["one", "two", "three"])
    np.testing.assert_allclose(batch[1], embedder.embed_query("two"), rtol=1e-6)


def test_empty_list_returns_empty_array_without_loading_model() -> None:
    model = FakeModel()
    result = Embedder(model=model).embed_texts([])
    assert result.shape == (0, EMBEDDING_DIM)
    assert model.calls == 0


def test_blank_text_rejected() -> None:
    with pytest.raises(ValueError):
        Embedder(model=FakeModel()).embed_texts(["ok", "   "])


def test_wrong_dimension_raises() -> None:
    with pytest.raises(EmbeddingError, match="expected shape"):
        Embedder(model=FakeModel(dim=10)).embed_texts(["hello"])


def test_zero_vector_raises() -> None:
    class ZeroModel:
        def encode(self, sentences: list[str], **kwargs: Any) -> np.ndarray:
            return np.zeros((len(sentences), EMBEDDING_DIM))

    with pytest.raises(EmbeddingError, match="zero vector"):
        Embedder(model=ZeroModel()).embed_texts(["hello"])


def test_real_model_understands_meaning() -> None:
    pytest.importorskip("sentence_transformers")
    embedder = Embedder()
    query = embedder.embed_query("where is my parcel")
    track, password = embedder.embed_texts(
        ["How to track your order", "How to reset your password"]
    )
    assert embedder.embed_query("x").shape == (EMBEDDING_DIM,)
    # cosine == dot product because vectors are normalised
    assert float(query @ track) > float(query @ password)