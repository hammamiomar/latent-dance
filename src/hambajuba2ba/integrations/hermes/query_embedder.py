"""Lazy local query embedding for Hermes feature search."""

from __future__ import annotations

from importlib.util import find_spec
import os
from typing import Any

import numpy as np

DEFAULT_QUERY_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


class SentenceTransformerQueryEmbedder:
    """Small local text embedder loaded only when semantic search is used."""

    def __init__(self, model_name: str = DEFAULT_QUERY_EMBEDDING_MODEL) -> None:
        self.model_name = model_name
        self._model: Any | None = None

    def encode(self, text: str) -> np.ndarray:
        model = self._load_model()
        vector = model.encode(
            text,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return np.asarray(vector, dtype=np.float32)

    def _load_model(self) -> Any:
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name)
        return self._model


def create_default_query_embedder(
    model_name: str | None = None,
) -> SentenceTransformerQueryEmbedder | None:
    if os.getenv("HAMBA_FEATURE_SEMANTIC", "").strip().lower() in {
        "0",
        "false",
        "no",
        "off",
    }:
        return None
    if find_spec("sentence_transformers") is None:
        return None
    resolved_model = os.getenv(
        "HAMBA_FEATURE_QUERY_MODEL",
        model_name or DEFAULT_QUERY_EMBEDDING_MODEL,
    )
    return SentenceTransformerQueryEmbedder(model_name=resolved_model)
