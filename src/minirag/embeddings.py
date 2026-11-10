"""Optional neural embeddings via sentence-transformers.

Install with ``pip install "minirag[embeddings]"``. The model is loaded on
first use, so importing this module stays cheap and dependency-free.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


class SentenceTransformerEmbedder:
    """Embeds text with a sentence-transformers model, returning unit-length rows.

    Unit-length rows make the dot products in :class:`~minirag.VectorRetriever`
    equal to cosine similarity.
    """

    def __init__(self, model_name: str = DEFAULT_MODEL, batch_size: int = 32, device: str | None = None):
        self.model_name = model_name
        self.batch_size = batch_size
        self.device = device
        self._model: Any = None

    def _load(self) -> Any:
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as e:
                raise ImportError(
                    'SentenceTransformerEmbedder needs the "embeddings" extra: '
                    'pip install "minirag[embeddings]"'
                ) from e
            self._model = SentenceTransformer(self.model_name, device=self.device)
        return self._model

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, 0))
        vectors = np.asarray(
            self._load().encode(list(texts), batch_size=self.batch_size, show_progress_bar=False),
            dtype=float,
        )
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        return vectors / np.where(norms == 0, 1, norms)
