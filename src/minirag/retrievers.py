"""Retrievers: lexical (BM25), vector (TF-IDF cosine) and hybrid fusion."""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass
from typing import Protocol, Sequence

import numpy as np

from minirag.chunking import Chunk
from minirag.text import tokenize


@dataclass(frozen=True)
class Hit:
    """A retrieved chunk and its retriever-specific score."""

    chunk: Chunk
    score: float


class Retriever(Protocol):
    def index(self, chunks: Sequence[Chunk]) -> None: ...

    def search(self, query: str, k: int = 5) -> list[Hit]: ...


def _top_k(scores: np.ndarray, chunks: Sequence[Chunk], k: int) -> list[Hit]:
    if k <= 0 or len(chunks) == 0:
        return []
    k = min(k, len(chunks))
    idx = np.argpartition(-scores, k - 1)[:k]
    # Sort by score, breaking ties by corpus order so results are deterministic.
    idx = sorted(idx, key=lambda i: (-scores[i], i))
    return [Hit(chunks[i], float(scores[i])) for i in idx if scores[i] > 0]


class BM25Retriever:
    """Okapi BM25 over tokenised chunks.

    ``k1`` controls term-frequency saturation and ``b`` the strength of
    document-length normalisation; the defaults are the usual 1.5 and 0.75.
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self._chunks: list[Chunk] = []
        self._tfs: list[Counter[str]] = []
        self._lengths = np.zeros(0)
        self._idf: dict[str, float] = {}

    def index(self, chunks: Sequence[Chunk]) -> None:
        self._chunks = list(chunks)
        self._tfs = [Counter(tokenize(c.text)) for c in self._chunks]
        self._lengths = np.array([sum(tf.values()) for tf in self._tfs], dtype=float)
        n = len(self._chunks)
        df: Counter[str] = Counter()
        for tf in self._tfs:
            df.update(tf.keys())
        # The "+1" variant of BM25 idf keeps scores non-negative for common terms.
        self._idf = {t: math.log(1 + (n - d + 0.5) / (d + 0.5)) for t, d in df.items()}

    def search(self, query: str, k: int = 5) -> list[Hit]:
        if not self._chunks:
            return []
        avg_len = self._lengths.mean() or 1.0
        norm = self.k1 * (1 - self.b + self.b * self._lengths / avg_len)
        scores = np.zeros(len(self._chunks))
        for term in set(tokenize(query)):
            idf = self._idf.get(term)
            if idf is None:
                continue
            tf = np.array([d.get(term, 0) for d in self._tfs], dtype=float)
            scores += idf * tf * (self.k1 + 1) / (tf + norm)
        return _top_k(scores, self._chunks, k)


class TfidfRetriever:
    """Cosine similarity between L2-normalised TF-IDF vectors.

    This is a stand-in for a dense embedding model that needs no downloads;
    any object with an ``embed(list[str]) -> np.ndarray`` method can be plugged
    into :class:`VectorRetriever` instead.
    """

    def __init__(self, sublinear_tf: bool = True) -> None:
        self.sublinear_tf = sublinear_tf
        self.vocab: dict[str, int] = {}
        self.idf = np.zeros(0)

    def fit(self, texts: Sequence[str]) -> "TfidfRetriever":
        docs = [set(tokenize(t)) for t in texts]
        self.vocab = {t: i for i, t in enumerate(sorted(set().union(*docs)))}
        df = np.zeros(len(self.vocab))
        for d in docs:
            for t in d:
                df[self.vocab[t]] += 1
        self.idf = np.log((1 + len(docs)) / (1 + df)) + 1
        return self

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        out = np.zeros((len(texts), len(self.vocab)))
        for row, text in enumerate(texts):
            for term, count in Counter(tokenize(text)).items():
                col = self.vocab.get(term)
                if col is not None:
                    out[row, col] = 1 + math.log(count) if self.sublinear_tf else count
        out *= self.idf
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        return np.divide(out, norms, out=np.zeros_like(out), where=norms > 0)


class Embedder(Protocol):
    def embed(self, texts: Sequence[str]) -> np.ndarray: ...


class VectorRetriever:
    """Brute-force cosine search over embeddings produced by ``embedder``.

    If no embedder is given, a :class:`TfidfRetriever` is fitted on the corpus.
    """

    def __init__(self, embedder: Embedder | None = None) -> None:
        self.embedder = embedder
        self._chunks: list[Chunk] = []
        self._matrix = np.zeros((0, 0))

    def index(self, chunks: Sequence[Chunk]) -> None:
        self._chunks = list(chunks)
        texts = [c.text for c in self._chunks]
        if self.embedder is None:
            self.embedder = TfidfRetriever().fit(texts)
        self._matrix = self.embedder.embed(texts)

    def search(self, query: str, k: int = 5) -> list[Hit]:
        if not self._chunks:
            return []
        q = self.embedder.embed([query])[0]
        return _top_k(self._matrix @ q, self._chunks, k)
