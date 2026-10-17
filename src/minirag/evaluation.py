"""Retrieval metrics for comparing retrievers on a labelled query set."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Sequence

from minirag.retrievers import Retriever


def recall_at_k(retrieved: Sequence[str], relevant: set[str], k: int) -> float:
    """Fraction of relevant ids that appear in the first ``k`` results."""
    if not relevant:
        return 0.0
    return len(set(retrieved[:k]) & relevant) / len(relevant)


def reciprocal_rank(retrieved: Sequence[str], relevant: set[str]) -> float:
    """1 / rank of the first relevant result, or 0 if none was retrieved."""
    for rank, rid in enumerate(retrieved, start=1):
        if rid in relevant:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(retrieved: Sequence[str], relevant: set[str], k: int) -> float:
    """Binary-relevance normalised discounted cumulative gain."""
    dcg = sum(1.0 / math.log2(i + 2) for i, rid in enumerate(retrieved[:k]) if rid in relevant)
    ideal = sum(1.0 / math.log2(i + 2) for i in range(min(len(relevant), k)))
    return dcg / ideal if ideal else 0.0


@dataclass(frozen=True)
class EvalReport:
    queries: int
    recall: float
    mrr: float
    ndcg: float
    k: int

    def __str__(self) -> str:
        return (
            f"queries={self.queries}  recall@{self.k}={self.recall:.3f}  "
            f"mrr={self.mrr:.3f}  ndcg@{self.k}={self.ndcg:.3f}"
        )


def evaluate(
    retriever: Retriever,
    labelled: Iterable[tuple[str, set[str]]],
    k: int = 5,
    key: str = "doc_id",
) -> EvalReport:
    """Score ``retriever`` on ``(query, relevant_ids)`` pairs.

    ``key`` picks which chunk attribute is compared with the relevant ids:
    ``"doc_id"`` (the default) or ``"id"`` for chunk-level labels. When
    comparing by document, repeated chunks of the same document are collapsed
    so each document is counted once at its best rank.
    """
    recalls, rrs, ndcgs = [], [], []
    for query, relevant in labelled:
        ids: list[str] = []
        for hit in retriever.search(query, k=k * 3 if key == "doc_id" else k):
            rid = getattr(hit.chunk, key)
            if rid not in ids:
                ids.append(rid)
        recalls.append(recall_at_k(ids, relevant, k))
        rrs.append(reciprocal_rank(ids[:k], relevant))
        ndcgs.append(ndcg_at_k(ids, relevant, k))
    n = len(recalls)
    mean = lambda xs: sum(xs) / n if n else 0.0  # noqa: E731
    return EvalReport(queries=n, recall=mean(recalls), mrr=mean(rrs), ndcg=mean(ndcgs), k=k)
