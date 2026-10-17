import math

import pytest

from minirag.chunking import Chunk
from minirag.evaluation import evaluate, ndcg_at_k, recall_at_k, reciprocal_rank
from minirag.retrievers import BM25Retriever


def test_recall_at_k():
    assert recall_at_k(["a", "b", "c"], {"a", "c"}, 2) == 0.5
    assert recall_at_k(["a", "b", "c"], {"a", "c"}, 3) == 1.0
    assert recall_at_k(["a"], set(), 1) == 0.0


def test_reciprocal_rank():
    assert reciprocal_rank(["x", "y", "a"], {"a"}) == pytest.approx(1 / 3)
    assert reciprocal_rank(["x"], {"a"}) == 0.0


def test_ndcg():
    assert ndcg_at_k(["a", "b"], {"a"}, 2) == 1.0
    assert ndcg_at_k(["b", "a"], {"a"}, 2) == pytest.approx(1 / math.log2(3))
    assert ndcg_at_k(["b"], set(), 2) == 0.0


def test_evaluate_collapses_chunks_by_document():
    chunks = [
        Chunk("d1#0", "d1", "redis is an in-memory key value store", 0, 0),
        Chunk("d1#1", "d1", "redis supports pub sub messaging", 0, 0),
        Chunk("d2#0", "d2", "postgres is a relational database", 0, 0),
    ]
    r = BM25Retriever()
    r.index(chunks)
    report = evaluate(r, [("redis key value", {"d1"}), ("relational database", {"d2"})], k=2)
    assert report.queries == 2
    assert report.recall == 1.0
    assert report.mrr == 1.0
    assert "recall@2=1.000" in str(report)
