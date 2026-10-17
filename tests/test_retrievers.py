import numpy as np
import pytest

from minirag.chunking import Chunk
from minirag.retrievers import BM25Retriever, TfidfRetriever, VectorRetriever

CORPUS = [
    "Kubernetes schedules containers onto nodes in a cluster.",
    "Gradient descent minimises a loss function by following its gradient.",
    "Transformers use self-attention to model relationships between tokens.",
    "A load balancer spreads incoming requests across several servers.",
    "Overfitting happens when a model memorises training data instead of generalising.",
]


@pytest.fixture
def chunks():
    return [Chunk(id=f"c{i}", doc_id=f"d{i}", text=t, start=0, end=len(t)) for i, t in enumerate(CORPUS)]


@pytest.mark.parametrize("retriever_cls", [BM25Retriever, VectorRetriever])
@pytest.mark.parametrize(
    "query,expected",
    [
        ("how are containers scheduled in kubernetes", "c0"),
        ("gradient of the loss", "c1"),
        ("self-attention transformers", "c2"),
        ("spread requests across servers", "c3"),
        ("model memorises training data", "c4"),
    ],
)
def test_top_hit(retriever_cls, chunks, query, expected):
    r = retriever_cls()
    r.index(chunks)
    hits = r.search(query, k=3)
    assert hits[0].chunk.id == expected
    assert [h.score for h in hits] == sorted((h.score for h in hits), reverse=True)


@pytest.mark.parametrize("retriever_cls", [BM25Retriever, VectorRetriever])
def test_no_overlap_returns_nothing(retriever_cls, chunks):
    r = retriever_cls()
    r.index(chunks)
    assert r.search("zebra pancake", k=3) == []


@pytest.mark.parametrize("retriever_cls", [BM25Retriever, VectorRetriever])
def test_empty_index(retriever_cls):
    r = retriever_cls()
    r.index([])
    assert r.search("anything") == []


def test_k_is_capped_by_corpus_size(chunks):
    r = BM25Retriever()
    r.index(chunks)
    assert len(r.search("a model of servers and nodes", k=50)) <= len(chunks)


def test_tfidf_embeddings_are_unit_length():
    emb = TfidfRetriever().fit(CORPUS).embed(CORPUS + ["unknown words only"])
    norms = np.linalg.norm(emb, axis=1)
    assert np.allclose(norms[:-1], 1.0)
    assert norms[-1] == 0.0


def test_custom_embedder_is_used(chunks):
    class Fixed:
        def embed(self, texts):
            # Every text maps to the same direction except the query, which
            # points at the third chunk's one-hot vector.
            out = np.eye(len(CORPUS))[: len(texts)]
            return out if len(texts) > 1 else np.eye(len(CORPUS))[[2]]

    r = VectorRetriever(embedder=Fixed())
    r.index(chunks)
    assert r.search("whatever", k=1)[0].chunk.id == "c2"
