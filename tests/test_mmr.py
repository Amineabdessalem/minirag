import pytest

from minirag.chunking import Chunk
from minirag.retrievers import BM25Retriever, MMRRetriever

TEXTS = {
    "a1": "Redis is an in-memory key value store used as a cache.",
    "a2": "Redis is an in-memory key value store, often used as a cache.",
    "b": "Memcached is another in-memory cache with a simple key value API.",
    "c": "Postgres is a relational database with strong consistency.",
}


@pytest.fixture
def chunks():
    return [Chunk(id=k, doc_id=k, text=t, start=0, end=len(t)) for k, t in TEXTS.items()]


def test_mmr_skips_near_duplicates(chunks):
    plain = BM25Retriever()
    plain.index(chunks)
    # Without MMR the two near-identical Redis chunks fill both slots.
    assert {h.chunk.id for h in plain.search("redis in-memory key value cache", k=2)} == {"a1", "a2"}

    mmr = MMRRetriever(BM25Retriever(), lambda_=0.5)
    mmr.index(chunks)
    ids = [h.chunk.id for h in mmr.search("redis in-memory key value cache", k=2)]
    assert ids[0] in {"a1", "a2"}
    assert ids[1] == "b"


def test_lambda_one_keeps_relevance_order(chunks):
    mmr = MMRRetriever(BM25Retriever(), lambda_=1.0)
    mmr.index(chunks)
    hits = mmr.search("redis in-memory key value cache", k=3)
    assert [h.score for h in hits] == sorted((h.score for h in hits), reverse=True)


def test_empty_results_and_validation(chunks):
    mmr = MMRRetriever(BM25Retriever())
    mmr.index(chunks)
    assert mmr.search("zebra pancake") == []
    with pytest.raises(ValueError):
        MMRRetriever(BM25Retriever(), lambda_=1.5)
