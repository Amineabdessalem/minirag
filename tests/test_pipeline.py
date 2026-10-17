import pytest

from minirag.chunking import Chunk
from minirag.pipeline import RAGPipeline
from minirag.retrievers import BM25Retriever, HybridRetriever, VectorRetriever

DOCS = [
    ("k8s.md", "Kubernetes schedules pods onto nodes. A pod is the smallest deployable unit."),
    ("ml.md", "Dropout randomly zeroes activations during training to reduce overfitting."),
    ("net.md", "TCP provides reliable, ordered delivery of a stream of bytes."),
]


def test_hybrid_fuses_and_ranks():
    chunks = [Chunk(id=d, doc_id=d, text=t, start=0, end=len(t)) for d, t in DOCS]
    r = HybridRetriever([BM25Retriever(), VectorRetriever()])
    r.index(chunks)
    hits = r.search("what is a pod in kubernetes", k=2)
    assert hits[0].chunk.id == "k8s.md"
    # A chunk ranked first by both retrievers gets the maximum fused score.
    assert hits[0].score == pytest.approx(2 / 61)


def test_hybrid_validates_arguments():
    with pytest.raises(ValueError):
        HybridRetriever([])
    with pytest.raises(ValueError):
        HybridRetriever([BM25Retriever()], weights=[1.0, 2.0])


def test_pipeline_builds_cited_prompt_and_calls_generator():
    seen = {}

    def fake_llm(prompt):
        seen["prompt"] = prompt
        return "Dropout reduces overfitting [1]."

    rag = RAGPipeline(generator=fake_llm)
    assert rag.add_documents(DOCS) == 3
    answer = rag.ask("how does dropout help training?", k=2)
    assert answer.hits[0].chunk.doc_id == "ml.md"
    assert "[1] (ml.md) Dropout" in answer.prompt
    assert answer.prompt == seen["prompt"]
    assert answer.text == "Dropout reduces overfitting [1]."


def test_pipeline_without_generator_returns_prompt_only():
    rag = RAGPipeline()
    rag.add_documents(DOCS)
    answer = rag.ask("zebra pancake")
    assert answer.text is None
    assert "(no context found)" in answer.prompt


def test_add_directory(tmp_path):
    (tmp_path / "a.md").write_text("Rust has an ownership model for memory safety.")
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "b.md").write_text("Go uses goroutines for concurrency.")
    (tmp_path / "ignored.txt").write_text("not markdown")
    rag = RAGPipeline()
    assert rag.add_directory(tmp_path) == 2
    assert rag.ask("goroutines concurrency").hits[0].chunk.doc_id == "sub/b.md"
