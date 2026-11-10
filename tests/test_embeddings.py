import sys
import types

import numpy as np
import pytest

from minirag import RAGPipeline, VectorRetriever
from minirag.embeddings import SentenceTransformerEmbedder


class FakeModel:
    """Bag-of-letters 'model' so tests run without downloading weights."""

    def __init__(self, name, device=None):
        self.name = name

    def encode(self, texts, batch_size=32, show_progress_bar=False):
        out = np.zeros((len(texts), 26))
        for i, t in enumerate(texts):
            for ch in t.lower():
                if "a" <= ch <= "z":
                    out[i, ord(ch) - 97] += 1
        return out


@pytest.fixture()
def fake_st(monkeypatch):
    mod = types.ModuleType("sentence_transformers")
    mod.SentenceTransformer = FakeModel
    monkeypatch.setitem(sys.modules, "sentence_transformers", mod)


def test_rows_are_unit_length(fake_st):
    vecs = SentenceTransformerEmbedder().embed(["hello world", "zzz", "123"])
    norms = np.linalg.norm(vecs, axis=1)
    assert np.allclose(norms[:2], 1) and norms[2] == 0


def test_model_loads_once_and_lazily(fake_st):
    emb = SentenceTransformerEmbedder("my/model")
    assert emb._model is None
    emb.embed(["a"])
    first = emb._model
    emb.embed(["b"])
    assert emb._model is first and first.name == "my/model"


def test_plugs_into_vector_retriever(fake_st):
    rag = RAGPipeline(retriever=VectorRetriever(embedder=SentenceTransformerEmbedder()))
    rag.add_documents([("a.txt", "zebra zoo zigzag"), ("b.txt", "apple banana")])
    assert rag.search("zz zebra", k=1)[0].chunk.doc_id == "a.txt"


def test_missing_extra_gives_install_hint(monkeypatch):
    monkeypatch.setitem(sys.modules, "sentence_transformers", None)
    with pytest.raises(ImportError, match=r"minirag\[embeddings\]"):
        SentenceTransformerEmbedder().embed(["x"])
