"""minirag: a small, dependency-light retrieval-augmented generation toolkit."""

from minirag.chunking import Chunk, chunk_text
from minirag.pipeline import Answer, RAGPipeline
from minirag.retrievers import (
    BM25Retriever,
    Hit,
    HybridRetriever,
    MMRRetriever,
    TfidfRetriever,
    VectorRetriever,
)

__all__ = [
    "Answer",
    "BM25Retriever",
    "Chunk",
    "Hit",
    "HybridRetriever",
    "MMRRetriever",
    "RAGPipeline",
    "TfidfRetriever",
    "VectorRetriever",
    "chunk_text",
]
__version__ = "0.1.0"
