"""End-to-end pipeline: ingest documents, retrieve context, build a prompt."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable

from minirag.chunking import Chunk, chunk_text
from minirag.retrievers import BM25Retriever, Hit, HybridRetriever, Retriever, VectorRetriever

PROMPT_TEMPLATE = """Answer the question using only the context below. \
If the context does not contain the answer, say you don't know.
Cite sources as [n].

Context:
{context}

Question: {question}
Answer:"""

Generator = Callable[[str], str]


@dataclass
class Answer:
    question: str
    prompt: str
    hits: list[Hit]
    text: str | None = None


@dataclass
class RAGPipeline:
    """Glue between chunking, retrieval and an optional text generator.

    ``generator`` is any callable that maps a prompt to a completion, so the
    pipeline works with a hosted LLM API, a local model or a test stub.
    """

    retriever: Retriever = field(
        default_factory=lambda: HybridRetriever([BM25Retriever(), VectorRetriever()])
    )
    generator: Generator | None = None
    max_words: int = 120
    overlap_words: int = 20
    chunks: list[Chunk] = field(default_factory=list)

    def add_documents(self, docs: Iterable[tuple[str, str]]) -> int:
        """Chunk ``(doc_id, text)`` pairs and rebuild the index. Returns the chunk count."""
        for doc_id, text in docs:
            self.chunks.extend(
                chunk_text(text, doc_id, max_words=self.max_words, overlap_words=self.overlap_words)
            )
        self.retriever.index(self.chunks)
        return len(self.chunks)

    def add_directory(self, path: str | Path, pattern: str = "**/*.md") -> int:
        """Ingest every file under ``path`` matching ``pattern`` (Markdown by default)."""
        root = Path(path)
        files = sorted(p for p in root.glob(pattern) if p.is_file())
        return self.add_documents((str(p.relative_to(root)), p.read_text(encoding="utf-8")) for p in files)

    def build_prompt(self, question: str, hits: list[Hit]) -> str:
        context = "\n\n".join(f"[{i}] ({h.chunk.doc_id}) {h.chunk.text}" for i, h in enumerate(hits, 1))
        return PROMPT_TEMPLATE.format(context=context or "(no context found)", question=question)

    def ask(self, question: str, k: int = 4) -> Answer:
        hits = self.retriever.search(question, k=k)
        prompt = self.build_prompt(question, hits)
        text = self.generator(prompt) if self.generator else None
        return Answer(question=question, prompt=prompt, hits=hits, text=text)
