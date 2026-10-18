"""End-to-end pipeline: ingest documents, retrieve context, build a prompt."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
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

INDEX_FORMAT_VERSION = 1


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

    def save(self, path: str | Path) -> None:
        """Write the chunks and chunking settings to ``path`` (a directory).

        Only the chunks are stored; retriever indexes are rebuilt on load,
        which keeps the format independent of the retriever in use and is fast
        for the BM25 and TF-IDF retrievers.
        """
        out = Path(path)
        out.mkdir(parents=True, exist_ok=True)
        meta = {
            "format_version": INDEX_FORMAT_VERSION,
            "max_words": self.max_words,
            "overlap_words": self.overlap_words,
            "chunks": len(self.chunks),
        }
        (out / "meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
        with (out / "chunks.jsonl").open("w", encoding="utf-8") as f:
            for chunk in self.chunks:
                f.write(json.dumps(asdict(chunk), ensure_ascii=False) + "\n")

    @classmethod
    def load(
        cls,
        path: str | Path,
        retriever: Retriever | None = None,
        generator: Generator | None = None,
    ) -> "RAGPipeline":
        """Rebuild a pipeline saved with :meth:`save`."""
        root = Path(path)
        meta = json.loads((root / "meta.json").read_text(encoding="utf-8"))
        if meta.get("format_version") != INDEX_FORMAT_VERSION:
            raise ValueError(f"unsupported index format: {meta.get('format_version')!r}")
        with (root / "chunks.jsonl").open(encoding="utf-8") as f:
            chunks = [Chunk(**json.loads(line)) for line in f if line.strip()]
        kwargs = {"retriever": retriever} if retriever is not None else {}
        rag = cls(
            generator=generator,
            max_words=meta["max_words"],
            overlap_words=meta["overlap_words"],
            chunks=chunks,
            **kwargs,
        )
        rag.retriever.index(rag.chunks)
        return rag
