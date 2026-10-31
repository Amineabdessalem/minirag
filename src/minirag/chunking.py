"""Split documents into overlapping chunks suitable for retrieval."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")


@dataclass(frozen=True)
class Chunk:
    """A contiguous piece of a source document."""

    id: str
    doc_id: str
    text: str
    start: int
    end: int
    metadata: dict = field(default_factory=dict, compare=False)


def split_sentences(text: str) -> list[str]:
    """Split text on sentence-ending punctuation, dropping empty pieces."""
    return [s.strip() for s in _SENTENCE_RE.split(text) if s.strip()]


def chunk_text(
    text: str,
    doc_id: str,
    *,
    max_words: int = 120,
    overlap_words: int = 20,
    metadata: dict | None = None,
) -> list[Chunk]:
    """Pack whole sentences into chunks of at most ``max_words`` words.

    Consecutive chunks share roughly ``overlap_words`` words so that an answer
    straddling a boundary is still retrievable. A single sentence longer than
    ``max_words`` is split on word boundaries.
    """
    if max_words <= 0:
        raise ValueError("max_words must be positive")
    if not 0 <= overlap_words < max_words:
        raise ValueError("overlap_words must be in [0, max_words)")

    words: list[tuple[str, int, int]] = []  # (word, start offset, end offset)
    sentence_ends: set[int] = set()  # word indices that end a sentence
    for m in re.finditer(r"\S+", text):
        words.append((m.group(), m.start(), m.end()))
        if re.search(r"[.!?][\"')\]]*$", m.group()):
            sentence_ends.add(len(words) - 1)
    if not words:
        return []
    sentence_ends.add(len(words) - 1)

    chunks: list[Chunk] = []
    start = 0
    while start < len(words):
        hard_end = min(start + max_words, len(words)) - 1
        # Prefer to stop at the last sentence boundary inside the window.
        end = max((i for i in sentence_ends if start <= i <= hard_end), default=hard_end)
        chunk_words = words[start : end + 1]
        chunks.append(
            Chunk(
                id=f"{doc_id}#{len(chunks)}",
                doc_id=doc_id,
                text=text[chunk_words[0][1] : chunk_words[-1][2]],
                start=chunk_words[0][1],
                end=chunk_words[-1][2],
                metadata=dict(metadata or {}),
            )
        )
        if end == len(words) - 1:
            break
        start = max(end + 1 - overlap_words, start + 1)
    return chunks


_HEADING_RE = re.compile(r"^(#{1,6})[ \t]+(.+?)[ \t#]*$", re.MULTILINE)


def chunk_markdown(
    text: str,
    doc_id: str,
    *,
    max_words: int = 120,
    overlap_words: int = 20,
    metadata: dict | None = None,
) -> list[Chunk]:
    """Chunk Markdown section by section so no chunk spans two headings.

    Each chunk gets ``metadata["heading"]``, the path of headings above it
    (e.g. ``"Install > From source"``), and offsets into the original text.
    Headings inside fenced code blocks are ignored.
    """
    fences = [(m.start(), m.end()) for m in re.finditer(r"^```.*?^```", text, re.MULTILINE | re.DOTALL)]
    headings = [m for m in _HEADING_RE.finditer(text) if not any(s <= m.start() < e for s, e in fences)]

    sections: list[tuple[int, int, str]] = []  # (body start, body end, heading path)
    path: list[tuple[int, str]] = []
    prev_end, prev_path = 0, ""
    for m in headings:
        sections.append((prev_end, m.start(), prev_path))
        level = len(m.group(1))
        path = [p for p in path if p[0] < level] + [(level, m.group(2).strip())]
        prev_end, prev_path = m.end(), " > ".join(title for _, title in path)
    sections.append((prev_end, len(text), prev_path))

    chunks: list[Chunk] = []
    for start, end, heading in sections:
        meta = dict(metadata or {})
        if heading:
            meta["heading"] = heading
        for c in chunk_text(text[start:end], doc_id, max_words=max_words,
                            overlap_words=overlap_words, metadata=meta):
            chunks.append(Chunk(id=f"{doc_id}#{len(chunks)}", doc_id=doc_id, text=c.text,
                                start=start + c.start, end=start + c.end, metadata=c.metadata))
    return chunks
