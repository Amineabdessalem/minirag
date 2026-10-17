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
