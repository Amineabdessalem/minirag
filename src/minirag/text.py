"""Text normalisation and tokenisation shared by every retriever."""

from __future__ import annotations

import re

_TOKEN_RE = re.compile(r"[a-z0-9]+")

# A deliberately small English stopword list: enough to keep scores from being
# dominated by glue words, small enough that queries like "how to" still work.
STOPWORDS = frozenset(
    """
    a an and are as at be but by for from has have i in is it its of on or
    that the this to was were will with
    """.split()
)


def tokenize(text: str, *, drop_stopwords: bool = True) -> list[str]:
    """Lowercase ``text`` and split it into alphanumeric tokens."""
    tokens = _TOKEN_RE.findall(text.lower())
    if drop_stopwords:
        tokens = [t for t in tokens if t not in STOPWORDS]
    return tokens
