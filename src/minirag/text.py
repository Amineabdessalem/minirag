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


# Suffixes stripped by :func:`stem`, longest first. Each needs a stem of at
# least three characters left behind so short words ("is", "bed") survive.
_SUFFIXES = ("ations", "ation", "ings", "ing", "ers", "est", "ies", "ed", "er", "es", "ly", "s")


def stem(token: str) -> str:
    """Strip common English inflections so "containers" matches "container".

    This is a deliberately light stemmer: it only needs to map inflected forms
    onto a shared key, not to produce dictionary words.
    """
    if token.isdigit():
        return token
    for suffix in _SUFFIXES:
        if token.endswith(suffix) and len(token) - len(suffix) >= 3:
            base = token[: -len(suffix)]
            if suffix == "ies":
                return base + "y"
            if suffix == "es" and not base.endswith(("s", "x", "z", "ch", "sh")):
                base = token[:-1]  # "images" -> "image", but "boxes" -> "box"
            elif suffix == "s" and base.endswith(("s", "u", "i")):
                return token  # "class", "status", "analysis"
            # Collapse a doubled consonant the suffix left behind ("running" -> "run").
            if len(base) > 3 and base[-1] == base[-2] and base[-1] not in "lsz":
                base = base[:-1]
            return _drop_final_e(base)
    return _drop_final_e(token)


def _drop_final_e(token: str) -> str:
    # "schedule" and "scheduled" should share a key.
    return token[:-1] if len(token) > 4 and token.endswith("e") else token


def tokenize(text: str, *, drop_stopwords: bool = True, stemming: bool = True) -> list[str]:
    """Lowercase ``text``, split it into alphanumeric tokens and stem them."""
    tokens = _TOKEN_RE.findall(text.lower())
    if drop_stopwords:
        tokens = [t for t in tokens if t not in STOPWORDS]
    if stemming:
        tokens = [stem(t) for t in tokens]
    return tokens
