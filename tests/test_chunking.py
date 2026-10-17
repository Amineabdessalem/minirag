import pytest

from minirag.chunking import chunk_text, split_sentences
from minirag.text import stem, tokenize


def test_tokenize_lowercases_and_drops_stopwords():
    assert tokenize("The Cat is on the MAT!") == ["cat", "mat"]
    assert tokenize("The cat", drop_stopwords=False) == ["the", "cat"]


def test_split_sentences():
    assert split_sentences("One. Two!  Three?") == ["One.", "Two!", "Three?"]


def test_short_text_is_one_chunk():
    chunks = chunk_text("Hello world. Goodbye.", "doc")
    assert len(chunks) == 1
    assert chunks[0].text == "Hello world. Goodbye."
    assert chunks[0].id == "doc#0"


def test_empty_text_has_no_chunks():
    assert chunk_text("   ", "doc") == []


def test_chunks_respect_max_words_and_offsets():
    text = " ".join(f"Sentence number {i} is here." for i in range(50))
    chunks = chunk_text(text, "d", max_words=30, overlap_words=5)
    assert len(chunks) > 1
    for c in chunks:
        assert len(c.text.split()) <= 30
        assert text[c.start : c.end] == c.text
    # Chunks end on sentence boundaries when one is available.
    assert all(c.text.endswith(".") for c in chunks)


def test_chunks_overlap():
    text = " ".join(f"w{i}." for i in range(100))
    chunks = chunk_text(text, "d", max_words=20, overlap_words=5)
    for a, b in zip(chunks, chunks[1:]):
        assert b.start < a.end


def test_long_sentence_is_split():
    text = " ".join(["word"] * 250)
    chunks = chunk_text(text, "d", max_words=100, overlap_words=0)
    assert [len(c.text.split()) for c in chunks] == [100, 100, 50]


@pytest.mark.parametrize("max_words,overlap", [(0, 0), (10, 10), (10, -1)])
def test_invalid_parameters(max_words, overlap):
    with pytest.raises(ValueError):
        chunk_text("text", "d", max_words=max_words, overlap_words=overlap)


@pytest.mark.parametrize(
    "a,b",
    [
        ("containers", "container"),
        ("scheduled", "schedule"),
        ("images", "image"),
        ("queries", "query"),
        ("running", "run"),
        ("boxes", "box"),
    ],
)
def test_stem_maps_inflections_together(a, b):
    assert stem(a) == stem(b)


@pytest.mark.parametrize("word", ["class", "status", "is", "bed", "2026"])
def test_stem_leaves_short_and_invariant_words(word):
    assert stem(word) == word
