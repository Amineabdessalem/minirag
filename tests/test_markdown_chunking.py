from minirag import RAGPipeline, chunk_markdown

DOC = """Intro paragraph before any heading.

# Install

Use pip to install the package.

## From source

Clone the repository and run pip install -e.

```bash
# not a heading
make build
```

# Usage

Call ask with a question.
"""


def test_sections_and_heading_paths():
    chunks = chunk_markdown(DOC, "guide.md")
    headings = [c.metadata.get("heading") for c in chunks]
    assert headings == [None, "Install", "Install > From source", "Usage"]
    assert "make build" in chunks[2].text and "# not a heading" in chunks[2].text


def test_offsets_point_into_original_text():
    for c in chunk_markdown(DOC, "guide.md"):
        assert DOC[c.start:c.end] == c.text


def test_ids_are_sequential_and_metadata_merged():
    chunks = chunk_markdown(DOC, "g.md", metadata={"lang": "en"})
    assert [c.id for c in chunks] == [f"g.md#{i}" for i in range(len(chunks))]
    assert all(c.metadata["lang"] == "en" for c in chunks)


def test_pipeline_uses_markdown_chunker_for_md_files():
    p = RAGPipeline()
    p.add_documents([("guide.md", DOC), ("notes.txt", "# Not markdown here. Plain text.")])
    md = [c for c in p.chunks if c.doc_id == "guide.md"]
    assert md[-1].metadata["heading"] == "Usage"
    assert "heading" not in [c for c in p.chunks if c.doc_id == "notes.txt"][0].metadata
