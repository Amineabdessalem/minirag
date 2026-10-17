# minirag

[![CI](https://github.com/Amineabdessalem/minirag/actions/workflows/ci.yml/badge.svg)](https://github.com/Amineabdessalem/minirag/actions/workflows/ci.yml)

A small, readable retrieval-augmented generation (RAG) toolkit in Python. It
covers the retrieval half of RAG end to end (chunking, lexical and vector
search, hybrid fusion, prompt building and evaluation) with NumPy as its only
dependency, so every step is easy to read and test.

## Features

- **Sentence-aware chunking** with configurable size and overlap, keeping
  character offsets back to the source document.
- **BM25** lexical retrieval (Okapi, with the non-negative idf variant).
- **Vector retrieval** with cosine similarity. A TF-IDF embedder is built in;
  plug in any object with `embed(texts) -> np.ndarray` to use a real
  embedding model.
- **Hybrid search** via Reciprocal Rank Fusion, with per-retriever weights.
- **Evaluation**: recall@k, MRR and nDCG@k over a labelled JSONL query set.
- **Generator-agnostic pipeline**: pass any `prompt -> text` callable (an LLM
  API client, a local model or a test stub).

## Install

```bash
git clone https://github.com/Amineabdessalem/minirag
cd minirag
pip install -e ".[dev]"
```

## Quick start

```python
from minirag import RAGPipeline

rag = RAGPipeline()
rag.add_directory("examples/docs")

answer = rag.ask("How do I make container images smaller?", k=2)
for hit in answer.hits:
    print(hit.chunk.doc_id, round(hit.score, 4))
print(answer.prompt)
```

To generate an answer, pass a generator. For example with the Anthropic SDK:

```python
import anthropic

client = anthropic.Anthropic()

def claude(prompt: str) -> str:
    msg = client.messages.create(
        model="claude-sonnet-5-5",
        max_tokens=512,
        messages=[{"role": "user", "content": prompt}],
    )
    return msg.content[0].text

rag = RAGPipeline(generator=claude)
```

## CLI

```bash
# Show the top chunks for a question
minirag ask --docs examples/docs "what is the smallest deployable unit in kubernetes"

# Print the full prompt that would be sent to an LLM
minirag ask --docs examples/docs --prompt "why do transformers need positional encodings"

# Compare retrievers on a labelled query set
minirag eval --docs examples/docs --queries examples/queries.jsonl --compare bm25 vector hybrid -k 3
```

On the bundled example set (8 documents, 17 queries):

| retriever | recall@3 | MRR   | nDCG@3 |
|-----------|---------:|------:|-------:|
| bm25      | 1.000    | 0.971 | 0.978  |
| vector    | 1.000    | 0.971 | 0.978  |
| hybrid    | 1.000    | 0.971 | 0.978  |

The example corpus is small enough that all three agree; the evaluation
harness is there so larger corpora and real embedding models can be compared
on equal terms.

## Project layout

```
src/minirag/
  text.py         tokenisation, stopwords and a light stemmer
  chunking.py     sentence-aware chunking with overlap
  retrievers.py   BM25, TF-IDF vector and hybrid (RRF) retrievers
  pipeline.py     ingest -> retrieve -> prompt -> generate
  evaluation.py   recall@k, MRR, nDCG@k
  cli.py          `minirag ask` and `minirag eval`
```

## Roadmap

See [ROADMAP.md](ROADMAP.md).

## License

MIT
