# Roadmap

Planned work, roughly in order. Each item should land as its own tested commit.

- [x] Persist and reload an index (`save`/`load`, chunks as JSONL + JSON metadata)
- [ ] Markdown-aware chunking that keeps headings as chunk metadata
- [x] Maximal Marginal Relevance (MMR) re-ranking for more diverse context
- [ ] Optional sentence-transformers embedder behind an extra (`pip install minirag[embeddings]`)
- [ ] Larger benchmark: a public QA dataset subset with a reproducible eval script
- [ ] Query-time metadata filters (e.g. only search `doc_id` prefixes)
- [ ] Simple HTTP API (FastAPI) exposing `/ask` and `/search`
- [ ] Type-checking with mypy and linting with ruff in CI
