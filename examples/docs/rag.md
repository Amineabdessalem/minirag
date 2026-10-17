# Retrieval-augmented generation

Retrieval-augmented generation (RAG) grounds a language model in external documents. Documents are split into chunks, indexed, and the most relevant chunks are placed into the prompt at query time.

Lexical retrieval such as BM25 matches exact terms and handles rare names and identifiers well. Dense retrieval embeds text into vectors and finds semantically similar passages even without shared words. Hybrid search merges both, commonly with reciprocal rank fusion.

Retrieval quality is measured with recall at k, mean reciprocal rank and nDCG on a labelled set of queries.
