# Vector databases

Vector databases store embeddings and answer nearest-neighbour queries. Exact search compares the query with every vector, which is accurate but slow for large collections.

Approximate nearest neighbour indexes trade a little recall for large speedups. HNSW builds a layered proximity graph, IVF clusters vectors and searches only the closest lists, and product quantisation compresses vectors to save memory.
