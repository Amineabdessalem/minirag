# Transformers

The transformer architecture replaces recurrence with self-attention. Each token produces query, key and value vectors; attention weights come from the softmax of scaled query-key dot products.

Multi-head attention runs several attention operations in parallel so the model can attend to different relationships at once. Positional encodings inject word order, since attention itself is permutation invariant.

Decoder-only transformers such as GPT-style language models are trained to predict the next token. The KV cache stores keys and values from earlier tokens so generation does not recompute them.
