# RAG Glossary

A short reference to the retrieval techniques this copilot uses. Replace or extend this folder
with your own company documents, then run `python cli.py ingest`.

## Hybrid search

Hybrid search runs two searches for every question and combines the results. Dense (vector)
search finds passages with a similar meaning, even when the words differ. Keyword search with
BM25 finds passages that contain the exact words of the question, which matters for product
names, error codes and function names. Each search covers the other's blind spot.

## Reciprocal Rank Fusion

Reciprocal Rank Fusion (RRF) merges several ranked result lists. Each list gives a passage a
score of 1 divided by (60 plus its rank), and the scores are added together. A passage that
several searches agree on rises to the top. RRF uses positions only, so scores on different
scales never need to be compared.

## Cross-encoder reranking

A cross-encoder reads the question and one passage together and scores how well they match.
It is more accurate than comparing two separate embeddings but slower, so it only re-scores the
top candidates from the first-stage search. This pattern is called "retrieve wide, rerank narrow".

## Corrective RAG

Corrective RAG (CRAG) grades retrieved passages before answering. Relevant passages are kept and
irrelevant ones are dropped. When the passages are enough, the answer is generated from them.
When they are partial, web results are added. When nothing is relevant, the search is rewritten
and retried before falling back to the web.

## Groundedness check

After an answer is written, a separate check confirms that every claim is supported by the cited
passages. If a claim is not supported, the answer is rewritten once with the unsupported claims
listed. This is the idea behind Self-RAG.
