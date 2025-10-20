# Docs2DB-API - RAG Server For a Docs2DB Database

Docs2DB-API is an RAG server for use with Docs2DB databases.

See https://github.com/rhel-lightspeed/docs2db

See `demos/llama-stack/README.md`

Summary:
* Gather a folder of your source data documents
* Run Docs2DB on that folder to get a database
* Import Docs2DB-API in your LLM applications to use your source data in RAG.

## RAG Algorithm

Docs2DB-API implements modern retrieval techniques:

- Contextual chunks with document-level context prepended to each chunk
- Hybrid search combining BM25 (lexical) and vector embeddings (semantic)
- Reciprocal Rank Fusion (RRF) for result combination
- Cross-encoder reranker for improved result quality
- PostgreSQL full-text search with tsvector and GIN indexing
- pgvector for similarity search
- Granite embedding models (30M parameters, 384 dimensions)
- Normalized model metadata storage
- Question refinement for improved query expansion
- Universal RAG engine adaptable to multiple API frameworks