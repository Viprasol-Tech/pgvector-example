"""Part of pgvector Example by Viprasol Tech Private Limited (https://viprasol.com).

A practical Postgres ``pgvector`` example: schema/DDL, pluggable embedders, and a
vector store with a real psycopg/pgvector backend plus an in-memory cosine
fallback so the full test suite runs offline with no database.
"""

from __future__ import annotations

from pgvector_example.embedder import (
    Embedder,
    FakeEmbedder,
    OpenAIEmbedder,
    SentenceTransformerEmbedder,
)
from pgvector_example.schema import (
    DEFAULT_DIM,
    DEFAULT_TABLE,
    create_index_sql,
    create_table_sql,
    enable_extension_sql,
)
from pgvector_example.store import (
    InMemoryVectorStore,
    PostgresVectorStore,
    SearchResult,
    VectorStore,
)

__version__ = "0.1.0"

__all__ = [
    "DEFAULT_DIM",
    "DEFAULT_TABLE",
    "Embedder",
    "FakeEmbedder",
    "InMemoryVectorStore",
    "OpenAIEmbedder",
    "PostgresVectorStore",
    "SearchResult",
    "SentenceTransformerEmbedder",
    "VectorStore",
    "__version__",
    "create_index_sql",
    "create_table_sql",
    "enable_extension_sql",
]
