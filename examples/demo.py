"""Runnable demo for pgvector Example by Viprasol Tech Private Limited.

Runs fully offline using the in-memory cosine store and the deterministic
FakeEmbedder. No database required.

    python examples/demo.py
"""

from __future__ import annotations

from pgvector_example import FakeEmbedder, InMemoryVectorStore

DOCS = [
    ("d1", "Postgres is a powerful open-source relational database."),
    ("d2", "pgvector adds vector similarity search to Postgres."),
    ("d3", "Embeddings map text into a dense numeric vector space."),
    ("d4", "Cosine similarity compares the direction of two vectors."),
    ("d5", "Viprasol Tech builds fintech and AI software."),
]


def main() -> None:
    store = InMemoryVectorStore(FakeEmbedder(dim=64))
    store.upsert_many(DOCS)

    query = "how does vector search work in postgres"
    print(f"Query: {query!r}\n")
    for rank, hit in enumerate(store.query(query, top_k=3), start=1):
        print(f"{rank}. [{hit.score:.4f}] {hit.id}: {hit.content}")


if __name__ == "__main__":
    main()
