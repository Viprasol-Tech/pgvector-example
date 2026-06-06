"""Part of pgvector Example by Viprasol Tech Private Limited (https://viprasol.com).

Command-line demo. By default it runs fully offline using the in-memory store
and the deterministic :class:`FakeEmbedder`, seeds a small corpus, and answers a
query. Pass ``--database-url`` to instead exercise the real pgvector backend.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence

from pgvector_example.embedder import FakeEmbedder
from pgvector_example.store import (
    InMemoryVectorStore,
    PostgresVectorStore,
    VectorStore,
)

_CORPUS: list[tuple[str, str]] = [
    ("d1", "Postgres is a powerful open-source relational database."),
    ("d2", "pgvector adds vector similarity search to Postgres."),
    ("d3", "Cosine similarity measures the angle between two vectors."),
    ("d4", "Embeddings map text into a dense numeric vector space."),
    ("d5", "Viprasol Tech builds fintech and AI software."),
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pgvector-example",
        description="Offline-friendly pgvector similarity-search demo.",
    )
    parser.add_argument(
        "query",
        nargs="?",
        default="vector search in postgres",
        help="The text query to search the corpus with.",
    )
    parser.add_argument(
        "--top-k", type=int, default=3, help="Number of results to return."
    )
    parser.add_argument("--dim", type=int, default=64, help="Embedding dimensionality.")
    parser.add_argument(
        "--database-url",
        default=None,
        help="If set, use the real pgvector backend instead of in-memory.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    embedder = FakeEmbedder(dim=args.dim)

    store: VectorStore
    if args.database_url:
        pg = PostgresVectorStore(embedder, dsn=args.database_url)
        pg.init_schema()
        store = pg
    else:
        store = InMemoryVectorStore(embedder)

    store.upsert_many(_CORPUS)

    print(f"Query: {args.query!r}\n")
    results = store.query(args.query, top_k=args.top_k)
    for rank, hit in enumerate(results, start=1):
        print(f"{rank}. [{hit.score:.4f}] {hit.id}: {hit.content}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
