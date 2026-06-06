"""Part of pgvector Example by Viprasol Tech Private Limited (https://viprasol.com).

Vector stores: a common :class:`VectorStore` protocol with two backends.

* :class:`PostgresVectorStore` — a real psycopg/pgvector backend (driver imported
  lazily so the package works without a database).
* :class:`InMemoryVectorStore` — a pure-Python cosine fallback so the full test
  suite and the demo run completely offline.

Both support ``upsert`` and a top-k cosine ``query``.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from pgvector_example import schema
from pgvector_example.embedder import Embedder, Vector


@dataclass(frozen=True)
class SearchResult:
    """A single similarity-search hit."""

    id: str
    content: str
    score: float
    metadata: dict[str, Any] = field(default_factory=dict)


class VectorStore(ABC):
    """Common interface shared by all vector-store backends."""

    embedder: Embedder

    @abstractmethod
    def upsert(
        self,
        doc_id: str,
        content: str,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        """Insert or update a document, embedding its ``content``."""

    @abstractmethod
    def query(self, text: str, top_k: int = 5) -> list[SearchResult]:
        """Return up to ``top_k`` documents most similar to ``text``."""

    def upsert_many(self, items: Sequence[tuple[str, str]]) -> None:
        """Convenience: upsert a sequence of ``(id, content)`` pairs."""
        for doc_id, content in items:
            self.upsert(doc_id, content)


class InMemoryVectorStore(VectorStore):
    """Offline cosine-similarity store backed by a plain dict.

    Useful for tests, demos, and small workloads where a database is overkill.
    Vectors from the embedder are L2-normalised on insert so the dot product is
    the cosine similarity.
    """

    def __init__(self, embedder: Embedder) -> None:
        self.embedder = embedder
        self._docs: dict[str, tuple[str, Vector, dict[str, Any]]] = {}

    def __len__(self) -> int:
        return len(self._docs)

    def upsert(
        self,
        doc_id: str,
        content: str,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        vec = _normalize(self.embedder.embed(content))
        self._docs[doc_id] = (content, vec, dict(metadata or {}))

    def query(self, text: str, top_k: int = 5) -> list[SearchResult]:
        if top_k < 1:
            msg = "top_k must be >= 1"
            raise ValueError(msg)
        q = _normalize(self.embedder.embed(text))
        scored = [
            SearchResult(
                id=doc_id,
                content=content,
                score=_dot(q, vec),
                metadata=dict(meta),
            )
            for doc_id, (content, vec, meta) in self._docs.items()
        ]
        scored.sort(key=lambda r: r.score, reverse=True)
        return scored[:top_k]


class PostgresVectorStore(VectorStore):
    """Real pgvector backend over psycopg 3 (driver imported lazily).

    Pass a ``DATABASE_URL`` (or rely on ``$DATABASE_URL``). Call
    :meth:`init_schema` once to enable the extension and create the table/index.
    Install with ``pip install pgvector-example[postgres]``.
    """

    def __init__(
        self,
        embedder: Embedder,
        dsn: str,
        table: str = schema.DEFAULT_TABLE,
        distance: schema.DistanceOp = "cosine",
    ) -> None:
        try:
            import psycopg
            from pgvector.psycopg import register_vector
        except ImportError as exc:  # pragma: no cover - optional dependency
            msg = (
                "psycopg and pgvector are required for PostgresVectorStore. "
                "Install with 'pip install pgvector-example[postgres]'."
            )
            raise ImportError(msg) from exc
        self.embedder = embedder
        self._table = table
        self._distance = distance
        schema._validate_identifier(table)
        self._conn = psycopg.connect(dsn)
        self._conn.execute(schema.enable_extension_sql())
        self._conn.commit()
        register_vector(self._conn)

    def init_schema(self) -> None:  # pragma: no cover - requires a database
        """Create the table and IVFFlat index if they do not exist."""
        self._conn.execute(schema.create_table_sql(self._table, self.embedder.dim))
        self._conn.execute(schema.create_index_sql(self._table, self._distance))
        self._conn.commit()

    def upsert(  # pragma: no cover - requires a database
        self,
        doc_id: str,
        content: str,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        import json

        vec = self.embedder.embed(content)
        self._conn.execute(
            f"INSERT INTO {self._table} (id, content, metadata, embedding) "  # noqa: S608
            "VALUES (%s, %s, %s, %s) "
            "ON CONFLICT (id) DO UPDATE SET "
            "content = EXCLUDED.content, "
            "metadata = EXCLUDED.metadata, "
            "embedding = EXCLUDED.embedding;",
            (doc_id, content, json.dumps(dict(metadata or {})), vec),
        )
        self._conn.commit()

    def query(  # pragma: no cover - requires a database
        self, text: str, top_k: int = 5
    ) -> list[SearchResult]:
        if top_k < 1:
            msg = "top_k must be >= 1"
            raise ValueError(msg)
        op = schema.query_operator(self._distance)
        q = self.embedder.embed(text)
        rows = self._conn.execute(
            f"SELECT id, content, metadata, 1 - (embedding {op} %s) AS score "  # noqa: S608
            f"FROM {self._table} ORDER BY embedding {op} %s LIMIT %s;",
            (q, q, top_k),
        ).fetchall()
        return [
            SearchResult(
                id=row[0],
                content=row[1],
                metadata=dict(row[2] or {}),
                score=float(row[3]),
            )
            for row in rows
        ]

    def close(self) -> None:  # pragma: no cover - requires a database
        """Close the underlying database connection."""
        self._conn.close()


def _dot(a: Vector, b: Vector) -> float:
    return math.fsum(x * y for x, y in zip(a, b, strict=True))


def _normalize(vec: Vector) -> Vector:
    norm = math.sqrt(math.fsum(x * x for x in vec))
    if norm == 0.0:
        return list(vec)
    return [x / norm for x in vec]
