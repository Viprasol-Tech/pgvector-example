"""Part of pgvector Example by Viprasol Tech Private Limited (https://viprasol.com).

Schema / DDL helpers for a pgvector-backed table. These functions return SQL
strings so they can be inspected, logged, or executed by any driver. The table
holds an ``id``, free-form ``content``, optional JSONB ``metadata``, and a
fixed-width ``vector`` column indexed for fast approximate nearest-neighbour
search.
"""

from __future__ import annotations

from typing import Literal

DEFAULT_TABLE = "documents"
DEFAULT_DIM = 64

DistanceOp = Literal["cosine", "l2", "ip"]

# Map a logical distance to the pgvector operator class and query operator.
_OPCLASS: dict[DistanceOp, str] = {
    "cosine": "vector_cosine_ops",
    "l2": "vector_l2_ops",
    "ip": "vector_ip_ops",
}
_OPERATOR: dict[DistanceOp, str] = {
    "cosine": "<=>",
    "l2": "<->",
    "ip": "<#>",
}


def enable_extension_sql() -> str:
    """Return SQL that enables the ``vector`` extension (idempotent)."""
    return "CREATE EXTENSION IF NOT EXISTS vector;"


def create_table_sql(table: str = DEFAULT_TABLE, dim: int = DEFAULT_DIM) -> str:
    """Return ``CREATE TABLE`` DDL for a documents table with a vector column.

    Args:
        table: Table name. Must be a valid SQL identifier.
        dim: Dimensionality of the stored embedding vectors.
    """
    _validate_identifier(table)
    _validate_dim(dim)
    return (
        f"CREATE TABLE IF NOT EXISTS {table} (\n"
        "    id TEXT PRIMARY KEY,\n"
        "    content TEXT NOT NULL,\n"
        "    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,\n"
        f"    embedding vector({dim}) NOT NULL\n"
        ");"
    )


def create_index_sql(
    table: str = DEFAULT_TABLE,
    distance: DistanceOp = "cosine",
    lists: int = 100,
) -> str:
    """Return ``CREATE INDEX`` DDL building an IVFFlat index on the vector column.

    Args:
        table: Table name.
        distance: Distance metric: ``"cosine"``, ``"l2"``, or ``"ip"``.
        lists: Number of IVFFlat lists (tune for table size).
    """
    _validate_identifier(table)
    if distance not in _OPCLASS:
        msg = f"unknown distance {distance!r}"
        raise ValueError(msg)
    if lists < 1:
        msg = "lists must be >= 1"
        raise ValueError(msg)
    opclass = _OPCLASS[distance]
    return (
        f"CREATE INDEX IF NOT EXISTS {table}_embedding_idx\n"
        f"    ON {table} USING ivfflat (embedding {opclass})\n"
        f"    WITH (lists = {lists});"
    )


def query_operator(distance: DistanceOp = "cosine") -> str:
    """Return the pgvector query operator for the given distance metric."""
    if distance not in _OPERATOR:
        msg = f"unknown distance {distance!r}"
        raise ValueError(msg)
    return _OPERATOR[distance]


def _validate_identifier(name: str) -> None:
    if not name or not all(c.isalnum() or c == "_" for c in name) or name[0].isdigit():
        msg = f"invalid SQL identifier: {name!r}"
        raise ValueError(msg)


def _validate_dim(dim: int) -> None:
    if dim < 1:
        msg = "dim must be >= 1"
        raise ValueError(msg)
