"""Offline tests for the schema/DDL helpers."""

from __future__ import annotations

import pytest

from pgvector_example import schema


def test_enable_extension_sql() -> None:
    assert schema.enable_extension_sql() == "CREATE EXTENSION IF NOT EXISTS vector;"


def test_create_table_sql_contains_vector_column() -> None:
    sql = schema.create_table_sql("docs", dim=128)
    assert "CREATE TABLE IF NOT EXISTS docs" in sql
    assert "embedding vector(128) NOT NULL" in sql
    assert "id TEXT PRIMARY KEY" in sql
    assert "metadata JSONB" in sql


def test_create_index_sql_cosine() -> None:
    sql = schema.create_index_sql("docs", distance="cosine", lists=50)
    assert "ivfflat (embedding vector_cosine_ops)" in sql
    assert "lists = 50" in sql


@pytest.mark.parametrize(
    ("distance", "opclass"),
    [
        ("cosine", "vector_cosine_ops"),
        ("l2", "vector_l2_ops"),
        ("ip", "vector_ip_ops"),
    ],
)
def test_index_opclasses(distance: schema.DistanceOp, opclass: str) -> None:
    assert opclass in schema.create_index_sql("t", distance=distance)


@pytest.mark.parametrize(
    ("distance", "operator"),
    [("cosine", "<=>"), ("l2", "<->"), ("ip", "<#>")],
)
def test_query_operator(distance: schema.DistanceOp, operator: str) -> None:
    assert schema.query_operator(distance) == operator


def test_invalid_identifier_rejected() -> None:
    with pytest.raises(ValueError, match="invalid SQL identifier"):
        schema.create_table_sql("1bad")
    with pytest.raises(ValueError, match="invalid SQL identifier"):
        schema.create_table_sql("drop table;")


def test_invalid_dim_rejected() -> None:
    with pytest.raises(ValueError, match="dim must be"):
        schema.create_table_sql("docs", dim=0)


def test_invalid_distance_rejected() -> None:
    with pytest.raises(ValueError, match="unknown distance"):
        schema.create_index_sql("docs", distance="banana")  # type: ignore[arg-type]
