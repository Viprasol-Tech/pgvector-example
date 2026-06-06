"""Offline tests for the in-memory vector store."""

from __future__ import annotations

import pytest

from pgvector_example.embedder import FakeEmbedder
from pgvector_example.store import InMemoryVectorStore, SearchResult, VectorStore


@pytest.fixture
def store() -> InMemoryVectorStore:
    s = InMemoryVectorStore(FakeEmbedder(dim=128))
    s.upsert("d1", "Postgres relational database management system")
    s.upsert("d2", "pgvector similarity search over embeddings")
    s.upsert("d3", "cooking pasta with tomato sauce", {"topic": "food"})
    return s


def test_is_vectorstore(store: InMemoryVectorStore) -> None:
    assert isinstance(store, VectorStore)


def test_len(store: InMemoryVectorStore) -> None:
    assert len(store) == 3


def test_upsert_updates_in_place() -> None:
    s = InMemoryVectorStore(FakeEmbedder(dim=64))
    s.upsert("x", "first content")
    s.upsert("x", "second content", {"v": 2})
    assert len(s) == 1
    top = s.query("second content", top_k=1)[0]
    assert top.id == "x"
    assert top.metadata == {"v": 2}


def test_query_returns_search_results(store: InMemoryVectorStore) -> None:
    results = store.query("vector embeddings search", top_k=2)
    assert all(isinstance(r, SearchResult) for r in results)
    assert len(results) == 2


def test_query_top_k_limits(store: InMemoryVectorStore) -> None:
    assert len(store.query("anything", top_k=1)) == 1
    assert len(store.query("anything", top_k=10)) == 3


def test_query_ranks_relevant_first(store: InMemoryVectorStore) -> None:
    results = store.query("pgvector embeddings similarity search", top_k=3)
    assert results[0].id == "d2"


def test_scores_sorted_descending(store: InMemoryVectorStore) -> None:
    scores = [r.score for r in store.query("database", top_k=3)]
    assert scores == sorted(scores, reverse=True)


def test_metadata_preserved(store: InMemoryVectorStore) -> None:
    hit = store.query("cooking pasta tomato", top_k=1)[0]
    assert hit.metadata == {"topic": "food"}


def test_invalid_top_k(store: InMemoryVectorStore) -> None:
    with pytest.raises(ValueError, match="top_k must be"):
        store.query("x", top_k=0)


def test_empty_store_returns_empty() -> None:
    s = InMemoryVectorStore(FakeEmbedder(dim=32))
    assert s.query("anything") == []


def test_upsert_many() -> None:
    s = InMemoryVectorStore(FakeEmbedder(dim=32))
    s.upsert_many([("a", "alpha"), ("b", "beta")])
    assert len(s) == 2


def test_scores_in_unit_range(store: InMemoryVectorStore) -> None:
    for r in store.query("database search", top_k=3):
        assert -1.0001 <= r.score <= 1.0001
