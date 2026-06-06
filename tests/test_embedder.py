"""Offline tests for the FakeEmbedder."""

from __future__ import annotations

import math

import pytest

from pgvector_example.embedder import Embedder, FakeEmbedder


def test_dim_and_length() -> None:
    emb = FakeEmbedder(dim=32)
    assert emb.dim == 32
    assert len(emb.embed("hello world")) == 32


def test_is_embedder_subclass() -> None:
    assert issubclass(FakeEmbedder, Embedder)


def test_deterministic() -> None:
    emb = FakeEmbedder(dim=64)
    assert emb.embed("repeatable text") == emb.embed("repeatable text")


def test_normalized_to_unit_length() -> None:
    emb = FakeEmbedder(dim=64)
    vec = emb.embed("postgres vector search")
    assert math.isclose(math.sqrt(sum(x * x for x in vec)), 1.0, rel_tol=1e-9)


def test_similar_text_more_similar_than_unrelated() -> None:
    emb = FakeEmbedder(dim=128)

    def cos(a: list[float], b: list[float]) -> float:
        return sum(x * y for x, y in zip(a, b, strict=True))

    base = emb.embed("the quick brown fox jumps")
    near = emb.embed("the quick brown fox runs")
    far = emb.embed("completely unrelated database tables")
    assert cos(base, near) > cos(base, far)


def test_empty_text_is_stable_unit_vector() -> None:
    emb = FakeEmbedder(dim=16)
    vec = emb.embed("")
    assert math.isclose(math.sqrt(sum(x * x for x in vec)), 1.0, rel_tol=1e-9)


def test_embed_many() -> None:
    emb = FakeEmbedder(dim=8)
    out = emb.embed_many(["a", "b", "c"])
    assert len(out) == 3
    assert all(len(v) == 8 for v in out)


def test_invalid_dim() -> None:
    with pytest.raises(ValueError, match="dim must be"):
        FakeEmbedder(dim=0)
