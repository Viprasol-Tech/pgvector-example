"""Part of pgvector Example by Viprasol Tech Private Limited (https://viprasol.com).

Embedders turn text into fixed-width float vectors. The :class:`Embedder` ABC
defines the contract. :class:`FakeEmbedder` is a deterministic, dependency-free
hashing embedder used for offline tests and the demo. Optional backends
(``sentence-transformers``, OpenAI) import their heavy dependencies lazily so the
package stays importable without them.
"""

from __future__ import annotations

import hashlib
import math
import os
from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import ClassVar

Vector = list[float]


class Embedder(ABC):
    """Abstract base class for text embedders."""

    @property
    @abstractmethod
    def dim(self) -> int:
        """Dimensionality of the produced vectors."""

    @abstractmethod
    def embed(self, text: str) -> Vector:
        """Embed a single string into a vector of length :attr:`dim`."""

    def embed_many(self, texts: Sequence[str]) -> list[Vector]:
        """Embed a sequence of strings. Override for batched backends."""
        return [self.embed(t) for t in texts]


class FakeEmbedder(Embedder):
    """Deterministic, offline embedder.

    Hashes token n-grams into a fixed number of buckets, then L2-normalises the
    result. It needs no network or model download, yet produces stable,
    semantically-flavoured vectors (shared words yield higher cosine similarity),
    which makes it ideal for tests and reproducible demos.
    """

    def __init__(self, dim: int = 64) -> None:
        if dim < 1:
            msg = "dim must be >= 1"
            raise ValueError(msg)
        self._dim = dim

    @property
    def dim(self) -> int:
        return self._dim

    def embed(self, text: str) -> Vector:
        vec = [0.0] * self._dim
        tokens = _tokenize(text)
        for token in tokens:
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            bucket = int.from_bytes(digest[:4], "big") % self._dim
            sign = 1.0 if digest[4] & 1 else -1.0
            vec[bucket] += sign
        norm = math.sqrt(sum(x * x for x in vec))
        if norm == 0.0:
            # Empty / unhashable input: return a stable unit vector.
            vec[0] = 1.0
            return vec
        return [x / norm for x in vec]


class SentenceTransformerEmbedder(Embedder):
    """Embedder backed by ``sentence-transformers`` (imported lazily).

    Install with ``pip install pgvector-example[sentence-transformers]``.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # pragma: no cover - optional dependency
            msg = (
                "sentence-transformers is not installed. Install with "
                "'pip install pgvector-example[sentence-transformers]'."
            )
            raise ImportError(msg) from exc
        self._model = SentenceTransformer(model_name)
        self._dim = int(self._model.get_sentence_embedding_dimension())

    @property
    def dim(self) -> int:  # pragma: no cover - optional dependency
        return self._dim

    def embed(self, text: str) -> Vector:  # pragma: no cover - optional dependency
        vec = self._model.encode([text], normalize_embeddings=True)[0]
        return [float(x) for x in vec]

    def embed_many(
        self, texts: Sequence[str]
    ) -> list[Vector]:  # pragma: no cover - optional dependency
        vecs = self._model.encode(list(texts), normalize_embeddings=True)
        return [[float(x) for x in row] for row in vecs]


class OpenAIEmbedder(Embedder):
    """Embedder backed by the OpenAI embeddings API (imported lazily).

    Install with ``pip install pgvector-example[openai]`` and set
    ``OPENAI_API_KEY``.
    """

    _DIMS: ClassVar[dict[str, int]] = {
        "text-embedding-3-small": 1536,
        "text-embedding-3-large": 3072,
    }

    def __init__(
        self,
        model: str = "text-embedding-3-small",
        api_key: str | None = None,
    ) -> None:
        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover - optional dependency
            msg = (
                "openai is not installed. Install with "
                "'pip install pgvector-example[openai]'."
            )
            raise ImportError(msg) from exc
        self._client = OpenAI(api_key=api_key or os.environ.get("OPENAI_API_KEY"))
        self._model = model
        self._dim = self._DIMS.get(model, 1536)

    @property
    def dim(self) -> int:  # pragma: no cover - optional dependency
        return self._dim

    def embed(self, text: str) -> Vector:  # pragma: no cover - optional dependency
        resp = self._client.embeddings.create(model=self._model, input=text)
        return [float(x) for x in resp.data[0].embedding]


def _tokenize(text: str) -> list[str]:
    cleaned = "".join(c.lower() if c.isalnum() else " " for c in text)
    return [t for t in cleaned.split() if t]
