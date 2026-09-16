"""Embedding provider abstraction and OpenAI implementation.

Public surface
--------------
- `EmbeddingProvider` — structural `Protocol` (no inheritance needed).
- `OpenAIEmbeddingProvider` — batched, retry-capable OpenAI implementation.
- `make_openai_provider` — production factory that reads from `Settings`.

Callers should depend on `EmbeddingProvider`, not on the concrete class,
so the implementation can be swapped or mocked in tests without changes
to the call-site.
"""

from src.retrieval.embedding.base import EmbeddingProvider
from src.retrieval.embedding.openai_provider import OpenAIEmbeddingProvider, make_openai_provider

__all__ = [
    "EmbeddingProvider",
    "OpenAIEmbeddingProvider",
    "make_openai_provider",
]
