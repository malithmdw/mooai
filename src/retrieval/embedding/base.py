"""Abstract embedding provider interface.

`EmbeddingProvider` is a structural `Protocol` — any object that exposes
the right methods satisfies it without inheriting from this class.  That
makes it straightforward to inject a lightweight stub or an `AsyncMock`
in tests without any coupling to the concrete OpenAI implementation.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, runtime_checkable


@runtime_checkable
class EmbeddingProvider(Protocol):
    """Structural interface for text embedding providers.

    Implementations must be safe to call concurrently from multiple
    coroutines: they must not mutate shared state between calls.
    """

    @property
    def model(self) -> str:
        """Name of the underlying embedding model, for logging and attribution."""
        ...

    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed *texts* in one or more batches.

        Returns one float vector per input, in the same order as the input.
        An empty sequence returns an empty list (no API call is made).
        """
        ...

    async def embed_one(self, text: str) -> list[float]:
        """Convenience wrapper: embed a single text and return its vector."""
        ...
