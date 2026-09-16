"""Async BM25 service — wraps BM25Corpus for use in async application code.

``BM25Corpus.search`` is synchronous (pure CPU work on in-memory data).
``BM25Service`` runs it in a thread pool via ``asyncio.to_thread`` so it
never blocks the event loop, consistent with how ``PineconeIndexService``
wraps the synchronous Pinecone SDK.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Sequence

from src.core.logging import get_logger
from src.models.enums import AccessLevel, Role
from src.retrieval.bm25.corpus import BM25Corpus, BM25Result
from src.retrieval.ingestion.models import DocumentChunk

logger: logging.Logger = get_logger(__name__)


class BM25Service:
    """Async facade over ``BM25Corpus``.

    Build one service per corpus snapshot and rebuild whenever documents
    change — the corpus is immutable after construction.
    """

    def __init__(self, corpus: BM25Corpus) -> None:
        self._corpus = corpus

    @classmethod
    def from_chunks(cls, chunks: Sequence[DocumentChunk]) -> "BM25Service":
        """Build a service directly from a chunk list (convenience factory)."""
        return cls(BM25Corpus(chunks))

    @property
    def chunk_count(self) -> int:
        """Total number of chunks indexed in this service."""
        return self._corpus.chunk_count

    async def search(
        self,
        query: str,
        *,
        top_k: int = 10,
        roles: Sequence[Role] | None = None,
        access_levels: Sequence[AccessLevel] | None = None,
        department: str | None = None,
        document_type: str | None = None,
    ) -> list[BM25Result]:
        """Search the BM25 corpus asynchronously.

        All parameters are forwarded to ``BM25Corpus.search`` — see that
        method for full parameter documentation.  The call runs in a thread
        pool so it does not block the event loop.
        """
        return await asyncio.to_thread(
            self._corpus.search,
            query,
            top_k=top_k,
            roles=roles,
            access_levels=access_levels,
            department=department,
            document_type=document_type,
        )
