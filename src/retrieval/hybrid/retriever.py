"""Hybrid retriever: dense (Pinecone) + sparse (BM25) with RRF fusion.

Queries both retrievers in parallel via ``asyncio.gather``, converts
Pinecone matches back to ``DocumentChunk`` objects using
``metadata_to_chunk``, then merges the two ranked lists with
Reciprocal Rank Fusion.

Each retriever receives its own access filter so RBAC is enforced at both
layers — not just in one.  The Pinecone filter uses MongoDB-style predicates
(``$in``, ``$eq``); the BM25 filter uses the same domain types passed through
to ``BM25Corpus.search``.

Usage
-----
    from src.retrieval.hybrid import HybridRetriever
    from src.retrieval.indexing import make_pinecone_service
    from src.retrieval.embedding import make_openai_provider
    from src.retrieval.bm25 import BM25Service

    retriever = HybridRetriever(
        pinecone_service=make_pinecone_service(settings),
        embedder=make_openai_provider(settings),
        bm25_service=BM25Service.from_chunks(chunks),
    )
    results = await retriever.search("certificate expiry FPS", roles=[Role.ENGINEER])
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Sequence

from src.core.logging import get_logger, log_debug, log_info
from src.models.enums import AccessLevel, Role
from src.retrieval.bm25.service import BM25Service
from src.retrieval.embedding.base import EmbeddingProvider
from src.retrieval.hybrid.fusion import (
    DenseSearchResult,
    HybridResult,
    reciprocal_rank_fusion,
)
from src.retrieval.indexing.metadata import metadata_to_chunk
from src.retrieval.indexing.service import PineconeIndexService

logger: logging.Logger = get_logger(__name__)

_DEFAULT_RETRIEVER_MULTIPLIER: int = 3


class HybridRetriever:
    """Combines Pinecone dense search and BM25 sparse search via RRF.

    Parameters
    ----------
    pinecone_service:
        ``PineconeIndexService`` configured for the target index/namespace.
    embedder:
        Embedding provider used to turn the query string into a vector for
        the Pinecone dense search.
    bm25_service:
        ``BM25Service`` built from the same corpus that was indexed into
        Pinecone.
    """

    def __init__(
        self,
        pinecone_service: PineconeIndexService,
        embedder: EmbeddingProvider,
        bm25_service: BM25Service,
    ) -> None:
        self._pinecone = pinecone_service
        self._embedder = embedder
        self._bm25 = bm25_service

    async def search(
        self,
        query: str,
        *,
        top_k: int = 10,
        roles: Sequence[Role] | None = None,
        access_levels: Sequence[AccessLevel] | None = None,
        department: str | None = None,
        document_type: str | None = None,
        retriever_top_k: int | None = None,
    ) -> list[HybridResult]:
        """Search the hybrid index and return the top-*k* fused results.

        Both retrievers are queried in parallel (after the embedding step).
        Results are merged with Reciprocal Rank Fusion and the top *top_k*
        are returned.

        Parameters
        ----------
        query:
            Natural-language or keyword query string.
        top_k:
            Number of results to return from the fused ranking.
        roles:
            If provided, only chunks accessible to these roles are returned.
            Applied to both Pinecone (via metadata filter) and BM25 (via
            corpus filter).
        access_levels:
            If provided, restrict to these access levels.  Applied to both
            retrievers.
        department:
            If provided, restrict to this department (exact match).
        document_type:
            If provided, restrict to this document type (exact match).
        retriever_top_k:
            How many candidates each retriever fetches before fusion.
            Defaults to ``top_k * 3`` to give RRF enough material to work
            with, especially when overlap between the two lists is low.
        """
        rk = retriever_top_k if retriever_top_k is not None else top_k * _DEFAULT_RETRIEVER_MULTIPLIER

        log_debug(
            logger,
            "hybrid.search.start",
            "Hybrid search started",
            query=query,
            top_k=top_k,
            retriever_top_k=rk,
        )

        t0 = time.monotonic()

        # Build Pinecone metadata filter — all active predicates combined.
        pinecone_filter: dict[str, object] = {}
        if roles is not None:
            pinecone_filter["allowed_roles"] = {"$in": [r.value for r in roles]}
        if access_levels is not None:
            pinecone_filter["access_level"] = {"$in": [lv.value for lv in access_levels]}
        if department is not None:
            pinecone_filter["department"] = {"$eq": department}
        if document_type is not None:
            pinecone_filter["document_type"] = {"$eq": document_type}

        filter_arg: dict[str, object] | None = pinecone_filter or None

        # Embed the query, then search both retrievers in parallel.
        vector = await self._embedder.embed_one(query)

        dense_matches, sparse_results = await asyncio.gather(
            self._pinecone.search(vector, top_k=rk, filter=filter_arg),
            self._bm25.search(
                query,
                top_k=rk,
                roles=roles,
                access_levels=access_levels,
                department=department,
                document_type=document_type,
            ),
        )

        # Convert raw Pinecone matches to typed DenseSearchResult objects.
        dense_results: list[DenseSearchResult] = [
            DenseSearchResult(
                chunk=metadata_to_chunk(match.metadata),
                score=match.score,
                rank=i + 1,
            )
            for i, match in enumerate(dense_matches)
        ]

        results = reciprocal_rank_fusion(dense_results, sparse_results, top_k=top_k)

        elapsed_ms = round((time.monotonic() - t0) * 1000, 1)
        log_info(
            logger,
            "hybrid.search.completed",
            "Hybrid search completed",
            query=query,
            dense_count=len(dense_results),
            sparse_count=len(sparse_results),
            result_count=len(results),
            elapsed_ms=elapsed_ms,
        )
        return results
