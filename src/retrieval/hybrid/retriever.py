"""HybridRetriever — full pipeline: embed → search → normalise → RRF → filter → rerank.

Pipeline
--------
1. Embed the query (OpenAI, one call).
2. Search Pinecone (dense) and BM25 (sparse) in parallel — both retrievers
   receive RBAC filters as an early-exit optimisation.
3. Build ``DenseHit`` and ``SparseHit`` lists from the results.
4. ``reciprocal_rank_fusion`` — normalise scores, accumulate RRF scores,
   sort, emit ``RetrievalEvidence`` with full provenance.
5. ``apply_access_filter`` — authoritative RBAC gate; re-ranks survivors.
6. ``reranker.rerank`` — optional cross-encoder pass (no-op by default).

Access filtering is applied at TWO points intentionally:
- During retrieval (Pinecone filter, BM25 corpus filter): reduces the
  candidate pool for efficiency.
- Post-RRF (``apply_access_filter``): the authoritative enforcement check
  that cannot be bypassed by a misconfigured upstream layer.

The ``retriever_top_k`` parameter controls how many candidates each retriever
fetches.  Fetching more candidates than ``top_k`` gives RRF more material to
work with, especially when overlap between the two lists is low.  The default
is ``top_k * 3``.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Sequence

from src.core.logging import get_logger, log_info, log_debug
from src.models.enums import AccessLevel, Role
from src.retrieval.bm25.service import BM25Service
from src.retrieval.embedding.base import EmbeddingProvider
from src.retrieval.hybrid.filtering import apply_access_filter
from src.retrieval.hybrid.fusion import DenseHit, SparseHit, reciprocal_rank_fusion
from src.retrieval.hybrid.models import RetrievalEvidence
from src.retrieval.hybrid.reranking import IdentityReranker, Reranker
from src.retrieval.indexing.metadata import metadata_to_chunk
from src.retrieval.indexing.service import PineconeIndexService

logger: logging.Logger = get_logger(__name__)

_DEFAULT_RETRIEVER_MULTIPLIER: int = 3


class HybridRetriever:
    """Orchestrates the hybrid retrieval pipeline.

    Accepts injected dependencies so every component is independently
    replaceable and testable:

    - ``pinecone_service`` — dense vector search (Pinecone).
    - ``embedder`` — converts the query string to a dense vector.
    - ``bm25_service`` — sparse keyword search (BM25).
    - ``reranker`` — optional cross-encoder (defaults to no-op).

    The retriever is stateless after construction: calling ``search``
    multiple times with the same arguments always produces the same output
    for the same underlying index state.
    """

    def __init__(
        self,
        pinecone_service: PineconeIndexService,
        embedder: EmbeddingProvider,
        bm25_service: BM25Service,
        *,
        reranker: Reranker | None = None,
    ) -> None:
        self._pinecone = pinecone_service
        self._embedder = embedder
        self._bm25 = bm25_service
        self._reranker: Reranker = reranker if reranker is not None else IdentityReranker()

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
    ) -> list[RetrievalEvidence]:
        """Run the full hybrid retrieval pipeline and return ranked evidence.

        Parameters
        ----------
        query:
            User query string; embedded for dense search and tokenized for
            sparse search.
        top_k:
            Maximum number of ``RetrievalEvidence`` items to return after
            filtering and reranking.
        roles:
            RBAC roles of the requesting user.  Forwarded to both retrievers
            and to the post-RRF access filter.
        access_levels:
            Restrict results to these access levels.  Applied at both
            retrieval and post-RRF filter stages.
        department:
            Restrict to this department (exact match).
        document_type:
            Restrict to this document type (exact match).
        retriever_top_k:
            Candidates fetched from each retriever before fusion.  Defaults
            to ``top_k * 3``; increase when recall is low.
        """
        rk = retriever_top_k if retriever_top_k is not None else top_k * _DEFAULT_RETRIEVER_MULTIPLIER
        t0 = time.monotonic()

        log_debug(
            logger,
            "hybrid.search.start",
            "Hybrid search started",
            query=query,
            top_k=top_k,
            retriever_top_k=rk,
        )

        # --- Step 1: embed the query ---
        vector = await self._embedder.embed_one(query)

        # --- Step 2: search both retrievers in parallel ---
        # Build Pinecone metadata filter from all active predicates.
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

        dense_matches, sparse_bm25 = await asyncio.gather(
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

        # --- Step 3: build typed hit lists ---
        dense_hits: list[DenseHit] = [
            DenseHit(
                chunk=metadata_to_chunk(match.metadata),
                score=match.score,
                rank=i + 1,
            )
            for i, match in enumerate(dense_matches)
        ]

        sparse_hits: list[SparseHit] = [
            SparseHit(
                chunk=sr.chunk,
                score=sr.score,
                rank=sr.rank,
            )
            for sr in sparse_bm25
        ]

        # --- Step 4: RRF fusion (normalise + merge + rank) ---
        fused = reciprocal_rank_fusion(dense_hits, sparse_hits, top_k=rk)

        # --- Step 5: authoritative post-RRF access filter ---
        filtered = apply_access_filter(
            fused,
            roles=roles,
            access_levels=access_levels,
            department=department,
            document_type=document_type,
        )

        # --- Step 6: optional reranking ---
        results = await self._reranker.rerank(query, filtered, top_k=top_k)

        elapsed_ms = round((time.monotonic() - t0) * 1000, 1)
        log_info(
            logger,
            "hybrid.search.completed",
            "Hybrid search completed",
            query=query,
            dense_candidates=len(dense_hits),
            sparse_candidates=len(sparse_hits),
            fused_count=len(fused),
            filtered_count=len(filtered),
            result_count=len(results),
            elapsed_ms=elapsed_ms,
        )
        return results
