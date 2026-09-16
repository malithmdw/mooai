"""Reciprocal Rank Fusion for hybrid dense + sparse retrieval.

RRF merges ranked lists from two independent retrievers without requiring
their scores to be on the same scale.  Only the rank positions are used:

    RRF(d) = Σ_{r ∈ retrievers} 1 / (k + rank_r(d))

where k (default 60) dampens the effect of the highest-ranked documents.
Chunks that appear in only one retriever still receive an RRF contribution
from that retriever; the other contribution is implicitly zero.

Reference: Cormack, Clarke, Buettcher (2009) — "Reciprocal Rank Fusion
outperforms Condorcet and individual rank learning methods".
"""

from __future__ import annotations

from typing import Final

from pydantic import BaseModel, ConfigDict, Field

from src.retrieval.bm25.corpus import BM25Result
from src.retrieval.ingestion.models import DocumentChunk

RRF_K: Final[int] = 60


class DenseSearchResult(BaseModel):
    """One result from a Pinecone dense vector similarity search."""

    model_config = ConfigDict(frozen=True)

    chunk: DocumentChunk
    score: float = Field(ge=0.0, description="Cosine similarity score from Pinecone.")
    rank: int = Field(ge=1, description="1-based rank within the dense result list.")


class HybridResult(BaseModel):
    """One chunk returned by hybrid (dense + sparse) retrieval.

    ``rrf_score`` is the sum of Reciprocal Rank Fusion contributions from
    both retrievers.  ``dense_rank`` / ``sparse_rank`` are ``None`` when the
    chunk was not returned by that retriever.
    """

    model_config = ConfigDict(frozen=True)

    chunk: DocumentChunk
    rrf_score: float = Field(ge=0.0, description="Combined RRF score.")
    rank: int = Field(ge=1, description="1-based rank in the merged result list.")
    dense_rank: int | None = Field(
        default=None,
        description="Rank in the dense results (None if absent).",
    )
    sparse_rank: int | None = Field(
        default=None,
        description="Rank in the sparse results (None if absent).",
    )


def reciprocal_rank_fusion(
    dense: list[DenseSearchResult],
    sparse: list[BM25Result],
    *,
    k: int = RRF_K,
    top_k: int = 10,
) -> list[HybridResult]:
    """Merge *dense* and *sparse* result lists using Reciprocal Rank Fusion.

    Parameters
    ----------
    dense:
        Results from the Pinecone dense index, in score-descending order.
    sparse:
        Results from BM25 sparse search, in score-descending order.
    k:
        RRF constant (default 60).  Lower values give more weight to
        top-ranked documents; higher values flatten the distribution.
    top_k:
        Maximum number of results to return.

    Returns
    -------
    list[HybridResult]
        Up to *top_k* results sorted by ``rrf_score`` descending.
        Tie-breaking: ``chunk_id`` ascending for determinism.
    """
    rrf_scores: dict[str, float] = {}
    dense_ranks: dict[str, int] = {}
    sparse_ranks: dict[str, int] = {}
    chunks: dict[str, DocumentChunk] = {}

    for dr in dense:
        cid = dr.chunk.chunk_id
        rrf_scores[cid] = rrf_scores.get(cid, 0.0) + 1.0 / (k + dr.rank)
        dense_ranks[cid] = dr.rank
        chunks[cid] = dr.chunk

    for sr in sparse:
        cid = sr.chunk.chunk_id
        rrf_scores[cid] = rrf_scores.get(cid, 0.0) + 1.0 / (k + sr.rank)
        sparse_ranks[cid] = sr.rank
        chunks[cid] = sr.chunk

    ordered = sorted(rrf_scores.items(), key=lambda t: (-t[1], t[0]))[:top_k]

    return [
        HybridResult(
            chunk=chunks[cid],
            rrf_score=score,
            rank=i + 1,
            dense_rank=dense_ranks.get(cid),
            sparse_rank=sparse_ranks.get(cid),
        )
        for i, (cid, score) in enumerate(ordered)
    ]
