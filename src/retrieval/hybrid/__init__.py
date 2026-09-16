"""Hybrid retrieval: dense (Pinecone) + sparse (BM25) → RRF → access filter → rerank.

Public surface
--------------
Models
- ``RetrievalEvidence`` — one retrieved chunk with full provenance:
  chunk_id, document_id, title, text, metadata, source, dense_score,
  sparse_score, final_score, rank.
- ``RetrievalSource`` — which retriever(s) contributed: DENSE, SPARSE, BOTH.

Fusion
- ``DenseHit`` / ``SparseHit`` — typed inputs to ``reciprocal_rank_fusion``.
- ``normalize_scores`` — min-max normalization helper.
- ``reciprocal_rank_fusion`` — explicit RRF implementation.
- ``RRF_K`` — default damping constant (60).

Filtering
- ``apply_access_filter`` — authoritative post-RRF RBAC and metadata filter.

Reranking
- ``Reranker`` — Protocol for cross-encoder rerankers.
- ``IdentityReranker`` — no-op reranker (default).

Retriever
- ``HybridRetriever`` — orchestrates the full pipeline.
"""

from src.retrieval.hybrid.filtering import apply_access_filter
from src.retrieval.hybrid.fusion import (
    DenseHit,
    RRF_K,
    SparseHit,
    normalize_scores,
    reciprocal_rank_fusion,
)
from src.retrieval.hybrid.models import RetrievalEvidence, RetrievalSource
from src.retrieval.hybrid.reranking import IdentityReranker, Reranker
from src.retrieval.hybrid.retriever import HybridRetriever

__all__ = [
    # models
    "RetrievalEvidence",
    "RetrievalSource",
    # fusion
    "DenseHit",
    "SparseHit",
    "normalize_scores",
    "reciprocal_rank_fusion",
    "RRF_K",
    # filtering
    "apply_access_filter",
    # reranking
    "Reranker",
    "IdentityReranker",
    # retriever
    "HybridRetriever",
]
