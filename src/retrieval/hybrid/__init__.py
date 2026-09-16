"""Hybrid retrieval: dense (Pinecone) + sparse (BM25) with RRF fusion.

Public surface
--------------
- ``HybridRetriever`` — async retriever that queries both indexes in parallel
  and merges results with Reciprocal Rank Fusion.
- ``HybridResult`` — one fused result with chunk, RRF score, rank, and
  per-retriever rank provenance.
- ``DenseSearchResult`` — one Pinecone dense search result (chunk + cosine
  score + rank).
- ``reciprocal_rank_fusion`` — pure RRF merge function; useful for testing
  or custom retrieval pipelines.
- ``RRF_K`` — the damping constant used by default (60).
"""

from src.retrieval.hybrid.fusion import (
    DenseSearchResult,
    HybridResult,
    RRF_K,
    reciprocal_rank_fusion,
)
from src.retrieval.hybrid.retriever import HybridRetriever

__all__ = [
    "DenseSearchResult",
    "HybridResult",
    "HybridRetriever",
    "RRF_K",
    "reciprocal_rank_fusion",
]
