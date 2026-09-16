"""Optional cross-encoder reranking stage.

Reranking sits after access filtering so a reranker never scores
unauthorized documents.  For this POC, only ``IdentityReranker`` (a no-op)
is provided.  A real cross-encoder (Cohere Rerank, a local model, etc.)
would implement ``Reranker`` and be injected into ``HybridRetriever``.

The ``Reranker`` Protocol is ``@runtime_checkable`` so ``isinstance`` checks
work without subclassing — consistent with the ``EmbeddingProvider`` Protocol
pattern in ``src.retrieval.embedding.base``.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from src.retrieval.hybrid.models import RetrievalEvidence


@runtime_checkable
class Reranker(Protocol):
    """Structural protocol for optional cross-encoder rerankers.

    Implementors receive the query string and the access-filtered evidence
    list (already ranked by RRF).  They return a new list, possibly shorter,
    re-ordered by cross-encoder relevance.
    """

    async def rerank(
        self,
        query: str,
        evidence: list[RetrievalEvidence],
        *,
        top_k: int,
    ) -> list[RetrievalEvidence]: ...


class IdentityReranker:
    """No-op reranker — returns the first *top_k* items unchanged.

    Used as the default in ``HybridRetriever`` so the reranking slot is
    always exercised but has zero cost unless a real reranker is injected.
    """

    async def rerank(
        self,
        query: str,
        evidence: list[RetrievalEvidence],
        *,
        top_k: int,
    ) -> list[RetrievalEvidence]:
        """Return the top-*top_k* items from *evidence* in their current order."""
        return evidence[:top_k]
