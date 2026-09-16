"""Data models for hybrid retrieval results.

Every piece of evidence returned to the agent carries the full provenance
chain — which retriever(s) found it, the per-retriever scores, and the
final RRF score — so downstream consumers can reason about why a chunk
was retrieved without inspecting internals.

``RetrievalEvidence`` is the authoritative output type of the retrieval
layer.  Agents and services depend on it; they never reach into the
individual Pinecone or BM25 result types.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field

from src.models.documents import DocumentMetadata


class RetrievalSource(StrEnum):
    """Which retriever(s) contributed this evidence item."""

    DENSE = "dense"    # found only by Pinecone vector search
    SPARSE = "sparse"  # found only by BM25 keyword search
    BOTH = "both"      # found by both retrievers (higher RRF weight)


class RetrievalEvidence(BaseModel):
    """One retrieved chunk, fully annotated with retrieval provenance.

    Fields follow the spec exactly so agents can explain their answers:

    - ``source``       — which retriever(s) surfaced this chunk.
    - ``dense_score``  — min-max normalised cosine similarity (0–1), or
                         ``None`` if the dense retriever did not return it.
    - ``sparse_score`` — min-max normalised BM25 score (0–1), or ``None``
                         if the sparse retriever did not return it.
    - ``final_score``  — RRF score after fusion; the primary sort key.
    - ``rank``         — 1-based position in the final result list.

    The model is frozen so it can be safely shared across coroutines and
    used as a dict key or set element.
    """

    model_config = ConfigDict(frozen=True)

    chunk_id: str = Field(min_length=1)
    document_id: str = Field(min_length=1)
    title: str = Field(min_length=1)
    text: str = Field(min_length=1)
    metadata: DocumentMetadata

    source: RetrievalSource
    dense_score: float | None = Field(
        default=None,
        description="Normalised cosine similarity from Pinecone (0–1), or None.",
    )
    sparse_score: float | None = Field(
        default=None,
        description="Normalised BM25 score (0–1), or None.",
    )
    final_score: float = Field(ge=0.0, description="RRF score after fusion.")
    rank: int = Field(ge=1, description="1-based rank in the final result list.")
