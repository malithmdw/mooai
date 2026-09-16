"""Domain models for the Response Agent structured output."""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field


class ConfidenceLevel(StrEnum):
    """How well the retrieved evidence supports the answer."""

    HIGH = "high"              # direct, clear evidence in the retrieved documents
    MEDIUM = "medium"          # partial or indirect evidence requires synthesis
    LOW = "low"                # weak or tangential evidence; answer may not hold
    UNSUPPORTED = "unsupported"  # retrieved documents do not support the answer


class CitedChunk(BaseModel):
    """One citation the LLM produced, referencing a retrieved document chunk.

    ``chunk_id`` is cross-validated against ``GraphState.retrieved_documents``
    by the response node after construction.  Any ``chunk_id`` that does not
    appear in the retrieved set is treated as a hallucinated citation and
    stripped before the answer reaches the user.
    """

    model_config = ConfigDict(frozen=True)

    chunk_id: str = Field(
        min_length=1,
        description="Must exactly match a chunk_id in GraphState.retrieved_documents.",
    )
    document_id: str = Field(min_length=1)
    excerpt: str = Field(
        min_length=1,
        description="The specific passage excerpt supporting this claim.",
    )
    reference_number: int = Field(
        ge=1,
        description="1-based marker used inline in the answer text as [N].",
    )


class ResponseDecision(BaseModel):
    """Structured LLM output from the Response Agent.

    Pydantic validates structure; the node performs the separate citation
    cross-reference check (chunk_ids must exist in the retrieved set).

    ``reasoning_summary`` is internal — it is logged for observability but
    never included in the user-facing response or the graph state.
    """

    model_config = ConfigDict(frozen=True)

    answer: str = Field(
        min_length=1,
        description="The user-facing answer text, with inline [N] citation markers.",
    )
    cited_chunks: list[CitedChunk] = Field(
        default_factory=list,
        description="Citations to retrieved document chunks backing the answer.",
    )
    confidence: ConfidenceLevel = Field(
        description="How well the available evidence supports this answer.",
    )
    limitations: str = Field(
        min_length=1,
        description=(
            "Explicit statement of what the answer cannot address, gaps in evidence, "
            "or uncertainty the user should know about."
        ),
    )
    reasoning_summary: str = Field(
        min_length=1,
        description=(
            "High-level explanation of how the answer was derived from the evidence. "
            "Internal transparency only — not shown to the user."
        ),
    )
