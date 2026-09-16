"""Domain models for the Research Agent's structured LLM outputs.

Three shapes cover all four LLM-backed stages of the recursive pipeline
(see `src.agents.research.node`):

- ``QuestionAnalysis``  — Stage 1 (research question analysis).
- ``ResearchFinding``   — shared by Stage 6 (recursive/sub-agent analysis)
  and Stage 7 (intermediate result aggregation): both produce "a condensed
  take on the evidence/summaries I was given," just from different inputs.
- ``FinalSynthesis``    — Stage 9 (final evidence synthesis).

``Claim`` and ``Contradiction`` support Stage 8 (contradiction detection),
which is deterministic (no LLM call) — see
`src.agents.research.node._detect_contradictions`.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field


class ResearchConfidence(StrEnum):
    """How well the examined evidence supports a finding at a given tree node."""

    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    UNSUPPORTED = "unsupported"


class QuestionAnalysis(BaseModel):
    """Structured output of Stage 1 — research question analysis."""

    model_config = ConfigDict(frozen=True)

    sub_questions: list[str] = Field(
        min_length=1,
        max_length=6,
        description="Focused sub-questions that together answer the research question.",
    )
    key_entities: list[str] = Field(
        default_factory=list,
        description="Named entities (incident IDs, systems, dates) central to the question.",
    )
    scope_notes: str = Field(
        min_length=1,
        description="Brief note on scope or ambiguity the search plan should account for.",
    )


class Claim(BaseModel):
    """One short, structured factual assertion extracted from a batch of evidence.

    ``subject`` + ``attribute`` together identify what the claim is about
    (e.g. subject="INC-2024-001", attribute="root_cause"); ``value`` is the
    asserted answer. Two claims sharing a ``(subject, attribute)`` pair with
    different ``value``s are a contradiction — see
    `src.agents.research.node._detect_contradictions`.
    """

    model_config = ConfigDict(frozen=True)

    subject: str = Field(min_length=1)
    attribute: str = Field(min_length=1)
    value: str = Field(min_length=1)
    chunk_ids: list[str] = Field(default_factory=list)


class ResearchFinding(BaseModel):
    """Structured output shared by Stage 6 (batch analysis) and Stage 7
    (aggregation).

    Never contains raw document text — only a synthesized summary and the
    structured claims extracted from it. This is what keeps every level of
    the recursion bounded: a parent task only ever sees its children's
    ``ResearchFinding``s, never the underlying evidence chunks.
    """

    model_config = ConfigDict(frozen=True)

    summary: str = Field(min_length=1)
    claims: list[Claim] = Field(default_factory=list)
    confidence: ResearchConfidence


class Contradiction(BaseModel):
    """One detected conflict between claims sharing the same subject/attribute."""

    model_config = ConfigDict(frozen=True)

    subject: str = Field(min_length=1)
    attribute: str = Field(min_length=1)
    conflicting_values: tuple[str, ...] = Field(min_length=2)
    chunk_ids: tuple[str, ...] = Field(default_factory=tuple)

    @property
    def description(self) -> str:
        """Human-readable summary, suitable for `ResearchResult.contradictions`."""
        values = " vs. ".join(self.conflicting_values)
        return f"Conflicting {self.attribute} reported for {self.subject}: {values}"


class FinalSynthesis(BaseModel):
    """Structured output of Stage 9 — the final, corpus-wide synthesis."""

    model_config = ConfigDict(frozen=True)

    summary: str = Field(min_length=1)
    key_findings: list[str] = Field(min_length=1)
    limitations: str = Field(min_length=1)
    confidence: ResearchConfidence
