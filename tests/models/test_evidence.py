"""Tests for `src.models.evidence.Evidence` and `Citation`."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.models.evidence import Citation, Evidence


def test_valid_evidence() -> None:
    evidence = Evidence(
        evidence_id="ev-1",
        document_id="doc-1",
        excerpt="Loans over $1M require dual sign-off.",
        relevance_score=0.9,
    )
    assert evidence.relevance_score == 0.9


def test_evidence_rejects_empty_excerpt() -> None:
    with pytest.raises(ValidationError, match="excerpt"):
        Evidence(evidence_id="ev-1", document_id="doc-1", excerpt="  ", relevance_score=0.9)


def test_valid_citation() -> None:
    citation = Citation(evidence_id="ev-1", reference_number=1)
    assert citation.reference_number == 1


def test_citation_rejects_non_positive_reference_number() -> None:
    with pytest.raises(ValidationError, match="reference_number"):
        Citation(evidence_id="ev-1", reference_number=0)
