"""Tests for `src.models.documents.DocumentMetadata` and `RetrievedDocument`."""

from __future__ import annotations

from datetime import date, timedelta

import pytest
from pydantic import ValidationError

from src.models.documents import DocumentMetadata, RetrievedDocument
from src.models.enums import AccessLevel, Role


def _metadata(**overrides: object) -> dict[str, object]:
    return {
        "document_id": "doc-1",
        "title": "Retail Lending Policy",
        "department": "Risk",
        "document_type": "POLICY",
        "access_level": AccessLevel.INTERNAL,
        "created_date": date(2024, 1, 15),
        "allowed_roles": (Role.ANALYST, Role.ADMINISTRATOR),
        **overrides,
    }


def test_valid_document_metadata_is_immutable() -> None:
    metadata = DocumentMetadata.model_validate(_metadata())

    assert metadata.access_level is AccessLevel.INTERNAL
    with pytest.raises(ValidationError):
        metadata.title = "New Title"  # type: ignore[misc]


def test_rejects_future_created_date() -> None:
    tomorrow = date.today() + timedelta(days=1)
    with pytest.raises(ValidationError, match="cannot be in the future"):
        DocumentMetadata.model_validate(_metadata(created_date=tomorrow))


def test_rejects_malformed_created_date_string() -> None:
    with pytest.raises(ValidationError):
        DocumentMetadata.model_validate(_metadata(created_date="not-a-date"))


def test_rejects_empty_allowed_roles() -> None:
    with pytest.raises(ValidationError, match="at least one role"):
        DocumentMetadata.model_validate(_metadata(allowed_roles=()))


def test_rejects_invalid_access_level() -> None:
    with pytest.raises(ValidationError):
        DocumentMetadata.model_validate(_metadata(access_level="TOP_SECRET"))


def test_rejects_malformed_document_id() -> None:
    with pytest.raises(ValidationError, match="document_id"):
        DocumentMetadata.model_validate(_metadata(document_id=""))


class TestRetrievedDocument:
    def test_valid_retrieved_document(self) -> None:
        document = RetrievedDocument(
            metadata=DocumentMetadata.model_validate(_metadata()),
            content="Loans over $1M require dual sign-off.",
            relevance_score=0.87,
        )
        assert document.relevance_score == 0.87

    def test_rejects_empty_content(self) -> None:
        with pytest.raises(ValidationError, match="content"):
            RetrievedDocument(
                metadata=DocumentMetadata.model_validate(_metadata()),
                content="   ",
                relevance_score=0.5,
            )

    @pytest.mark.parametrize("score", [-0.01, 1.01])
    def test_rejects_out_of_range_relevance_score(self, score: float) -> None:
        with pytest.raises(ValidationError, match="relevance_score"):
            RetrievedDocument(
                metadata=DocumentMetadata.model_validate(_metadata()),
                content="Some retrieved text.",
                relevance_score=score,
            )
