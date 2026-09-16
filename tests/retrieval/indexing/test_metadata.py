"""Tests for chunk_to_metadata, metadata_to_chunk, and build_access_filter."""

from __future__ import annotations

from datetime import date

import pytest

from src.models.documents import DocumentMetadata
from src.models.enums import AccessLevel, Role
from src.retrieval.indexing.metadata import (
    build_access_filter,
    chunk_to_metadata,
    metadata_to_chunk,
)
from src.retrieval.ingestion.models import DocumentChunk


# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------


def _make_chunk(
    *,
    chunk_id: str = "ARCH-001-chunk-0000",
    document_id: str = "ARCH-001",
    title: str = "Test Document",
    section: str = "## Overview",
    chunk_index: int = 0,
    chunk_total: int = 3,
    text: str = "Chunk text content.",
    department: str = "Engineering",
    document_type: str = "architecture_document",
    access_level: AccessLevel = AccessLevel.INTERNAL,
    created_date: date = date(2024, 1, 15),
    allowed_roles: tuple[Role, ...] = (Role.ENGINEER, Role.ANALYST),
) -> DocumentChunk:
    meta = DocumentMetadata(
        document_id=document_id,
        title=title,
        department=department,
        document_type=document_type,
        access_level=access_level,
        created_date=created_date,
        allowed_roles=allowed_roles,
    )
    return DocumentChunk(
        chunk_id=chunk_id,
        document_id=document_id,
        title=title,
        section=section,
        chunk_index=chunk_index,
        chunk_total=chunk_total,
        text=text,
        metadata=meta,
    )


# ---------------------------------------------------------------------------
# chunk_to_metadata — field presence and types
# ---------------------------------------------------------------------------


class TestChunkToMetadata:
    def test_returns_dict(self) -> None:
        result = chunk_to_metadata(_make_chunk())
        assert isinstance(result, dict)

    def test_chunk_id(self) -> None:
        result = chunk_to_metadata(_make_chunk(chunk_id="ARCH-001-chunk-0007"))
        assert result["chunk_id"] == "ARCH-001-chunk-0007"

    def test_document_id(self) -> None:
        result = chunk_to_metadata(_make_chunk(document_id="ARCH-001"))
        assert result["document_id"] == "ARCH-001"

    def test_title(self) -> None:
        result = chunk_to_metadata(_make_chunk(title="Core Banking Ledger"))
        assert result["title"] == "Core Banking Ledger"

    def test_section(self) -> None:
        result = chunk_to_metadata(_make_chunk(section="## Root Cause"))
        assert result["section"] == "## Root Cause"

    def test_chunk_index_is_int(self) -> None:
        result = chunk_to_metadata(_make_chunk(chunk_index=2))
        assert result["chunk_index"] == 2
        assert isinstance(result["chunk_index"], int)

    def test_chunk_total_is_int(self) -> None:
        result = chunk_to_metadata(_make_chunk(chunk_total=5))
        assert result["chunk_total"] == 5
        assert isinstance(result["chunk_total"], int)

    def test_text_stored(self) -> None:
        result = chunk_to_metadata(_make_chunk(text="Unique text value."))
        assert result["text"] == "Unique text value."

    def test_department(self) -> None:
        result = chunk_to_metadata(_make_chunk(department="Payments Engineering"))
        assert result["department"] == "Payments Engineering"

    def test_document_type(self) -> None:
        result = chunk_to_metadata(_make_chunk(document_type="incident_report"))
        assert result["document_type"] == "incident_report"

    def test_access_level_is_string(self) -> None:
        result = chunk_to_metadata(_make_chunk(access_level=AccessLevel.RESTRICTED))
        assert result["access_level"] == "RESTRICTED"
        assert isinstance(result["access_level"], str)

    def test_created_date_is_iso_string(self) -> None:
        result = chunk_to_metadata(_make_chunk(created_date=date(2024, 3, 7)))
        assert result["created_date"] == "2024-03-07"
        assert isinstance(result["created_date"], str)

    def test_allowed_roles_is_list_of_strings(self) -> None:
        result = chunk_to_metadata(
            _make_chunk(allowed_roles=(Role.ENGINEER, Role.ANALYST, Role.ADMINISTRATOR))
        )
        roles = result["allowed_roles"]
        assert isinstance(roles, list)
        assert roles == ["ENGINEER", "ANALYST", "ADMINISTRATOR"]

    def test_single_allowed_role(self) -> None:
        result = chunk_to_metadata(_make_chunk(allowed_roles=(Role.ADMINISTRATOR,)))
        assert result["allowed_roles"] == ["ADMINISTRATOR"]

    def test_all_required_keys_present(self) -> None:
        required = {
            "chunk_id",
            "document_id",
            "title",
            "section",
            "chunk_index",
            "chunk_total",
            "text",
            "department",
            "document_type",
            "access_level",
            "created_date",
            "allowed_roles",
        }
        result = chunk_to_metadata(_make_chunk())
        assert required.issubset(result.keys())

    def test_public_access_level(self) -> None:
        result = chunk_to_metadata(_make_chunk(access_level=AccessLevel.PUBLIC))
        assert result["access_level"] == "PUBLIC"

    def test_internal_access_level(self) -> None:
        result = chunk_to_metadata(_make_chunk(access_level=AccessLevel.INTERNAL))
        assert result["access_level"] == "INTERNAL"

    def test_confidential_access_level(self) -> None:
        result = chunk_to_metadata(_make_chunk(access_level=AccessLevel.CONFIDENTIAL))
        assert result["access_level"] == "CONFIDENTIAL"

    def test_metadata_is_independent_of_source(self) -> None:
        chunk = _make_chunk()
        result1 = chunk_to_metadata(chunk)
        result2 = chunk_to_metadata(chunk)
        assert result1 == result2


# ---------------------------------------------------------------------------
# build_access_filter — filter structure
# ---------------------------------------------------------------------------


class TestBuildAccessFilter:
    def test_single_role_produces_in_filter(self) -> None:
        result = build_access_filter([Role.ENGINEER])
        assert result == {"allowed_roles": {"$in": ["ENGINEER"]}}

    def test_multiple_roles_all_included(self) -> None:
        result = build_access_filter([Role.ENGINEER, Role.ANALYST])
        assert set(result["allowed_roles"]["$in"]) == {"ENGINEER", "ANALYST"}  # type: ignore[index]

    def test_roles_are_string_values(self) -> None:
        result = build_access_filter([Role.ADMINISTRATOR])
        roles = result["allowed_roles"]["$in"]  # type: ignore[index]
        assert all(isinstance(r, str) for r in roles)

    def test_no_access_levels_omits_access_level_key(self) -> None:
        result = build_access_filter([Role.ENGINEER])
        assert "access_level" not in result

    def test_with_access_levels_includes_key(self) -> None:
        result = build_access_filter(
            [Role.ANALYST],
            access_levels=[AccessLevel.INTERNAL],
        )
        assert "access_level" in result

    def test_access_level_filter_uses_in_operator(self) -> None:
        result = build_access_filter(
            [Role.ANALYST],
            access_levels=[AccessLevel.INTERNAL],
        )
        assert result["access_level"] == {"$in": ["INTERNAL"]}  # type: ignore[index]

    def test_multiple_access_levels(self) -> None:
        result = build_access_filter(
            [Role.ENGINEER],
            access_levels=[AccessLevel.PUBLIC, AccessLevel.INTERNAL],
        )
        levels = result["access_level"]["$in"]  # type: ignore[index]
        assert set(levels) == {"PUBLIC", "INTERNAL"}

    def test_access_levels_are_strings(self) -> None:
        result = build_access_filter(
            [Role.ANALYST],
            access_levels=[AccessLevel.RESTRICTED],
        )
        levels = result["access_level"]["$in"]  # type: ignore[index]
        assert all(isinstance(lv, str) for lv in levels)

    def test_empty_access_levels_sequence_treated_as_absent(self) -> None:
        result = build_access_filter([Role.ENGINEER], access_levels=[])
        assert "access_level" not in result

    def test_all_roles_included(self) -> None:
        all_roles = list(Role)
        result = build_access_filter(all_roles)
        role_strings = set(result["allowed_roles"]["$in"])  # type: ignore[index]
        assert role_strings == {r.value for r in Role}


# ---------------------------------------------------------------------------
# metadata_to_chunk — inverse of chunk_to_metadata
# ---------------------------------------------------------------------------


class TestMetadataToChunk:
    def test_round_trip(self) -> None:
        original = _make_chunk()
        meta = chunk_to_metadata(original)
        # metadata_to_chunk accepts dict[str, object]
        reconstructed = metadata_to_chunk(dict(meta))  # type: ignore[arg-type]
        assert reconstructed.chunk_id == original.chunk_id
        assert reconstructed.document_id == original.document_id
        assert reconstructed.title == original.title
        assert reconstructed.section == original.section
        assert reconstructed.chunk_index == original.chunk_index
        assert reconstructed.chunk_total == original.chunk_total
        assert reconstructed.text == original.text

    def test_metadata_fields_round_trip(self) -> None:
        original = _make_chunk(
            department="Payments Engineering",
            document_type="incident_report",
            access_level=AccessLevel.CONFIDENTIAL,
            created_date=date(2024, 6, 30),
            allowed_roles=(Role.ANALYST, Role.ADMINISTRATOR),
        )
        meta = chunk_to_metadata(original)
        reconstructed = metadata_to_chunk(dict(meta))  # type: ignore[arg-type]
        assert reconstructed.metadata.department == "Payments Engineering"
        assert reconstructed.metadata.document_type == "incident_report"
        assert reconstructed.metadata.access_level == AccessLevel.CONFIDENTIAL
        assert reconstructed.metadata.created_date == date(2024, 6, 30)
        assert set(reconstructed.metadata.allowed_roles) == {Role.ANALYST, Role.ADMINISTRATOR}

    def test_chunk_index_preserved(self) -> None:
        original = _make_chunk(chunk_index=7, chunk_total=10)
        meta = chunk_to_metadata(original)
        reconstructed = metadata_to_chunk(dict(meta))  # type: ignore[arg-type]
        assert reconstructed.chunk_index == 7
        assert reconstructed.chunk_total == 10

    def test_chunk_index_is_int_in_result(self) -> None:
        meta = chunk_to_metadata(_make_chunk(chunk_index=3))
        reconstructed = metadata_to_chunk(dict(meta))  # type: ignore[arg-type]
        assert isinstance(reconstructed.chunk_index, int)

    def test_chunk_index_from_float_value(self) -> None:
        # Pinecone may return numeric metadata as float
        meta: dict[str, object] = dict(chunk_to_metadata(_make_chunk(chunk_index=2)))  # type: ignore[arg-type]
        meta["chunk_index"] = 2.0
        meta["chunk_total"] = 5.0
        reconstructed = metadata_to_chunk(meta)
        assert reconstructed.chunk_index == 2
        assert reconstructed.chunk_total == 5

    def test_raises_on_non_numeric_chunk_index(self) -> None:
        meta: dict[str, object] = dict(chunk_to_metadata(_make_chunk()))  # type: ignore[arg-type]
        meta["chunk_index"] = "not-a-number"
        with pytest.raises(ValueError, match="chunk_index"):
            metadata_to_chunk(meta)

    def test_raises_on_non_list_allowed_roles(self) -> None:
        meta: dict[str, object] = dict(chunk_to_metadata(_make_chunk()))  # type: ignore[arg-type]
        meta["allowed_roles"] = "ENGINEER"
        with pytest.raises(ValueError, match="allowed_roles"):
            metadata_to_chunk(meta)

    def test_allowed_roles_order_preserved(self) -> None:
        original = _make_chunk(allowed_roles=(Role.ENGINEER, Role.ANALYST, Role.ADMINISTRATOR))
        meta = chunk_to_metadata(original)
        reconstructed = metadata_to_chunk(dict(meta))  # type: ignore[arg-type]
        assert list(reconstructed.metadata.allowed_roles) == [
            Role.ENGINEER, Role.ANALYST, Role.ADMINISTRATOR
        ]

    def test_all_access_levels_round_trip(self) -> None:
        for level in AccessLevel:
            original = _make_chunk(access_level=level)
            meta = chunk_to_metadata(original)
            reconstructed = metadata_to_chunk(dict(meta))  # type: ignore[arg-type]
            assert reconstructed.metadata.access_level == level
