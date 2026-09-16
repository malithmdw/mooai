"""Tests for apply_access_filter.

Access filtering is the authoritative post-RRF RBAC gate.  Tests verify:
- Metadata filtering (department, document_type)
- Role-based access control (unauthorized documents removed)
- Access-level filtering
- Combined predicates
- Rank re-assignment after filtering
- No-filter passthrough
"""

from __future__ import annotations

from datetime import date

import pytest

from src.models.documents import DocumentMetadata
from src.models.enums import AccessLevel, Role
from src.retrieval.hybrid.filtering import apply_access_filter
from src.retrieval.hybrid.models import RetrievalEvidence, RetrievalSource
from src.retrieval.ingestion.models import DocumentChunk


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _meta(
    document_id: str = "DOC-001",
    department: str = "Engineering",
    document_type: str = "architecture_document",
    access_level: AccessLevel = AccessLevel.INTERNAL,
    allowed_roles: tuple[Role, ...] = (Role.ENGINEER,),
) -> DocumentMetadata:
    return DocumentMetadata(
        document_id=document_id,
        title="Test Document",
        department=department,
        document_type=document_type,
        access_level=access_level,
        created_date=date(2024, 1, 1),
        allowed_roles=allowed_roles,
    )


def _evidence(
    chunk_id: str = "DOC-001-chunk-0000",
    document_id: str = "DOC-001",
    rank: int = 1,
    source: RetrievalSource = RetrievalSource.DENSE,
    final_score: float = 0.5,
    metadata: DocumentMetadata | None = None,
) -> RetrievalEvidence:
    meta = metadata or _meta(document_id=document_id)
    return RetrievalEvidence(
        chunk_id=chunk_id,
        document_id=document_id,
        title="Test Document",
        text="Sample text content.",
        metadata=meta,
        source=source,
        dense_score=0.9,
        sparse_score=None,
        final_score=final_score,
        rank=rank,
    )


# ---------------------------------------------------------------------------
# Passthrough — no filters active
# ---------------------------------------------------------------------------


class TestNoFilters:
    def test_no_filter_returns_all(self) -> None:
        ev = [_evidence("A-chunk-0000"), _evidence("B-chunk-0000", rank=2)]
        result = apply_access_filter(ev)
        assert len(result) == 2

    def test_empty_input_returns_empty(self) -> None:
        assert apply_access_filter([]) == []

    def test_preserves_all_fields_except_rank(self) -> None:
        ev = _evidence("A-chunk-0000", final_score=0.75, source=RetrievalSource.BOTH)
        result = apply_access_filter([ev])
        assert result[0].chunk_id == ev.chunk_id
        assert result[0].final_score == ev.final_score
        assert result[0].source == ev.source


# ---------------------------------------------------------------------------
# Role-based access control — unauthorized documents removed
# ---------------------------------------------------------------------------


class TestRoleFiltering:
    def test_authorized_role_passes(self) -> None:
        ev = _evidence(metadata=_meta(allowed_roles=(Role.ENGINEER,)))
        result = apply_access_filter([ev], roles=[Role.ENGINEER])
        assert len(result) == 1

    def test_unauthorized_role_removed(self) -> None:
        ev = _evidence(metadata=_meta(allowed_roles=(Role.ADMINISTRATOR,)))
        result = apply_access_filter([ev], roles=[Role.ENGINEER])
        assert result == []

    def test_multi_role_document_accessible_to_any_permitted_role(self) -> None:
        ev = _evidence(
            metadata=_meta(allowed_roles=(Role.ENGINEER, Role.ANALYST))
        )
        assert len(apply_access_filter([ev], roles=[Role.ENGINEER])) == 1
        assert len(apply_access_filter([ev], roles=[Role.ANALYST])) == 1

    def test_viewer_blocked_from_engineer_only_document(self) -> None:
        ev = _evidence(metadata=_meta(allowed_roles=(Role.ENGINEER,)))
        result = apply_access_filter([ev], roles=[Role.VIEWER])
        assert result == []

    def test_mixed_authorized_and_unauthorized(self) -> None:
        allowed = _evidence(
            "DOC-001-chunk-0000", rank=1,
            metadata=_meta("DOC-001", allowed_roles=(Role.ENGINEER,)),
        )
        blocked = _evidence(
            "DOC-002-chunk-0000", rank=2,
            metadata=_meta("DOC-002", allowed_roles=(Role.ADMINISTRATOR,)),
        )
        result = apply_access_filter([allowed, blocked], roles=[Role.ENGINEER])
        assert len(result) == 1
        assert result[0].chunk_id == "DOC-001-chunk-0000"

    def test_empty_role_set_blocks_everything(self) -> None:
        ev = _evidence(metadata=_meta(allowed_roles=(Role.ENGINEER,)))
        result = apply_access_filter([ev], roles=[])
        assert result == []


# ---------------------------------------------------------------------------
# Access-level filtering
# ---------------------------------------------------------------------------


class TestAccessLevelFiltering:
    def test_matching_access_level_passes(self) -> None:
        ev = _evidence(metadata=_meta(access_level=AccessLevel.INTERNAL))
        result = apply_access_filter([ev], access_levels=[AccessLevel.INTERNAL])
        assert len(result) == 1

    def test_non_matching_access_level_removed(self) -> None:
        ev = _evidence(metadata=_meta(access_level=AccessLevel.CONFIDENTIAL))
        result = apply_access_filter([ev], access_levels=[AccessLevel.INTERNAL])
        assert result == []

    def test_multiple_access_levels_allowed(self) -> None:
        public_ev = _evidence(
            "PUB-chunk-0000",
            metadata=_meta("PUB", access_level=AccessLevel.PUBLIC),
        )
        internal_ev = _evidence(
            "INT-chunk-0000",
            metadata=_meta("INT", access_level=AccessLevel.INTERNAL),
        )
        restricted_ev = _evidence(
            "RES-chunk-0000",
            metadata=_meta("RES", access_level=AccessLevel.RESTRICTED),
        )
        result = apply_access_filter(
            [public_ev, internal_ev, restricted_ev],
            access_levels=[AccessLevel.PUBLIC, AccessLevel.INTERNAL],
        )
        ids = {r.chunk_id for r in result}
        assert "PUB-chunk-0000" in ids
        assert "INT-chunk-0000" in ids
        assert "RES-chunk-0000" not in ids


# ---------------------------------------------------------------------------
# Metadata filtering — department and document_type
# ---------------------------------------------------------------------------


class TestMetadataFiltering:
    def test_matching_department_passes(self) -> None:
        ev = _evidence(metadata=_meta(department="Core Banking"))
        result = apply_access_filter([ev], department="Core Banking")
        assert len(result) == 1

    def test_non_matching_department_removed(self) -> None:
        ev = _evidence(metadata=_meta(department="FPS"))
        result = apply_access_filter([ev], department="Core Banking")
        assert result == []

    def test_matching_document_type_passes(self) -> None:
        ev = _evidence(metadata=_meta(document_type="runbook"))
        result = apply_access_filter([ev], document_type="runbook")
        assert len(result) == 1

    def test_non_matching_document_type_removed(self) -> None:
        ev = _evidence(metadata=_meta(document_type="architecture_document"))
        result = apply_access_filter([ev], document_type="runbook")
        assert result == []


# ---------------------------------------------------------------------------
# Combined predicates
# ---------------------------------------------------------------------------


class TestCombinedFilters:
    def test_all_predicates_must_pass(self) -> None:
        # All matching → passes
        ev = _evidence(
            metadata=_meta(
                allowed_roles=(Role.ENGINEER,),
                access_level=AccessLevel.INTERNAL,
                department="Engineering",
                document_type="runbook",
            )
        )
        assert len(apply_access_filter(
            [ev],
            roles=[Role.ENGINEER],
            access_levels=[AccessLevel.INTERNAL],
            department="Engineering",
            document_type="runbook",
        )) == 1

    def test_one_failing_predicate_removes_result(self) -> None:
        ev = _evidence(
            metadata=_meta(
                allowed_roles=(Role.ENGINEER,),
                access_level=AccessLevel.INTERNAL,
                department="FPS",   # ← wrong department
                document_type="runbook",
            )
        )
        assert apply_access_filter(
            [ev],
            roles=[Role.ENGINEER],
            department="Engineering",  # ← won't match "FPS"
        ) == []


# ---------------------------------------------------------------------------
# Rank re-assignment after filtering
# ---------------------------------------------------------------------------


class TestRankReassignment:
    def test_ranks_reassigned_consecutively_after_filter(self) -> None:
        # 3 results; middle one is filtered out
        ev1 = _evidence(
            "DOC-001-chunk-0000", rank=1,
            metadata=_meta("DOC-001", allowed_roles=(Role.ENGINEER,)),
        )
        ev2 = _evidence(
            "DOC-002-chunk-0000", rank=2,
            metadata=_meta("DOC-002", allowed_roles=(Role.ADMINISTRATOR,)),
        )
        ev3 = _evidence(
            "DOC-003-chunk-0000", rank=3,
            metadata=_meta("DOC-003", allowed_roles=(Role.ENGINEER,)),
        )
        result = apply_access_filter([ev1, ev2, ev3], roles=[Role.ENGINEER])
        assert len(result) == 2
        assert result[0].rank == 1
        assert result[1].rank == 2

    def test_single_survivor_gets_rank_one(self) -> None:
        ev = _evidence(
            metadata=_meta(allowed_roles=(Role.ENGINEER,)),
            rank=5,  # original rank 5
        )
        result = apply_access_filter([ev], roles=[Role.ENGINEER])
        assert result[0].rank == 1

    def test_no_filter_ranks_unchanged(self) -> None:
        ev = [_evidence("A-chunk-0000", rank=1), _evidence("B-chunk-0000", rank=2)]
        result = apply_access_filter(ev)
        assert result[0].rank == 1
        assert result[1].rank == 2
