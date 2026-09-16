"""Tests for normalize_scores and reciprocal_rank_fusion.

The fusion function is pure (no I/O), so tests use deterministic inputs
and verify exact numeric behaviour of the RRF formula.  Every result field
on RetrievalEvidence is checked.

Test cases cover:
- Dense-only results (sparse returns nothing)
- Sparse-only results (dense returns nothing)
- Overlapping results (same chunk found by both retrievers)
- RRF score formula correctness
- Rank ordering and tie-breaking
- Source field attribution (DENSE / SPARSE / BOTH)
- Score provenance (dense_score / sparse_score populated correctly)
- top_k limiting
"""

from __future__ import annotations

from datetime import date

import pytest

from src.models.documents import DocumentMetadata
from src.models.enums import AccessLevel, Role
from src.retrieval.hybrid.fusion import (
    DenseHit,
    RRF_K,
    SparseHit,
    normalize_scores,
    reciprocal_rank_fusion,
)
from src.retrieval.hybrid.models import RetrievalEvidence, RetrievalSource
from src.retrieval.ingestion.models import DocumentChunk


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _meta(document_id: str = "DOC-001") -> DocumentMetadata:
    return DocumentMetadata(
        document_id=document_id,
        title="Test Document",
        department="Engineering",
        document_type="architecture_document",
        access_level=AccessLevel.INTERNAL,
        created_date=date(2024, 1, 1),
        allowed_roles=(Role.ENGINEER,),
    )


def _chunk(
    chunk_id: str = "DOC-001-chunk-0000",
    document_id: str = "DOC-001",
    text: str = "sample text",
) -> DocumentChunk:
    return DocumentChunk(
        chunk_id=chunk_id,
        document_id=document_id,
        title="Test Document",
        section="## Overview",
        chunk_index=0,
        chunk_total=1,
        text=text,
        metadata=_meta(document_id),
    )


def _dense(chunk_id: str, score: float = 0.9, rank: int = 1) -> DenseHit:
    doc_id = chunk_id.split("-chunk-")[0]
    return DenseHit(chunk=_chunk(chunk_id=chunk_id, document_id=doc_id), score=score, rank=rank)


def _sparse(chunk_id: str, score: float = 2.0, rank: int = 1) -> SparseHit:
    doc_id = chunk_id.split("-chunk-")[0]
    return SparseHit(chunk=_chunk(chunk_id=chunk_id, document_id=doc_id), score=score, rank=rank)


# ---------------------------------------------------------------------------
# normalize_scores
# ---------------------------------------------------------------------------


class TestNormalizeScores:
    def test_empty_list_returns_empty(self) -> None:
        assert normalize_scores([]) == []

    def test_single_item_returns_one(self) -> None:
        assert normalize_scores([5.0]) == [1.0]

    def test_all_equal_returns_all_ones(self) -> None:
        assert normalize_scores([3.0, 3.0, 3.0]) == [1.0, 1.0, 1.0]

    def test_min_maps_to_zero(self) -> None:
        result = normalize_scores([0.0, 1.0])
        assert result[0] == pytest.approx(0.0)

    def test_max_maps_to_one(self) -> None:
        result = normalize_scores([0.0, 1.0])
        assert result[1] == pytest.approx(1.0)

    def test_midpoint_maps_to_half(self) -> None:
        result = normalize_scores([0.0, 0.5, 1.0])
        assert result[1] == pytest.approx(0.5)

    def test_output_length_matches_input(self) -> None:
        scores = [1.0, 2.0, 3.0, 4.0, 5.0]
        assert len(normalize_scores(scores)) == len(scores)

    def test_all_values_in_zero_one_range(self) -> None:
        result = normalize_scores([10.5, 3.2, 7.8, 1.1, 9.0])
        assert all(0.0 <= v <= 1.0 for v in result)

    def test_order_preserved(self) -> None:
        scores = [3.0, 1.0, 2.0]
        result = normalize_scores(scores)
        # 3.0 → 1.0, 1.0 → 0.0, 2.0 → 0.5
        assert result[0] == pytest.approx(1.0)
        assert result[1] == pytest.approx(0.0)
        assert result[2] == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# reciprocal_rank_fusion — empty / edge cases
# ---------------------------------------------------------------------------


class TestRRFEdgeCases:
    def test_both_empty_returns_empty(self) -> None:
        assert reciprocal_rank_fusion([], []) == []

    def test_empty_sparse_uses_dense_only(self) -> None:
        dense = [_dense("A-chunk-0000"), _dense("B-chunk-0000", rank=2)]
        results = reciprocal_rank_fusion(dense, [])
        ids = [r.chunk_id for r in results]
        assert "A-chunk-0000" in ids
        assert "B-chunk-0000" in ids

    def test_empty_dense_uses_sparse_only(self) -> None:
        sparse = [_sparse("A-chunk-0000"), _sparse("B-chunk-0000", rank=2)]
        results = reciprocal_rank_fusion([], sparse)
        ids = [r.chunk_id for r in results]
        assert "A-chunk-0000" in ids
        assert "B-chunk-0000" in ids

    def test_top_k_limits_output(self) -> None:
        dense = [_dense(f"D-{i:03d}-chunk-0000", rank=i + 1) for i in range(10)]
        results = reciprocal_rank_fusion(dense, [], top_k=3)
        assert len(results) <= 3

    def test_returns_retrieval_evidence_instances(self) -> None:
        dense = [_dense("A-chunk-0000")]
        results = reciprocal_rank_fusion(dense, [])
        assert all(isinstance(r, RetrievalEvidence) for r in results)


# ---------------------------------------------------------------------------
# Dense-only results
# ---------------------------------------------------------------------------


class TestDenseOnlyResults:
    def test_source_is_dense(self) -> None:
        results = reciprocal_rank_fusion([_dense("A-chunk-0000")], [])
        assert results[0].source == RetrievalSource.DENSE

    def test_dense_score_populated(self) -> None:
        results = reciprocal_rank_fusion([_dense("A-chunk-0000", score=0.9)], [])
        assert results[0].dense_score is not None

    def test_sparse_score_is_none(self) -> None:
        results = reciprocal_rank_fusion([_dense("A-chunk-0000")], [])
        assert results[0].sparse_score is None

    def test_final_score_is_rrf_formula(self) -> None:
        # rank 1, k=60 → 1/(60+1)
        results = reciprocal_rank_fusion([_dense("A-chunk-0000", rank=1)], [], k=60)
        expected = 1.0 / (60 + 1)
        assert results[0].final_score == pytest.approx(expected)

    def test_rank_starts_at_one(self) -> None:
        results = reciprocal_rank_fusion([_dense("A-chunk-0000")], [])
        assert results[0].rank == 1

    def test_all_required_fields_present(self) -> None:
        chunk = _chunk()
        dense = [DenseHit(chunk=chunk, score=0.85, rank=1)]
        results = reciprocal_rank_fusion(dense, [])
        ev = results[0]
        assert ev.chunk_id == chunk.chunk_id
        assert ev.document_id == chunk.document_id
        assert ev.title == chunk.title
        assert ev.text == chunk.text
        assert ev.metadata == chunk.metadata
        assert ev.final_score > 0.0


# ---------------------------------------------------------------------------
# Sparse-only results
# ---------------------------------------------------------------------------


class TestSparseOnlyResults:
    def test_source_is_sparse(self) -> None:
        results = reciprocal_rank_fusion([], [_sparse("A-chunk-0000")])
        assert results[0].source == RetrievalSource.SPARSE

    def test_sparse_score_populated(self) -> None:
        results = reciprocal_rank_fusion([], [_sparse("A-chunk-0000", score=3.5)])
        assert results[0].sparse_score is not None

    def test_dense_score_is_none(self) -> None:
        results = reciprocal_rank_fusion([], [_sparse("A-chunk-0000")])
        assert results[0].dense_score is None

    def test_final_score_matches_rrf(self) -> None:
        results = reciprocal_rank_fusion([], [_sparse("A-chunk-0000", rank=1)], k=60)
        expected = 1.0 / (60 + 1)
        assert results[0].final_score == pytest.approx(expected)


# ---------------------------------------------------------------------------
# Overlapping results — same chunk in both retrievers
# ---------------------------------------------------------------------------


class TestOverlappingResults:
    def test_source_is_both(self) -> None:
        dense = [_dense("A-chunk-0000", rank=1)]
        sparse = [_sparse("A-chunk-0000", rank=1)]
        results = reciprocal_rank_fusion(dense, sparse)
        assert results[0].source == RetrievalSource.BOTH

    def test_appears_exactly_once(self) -> None:
        dense = [_dense("A-chunk-0000", rank=1)]
        sparse = [_sparse("A-chunk-0000", rank=1)]
        results = reciprocal_rank_fusion(dense, sparse)
        ids = [r.chunk_id for r in results]
        assert ids.count("A-chunk-0000") == 1

    def test_both_scores_populated(self) -> None:
        dense = [_dense("A-chunk-0000", rank=1)]
        sparse = [_sparse("A-chunk-0000", rank=1)]
        results = reciprocal_rank_fusion(dense, sparse)
        assert results[0].dense_score is not None
        assert results[0].sparse_score is not None

    def test_overlap_score_is_sum_of_contributions(self) -> None:
        k = 60
        dense = [_dense("A-chunk-0000", rank=1)]
        sparse = [_sparse("A-chunk-0000", rank=1)]
        results = reciprocal_rank_fusion(dense, sparse, k=k)
        expected = 1.0 / (k + 1) + 1.0 / (k + 1)
        assert results[0].final_score == pytest.approx(expected)

    def test_overlap_ranks_higher_than_single_retriever_same_rank(self) -> None:
        # Chunk A is in both at rank 2; Chunk B is only in dense at rank 1.
        dense = [_dense("B-chunk-0000", rank=1), _dense("A-chunk-0000", rank=2)]
        sparse = [_sparse("A-chunk-0000", rank=1)]
        results = reciprocal_rank_fusion(dense, sparse)
        result_map = {r.chunk_id: r for r in results}
        # A gets 1/(k+2) + 1/(k+1); B gets 1/(k+1)
        # A's combined score > B's single score since 1/(k+2) > 0
        assert result_map["A-chunk-0000"].final_score > result_map["B-chunk-0000"].final_score


# ---------------------------------------------------------------------------
# Ranking
# ---------------------------------------------------------------------------


class TestRanking:
    def test_higher_rrf_score_gets_lower_rank_number(self) -> None:
        # rank 1 in dense beats rank 3 in dense
        dense = [
            _dense("A-chunk-0000", rank=1),
            _dense("B-chunk-0000", rank=3),
        ]
        results = reciprocal_rank_fusion(dense, [])
        assert results[0].chunk_id == "A-chunk-0000"
        assert results[1].chunk_id == "B-chunk-0000"

    def test_ranks_are_consecutive_from_one(self) -> None:
        dense = [_dense(f"D-{i:03d}-chunk-0000", rank=i + 1) for i in range(5)]
        results = reciprocal_rank_fusion(dense, [])
        for i, r in enumerate(results):
            assert r.rank == i + 1

    def test_final_scores_non_increasing(self) -> None:
        dense = [_dense(f"D-{i:03d}-chunk-0000", rank=i + 1) for i in range(5)]
        results = reciprocal_rank_fusion(dense, [])
        scores = [r.final_score for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_tie_breaking_by_chunk_id_ascending(self) -> None:
        # Two chunks at the same rank in the same retriever → equal scores
        dense = [
            _dense("Z-chunk-0000", rank=1),
            _dense("A-chunk-0000", rank=1),
        ]
        results = reciprocal_rank_fusion(dense, [])
        # Tie-break: chunk_id ascending → A before Z
        assert results[0].chunk_id == "A-chunk-0000"
        assert results[1].chunk_id == "Z-chunk-0000"

    def test_deterministic_output_same_input(self) -> None:
        dense = [_dense(f"D-{i:03d}-chunk-0000", rank=i + 1) for i in range(5)]
        sparse = [_sparse(f"S-{i:03d}-chunk-0000", rank=i + 1) for i in range(3)]
        r1 = [ev.chunk_id for ev in reciprocal_rank_fusion(dense, sparse)]
        r2 = [ev.chunk_id for ev in reciprocal_rank_fusion(dense, sparse)]
        assert r1 == r2

    def test_custom_k_affects_score(self) -> None:
        dense = [_dense("A-chunk-0000", rank=1)]
        r_k10 = reciprocal_rank_fusion(dense, [], k=10)
        r_k60 = reciprocal_rank_fusion(dense, [], k=60)
        # Lower k → larger score (less damping of rank 1)
        assert r_k10[0].final_score > r_k60[0].final_score

    def test_no_overlap_all_chunks_returned(self) -> None:
        dense = [_dense("A-chunk-0000")]
        sparse = [_sparse("B-chunk-0000")]
        results = reciprocal_rank_fusion(dense, sparse, top_k=10)
        ids = {r.chunk_id for r in results}
        assert "A-chunk-0000" in ids
        assert "B-chunk-0000" in ids
