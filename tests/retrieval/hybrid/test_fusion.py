"""Unit tests for reciprocal_rank_fusion and related models.

The fusion function is pure (no I/O), making it straightforward to test
with deterministic inputs.  Tests cover: empty inputs, single-retriever
results, overlapping results, RRF score ordering, top-k limiting,
tie-breaking, and rank provenance (dense_rank / sparse_rank).
"""

from __future__ import annotations

from datetime import date

import pytest

from src.models.documents import DocumentMetadata
from src.models.enums import AccessLevel, Role
from src.retrieval.bm25.corpus import BM25Result
from src.retrieval.hybrid.fusion import (
    DenseSearchResult,
    HybridResult,
    RRF_K,
    reciprocal_rank_fusion,
)
from src.retrieval.ingestion.models import DocumentChunk


# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------


def _make_meta(document_id: str = "DOC-001") -> DocumentMetadata:
    return DocumentMetadata(
        document_id=document_id,
        title="Test Document",
        department="Engineering",
        document_type="architecture_document",
        access_level=AccessLevel.INTERNAL,
        created_date=date(2024, 1, 1),
        allowed_roles=(Role.ENGINEER,),
    )


def _make_chunk(
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
        metadata=_make_meta(document_id),
    )


def _dense(chunk_id: str, score: float, rank: int) -> DenseSearchResult:
    return DenseSearchResult(
        chunk=_make_chunk(chunk_id=chunk_id, document_id=chunk_id.split("-chunk-")[0]),
        score=score,
        rank=rank,
    )


def _sparse(chunk_id: str, score: float, rank: int) -> BM25Result:
    return BM25Result(
        chunk=_make_chunk(chunk_id=chunk_id, document_id=chunk_id.split("-chunk-")[0]),
        score=score,
        rank=rank,
    )


# ---------------------------------------------------------------------------
# DenseSearchResult model
# ---------------------------------------------------------------------------


class TestDenseSearchResult:
    def test_is_frozen(self) -> None:
        chunk = _make_chunk()
        dr = DenseSearchResult(chunk=chunk, score=0.9, rank=1)
        with pytest.raises(Exception):
            dr.score = 0.1  # type: ignore[misc]

    def test_score_ge_zero(self) -> None:
        chunk = _make_chunk()
        with pytest.raises(Exception):
            DenseSearchResult(chunk=chunk, score=-0.1, rank=1)

    def test_rank_ge_one(self) -> None:
        chunk = _make_chunk()
        with pytest.raises(Exception):
            DenseSearchResult(chunk=chunk, score=0.9, rank=0)


# ---------------------------------------------------------------------------
# HybridResult model
# ---------------------------------------------------------------------------


class TestHybridResult:
    def test_is_frozen(self) -> None:
        chunk = _make_chunk()
        hr = HybridResult(chunk=chunk, rrf_score=0.5, rank=1)
        with pytest.raises(Exception):
            hr.rank = 2  # type: ignore[misc]

    def test_dense_rank_defaults_to_none(self) -> None:
        hr = HybridResult(chunk=_make_chunk(), rrf_score=0.5, rank=1)
        assert hr.dense_rank is None

    def test_sparse_rank_defaults_to_none(self) -> None:
        hr = HybridResult(chunk=_make_chunk(), rrf_score=0.5, rank=1)
        assert hr.sparse_rank is None


# ---------------------------------------------------------------------------
# reciprocal_rank_fusion
# ---------------------------------------------------------------------------


class TestReciprocalRankFusion:
    def test_empty_both_returns_empty(self) -> None:
        assert reciprocal_rank_fusion([], []) == []

    def test_empty_dense_returns_sparse_only(self) -> None:
        sparse = [_sparse("A-chunk-0000", 2.5, 1), _sparse("B-chunk-0000", 1.5, 2)]
        results = reciprocal_rank_fusion([], sparse)
        ids = [r.chunk.chunk_id for r in results]
        assert "A-chunk-0000" in ids
        assert "B-chunk-0000" in ids

    def test_empty_sparse_returns_dense_only(self) -> None:
        dense = [_dense("A-chunk-0000", 0.9, 1), _dense("B-chunk-0000", 0.8, 2)]
        results = reciprocal_rank_fusion(dense, [])
        ids = [r.chunk.chunk_id for r in results]
        assert "A-chunk-0000" in ids
        assert "B-chunk-0000" in ids

    def test_dense_only_ordered_by_rrf_score(self) -> None:
        dense = [
            _dense("A-chunk-0000", 0.9, 1),
            _dense("B-chunk-0000", 0.8, 2),
            _dense("C-chunk-0000", 0.7, 3),
        ]
        results = reciprocal_rank_fusion(dense, [])
        # Higher dense rank (lower rank number) → higher RRF score
        assert results[0].chunk.chunk_id == "A-chunk-0000"
        assert results[1].chunk.chunk_id == "B-chunk-0000"
        assert results[2].chunk.chunk_id == "C-chunk-0000"

    def test_overlap_gives_higher_rrf_score(self) -> None:
        # Chunk A appears in both; Chunk B appears only in dense
        chunk_a_dense = _dense("A-chunk-0000", 0.85, 2)  # rank 2 in dense
        chunk_b_dense = _dense("B-chunk-0000", 0.90, 1)  # rank 1 in dense only
        chunk_a_sparse = _sparse("A-chunk-0000", 3.0, 1)  # rank 1 in sparse

        results = reciprocal_rank_fusion([chunk_a_dense, chunk_b_dense], [chunk_a_sparse])

        # A: 1/(k+2) + 1/(k+1), B: 1/(k+1) only
        # A's combined score > B's single-retriever score
        result_map = {r.chunk.chunk_id: r for r in results}
        assert result_map["A-chunk-0000"].rrf_score > result_map["B-chunk-0000"].rrf_score

    def test_rrf_score_formula(self) -> None:
        k = RRF_K
        dense = [_dense("A-chunk-0000", 0.9, 1)]
        sparse = [_sparse("A-chunk-0000", 2.0, 1)]
        results = reciprocal_rank_fusion(dense, sparse, k=k)
        expected = 1.0 / (k + 1) + 1.0 / (k + 1)
        assert abs(results[0].rrf_score - expected) < 1e-12

    def test_custom_k_parameter(self) -> None:
        dense = [_dense("A-chunk-0000", 0.9, 1)]
        results_k10 = reciprocal_rank_fusion(dense, [], k=10)
        results_k60 = reciprocal_rank_fusion(dense, [], k=60)
        # Lower k → higher RRF score for same rank
        assert results_k10[0].rrf_score > results_k60[0].rrf_score

    def test_top_k_limits_results(self) -> None:
        dense = [_dense(f"DOC-{i:03d}-chunk-0000", 0.9 - i * 0.05, i + 1) for i in range(10)]
        results = reciprocal_rank_fusion(dense, [], top_k=3)
        assert len(results) <= 3

    def test_ranks_are_sequential_from_one(self) -> None:
        dense = [_dense(f"DOC-{i:03d}-chunk-0000", 0.9 - i * 0.1, i + 1) for i in range(4)]
        results = reciprocal_rank_fusion(dense, [])
        for i, r in enumerate(results):
            assert r.rank == i + 1

    def test_results_are_hybrid_result_instances(self) -> None:
        dense = [_dense("A-chunk-0000", 0.9, 1)]
        results = reciprocal_rank_fusion(dense, [])
        assert all(isinstance(r, HybridResult) for r in results)

    def test_dense_rank_provenance(self) -> None:
        dense = [_dense("A-chunk-0000", 0.9, 1)]
        results = reciprocal_rank_fusion(dense, [])
        assert results[0].dense_rank == 1

    def test_sparse_rank_provenance(self) -> None:
        sparse = [_sparse("A-chunk-0000", 2.0, 1)]
        results = reciprocal_rank_fusion([], sparse)
        assert results[0].sparse_rank == 1

    def test_dense_only_chunk_has_no_sparse_rank(self) -> None:
        dense = [_dense("A-chunk-0000", 0.9, 1)]
        results = reciprocal_rank_fusion(dense, [])
        assert results[0].sparse_rank is None

    def test_sparse_only_chunk_has_no_dense_rank(self) -> None:
        sparse = [_sparse("A-chunk-0000", 2.0, 1)]
        results = reciprocal_rank_fusion([], sparse)
        assert results[0].dense_rank is None

    def test_overlap_chunk_has_both_ranks(self) -> None:
        dense = [_dense("A-chunk-0000", 0.9, 2)]
        sparse = [_sparse("A-chunk-0000", 2.0, 3)]
        results = reciprocal_rank_fusion(dense, sparse)
        result = results[0]
        assert result.dense_rank == 2
        assert result.sparse_rank == 3

    def test_deterministic_tie_breaking_by_chunk_id(self) -> None:
        # Three chunks at the same dense rank — should sort by chunk_id ascending
        dense = [
            _dense("Z-chunk-0000", 0.9, 1),
            _dense("A-chunk-0000", 0.9, 1),
            _dense("M-chunk-0000", 0.9, 1),
        ]
        results_1 = reciprocal_rank_fusion(dense, [])
        results_2 = reciprocal_rank_fusion(dense, [])
        assert [r.chunk.chunk_id for r in results_1] == [r.chunk.chunk_id for r in results_2]

    def test_all_chunks_returned_when_no_overlap(self) -> None:
        dense = [_dense("A-chunk-0000", 0.9, 1)]
        sparse = [_sparse("B-chunk-0000", 2.0, 1)]
        results = reciprocal_rank_fusion(dense, sparse, top_k=10)
        ids = {r.chunk.chunk_id for r in results}
        assert "A-chunk-0000" in ids
        assert "B-chunk-0000" in ids
