"""Tests for HybridRetriever — full pipeline integration.

All external dependencies (Pinecone, embedder, BM25) are mocked so tests
run without network access.  Tests verify the full pipeline:
embed → search-in-parallel → RRF → access filter → rerank.

Covers:
- Dense-only and sparse-only pipeline paths
- Overlapping results (same chunk from both retrievers)
- Metadata and access-control filter forwarding
- Unauthorized documents removed before results are returned
- Ranking of final results
- Reranker injection
"""

from __future__ import annotations

from datetime import date
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.models.documents import DocumentMetadata
from src.models.enums import AccessLevel, Role
from src.retrieval.bm25.corpus import BM25Result
from src.retrieval.bm25.service import BM25Service
from src.retrieval.embedding.base import EmbeddingProvider
from src.retrieval.hybrid.models import RetrievalEvidence, RetrievalSource
from src.retrieval.hybrid.reranking import IdentityReranker, Reranker
from src.retrieval.hybrid.retriever import HybridRetriever
from src.retrieval.indexing.service import PineconeIndexService, PineconeMatch
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


def _chunk(
    chunk_id: str = "DOC-001-chunk-0000",
    document_id: str = "DOC-001",
    text: str = "sample text",
    metadata: DocumentMetadata | None = None,
) -> DocumentChunk:
    return DocumentChunk(
        chunk_id=chunk_id,
        document_id=document_id,
        title="Test Document",
        section="## Overview",
        chunk_index=0,
        chunk_total=1,
        text=text,
        metadata=metadata or _meta(document_id),
    )


def _pinecone_meta(chunk: DocumentChunk) -> dict[str, object]:
    """Build a Pinecone-compatible metadata dict for a chunk."""
    return {
        "chunk_id": chunk.chunk_id,
        "document_id": chunk.document_id,
        "title": chunk.title,
        "section": chunk.section,
        "chunk_index": chunk.chunk_index,
        "chunk_total": chunk.chunk_total,
        "text": chunk.text,
        "department": chunk.metadata.department,
        "document_type": chunk.metadata.document_type,
        "access_level": chunk.metadata.access_level.value,
        "created_date": chunk.metadata.created_date.isoformat(),
        "allowed_roles": [r.value for r in chunk.metadata.allowed_roles],
    }


def _match(chunk: DocumentChunk, score: float = 0.9) -> PineconeMatch:
    return PineconeMatch(
        chunk_id=chunk.chunk_id,
        score=score,
        metadata=_pinecone_meta(chunk),
    )


def _bm25_result(chunk: DocumentChunk, score: float = 2.0, rank: int = 1) -> BM25Result:
    return BM25Result(chunk=chunk, score=score, rank=rank)


def _make_retriever(
    *,
    pinecone_matches: list[PineconeMatch] | None = None,
    sparse_results: list[BM25Result] | None = None,
    embed_vector: list[float] | None = None,
    reranker: Reranker | None = None,
) -> HybridRetriever:
    mock_pinecone = AsyncMock(spec=PineconeIndexService)
    mock_pinecone.search.return_value = pinecone_matches or []

    mock_embedder = AsyncMock(spec=EmbeddingProvider)
    mock_embedder.embed_one.return_value = embed_vector or [0.1, 0.2, 0.3]

    mock_bm25 = AsyncMock(spec=BM25Service)
    mock_bm25.search.return_value = sparse_results or []

    return HybridRetriever(
        pinecone_service=mock_pinecone,
        embedder=mock_embedder,
        bm25_service=mock_bm25,
        reranker=reranker,
    )


# ---------------------------------------------------------------------------
# Basic pipeline
# ---------------------------------------------------------------------------


class TestPipelineBasics:
    async def test_returns_list(self) -> None:
        retriever = _make_retriever()
        result = await retriever.search("payment timeout")
        assert isinstance(result, list)

    async def test_empty_when_both_return_nothing(self) -> None:
        retriever = _make_retriever()
        assert await retriever.search("payment timeout") == []

    async def test_embed_called_with_query(self) -> None:
        mock_pinecone = AsyncMock(spec=PineconeIndexService)
        mock_pinecone.search.return_value = []
        mock_embedder = AsyncMock(spec=EmbeddingProvider)
        mock_embedder.embed_one.return_value = [0.1]
        mock_bm25 = AsyncMock(spec=BM25Service)
        mock_bm25.search.return_value = []

        retriever = HybridRetriever(mock_pinecone, mock_embedder, mock_bm25)
        await retriever.search("certificate expiry FPS")

        mock_embedder.embed_one.assert_awaited_once_with("certificate expiry FPS")

    async def test_both_retrievers_called(self) -> None:
        mock_pinecone = AsyncMock(spec=PineconeIndexService)
        mock_pinecone.search.return_value = []
        mock_embedder = AsyncMock(spec=EmbeddingProvider)
        mock_embedder.embed_one.return_value = [0.1]
        mock_bm25 = AsyncMock(spec=BM25Service)
        mock_bm25.search.return_value = []

        retriever = HybridRetriever(mock_pinecone, mock_embedder, mock_bm25)
        await retriever.search("query")

        mock_pinecone.search.assert_awaited_once()
        mock_bm25.search.assert_awaited_once()

    async def test_results_are_retrieval_evidence(self) -> None:
        chunk = _chunk()
        retriever = _make_retriever(pinecone_matches=[_match(chunk)])
        results = await retriever.search("test query")
        assert all(isinstance(r, RetrievalEvidence) for r in results)


# ---------------------------------------------------------------------------
# Dense-only path
# ---------------------------------------------------------------------------


class TestDenseOnlyPath:
    async def test_dense_result_returned(self) -> None:
        chunk = _chunk(chunk_id="ARCH-001-chunk-0000", document_id="ARCH-001")
        retriever = _make_retriever(pinecone_matches=[_match(chunk, score=0.92)])
        results = await retriever.search("architecture gateway")
        assert len(results) == 1
        assert results[0].chunk_id == "ARCH-001-chunk-0000"

    async def test_source_is_dense(self) -> None:
        chunk = _chunk()
        retriever = _make_retriever(pinecone_matches=[_match(chunk)])
        results = await retriever.search("query")
        assert results[0].source == RetrievalSource.DENSE

    async def test_dense_score_present_sparse_absent(self) -> None:
        chunk = _chunk()
        retriever = _make_retriever(pinecone_matches=[_match(chunk)])
        results = await retriever.search("query")
        assert results[0].dense_score is not None
        assert results[0].sparse_score is None

    async def test_final_score_positive(self) -> None:
        chunk = _chunk()
        retriever = _make_retriever(pinecone_matches=[_match(chunk)])
        results = await retriever.search("query")
        assert results[0].final_score > 0.0

    async def test_all_required_fields_populated(self) -> None:
        chunk = _chunk(chunk_id="DOC-001-chunk-0000", document_id="DOC-001")
        retriever = _make_retriever(pinecone_matches=[_match(chunk)])
        results = await retriever.search("query")
        ev = results[0]
        assert ev.chunk_id == "DOC-001-chunk-0000"
        assert ev.document_id == "DOC-001"
        assert ev.title == "Test Document"
        assert ev.text
        assert ev.metadata is not None
        assert ev.rank == 1


# ---------------------------------------------------------------------------
# Sparse-only path
# ---------------------------------------------------------------------------


class TestSparseOnlyPath:
    async def test_sparse_result_returned(self) -> None:
        chunk = _chunk(chunk_id="INC-001-chunk-0000", document_id="INC-001")
        retriever = _make_retriever(sparse_results=[_bm25_result(chunk)])
        results = await retriever.search("ERR-429 FPS")
        assert len(results) == 1
        assert results[0].chunk_id == "INC-001-chunk-0000"

    async def test_source_is_sparse(self) -> None:
        chunk = _chunk()
        retriever = _make_retriever(sparse_results=[_bm25_result(chunk)])
        results = await retriever.search("query")
        assert results[0].source == RetrievalSource.SPARSE

    async def test_sparse_score_present_dense_absent(self) -> None:
        chunk = _chunk()
        retriever = _make_retriever(sparse_results=[_bm25_result(chunk)])
        results = await retriever.search("query")
        assert results[0].sparse_score is not None
        assert results[0].dense_score is None


# ---------------------------------------------------------------------------
# Overlapping results
# ---------------------------------------------------------------------------


class TestOverlappingResults:
    async def test_overlap_chunk_appears_once(self) -> None:
        chunk = _chunk(chunk_id="DOC-001-chunk-0000", document_id="DOC-001")
        retriever = _make_retriever(
            pinecone_matches=[_match(chunk, score=0.85)],
            sparse_results=[_bm25_result(chunk, score=2.5)],
        )
        results = await retriever.search("payment gateway")
        ids = [r.chunk_id for r in results]
        assert ids.count("DOC-001-chunk-0000") == 1

    async def test_overlap_source_is_both(self) -> None:
        chunk = _chunk()
        retriever = _make_retriever(
            pinecone_matches=[_match(chunk)],
            sparse_results=[_bm25_result(chunk)],
        )
        results = await retriever.search("query")
        assert results[0].source == RetrievalSource.BOTH

    async def test_overlap_both_scores_populated(self) -> None:
        chunk = _chunk()
        retriever = _make_retriever(
            pinecone_matches=[_match(chunk)],
            sparse_results=[_bm25_result(chunk)],
        )
        results = await retriever.search("query")
        assert results[0].dense_score is not None
        assert results[0].sparse_score is not None


# ---------------------------------------------------------------------------
# Unauthorized documents removed before results reach the agent
# ---------------------------------------------------------------------------


class TestAccessControl:
    async def test_unauthorized_chunk_not_returned(self) -> None:
        # Chunk is ADMINISTRATOR-only; caller is ENGINEER
        restricted_chunk = _chunk(
            chunk_id="SEC-001-chunk-0000",
            document_id="SEC-001",
            metadata=_meta("SEC-001", allowed_roles=(Role.ADMINISTRATOR,)),
        )
        retriever = _make_retriever(
            pinecone_matches=[_match(restricted_chunk)],
        )
        results = await retriever.search("query", roles=[Role.ENGINEER])
        ids = [r.chunk_id for r in results]
        assert "SEC-001-chunk-0000" not in ids

    async def test_authorized_chunk_returned(self) -> None:
        allowed_chunk = _chunk(
            chunk_id="DOC-001-chunk-0000",
            document_id="DOC-001",
            metadata=_meta("DOC-001", allowed_roles=(Role.ENGINEER,)),
        )
        retriever = _make_retriever(pinecone_matches=[_match(allowed_chunk)])
        results = await retriever.search("query", roles=[Role.ENGINEER])
        assert len(results) == 1
        assert results[0].chunk_id == "DOC-001-chunk-0000"

    async def test_mixed_authorized_and_unauthorized(self) -> None:
        allowed = _chunk(
            chunk_id="DOC-001-chunk-0000",
            document_id="DOC-001",
            metadata=_meta("DOC-001", allowed_roles=(Role.ENGINEER,)),
        )
        blocked = _chunk(
            chunk_id="SEC-001-chunk-0000",
            document_id="SEC-001",
            metadata=_meta("SEC-001", allowed_roles=(Role.ADMINISTRATOR,)),
        )
        retriever = _make_retriever(
            pinecone_matches=[_match(allowed, score=0.9), _match(blocked, score=0.8)],
        )
        results = await retriever.search("query", roles=[Role.ENGINEER])
        ids = [r.chunk_id for r in results]
        assert "DOC-001-chunk-0000" in ids
        assert "SEC-001-chunk-0000" not in ids

    async def test_access_level_filter_blocks_confidential(self) -> None:
        confidential = _chunk(
            metadata=_meta(access_level=AccessLevel.CONFIDENTIAL)
        )
        retriever = _make_retriever(pinecone_matches=[_match(confidential)])
        results = await retriever.search(
            "query",
            access_levels=[AccessLevel.INTERNAL],
        )
        assert results == []

    async def test_ranks_reassigned_after_unauthorized_removed(self) -> None:
        allowed = _chunk(
            chunk_id="DOC-001-chunk-0000",
            document_id="DOC-001",
            metadata=_meta("DOC-001", allowed_roles=(Role.ENGINEER,)),
        )
        blocked = _chunk(
            chunk_id="SEC-001-chunk-0000",
            document_id="SEC-001",
            metadata=_meta("SEC-001", allowed_roles=(Role.ADMINISTRATOR,)),
        )
        # blocked has higher dense score → would be rank 1 before filtering
        retriever = _make_retriever(
            pinecone_matches=[_match(blocked, score=0.95), _match(allowed, score=0.80)],
        )
        results = await retriever.search("query", roles=[Role.ENGINEER])
        # After filtering, only one result survives — must be rank 1
        assert len(results) == 1
        assert results[0].rank == 1


# ---------------------------------------------------------------------------
# Metadata filtering
# ---------------------------------------------------------------------------


class TestMetadataFiltering:
    async def test_department_filter_applied(self) -> None:
        fps_chunk = _chunk(
            chunk_id="FPS-001-chunk-0000",
            document_id="FPS-001",
            metadata=_meta("FPS-001", department="FPS"),
        )
        core_chunk = _chunk(
            chunk_id="CORE-001-chunk-0000",
            document_id="CORE-001",
            metadata=_meta("CORE-001", department="Core Banking"),
        )
        retriever = _make_retriever(
            pinecone_matches=[_match(fps_chunk), _match(core_chunk)],
        )
        results = await retriever.search("query", department="FPS")
        ids = [r.chunk_id for r in results]
        assert "FPS-001-chunk-0000" in ids
        assert "CORE-001-chunk-0000" not in ids

    async def test_document_type_filter_applied(self) -> None:
        runbook = _chunk(
            chunk_id="RB-001-chunk-0000",
            document_id="RB-001",
            metadata=_meta("RB-001", document_type="runbook"),
        )
        arch = _chunk(
            chunk_id="ARCH-001-chunk-0000",
            document_id="ARCH-001",
            metadata=_meta("ARCH-001", document_type="architecture_document"),
        )
        retriever = _make_retriever(
            pinecone_matches=[_match(runbook), _match(arch)],
        )
        results = await retriever.search("query", document_type="runbook")
        ids = [r.chunk_id for r in results]
        assert "RB-001-chunk-0000" in ids
        assert "ARCH-001-chunk-0000" not in ids


# ---------------------------------------------------------------------------
# Ranking
# ---------------------------------------------------------------------------


class TestRanking:
    async def test_ranks_are_consecutive_from_one(self) -> None:
        chunks = [
            _chunk(chunk_id=f"D-{i:03d}-chunk-0000", document_id=f"D-{i:03d}")
            for i in range(3)
        ]
        retriever = _make_retriever(
            pinecone_matches=[_match(c, score=0.9 - i * 0.1) for i, c in enumerate(chunks)]
        )
        results = await retriever.search("query")
        for i, r in enumerate(results):
            assert r.rank == i + 1

    async def test_final_scores_non_increasing(self) -> None:
        chunks = [
            _chunk(chunk_id=f"D-{i:03d}-chunk-0000", document_id=f"D-{i:03d}")
            for i in range(4)
        ]
        retriever = _make_retriever(
            pinecone_matches=[_match(c, score=0.9 - i * 0.05) for i, c in enumerate(chunks)]
        )
        results = await retriever.search("query")
        scores = [r.final_score for r in results]
        assert scores == sorted(scores, reverse=True)

    async def test_top_k_respected(self) -> None:
        chunks = [
            _chunk(chunk_id=f"D-{i:03d}-chunk-0000", document_id=f"D-{i:03d}")
            for i in range(10)
        ]
        retriever = _make_retriever(
            pinecone_matches=[_match(c) for c in chunks]
        )
        results = await retriever.search("query", top_k=3)
        assert len(results) <= 3


# ---------------------------------------------------------------------------
# Reranker injection
# ---------------------------------------------------------------------------


class TestRerankerInjection:
    async def test_identity_reranker_used_by_default(self) -> None:
        chunk = _chunk()
        retriever = _make_retriever(pinecone_matches=[_match(chunk)])
        results = await retriever.search("query")
        assert len(results) == 1

    async def test_custom_reranker_called(self) -> None:
        chunk = _chunk()
        mock_reranker = AsyncMock(spec=Reranker)
        mock_reranker.rerank.return_value = []

        retriever = _make_retriever(
            pinecone_matches=[_match(chunk)],
            reranker=mock_reranker,
        )
        await retriever.search("query", top_k=5)
        mock_reranker.rerank.assert_awaited_once()
        _, kwargs = mock_reranker.rerank.call_args
        assert kwargs["top_k"] == 5

    async def test_reranker_receives_filtered_evidence(self) -> None:
        allowed = _chunk(
            chunk_id="DOC-001-chunk-0000",
            metadata=_meta(allowed_roles=(Role.ENGINEER,)),
        )
        blocked = _chunk(
            chunk_id="SEC-001-chunk-0000",
            document_id="SEC-001",
            metadata=_meta("SEC-001", allowed_roles=(Role.ADMINISTRATOR,)),
        )
        captured: list[list[RetrievalEvidence]] = []

        class CapturingReranker:
            async def rerank(
                self, query: str, evidence: list[RetrievalEvidence], *, top_k: int
            ) -> list[RetrievalEvidence]:
                captured.append(evidence)
                return evidence[:top_k]

        retriever = _make_retriever(
            pinecone_matches=[_match(allowed), _match(blocked)],
            reranker=CapturingReranker(),
        )
        await retriever.search("query", roles=[Role.ENGINEER])
        assert len(captured) == 1
        reranker_ids = [ev.chunk_id for ev in captured[0]]
        assert "DOC-001-chunk-0000" in reranker_ids
        assert "SEC-001-chunk-0000" not in reranker_ids
