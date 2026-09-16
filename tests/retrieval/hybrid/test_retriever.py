"""Unit tests for HybridRetriever.

All external dependencies (Pinecone, embedder, BM25) are mocked so tests
run without network access or a real index.  Tests verify: delegation to
both retrievers, parallel execution, RRF fusion, filter forwarding, and
top-k limiting.
"""

from __future__ import annotations

from datetime import date
from unittest.mock import AsyncMock, MagicMock, call, patch

import pytest

from src.models.documents import DocumentMetadata
from src.models.enums import AccessLevel, Role
from src.retrieval.bm25.corpus import BM25Result
from src.retrieval.bm25.service import BM25Service
from src.retrieval.embedding.base import EmbeddingProvider
from src.retrieval.hybrid.fusion import DenseSearchResult, HybridResult
from src.retrieval.hybrid.retriever import HybridRetriever
from src.retrieval.indexing.service import PineconeIndexService, PineconeMatch
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
    text: str = "sample text content",
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


def _make_pinecone_meta(chunk: DocumentChunk) -> dict[str, object]:
    """Build a Pinecone-style metadata dict for the given chunk."""
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


def _make_retriever(
    *,
    pinecone_matches: list[PineconeMatch] | None = None,
    sparse_results: list[BM25Result] | None = None,
    embed_vector: list[float] | None = None,
) -> HybridRetriever:
    """Build a HybridRetriever with all dependencies mocked."""
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
    )


# ---------------------------------------------------------------------------
# Basic delegation
# ---------------------------------------------------------------------------


class TestHybridRetrieverDelegation:
    async def test_returns_list_of_hybrid_results(self) -> None:
        retriever = _make_retriever()
        results = await retriever.search("payment timeout")
        assert isinstance(results, list)

    async def test_empty_results_when_both_return_nothing(self) -> None:
        retriever = _make_retriever()
        results = await retriever.search("payment timeout")
        assert results == []

    async def test_embed_one_called_with_query(self) -> None:
        mock_pinecone = AsyncMock(spec=PineconeIndexService)
        mock_pinecone.search.return_value = []
        mock_embedder = AsyncMock(spec=EmbeddingProvider)
        mock_embedder.embed_one.return_value = [0.1, 0.2]
        mock_bm25 = AsyncMock(spec=BM25Service)
        mock_bm25.search.return_value = []

        retriever = HybridRetriever(mock_pinecone, mock_embedder, mock_bm25)
        await retriever.search("certificate expiry FPS")

        mock_embedder.embed_one.assert_awaited_once_with("certificate expiry FPS")

    async def test_pinecone_search_called(self) -> None:
        mock_pinecone = AsyncMock(spec=PineconeIndexService)
        mock_pinecone.search.return_value = []
        mock_embedder = AsyncMock(spec=EmbeddingProvider)
        mock_embedder.embed_one.return_value = [0.1, 0.2]
        mock_bm25 = AsyncMock(spec=BM25Service)
        mock_bm25.search.return_value = []

        retriever = HybridRetriever(mock_pinecone, mock_embedder, mock_bm25)
        await retriever.search("query")

        mock_pinecone.search.assert_awaited_once()

    async def test_bm25_search_called(self) -> None:
        mock_pinecone = AsyncMock(spec=PineconeIndexService)
        mock_pinecone.search.return_value = []
        mock_embedder = AsyncMock(spec=EmbeddingProvider)
        mock_embedder.embed_one.return_value = [0.1]
        mock_bm25 = AsyncMock(spec=BM25Service)
        mock_bm25.search.return_value = []

        retriever = HybridRetriever(mock_pinecone, mock_embedder, mock_bm25)
        await retriever.search("query")

        mock_bm25.search.assert_awaited_once()

    async def test_vector_passed_to_pinecone(self) -> None:
        mock_pinecone = AsyncMock(spec=PineconeIndexService)
        mock_pinecone.search.return_value = []
        mock_embedder = AsyncMock(spec=EmbeddingProvider)
        vector = [0.1, 0.2, 0.3, 0.4]
        mock_embedder.embed_one.return_value = vector
        mock_bm25 = AsyncMock(spec=BM25Service)
        mock_bm25.search.return_value = []

        retriever = HybridRetriever(mock_pinecone, mock_embedder, mock_bm25)
        await retriever.search("query")

        call_args = mock_pinecone.search.call_args
        assert call_args.args[0] == vector


# ---------------------------------------------------------------------------
# Dense-only results
# ---------------------------------------------------------------------------


class TestDenseOnlyResults:
    async def test_dense_only_results_returned(self) -> None:
        chunk = _make_chunk(chunk_id="ARCH-001-chunk-0000", document_id="ARCH-001")
        match = PineconeMatch(
            chunk_id=chunk.chunk_id,
            score=0.92,
            metadata=_make_pinecone_meta(chunk),
        )
        retriever = _make_retriever(pinecone_matches=[match])
        results = await retriever.search("architecture gateway")
        assert len(results) == 1
        assert results[0].chunk.chunk_id == "ARCH-001-chunk-0000"

    async def test_dense_result_has_dense_rank_set(self) -> None:
        chunk = _make_chunk(chunk_id="ARCH-001-chunk-0000", document_id="ARCH-001")
        match = PineconeMatch(
            chunk_id=chunk.chunk_id,
            score=0.9,
            metadata=_make_pinecone_meta(chunk),
        )
        retriever = _make_retriever(pinecone_matches=[match])
        results = await retriever.search("architecture")
        assert results[0].dense_rank == 1
        assert results[0].sparse_rank is None


# ---------------------------------------------------------------------------
# Sparse-only results
# ---------------------------------------------------------------------------


class TestSparseOnlyResults:
    async def test_sparse_only_results_returned(self) -> None:
        chunk = _make_chunk(chunk_id="INC-001-chunk-0000", document_id="INC-001")
        sparse = [BM25Result(chunk=chunk, score=3.5, rank=1)]
        retriever = _make_retriever(sparse_results=sparse)
        results = await retriever.search("ERR-429 FPS")
        assert len(results) == 1
        assert results[0].chunk.chunk_id == "INC-001-chunk-0000"

    async def test_sparse_result_has_sparse_rank_set(self) -> None:
        chunk = _make_chunk(chunk_id="INC-001-chunk-0000", document_id="INC-001")
        sparse = [BM25Result(chunk=chunk, score=3.5, rank=1)]
        retriever = _make_retriever(sparse_results=sparse)
        results = await retriever.search("ERR-429")
        assert results[0].sparse_rank == 1
        assert results[0].dense_rank is None


# ---------------------------------------------------------------------------
# Overlap — same chunk in both retrievers
# ---------------------------------------------------------------------------


class TestOverlapResults:
    async def test_overlap_chunk_appears_once(self) -> None:
        chunk = _make_chunk(chunk_id="DOC-001-chunk-0000", document_id="DOC-001")
        match = PineconeMatch(
            chunk_id=chunk.chunk_id,
            score=0.85,
            metadata=_make_pinecone_meta(chunk),
        )
        sparse = [BM25Result(chunk=chunk, score=2.5, rank=1)]
        retriever = _make_retriever(pinecone_matches=[match], sparse_results=sparse)
        results = await retriever.search("payment gateway")
        ids = [r.chunk.chunk_id for r in results]
        assert ids.count("DOC-001-chunk-0000") == 1

    async def test_overlap_chunk_has_both_rank_fields(self) -> None:
        chunk = _make_chunk(chunk_id="DOC-001-chunk-0000", document_id="DOC-001")
        match = PineconeMatch(
            chunk_id=chunk.chunk_id,
            score=0.85,
            metadata=_make_pinecone_meta(chunk),
        )
        sparse = [BM25Result(chunk=chunk, score=2.5, rank=1)]
        retriever = _make_retriever(pinecone_matches=[match], sparse_results=sparse)
        results = await retriever.search("payment gateway")
        result = next(r for r in results if r.chunk.chunk_id == "DOC-001-chunk-0000")
        assert result.dense_rank is not None
        assert result.sparse_rank is not None


# ---------------------------------------------------------------------------
# Filter forwarding
# ---------------------------------------------------------------------------


class TestFilterForwarding:
    async def test_role_filter_forwarded_to_pinecone(self) -> None:
        mock_pinecone = AsyncMock(spec=PineconeIndexService)
        mock_pinecone.search.return_value = []
        mock_embedder = AsyncMock(spec=EmbeddingProvider)
        mock_embedder.embed_one.return_value = [0.1]
        mock_bm25 = AsyncMock(spec=BM25Service)
        mock_bm25.search.return_value = []

        retriever = HybridRetriever(mock_pinecone, mock_embedder, mock_bm25)
        await retriever.search("query", roles=[Role.ENGINEER])

        _, kwargs = mock_pinecone.search.call_args
        filt = kwargs.get("filter", {})
        assert "allowed_roles" in filt
        assert "ENGINEER" in filt["allowed_roles"]["$in"]

    async def test_no_filter_when_no_constraints(self) -> None:
        mock_pinecone = AsyncMock(spec=PineconeIndexService)
        mock_pinecone.search.return_value = []
        mock_embedder = AsyncMock(spec=EmbeddingProvider)
        mock_embedder.embed_one.return_value = [0.1]
        mock_bm25 = AsyncMock(spec=BM25Service)
        mock_bm25.search.return_value = []

        retriever = HybridRetriever(mock_pinecone, mock_embedder, mock_bm25)
        await retriever.search("query")

        _, kwargs = mock_pinecone.search.call_args
        assert kwargs.get("filter") is None

    async def test_role_filter_forwarded_to_bm25(self) -> None:
        mock_pinecone = AsyncMock(spec=PineconeIndexService)
        mock_pinecone.search.return_value = []
        mock_embedder = AsyncMock(spec=EmbeddingProvider)
        mock_embedder.embed_one.return_value = [0.1]
        mock_bm25 = AsyncMock(spec=BM25Service)
        mock_bm25.search.return_value = []

        retriever = HybridRetriever(mock_pinecone, mock_embedder, mock_bm25)
        await retriever.search("query", roles=[Role.ANALYST])

        _, kwargs = mock_bm25.search.call_args
        assert kwargs["roles"] == [Role.ANALYST]

    async def test_department_filter_forwarded_to_both(self) -> None:
        mock_pinecone = AsyncMock(spec=PineconeIndexService)
        mock_pinecone.search.return_value = []
        mock_embedder = AsyncMock(spec=EmbeddingProvider)
        mock_embedder.embed_one.return_value = [0.1]
        mock_bm25 = AsyncMock(spec=BM25Service)
        mock_bm25.search.return_value = []

        retriever = HybridRetriever(mock_pinecone, mock_embedder, mock_bm25)
        await retriever.search("query", department="Core Banking")

        _, pine_kwargs = mock_pinecone.search.call_args
        filt = pine_kwargs.get("filter", {})
        assert filt.get("department") == {"$eq": "Core Banking"}

        _, bm25_kwargs = mock_bm25.search.call_args
        assert bm25_kwargs["department"] == "Core Banking"

    async def test_document_type_filter_forwarded_to_pinecone(self) -> None:
        mock_pinecone = AsyncMock(spec=PineconeIndexService)
        mock_pinecone.search.return_value = []
        mock_embedder = AsyncMock(spec=EmbeddingProvider)
        mock_embedder.embed_one.return_value = [0.1]
        mock_bm25 = AsyncMock(spec=BM25Service)
        mock_bm25.search.return_value = []

        retriever = HybridRetriever(mock_pinecone, mock_embedder, mock_bm25)
        await retriever.search("query", document_type="runbook")

        _, pine_kwargs = mock_pinecone.search.call_args
        filt = pine_kwargs.get("filter", {})
        assert filt.get("document_type") == {"$eq": "runbook"}


# ---------------------------------------------------------------------------
# Top-k and retriever_top_k
# ---------------------------------------------------------------------------


class TestTopK:
    async def test_top_k_limits_results(self) -> None:
        chunks = [
            _make_chunk(chunk_id=f"DOC-{i:03d}-chunk-0000", document_id=f"DOC-{i:03d}")
            for i in range(10)
        ]
        sparse = [BM25Result(chunk=c, score=float(10 - i), rank=i + 1) for i, c in enumerate(chunks)]
        retriever = _make_retriever(sparse_results=sparse)
        results = await retriever.search("payment", top_k=3)
        assert len(results) <= 3

    async def test_retriever_top_k_passed_to_pinecone(self) -> None:
        mock_pinecone = AsyncMock(spec=PineconeIndexService)
        mock_pinecone.search.return_value = []
        mock_embedder = AsyncMock(spec=EmbeddingProvider)
        mock_embedder.embed_one.return_value = [0.1]
        mock_bm25 = AsyncMock(spec=BM25Service)
        mock_bm25.search.return_value = []

        retriever = HybridRetriever(mock_pinecone, mock_embedder, mock_bm25)
        await retriever.search("query", top_k=5, retriever_top_k=20)

        _, kwargs = mock_pinecone.search.call_args
        assert kwargs["top_k"] == 20

    async def test_default_retriever_top_k_is_three_times_top_k(self) -> None:
        mock_pinecone = AsyncMock(spec=PineconeIndexService)
        mock_pinecone.search.return_value = []
        mock_embedder = AsyncMock(spec=EmbeddingProvider)
        mock_embedder.embed_one.return_value = [0.1]
        mock_bm25 = AsyncMock(spec=BM25Service)
        mock_bm25.search.return_value = []

        retriever = HybridRetriever(mock_pinecone, mock_embedder, mock_bm25)
        await retriever.search("query", top_k=5)

        _, kwargs = mock_pinecone.search.call_args
        assert kwargs["top_k"] == 15  # 5 * 3
