"""Unit tests for BM25Service.

BM25Service is a thin async facade over BM25Corpus.  Tests verify:
- Results match what the underlying corpus returns.
- The service delegates correctly to the corpus.
- `from_chunks` builds a working service.
- `chunk_count` reflects the corpus size.
- `search` is awaitable and returns a list of BM25Result.
"""

from __future__ import annotations

from datetime import date
from unittest.mock import MagicMock, patch

import pytest

from src.models.documents import DocumentMetadata
from src.models.enums import AccessLevel, Role
from src.retrieval.bm25.corpus import BM25Corpus, BM25Result
from src.retrieval.bm25.service import BM25Service
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
    *,
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


# ---------------------------------------------------------------------------
# from_chunks factory
# ---------------------------------------------------------------------------


class TestFromChunksFactory:
    def test_returns_bm25_service(self) -> None:
        service = BM25Service.from_chunks([_make_chunk()])
        assert isinstance(service, BM25Service)

    def test_chunk_count_reflects_corpus_size(self) -> None:
        chunks = [
            _make_chunk(chunk_id=f"DOC-{i:03d}-chunk-0000", document_id=f"DOC-{i:03d}")
            for i in range(7)
        ]
        service = BM25Service.from_chunks(chunks)
        assert service.chunk_count == 7

    def test_empty_chunk_list(self) -> None:
        service = BM25Service.from_chunks([])
        assert service.chunk_count == 0


# ---------------------------------------------------------------------------
# chunk_count property
# ---------------------------------------------------------------------------


class TestChunkCount:
    def test_delegates_to_corpus(self) -> None:
        mock_corpus = MagicMock(spec=BM25Corpus)
        mock_corpus.chunk_count = 42
        service = BM25Service(mock_corpus)
        assert service.chunk_count == 42


# ---------------------------------------------------------------------------
# async search
# ---------------------------------------------------------------------------


class TestAsyncSearch:
    async def test_search_is_awaitable_and_returns_list(self) -> None:
        chunk = _make_chunk(text="payment gateway timeout error processing")
        service = BM25Service.from_chunks([chunk])
        results = await service.search("payment timeout")
        assert isinstance(results, list)

    async def test_matching_chunk_returned(self) -> None:
        chunk = _make_chunk(
            chunk_id="INC-001-chunk-0000",
            document_id="INC-001",
            text="FPS payment gateway ERR-429 error certificate timeout",
        )
        service = BM25Service.from_chunks([chunk])
        results = await service.search("ERR-429")
        assert len(results) == 1
        assert results[0].chunk.chunk_id == "INC-001-chunk-0000"

    async def test_empty_query_returns_empty_list(self) -> None:
        service = BM25Service.from_chunks([_make_chunk()])
        results = await service.search("")
        assert results == []

    async def test_no_matching_chunks_returns_empty_list(self) -> None:
        service = BM25Service.from_chunks([_make_chunk(text="core banking ledger")])
        results = await service.search("xylophone")
        assert results == []

    async def test_results_are_bm25_result_instances(self) -> None:
        chunk = _make_chunk(text="payment processing timeout gateway FPS")
        service = BM25Service.from_chunks([chunk])
        results = await service.search("payment processing")
        for r in results:
            assert isinstance(r, BM25Result)

    async def test_top_k_parameter_respected(self) -> None:
        chunks = [
            _make_chunk(
                chunk_id=f"DOC-{i:03d}-chunk-0000",
                document_id=f"DOC-{i:03d}",
                text="payment processing error timeout gateway FPS certificate",
            )
            for i in range(10)
        ]
        service = BM25Service.from_chunks(chunks)
        results = await service.search("payment error", top_k=3)
        assert len(results) <= 3

    async def test_role_filter_forwarded_to_corpus(self) -> None:
        chunk_eng = _make_chunk(
            chunk_id="DOC-001-chunk-0000",
            document_id="DOC-001",
            text="payment gateway architecture certificate timeout",
        )
        corpus = BM25Corpus([chunk_eng])
        service = BM25Service(corpus)
        # ANALYST has no access to ENGINEER-only chunks
        results = await service.search("payment gateway", roles=[Role.ANALYST])
        assert results == []

    async def test_access_level_filter_forwarded_to_corpus(self) -> None:
        from src.models.enums import AccessLevel

        chunk = _make_chunk(text="payment processing timeout error gateway")
        # chunk has INTERNAL access; filter for CONFIDENTIAL only
        results_confidential = await BM25Service.from_chunks([chunk]).search(
            "payment timeout",
            access_levels=[AccessLevel.CONFIDENTIAL],
        )
        assert results_confidential == []

        results_internal = await BM25Service.from_chunks([chunk]).search(
            "payment timeout",
            access_levels=[AccessLevel.INTERNAL],
        )
        assert len(results_internal) == 1

    async def test_search_delegates_to_corpus(self) -> None:
        mock_corpus = MagicMock(spec=BM25Corpus)
        mock_corpus.search.return_value = []
        service = BM25Service(mock_corpus)
        await service.search("payment", top_k=5, roles=[Role.ENGINEER])
        mock_corpus.search.assert_called_once_with(
            "payment",
            top_k=5,
            roles=[Role.ENGINEER],
            access_levels=None,
            department=None,
            document_type=None,
        )
