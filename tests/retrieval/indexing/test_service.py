"""Unit tests for PineconeIndexService and make_pinecone_service.

All tests use MagicMock for the Pinecone index object — no real network
calls are made.  asyncio.to_thread runs the sync mock in a thread pool,
which works correctly in pytest-asyncio's async test context.
"""

from __future__ import annotations

from datetime import date
from unittest.mock import MagicMock, call, patch

import pytest

from src.models.documents import DocumentMetadata
from src.models.enums import AccessLevel, Role
from src.retrieval.indexing.service import (
    DEFAULT_BATCH_SIZE,
    PineconeIndexService,
    PineconeMatch,
    UpsertResult,
    make_pinecone_service,
)
from src.retrieval.ingestion.models import DocumentChunk


# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------


def _make_chunk(
    *,
    chunk_id: str = "ARCH-001-chunk-0000",
    document_id: str = "ARCH-001",
    chunk_index: int = 0,
    chunk_total: int = 1,
    text: str = "Sample text.",
) -> DocumentChunk:
    meta = DocumentMetadata(
        document_id=document_id,
        title="Test Document",
        department="Engineering",
        document_type="architecture_document",
        access_level=AccessLevel.INTERNAL,
        created_date=date(2024, 1, 1),
        allowed_roles=(Role.ENGINEER,),
    )
    return DocumentChunk(
        chunk_id=chunk_id,
        document_id=document_id,
        title="Test Document",
        section="## Overview",
        chunk_index=chunk_index,
        chunk_total=chunk_total,
        text=text,
        metadata=meta,
    )


def _upsert_response(upserted_count: int) -> MagicMock:
    r = MagicMock()
    r.upserted_count = upserted_count
    return r


def _stats_response(
    total: int = 0,
    dimension: int = 3072,
    namespaces: dict[str, int] | None = None,
) -> MagicMock:
    r = MagicMock()
    r.total_vector_count = total
    r.dimension = dimension
    ns: dict[str, MagicMock] = {}
    for name, count in (namespaces or {}).items():
        ns_info = MagicMock()
        ns_info.vector_count = count
        ns[name] = ns_info
    r.namespaces = ns
    return r


def _make_service(
    namespace: str = "",
    upserted_count: int = 1,
) -> tuple[PineconeIndexService, MagicMock]:
    mock_index = MagicMock()
    mock_index.upsert.return_value = _upsert_response(upserted_count)
    mock_index.delete.return_value = {}
    mock_index.describe_index_stats.return_value = _stats_response()
    service = PineconeIndexService(mock_index, namespace=namespace)
    return service, mock_index


# ---------------------------------------------------------------------------
# Namespace property
# ---------------------------------------------------------------------------


class TestNamespaceProperty:
    def test_default_namespace_is_empty_string(self) -> None:
        service, _ = _make_service()
        assert service.namespace == ""

    def test_custom_namespace(self) -> None:
        service, _ = _make_service(namespace="prod")
        assert service.namespace == "prod"


# ---------------------------------------------------------------------------
# UpsertResult model
# ---------------------------------------------------------------------------


class TestUpsertResult:
    def test_is_frozen(self) -> None:
        result = UpsertResult(upserted_count=5, batch_count=1)
        with pytest.raises(Exception):
            result.upserted_count = 99  # type: ignore[misc]

    def test_zero_counts_valid(self) -> None:
        result = UpsertResult(upserted_count=0, batch_count=0)
        assert result.upserted_count == 0


# ---------------------------------------------------------------------------
# upsert_chunks
# ---------------------------------------------------------------------------


class TestUpsertChunks:
    async def test_empty_input_is_noop(self) -> None:
        service, mock_index = _make_service()
        result = await service.upsert_chunks([], [])
        mock_index.upsert.assert_not_called()
        assert result.upserted_count == 0
        assert result.batch_count == 0

    async def test_raises_on_length_mismatch(self) -> None:
        service, _ = _make_service()
        chunk = _make_chunk()
        with pytest.raises(ValueError, match="same length"):
            await service.upsert_chunks([chunk], [])

    async def test_raises_when_more_vectors_than_chunks(self) -> None:
        service, _ = _make_service()
        with pytest.raises(ValueError, match="same length"):
            await service.upsert_chunks([], [[0.1, 0.2]])

    async def test_upsert_called_once_for_small_input(self) -> None:
        service, mock_index = _make_service(upserted_count=1)
        chunk = _make_chunk()
        await service.upsert_chunks([chunk], [[0.1, 0.2, 0.3]])
        mock_index.upsert.assert_called_once()

    async def test_vector_id_matches_chunk_id(self) -> None:
        service, mock_index = _make_service(upserted_count=1)
        chunk = _make_chunk(chunk_id="ARCH-001-chunk-0002")
        await service.upsert_chunks([chunk], [[0.1]])
        _, kwargs = mock_index.upsert.call_args
        vectors = kwargs["vectors"]
        assert vectors[0]["id"] == "ARCH-001-chunk-0002"

    async def test_vector_values_passed_correctly(self) -> None:
        service, mock_index = _make_service(upserted_count=1)
        embedding = [0.1, 0.2, 0.3, 0.4]
        chunk = _make_chunk()
        await service.upsert_chunks([chunk], [embedding])
        _, kwargs = mock_index.upsert.call_args
        assert kwargs["vectors"][0]["values"] == embedding

    async def test_metadata_included_in_upsert_payload(self) -> None:
        service, mock_index = _make_service(upserted_count=1)
        chunk = _make_chunk(document_id="ARCH-001")
        await service.upsert_chunks([chunk], [[0.1]])
        _, kwargs = mock_index.upsert.call_args
        meta = kwargs["vectors"][0]["metadata"]
        assert meta["document_id"] == "ARCH-001"
        assert "allowed_roles" in meta
        assert "access_level" in meta

    async def test_namespace_passed_to_upsert(self) -> None:
        service, mock_index = _make_service(namespace="finance", upserted_count=1)
        chunk = _make_chunk()
        await service.upsert_chunks([chunk], [[0.1]])
        _, kwargs = mock_index.upsert.call_args
        assert kwargs["namespace"] == "finance"

    async def test_result_upserted_count_from_response(self) -> None:
        service, mock_index = _make_service(upserted_count=3)
        chunks = [_make_chunk(chunk_id=f"D-chunk-{i:04d}") for i in range(3)]
        vectors = [[float(i)] for i in range(3)]
        result = await service.upsert_chunks(chunks, vectors)
        assert result.upserted_count == 3

    async def test_result_batch_count_is_one_for_small_input(self) -> None:
        service, mock_index = _make_service(upserted_count=2)
        chunks = [_make_chunk(chunk_id=f"D-chunk-{i:04d}") for i in range(2)]
        result = await service.upsert_chunks(chunks, [[0.1], [0.2]])
        assert result.batch_count == 1

    async def test_large_input_split_into_multiple_batches(self) -> None:
        batch_size = 3
        chunk_count = 7
        mock_index = MagicMock()
        mock_index.upsert.return_value = _upsert_response(batch_size)
        service = PineconeIndexService(mock_index, namespace="")

        chunks = [
            _make_chunk(chunk_id=f"DOC-chunk-{i:04d}", chunk_index=i, chunk_total=chunk_count)
            for i in range(chunk_count)
        ]
        vectors = [[float(i)] for i in range(chunk_count)]

        result = await service.upsert_chunks(chunks, vectors, batch_size=batch_size)

        expected_batches = (chunk_count + batch_size - 1) // batch_size
        assert mock_index.upsert.call_count == expected_batches
        assert result.batch_count == expected_batches

    async def test_each_batch_contains_correct_ids(self) -> None:
        batch_size = 2
        mock_index = MagicMock()
        mock_index.upsert.return_value = _upsert_response(2)
        service = PineconeIndexService(mock_index)

        chunks = [
            _make_chunk(chunk_id=f"DOC-chunk-{i:04d}", chunk_index=i, chunk_total=3)
            for i in range(3)
        ]
        await service.upsert_chunks(chunks, [[0.1], [0.2], [0.3]], batch_size=batch_size)

        first_call_ids = [v["id"] for v in mock_index.upsert.call_args_list[0].kwargs["vectors"]]
        second_call_ids = [v["id"] for v in mock_index.upsert.call_args_list[1].kwargs["vectors"]]
        assert first_call_ids == ["DOC-chunk-0000", "DOC-chunk-0001"]
        assert second_call_ids == ["DOC-chunk-0002"]

    async def test_upsert_accumulates_count_across_batches(self) -> None:
        mock_index = MagicMock()
        # First batch returns 2, second returns 1
        mock_index.upsert.side_effect = [_upsert_response(2), _upsert_response(1)]
        service = PineconeIndexService(mock_index)

        chunks = [
            _make_chunk(chunk_id=f"D-chunk-{i:04d}", chunk_index=i, chunk_total=3)
            for i in range(3)
        ]
        result = await service.upsert_chunks(chunks, [[0.1], [0.2], [0.3]], batch_size=2)
        assert result.upserted_count == 3


# ---------------------------------------------------------------------------
# delete_document
# ---------------------------------------------------------------------------


class TestDeleteDocument:
    async def test_calls_delete_with_document_id_filter(self) -> None:
        service, mock_index = _make_service()
        await service.delete_document("ARCH-001")
        mock_index.delete.assert_called_once()
        _, kwargs = mock_index.delete.call_args
        assert kwargs["filter"] == {"document_id": {"$eq": "ARCH-001"}}

    async def test_passes_namespace_to_delete(self) -> None:
        service, mock_index = _make_service(namespace="prod")
        await service.delete_document("INC-001")
        _, kwargs = mock_index.delete.call_args
        assert kwargs["namespace"] == "prod"

    async def test_delete_document_returns_none(self) -> None:
        service, _ = _make_service()
        result = await service.delete_document("DOC-001")
        assert result is None

    async def test_delete_called_exactly_once(self) -> None:
        service, mock_index = _make_service()
        await service.delete_document("ARCH-001")
        mock_index.delete.assert_called_once()


# ---------------------------------------------------------------------------
# delete_chunks
# ---------------------------------------------------------------------------


class TestDeleteChunks:
    async def test_empty_input_is_noop(self) -> None:
        service, mock_index = _make_service()
        await service.delete_chunks([])
        mock_index.delete.assert_not_called()

    async def test_calls_delete_with_ids(self) -> None:
        service, mock_index = _make_service()
        ids = ["ARCH-001-chunk-0000", "ARCH-001-chunk-0001"]
        await service.delete_chunks(ids)
        mock_index.delete.assert_called_once()
        _, kwargs = mock_index.delete.call_args
        assert kwargs["ids"] == ids

    async def test_ids_converted_to_list(self) -> None:
        service, mock_index = _make_service()
        await service.delete_chunks(("ARCH-001-chunk-0000",))
        _, kwargs = mock_index.delete.call_args
        assert isinstance(kwargs["ids"], list)

    async def test_passes_namespace(self) -> None:
        service, mock_index = _make_service(namespace="staging")
        await service.delete_chunks(["A-chunk-0000"])
        _, kwargs = mock_index.delete.call_args
        assert kwargs["namespace"] == "staging"

    async def test_single_id(self) -> None:
        service, mock_index = _make_service()
        await service.delete_chunks(["ONLY-chunk-0000"])
        _, kwargs = mock_index.delete.call_args
        assert kwargs["ids"] == ["ONLY-chunk-0000"]


# ---------------------------------------------------------------------------
# describe_stats
# ---------------------------------------------------------------------------


class TestDescribeStats:
    async def test_returns_dict(self) -> None:
        service, mock_index = _make_service()
        mock_index.describe_index_stats.return_value = _stats_response(total=50)
        result = await service.describe_stats()
        assert isinstance(result, dict)

    async def test_total_vector_count(self) -> None:
        service, mock_index = _make_service()
        mock_index.describe_index_stats.return_value = _stats_response(total=150)
        result = await service.describe_stats()
        assert result["total_vector_count"] == 150

    async def test_dimension(self) -> None:
        service, mock_index = _make_service()
        mock_index.describe_index_stats.return_value = _stats_response(dimension=1536)
        result = await service.describe_stats()
        assert result["dimension"] == 1536

    async def test_namespace_vector_count(self) -> None:
        service, mock_index = _make_service()
        mock_index.describe_index_stats.return_value = _stats_response(
            namespaces={"default": 42, "staging": 8}
        )
        result = await service.describe_stats()
        ns = result["namespaces"]
        assert isinstance(ns, dict)
        assert ns["default"]["vector_count"] == 42  # type: ignore[index]
        assert ns["staging"]["vector_count"] == 8  # type: ignore[index]

    async def test_empty_namespaces(self) -> None:
        service, mock_index = _make_service()
        mock_index.describe_index_stats.return_value = _stats_response(namespaces={})
        result = await service.describe_stats()
        assert result["namespaces"] == {}


# ---------------------------------------------------------------------------
# make_pinecone_service factory
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# search
# ---------------------------------------------------------------------------


def _scored_vector(
    vid: str,
    score: float,
    metadata: dict[str, object] | None = None,
) -> MagicMock:
    m = MagicMock()
    m.id = vid
    m.score = score
    m.metadata = metadata or {}
    return m


def _query_response(matches: list[MagicMock]) -> MagicMock:
    r = MagicMock()
    r.matches = matches
    return r


class TestSearch:
    async def test_returns_list_of_pinecone_matches(self) -> None:
        service, mock_index = _make_service()
        mock_index.query.return_value = _query_response([
            _scored_vector("A-chunk-0000", 0.95),
        ])
        results = await service.search([0.1, 0.2, 0.3])
        assert isinstance(results, list)
        assert all(isinstance(r, PineconeMatch) for r in results)

    async def test_empty_matches_returns_empty_list(self) -> None:
        service, mock_index = _make_service()
        mock_index.query.return_value = _query_response([])
        results = await service.search([0.1])
        assert results == []

    async def test_none_matches_attribute_returns_empty(self) -> None:
        service, mock_index = _make_service()
        r = MagicMock()
        r.matches = None
        mock_index.query.return_value = r
        results = await service.search([0.1])
        assert results == []

    async def test_chunk_id_extracted(self) -> None:
        service, mock_index = _make_service()
        mock_index.query.return_value = _query_response([
            _scored_vector("ARCH-001-chunk-0002", 0.88),
        ])
        results = await service.search([0.1])
        assert results[0].chunk_id == "ARCH-001-chunk-0002"

    async def test_score_extracted(self) -> None:
        service, mock_index = _make_service()
        mock_index.query.return_value = _query_response([
            _scored_vector("A-chunk-0000", 0.72),
        ])
        results = await service.search([0.1])
        assert abs(results[0].score - 0.72) < 1e-9

    async def test_metadata_extracted(self) -> None:
        service, mock_index = _make_service()
        meta: dict[str, object] = {"document_id": "ARCH-001", "text": "hello"}
        mock_index.query.return_value = _query_response([
            _scored_vector("A-chunk-0000", 0.9, metadata=meta),
        ])
        results = await service.search([0.1])
        assert results[0].metadata["document_id"] == "ARCH-001"

    async def test_top_k_passed_to_query(self) -> None:
        service, mock_index = _make_service()
        mock_index.query.return_value = _query_response([])
        await service.search([0.1], top_k=25)
        _, kwargs = mock_index.query.call_args
        assert kwargs["top_k"] == 25

    async def test_filter_passed_to_query(self) -> None:
        service, mock_index = _make_service()
        mock_index.query.return_value = _query_response([])
        filt = {"allowed_roles": {"$in": ["ENGINEER"]}}
        await service.search([0.1], filter=filt)
        _, kwargs = mock_index.query.call_args
        assert kwargs["filter"] == filt

    async def test_no_filter_passes_none(self) -> None:
        service, mock_index = _make_service()
        mock_index.query.return_value = _query_response([])
        await service.search([0.1])
        _, kwargs = mock_index.query.call_args
        assert kwargs["filter"] is None

    async def test_namespace_passed_to_query(self) -> None:
        service, mock_index = _make_service(namespace="staging")
        mock_index.query.return_value = _query_response([])
        await service.search([0.1])
        _, kwargs = mock_index.query.call_args
        assert kwargs["namespace"] == "staging"

    async def test_include_metadata_always_true(self) -> None:
        service, mock_index = _make_service()
        mock_index.query.return_value = _query_response([])
        await service.search([0.1])
        _, kwargs = mock_index.query.call_args
        assert kwargs["include_metadata"] is True

    async def test_multiple_matches_returned_in_order(self) -> None:
        service, mock_index = _make_service()
        mock_index.query.return_value = _query_response([
            _scored_vector("A-chunk-0000", 0.95),
            _scored_vector("B-chunk-0000", 0.80),
            _scored_vector("C-chunk-0000", 0.65),
        ])
        results = await service.search([0.1])
        ids = [r.chunk_id for r in results]
        assert ids == ["A-chunk-0000", "B-chunk-0000", "C-chunk-0000"]


class TestMakePineconeService:
    def test_returns_pinecone_index_service(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("PINECONE_API_KEY", "pc-test-key")
        monkeypatch.setenv("PINECONE_INDEX_NAME", "test-index")

        with patch("src.retrieval.indexing.service.Pinecone") as mock_pinecone_cls:
            mock_pc = MagicMock()
            mock_pinecone_cls.return_value = mock_pc

            from src.core.config import Settings

            service = make_pinecone_service(Settings(_env_file=None))

        assert isinstance(service, PineconeIndexService)

    def test_uses_pinecone_namespace_from_settings(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("PINECONE_API_KEY", "pc-test-key")
        monkeypatch.setenv("PINECONE_NAMESPACE", "my-namespace")

        with patch("src.retrieval.indexing.service.Pinecone") as mock_pinecone_cls:
            mock_pc = MagicMock()
            mock_pinecone_cls.return_value = mock_pc

            from src.core.config import Settings

            service = make_pinecone_service(Settings(_env_file=None))

        assert service.namespace == "my-namespace"

    def test_api_key_not_stored_on_service(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("PINECONE_API_KEY", "pc-super-secret-key")

        with patch("src.retrieval.indexing.service.Pinecone") as mock_pinecone_cls:
            mock_pc = MagicMock()
            mock_pinecone_cls.return_value = mock_pc

            from src.core.config import Settings

            service = make_pinecone_service(Settings(_env_file=None))

        assert "pc-super-secret-key" not in str(vars(service))
