"""Async-compatible Pinecone index service.

Pinecone's Python client is synchronous.  Each operation in this module is
dispatched to a thread-pool thread via ``asyncio.to_thread`` so the event
loop is never blocked.

Design choices that are deliberately visible — not hidden behind a
framework:

- Every Pinecone API call (``upsert``, ``delete``, ``describe_index_stats``,
  ``create_index``) is invoked by name, not through a generic dispatch layer.
- Batch boundaries are controlled here and logged individually so operators
  can see exactly how data flows into the index.
- The metadata format is defined in ``metadata.py``, not auto-generated —
  every field that lands in Pinecone is explicitly listed.
- ``ensure_index`` wraps ``pc.create_index`` directly; the ``ServerlessSpec``
  parameters are visible to callers.

Usage
-----
    # Production (reads Settings)
    from src.retrieval.indexing import make_pinecone_service, ensure_index
    await ensure_index()
    service = make_pinecone_service()
    result = await service.upsert_chunks(chunks, vectors)

    # Tests (inject a MagicMock index)
    from src.retrieval.indexing import PineconeIndexService
    service = PineconeIndexService(mock_index, namespace="test")
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Sequence
from typing import Any, Final

from pinecone import Pinecone, ServerlessSpec
from pydantic import BaseModel, ConfigDict, Field

from src.core.config import Settings, get_settings
from src.core.logging import get_logger, log_debug, log_info
from src.retrieval.indexing.metadata import chunk_to_metadata
from src.retrieval.ingestion.models import DocumentChunk

_logger = get_logger(__name__)

DEFAULT_BATCH_SIZE: Final[int] = 100
DEFAULT_EMBEDDING_DIM: Final[int] = 3072  # text-embedding-3-large

# Embedding dimension for known OpenAI models.  Unknown models fall back to
# DEFAULT_EMBEDDING_DIM so the CLI does not crash on custom model names.
_MODEL_DIMENSIONS: Final[dict[str, int]] = {
    "text-embedding-3-large": 3072,
    "text-embedding-3-small": 1536,
    "text-embedding-ada-002": 1536,
}


class UpsertResult(BaseModel):
    """Result of a `PineconeIndexService.upsert_chunks` call."""

    model_config = ConfigDict(frozen=True)

    upserted_count: int = Field(ge=0, description="Total vectors accepted by Pinecone.")
    batch_count: int = Field(ge=0, description="Number of Pinecone upsert API calls made.")


class PineconeMatch(BaseModel):
    """One result from a Pinecone vector similarity query."""

    model_config = ConfigDict(frozen=True)

    chunk_id: str = Field(min_length=1, description="Pinecone vector ID (equals chunk_id).")
    score: float = Field(ge=0.0, description="Cosine similarity score.")
    metadata: dict[str, object] = Field(description="All stored metadata fields.")


class PineconeIndexService:
    """Async-compatible service for Pinecone vector index operations.

    Each method wraps one or more synchronous Pinecone SDK calls in
    ``asyncio.to_thread`` so the calling coroutine never blocks.

    Instantiate via ``make_pinecone_service()`` in production.  In tests,
    construct directly with a ``MagicMock`` for *index*:

        service = PineconeIndexService(mock_index, namespace="test")

    The *index* parameter is typed as ``Any`` to avoid coupling to
    Pinecone's internal class hierarchy, which changes between minor
    releases.  The interface contract is behavioural: *index* must have
    ``upsert``, ``delete``, and ``describe_index_stats`` methods.
    """

    def __init__(
        self,
        index: Any,
        *,
        namespace: str = "",
    ) -> None:
        self._index = index
        self._namespace = namespace

    @property
    def namespace(self) -> str:
        """Pinecone namespace this service operates within."""
        return self._namespace

    async def upsert_chunks(
        self,
        chunks: Sequence[DocumentChunk],
        vectors: Sequence[Sequence[float]],
        *,
        batch_size: int = DEFAULT_BATCH_SIZE,
    ) -> UpsertResult:
        """Upsert chunk vectors and their metadata to Pinecone.

        *chunks* and *vectors* must be the same length — element *i* of
        *vectors* is the embedding for element *i* of *chunks*.

        Vectors are sent in batches of *batch_size* to stay within
        Pinecone's per-request size limits.  Each batch is a separate
        ``index.upsert`` call; partial failures surface as exceptions from
        that call.

        Raises ``ValueError`` if ``len(chunks) != len(vectors)``.
        Empty input is a no-op that returns a zero-count ``UpsertResult``.
        """
        if len(chunks) != len(vectors):
            raise ValueError(
                f"chunks and vectors must be the same length: "
                f"got {len(chunks)} chunks and {len(vectors)} vectors"
            )
        if not chunks:
            return UpsertResult(upserted_count=0, batch_count=0)

        records = [
            {
                "id": chunk.chunk_id,
                "values": list(vector),
                "metadata": chunk_to_metadata(chunk),
            }
            for chunk, vector in zip(chunks, vectors, strict=True)
        ]

        batches = [
            records[i : i + batch_size] for i in range(0, len(records), batch_size)
        ]

        log_info(
            _logger,
            "indexing.upsert_start",
            "Starting upsert to Pinecone",
            chunk_count=len(chunks),
            batch_count=len(batches),
            namespace=self._namespace,
        )

        t0 = time.monotonic()
        total_upserted = 0

        for batch_num, batch in enumerate(batches):
            response = await asyncio.to_thread(
                self._index.upsert,
                vectors=batch,
                namespace=self._namespace,
            )
            # Fall back to len(batch) if Pinecone does not return a count
            # (e.g. on some serverless configurations).
            count: int = getattr(response, "upserted_count", len(batch))
            total_upserted += count
            log_debug(
                _logger,
                "indexing.batch_upserted",
                "Batch upserted",
                batch_num=batch_num + 1,
                batch_total=len(batches),
                batch_upserted=count,
            )

        elapsed_ms = round((time.monotonic() - t0) * 1000, 1)
        log_info(
            _logger,
            "indexing.upsert_complete",
            "Upsert complete",
            upserted_count=total_upserted,
            batch_count=len(batches),
            elapsed_ms=elapsed_ms,
        )
        return UpsertResult(upserted_count=total_upserted, batch_count=len(batches))

    async def delete_document(self, document_id: str) -> None:
        """Delete all vectors that belong to *document_id*.

        Uses Pinecone's metadata filter ``{"document_id": {"$eq": ...}}``
        to target only the vectors for this document.  Every chunk stored
        by ``upsert_chunks`` carries a ``document_id`` metadata field, so
        this is always correct as long as metadata was not overwritten
        externally.

        Note: metadata-filter delete requires that the Pinecone index has
        metadata filtering enabled (the default for serverless indexes and
        pod indexes with ``metadata_config``).
        """
        log_info(
            _logger,
            "indexing.delete_document",
            "Deleting document from index",
            document_id=document_id,
            namespace=self._namespace,
        )
        await asyncio.to_thread(
            self._index.delete,
            filter={"document_id": {"$eq": document_id}},
            namespace=self._namespace,
        )

    async def delete_chunks(self, chunk_ids: Sequence[str]) -> None:
        """Delete specific vectors by their chunk IDs.

        Preferred over ``delete_document`` when the exact set of chunk IDs
        is already known — bypasses the metadata filter and works on all
        Pinecone index types, including those without metadata filtering.

        Empty input is a no-op.
        """
        if not chunk_ids:
            return
        log_info(
            _logger,
            "indexing.delete_chunks",
            "Deleting chunks by ID",
            chunk_count=len(chunk_ids),
            namespace=self._namespace,
        )
        await asyncio.to_thread(
            self._index.delete,
            ids=list(chunk_ids),
            namespace=self._namespace,
        )

    async def search(
        self,
        vector: list[float],
        *,
        top_k: int = 10,
        filter: dict[str, object] | None = None,
    ) -> list[PineconeMatch]:
        """Query the Pinecone index for vectors nearest to *vector*.

        Returns up to *top_k* results ordered by cosine similarity descending.
        Pass *filter* to restrict results via Pinecone's metadata filter
        syntax (build the filter with ``build_access_filter`` for RBAC).

        The ``include_metadata=True`` flag is always set so callers can
        reconstruct ``DocumentChunk`` objects without a secondary lookup via
        ``metadata_to_chunk``.
        """
        log_debug(
            _logger,
            "indexing.search",
            "Querying Pinecone index",
            top_k=top_k,
            namespace=self._namespace,
            has_filter=filter is not None,
        )

        def _query() -> list[PineconeMatch]:
            response = self._index.query(
                vector=vector,
                top_k=top_k,
                namespace=self._namespace,
                filter=filter,
                include_metadata=True,
            )
            return [
                PineconeMatch(
                    chunk_id=str(match.id),
                    score=float(match.score),
                    metadata=dict(match.metadata or {}),
                )
                for match in (response.matches or [])
            ]

        return await asyncio.to_thread(_query)

    async def describe_stats(self) -> dict[str, object]:
        """Return index statistics as a plain dict.

        Wraps Pinecone's ``describe_index_stats`` and extracts the fields
        most useful for operational monitoring: total vector count, index
        dimension, and per-namespace vector counts.
        """
        response = await asyncio.to_thread(self._index.describe_index_stats)
        namespaces: dict[str, object] = {}
        if response.namespaces:
            namespaces = {
                ns: {"vector_count": info.vector_count}
                for ns, info in response.namespaces.items()
            }
        return {
            "total_vector_count": response.total_vector_count,
            "dimension": response.dimension,
            "namespaces": namespaces,
        }


async def ensure_index(
    settings: Settings | None = None,
    *,
    metric: str = "cosine",
    cloud: str = "aws",
    region: str = "us-east-1",
) -> None:
    """Create the Pinecone index if it does not already exist.

    The index dimension is looked up from ``Settings.embedding_model``; any
    unrecognised model name falls back to ``DEFAULT_EMBEDDING_DIM`` (3072).

    This is an administrative operation — call it once during deployment or
    via the CLI (``python -m scripts.index --init``), not on every request.

    Parameters
    ----------
    settings:
        Override application settings (for tests or CLI use).
    metric:
        Distance metric for the index (``"cosine"``, ``"euclidean"``,
        ``"dotproduct"``).  Cosine is correct for OpenAI embeddings.
    cloud, region:
        Pinecone Serverless cloud provider and region.  Defaults to
        ``aws / us-east-1`` — change to match your Pinecone project.
    """
    s = settings or get_settings()
    dimension = _MODEL_DIMENSIONS.get(s.embedding_model, DEFAULT_EMBEDDING_DIM)

    pc = Pinecone(api_key=s.pinecone_api_key.get_secret_value())

    def _existing_names() -> list[str]:
        return [str(idx.name) for idx in pc.list_indexes()]

    existing = await asyncio.to_thread(_existing_names)

    if s.pinecone_index_name in existing:
        log_info(
            _logger,
            "indexing.index_exists",
            "Pinecone index already exists — skipping create",
            index_name=s.pinecone_index_name,
        )
        return

    log_info(
        _logger,
        "indexing.create_index",
        "Creating Pinecone index",
        index_name=s.pinecone_index_name,
        dimension=dimension,
        metric=metric,
        cloud=cloud,
        region=region,
    )
    await asyncio.to_thread(
        pc.create_index,
        name=s.pinecone_index_name,
        dimension=dimension,
        metric=metric,
        spec=ServerlessSpec(cloud=cloud, region=region),
    )
    log_info(
        _logger,
        "indexing.index_created",
        "Pinecone index created successfully",
        index_name=s.pinecone_index_name,
        dimension=dimension,
    )


def make_pinecone_service(settings: Settings | None = None) -> PineconeIndexService:
    """Build a `PineconeIndexService` from application settings.

    The Pinecone API key is read from ``Settings.pinecone_api_key``
    (``SecretStr``) and passed directly to the Pinecone constructor — it is
    never stored as an attribute of ``PineconeIndexService``.

    Tests should construct ``PineconeIndexService`` directly with a mock
    index rather than calling this factory.
    """
    s = settings or get_settings()
    pc = Pinecone(api_key=s.pinecone_api_key.get_secret_value())
    index = pc.Index(s.pinecone_index_name)
    return PineconeIndexService(index, namespace=s.pinecone_namespace)
