"""Pinecone vector index service.

Public surface
--------------
- `PineconeIndexService` — async-compatible upsert / delete / search / stats.
- `UpsertResult` — typed result of an upsert operation.
- `PineconeMatch` — one result from `PineconeIndexService.search`.
- `ensure_index` — create the Pinecone index when it does not already exist.
- `make_pinecone_service` — production factory that reads from `Settings`.
- `chunk_to_metadata` — serialize a `DocumentChunk` to Pinecone metadata dict.
- `metadata_to_chunk` — inverse: reconstruct a `DocumentChunk` from metadata.
- `build_access_filter` — build a Pinecone metadata filter for RBAC queries.

Callers should depend on `PineconeIndexService`, not on the Pinecone SDK
directly, so the concrete index client can be swapped or mocked in tests.
"""

from src.retrieval.indexing.metadata import (
    build_access_filter,
    chunk_to_metadata,
    metadata_to_chunk,
)
from src.retrieval.indexing.service import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_EMBEDDING_DIM,
    PineconeIndexService,
    PineconeMatch,
    UpsertResult,
    ensure_index,
    make_pinecone_service,
)

__all__ = [
    "build_access_filter",
    "chunk_to_metadata",
    "metadata_to_chunk",
    "DEFAULT_BATCH_SIZE",
    "DEFAULT_EMBEDDING_DIM",
    "PineconeIndexService",
    "PineconeMatch",
    "UpsertResult",
    "ensure_index",
    "make_pinecone_service",
]
