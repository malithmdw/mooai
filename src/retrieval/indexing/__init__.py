"""Pinecone vector index service.

Public surface
--------------
- `PineconeIndexService` — async-compatible upsert / delete / stats service.
- `UpsertResult` — typed result of an upsert operation.
- `ensure_index` — create the Pinecone index when it does not already exist.
- `make_pinecone_service` — production factory that reads from `Settings`.
- `chunk_to_metadata` — serialize a `DocumentChunk` to Pinecone metadata dict.
- `build_access_filter` — build a Pinecone metadata filter for RBAC queries.

Callers should depend on `PineconeIndexService`, not on the Pinecone SDK
directly, so the concrete index client can be swapped or mocked in tests.
"""

from src.retrieval.indexing.metadata import build_access_filter, chunk_to_metadata
from src.retrieval.indexing.service import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_EMBEDDING_DIM,
    PineconeIndexService,
    UpsertResult,
    ensure_index,
    make_pinecone_service,
)

__all__ = [
    "build_access_filter",
    "chunk_to_metadata",
    "DEFAULT_BATCH_SIZE",
    "DEFAULT_EMBEDDING_DIM",
    "PineconeIndexService",
    "UpsertResult",
    "ensure_index",
    "make_pinecone_service",
]
