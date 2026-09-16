"""Pinecone metadata serialisation and access-control filter helpers.

Pinecone metadata values are restricted to ``str``, ``bool``, ``int``,
``float``, or ``list[str]``.  This module converts ``DocumentChunk`` fields
to that schema and builds MongoDB-style filter expressions that Pinecone
accepts for RBAC-aware queries.

Both helpers are deliberately simple and fully explicit — every field name
that ends up in Pinecone is stated here, making the metadata schema easy to
audit.
"""

from __future__ import annotations

from collections.abc import Sequence

from src.models.enums import AccessLevel, Role
from src.retrieval.ingestion.models import DocumentChunk

# Type alias matching Pinecone's accepted metadata value types.
PineconeMetadata = dict[str, str | int | list[str]]


def chunk_to_metadata(chunk: DocumentChunk) -> PineconeMetadata:
    """Serialize *chunk* to a Pinecone-compatible metadata dict.

    All fields required to reconstruct a retrieval result are included so
    retrieved vectors are fully self-describing — no secondary database
    lookup is needed when a query returns a match.

    Field mapping
    -------------
    ``allowed_roles``  → ``list[str]`` — supports ``$in`` filter queries.
    ``access_level``   → ``str``       — supports equality / ``$in`` filters.
    ``created_date``   → ISO-8601 ``str`` — Pinecone has no native date type.
    ``chunk_index``    → ``int``       — for ordering sibling chunks.
    ``chunk_total``    → ``int``       — total chunks in the source document.
    """
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


def build_access_filter(
    roles: Sequence[Role],
    *,
    access_levels: Sequence[AccessLevel] | None = None,
) -> dict[str, object]:
    """Build a Pinecone metadata filter that enforces RBAC access control.

    The filter requires that *at least one* of the user's *roles* appears
    in the ``allowed_roles`` metadata list on a vector.  This ensures a
    query only surfaces documents the requesting user is permitted to read.

    Optionally restricts to a specific set of *access_levels* — useful when
    a caller wants to exclude ``CONFIDENTIAL`` or ``RESTRICTED`` documents
    regardless of the user's roles.

    Examples
    --------
    >>> build_access_filter([Role.ENGINEER])
    {'allowed_roles': {'$in': ['ENGINEER']}}

    >>> build_access_filter([Role.ANALYST], access_levels=[AccessLevel.INTERNAL])
    {'allowed_roles': {'$in': ['ANALYST']}, 'access_level': {'$in': ['INTERNAL']}}
    """
    result: dict[str, object] = {
        "allowed_roles": {"$in": [r.value for r in roles]},
    }
    if access_levels:
        result["access_level"] = {"$in": [a.value for a in access_levels]}
    return result
