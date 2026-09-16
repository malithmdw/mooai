"""Ingestion-internal data structures.

`SourceDocument` is the output of a parser (one per file).
`DocumentChunk` is the output of the chunker (one or more per source document).
Neither is stored or embedded here — they are handed off to downstream stages.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from src.models.documents import DocumentMetadata


class SourceDocument(BaseModel):
    """A parsed document before chunking.

    `body` is the renderable text content *without* frontmatter or file
    headers — ready to be fed to the chunker.
    """

    model_config = ConfigDict(frozen=True)

    file_path: Path
    metadata: DocumentMetadata
    body: str = Field(min_length=1)


class DocumentChunk(BaseModel):
    """One chunk produced by splitting a `SourceDocument`.

    Chunk identity:
    - `chunk_id`: ``{document_id}-chunk-{chunk_index:04d}`` — stable,
      sortable, and unique within the corpus as long as document IDs are.
    - `chunk_index`: 0-based position within the document.
    - `chunk_total`: total chunks for this document (same value on every
      sibling chunk — lets callers reconstruct order without scanning).

    Attribution:
    - `document_id`, `title`, and `metadata` are copied from the source
      document so every chunk is fully self-describing when retrieved in
      isolation.

    Section context:
    - `section` is the H2 heading that contains this chunk.  When a section
      is large enough to split into multiple chunks they all share the same
      ``section`` value; ``chunk_index`` provides ordering.
    """

    model_config = ConfigDict(frozen=True)

    chunk_id: str = Field(min_length=1)
    document_id: str = Field(min_length=1)
    title: str = Field(min_length=1)
    section: str = Field(min_length=1)
    chunk_index: int = Field(ge=0)
    chunk_total: int = Field(ge=1)
    text: str = Field(min_length=1)
    metadata: DocumentMetadata
