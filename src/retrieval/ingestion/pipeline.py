"""Ingestion pipeline: discover → parse → chunk.

`ingest()` is the top-level entry point.  It discovers all supported files
under a data directory, parses them into `SourceDocument` objects, and
chunks them into `DocumentChunk` objects.  No embedding or indexing is
performed — that is a downstream concern.

Supported file types
--------------------
- ``*.md``   — Markdown with YAML frontmatter (architecture, runbooks, …)
- ``*.json`` — JSON incident reports with a known schema

Discovery order is deterministic (alphabetical by relative path) so the
output list is stable across runs regardless of filesystem ordering.
"""

from __future__ import annotations

import logging
from pathlib import Path

from src.core.logging import get_logger, log_error, log_info
from src.retrieval.ingestion.chunker import chunk
from src.retrieval.ingestion.models import DocumentChunk, SourceDocument
from src.retrieval.ingestion.parsers.json_incident import parse_json_incident
from src.retrieval.ingestion.parsers.markdown import parse_markdown

logger: logging.Logger = get_logger(__name__)

_PARSERS = {
    ".md": parse_markdown,
    ".json": parse_json_incident,
}


def discover(data_dir: Path) -> list[Path]:
    """Return all ingestible files under *data_dir*, sorted alphabetically.

    Only files with a registered parser extension (``.md``, ``.json``) are
    returned.  ``.gitkeep`` and other housekeeping files are silently skipped.

    The sort is on the *relative* path string so the ordering is independent
    of where *data_dir* lives on disk.
    """
    paths = [
        p
        for p in data_dir.rglob("*")
        if p.is_file() and p.suffix in _PARSERS
    ]
    return sorted(paths, key=lambda p: str(p.relative_to(data_dir)))


def _parse(path: Path) -> SourceDocument:
    parser = _PARSERS[path.suffix]
    return parser(path)


def ingest(
    data_dir: Path,
    *,
    max_chars: int = 1500,
) -> list[DocumentChunk]:
    """Discover, parse, and chunk all documents under *data_dir*.

    Files that fail to parse are logged and skipped — the pipeline continues
    so a single malformed document does not block the entire corpus.

    Returns a flat list of `DocumentChunk` objects sorted by
    ``(document_id, chunk_index)`` for deterministic downstream processing.

    Args:
        data_dir:  Root directory to search for ingestible files.
        max_chars: Maximum characters per chunk passed to the chunker.
    """
    paths = discover(data_dir)
    log_info(logger, "ingestion.discovered", "Discovered files", count=len(paths))

    all_chunks: list[DocumentChunk] = []
    parse_errors = 0
    chunk_errors = 0

    for path in paths:
        try:
            doc = _parse(path)
        except Exception as exc:
            parse_errors += 1
            log_error(
                logger,
                "ingestion.parse_error",
                f"Failed to parse {path.name}",
                error=exc,
                file=str(path),
            )
            continue

        try:
            chunks = chunk(doc, max_chars=max_chars)
        except Exception as exc:
            chunk_errors += 1
            log_error(
                logger,
                "ingestion.chunk_error",
                f"Failed to chunk {doc.metadata.document_id}",
                error=exc,
                document_id=doc.metadata.document_id,
            )
            continue

        all_chunks.extend(chunks)

    all_chunks.sort(key=lambda c: (c.document_id, c.chunk_index))

    log_info(
        logger,
        "ingestion.complete",
        "Ingestion complete",
        files=len(paths),
        chunks=len(all_chunks),
        parse_errors=parse_errors,
        chunk_errors=chunk_errors,
    )
    return all_chunks
