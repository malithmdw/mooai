"""Deterministic, section-aware document chunker.

Algorithm
---------
1. Split the document body at H2 heading boundaries (lines matching ``^## ``).
   Text before the first H2 becomes an "Introduction" section (if non-empty).
2. Each section yields one or more chunks:
   a. If ``len(section_text) <= max_chars``: a single chunk.
   b. If ``len(section_text) > max_chars``: split at double-newline
      (``\\n\\n``) paragraph boundaries, accumulating paragraphs greedily
      until the next paragraph would push the chunk over *max_chars*,
      then start a new chunk.  The section heading is prepended to every
      sub-chunk so each chunk is self-contained.
3. Chunk IDs are ``{document_id}-chunk-{chunk_index:04d}`` — lexicographically
   sortable and stable as long as document content does not change.
4. ``chunk_total`` is set to the final chunk count after all sections are
   processed, and written back into every chunk.

Determinism guarantee: given the same ``SourceDocument`` and same
``max_chars``, this function always produces the same ordered list of
``DocumentChunk`` objects.  There is no randomness or timestamp dependency.
"""

from __future__ import annotations

import re
from typing import Final

from src.retrieval.ingestion.models import DocumentChunk, SourceDocument

# Split at lines that start with exactly two hashes followed by a space.
# H3+ headings (###) are kept inside sections; only H2 triggers a split.
_H2_SPLIT_RE: Final = re.compile(r"^(## .+)$", re.MULTILINE)

DEFAULT_MAX_CHARS: Final[int] = 1500


def _split_sections(body: str) -> list[tuple[str, str]]:
    """Split *body* into ``(heading, content)`` pairs at H2 boundaries.

    Text before the first H2 heading is labelled ``"Introduction"`` if it
    contains non-whitespace content; otherwise it is discarded.
    """
    parts = _H2_SPLIT_RE.split(body)
    # parts: [pre_text, "## H1", section1_body, "## H2", section2_body, ...]

    sections: list[tuple[str, str]] = []

    intro = parts[0].strip()
    if intro:
        sections.append(("Introduction", intro))

    for i in range(1, len(parts), 2):
        heading = parts[i].strip()
        content = parts[i + 1].strip() if i + 1 < len(parts) else ""
        if content:
            sections.append((heading, content))

    return sections


def _split_paragraphs(text: str, heading: str, max_chars: int) -> list[str]:
    """Divide *text* into chunks of at most *max_chars* at paragraph breaks.

    Each sub-chunk is prefixed with *heading* so it is self-contained when
    retrieved in isolation.  Paragraphs that individually exceed *max_chars*
    are included as a single oversized chunk rather than being split
    mid-sentence (hard-cutting prose is worse than a slightly oversized chunk).
    """
    paragraphs = [p.strip() for p in re.split(r"\n\n+", text) if p.strip()]

    chunks: list[str] = []
    current_parts: list[str] = []
    current_len: int = 0

    for para in paragraphs:
        # +2 accounts for the \n\n separator added between paragraphs
        addition = len(para) + (2 if current_parts else 0)
        if current_parts and current_len + addition > max_chars:
            chunks.append(f"{heading}\n\n" + "\n\n".join(current_parts))
            current_parts = [para]
            current_len = len(para)
        else:
            current_parts.append(para)
            current_len += addition

    if current_parts:
        chunks.append(f"{heading}\n\n" + "\n\n".join(current_parts))

    return chunks


def chunk(doc: SourceDocument, max_chars: int = DEFAULT_MAX_CHARS) -> list[DocumentChunk]:
    """Chunk *doc* into an ordered list of `DocumentChunk` objects.

    The returned list is sorted by ``chunk_index`` (0-based).  ``chunk_total``
    on every chunk equals ``len(result)``.

    Raises ``ValueError`` if the document body produces no non-empty sections.
    """
    sections = _split_sections(doc.body)
    if not sections:
        raise ValueError(
            f"Document '{doc.metadata.document_id}' produced no non-empty sections"
        )

    raw_chunks: list[tuple[str, str]] = []  # (section_heading, chunk_text)

    for heading, content in sections:
        section_text = f"{heading}\n\n{content}"
        if len(section_text) <= max_chars:
            raw_chunks.append((heading, section_text))
        else:
            for sub_chunk in _split_paragraphs(content, heading, max_chars):
                raw_chunks.append((heading, sub_chunk))

    document_id = doc.metadata.document_id
    title = doc.metadata.title
    total = len(raw_chunks)

    return [
        DocumentChunk(
            chunk_id=f"{document_id}-chunk-{index:04d}",
            document_id=document_id,
            title=title,
            section=section_heading,
            chunk_index=index,
            chunk_total=total,
            text=chunk_text,
            metadata=doc.metadata,
        )
        for index, (section_heading, chunk_text) in enumerate(raw_chunks)
    ]
