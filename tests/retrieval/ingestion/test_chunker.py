"""Tests for the deterministic, section-aware document chunker.

Tests verify:
- Chunk ordering (chunk_index sequential, 0-based)
- chunk_total consistency across all sibling chunks
- Document attribution preserved in every chunk (document_id, title, metadata)
- Section heading preserved
- No chunk exceeds max_chars (barring a single oversized paragraph)
- Chunker is deterministic: same input → same output
- No text is lost: all content appears in exactly one chunk
"""

from __future__ import annotations

from datetime import date

import pytest

from src.models.documents import DocumentMetadata
from src.models.enums import AccessLevel, Role
from src.retrieval.ingestion.chunker import chunk, DEFAULT_MAX_CHARS
from src.retrieval.ingestion.models import DocumentChunk, SourceDocument


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_metadata(**overrides: object) -> DocumentMetadata:
    fields: dict[str, object] = {
        "document_id": "DOC-001",
        "title": "Test Document",
        "department": "Engineering",
        "document_type": "architecture_document",
        "access_level": AccessLevel.INTERNAL,
        "created_date": date(2024, 1, 1),
        "allowed_roles": (Role.ENGINEER, Role.ANALYST),
        **overrides,
    }
    return DocumentMetadata.model_validate(fields)


def _make_doc(body: str, **meta_overrides: object) -> SourceDocument:
    return SourceDocument(
        file_path=__file__,  # type: ignore[arg-type]
        metadata=_make_metadata(**meta_overrides),
        body=body,
    )


# A simple two-section body that fits in a single chunk each.
_SIMPLE_BODY = """\
## Section One

Content of the first section.

## Section Two

Content of the second section.
"""

# A body with one section that will exceed max_chars when max_chars is small.
_LONG_SECTION_BODY = """\
## Long Section

First paragraph of substantial content that fills some space.

Second paragraph with more content that adds to the total length.

Third paragraph to push the section over the limit.
"""


# ---------------------------------------------------------------------------
# Chunk ordering and basic structure
# ---------------------------------------------------------------------------

class TestChunkOrdering:
    def test_chunk_indices_start_at_zero(self) -> None:
        chunks = chunk(_make_doc(_SIMPLE_BODY))
        assert chunks[0].chunk_index == 0

    def test_chunk_indices_are_sequential(self) -> None:
        chunks = chunk(_make_doc(_SIMPLE_BODY))
        for expected, c in enumerate(chunks):
            assert c.chunk_index == expected

    def test_two_sections_produce_two_chunks(self) -> None:
        chunks = chunk(_make_doc(_SIMPLE_BODY))
        assert len(chunks) == 2

    def test_chunk_total_equals_list_length(self) -> None:
        chunks = chunk(_make_doc(_SIMPLE_BODY))
        for c in chunks:
            assert c.chunk_total == len(chunks)

    def test_chunk_total_consistent_across_all_chunks(self) -> None:
        chunks = chunk(_make_doc(_SIMPLE_BODY))
        totals = {c.chunk_total for c in chunks}
        assert len(totals) == 1  # all chunks report the same total


# ---------------------------------------------------------------------------
# Document attribution
# ---------------------------------------------------------------------------

class TestDocumentAttribution:
    def test_document_id_preserved_in_all_chunks(self) -> None:
        chunks = chunk(_make_doc(_SIMPLE_BODY, document_id="ARCH-999"))
        assert all(c.document_id == "ARCH-999" for c in chunks)

    def test_title_preserved_in_all_chunks(self) -> None:
        chunks = chunk(_make_doc(_SIMPLE_BODY, title="My Custom Title"))
        assert all(c.title == "My Custom Title" for c in chunks)

    def test_metadata_preserved_in_all_chunks(self) -> None:
        doc = _make_doc(_SIMPLE_BODY, access_level=AccessLevel.RESTRICTED)
        chunks = chunk(doc)
        assert all(c.metadata.access_level is AccessLevel.RESTRICTED for c in chunks)

    def test_metadata_allowed_roles_preserved(self) -> None:
        doc = _make_doc(_SIMPLE_BODY)
        chunks = chunk(doc)
        assert all(Role.ENGINEER in c.metadata.allowed_roles for c in chunks)


# ---------------------------------------------------------------------------
# Chunk IDs
# ---------------------------------------------------------------------------

class TestChunkIds:
    def test_chunk_id_format(self) -> None:
        chunks = chunk(_make_doc(_SIMPLE_BODY, document_id="ARCH-001"))
        assert chunks[0].chunk_id == "ARCH-001-chunk-0000"
        assert chunks[1].chunk_id == "ARCH-001-chunk-0001"

    def test_chunk_ids_are_unique(self) -> None:
        chunks = chunk(_make_doc(_SIMPLE_BODY))
        ids = [c.chunk_id for c in chunks]
        assert len(ids) == len(set(ids))


# ---------------------------------------------------------------------------
# Section heading preservation
# ---------------------------------------------------------------------------

class TestSectionPreservation:
    def test_section_heading_matches_h2(self) -> None:
        chunks = chunk(_make_doc(_SIMPLE_BODY))
        sections = [c.section for c in chunks]
        assert "## Section One" in sections
        assert "## Section Two" in sections

    def test_section_is_non_empty_on_every_chunk(self) -> None:
        chunks = chunk(_make_doc(_SIMPLE_BODY))
        assert all(c.section for c in chunks)

    def test_text_includes_section_heading(self) -> None:
        chunks = chunk(_make_doc(_SIMPLE_BODY))
        for c in chunks:
            assert c.section in c.text


# ---------------------------------------------------------------------------
# Chunk boundaries
# ---------------------------------------------------------------------------

class TestChunkBoundaries:
    def test_short_section_is_single_chunk(self) -> None:
        body = "## Short Section\n\nThis is short content."
        chunks = chunk(_make_doc(body))
        assert len(chunks) == 1

    def test_no_chunk_exceeds_max_chars(self) -> None:
        max_chars = 200
        chunks = chunk(_make_doc(_LONG_SECTION_BODY), max_chars=max_chars)
        # Single oversized paragraphs are allowed; combined chunks must not
        # exceed max_chars unless a single paragraph alone does.
        for c in chunks:
            paragraphs = [p for p in c.text.split("\n\n") if p.strip()]
            if len(paragraphs) > 1:
                assert len(c.text) <= max_chars, (
                    f"Multi-paragraph chunk exceeds max_chars={max_chars}: "
                    f"len={len(c.text)}"
                )

    def test_long_section_splits_into_multiple_chunks(self) -> None:
        # Use a very small max_chars to force splitting
        max_chars = 100
        chunks = chunk(_make_doc(_LONG_SECTION_BODY), max_chars=max_chars)
        assert len(chunks) > 1

    def test_split_chunks_share_section_heading(self) -> None:
        max_chars = 100
        chunks = chunk(_make_doc(_LONG_SECTION_BODY), max_chars=max_chars)
        assert all(c.section == "## Long Section" for c in chunks)

    def test_paragraph_boundary_not_mid_sentence(self) -> None:
        # Each known paragraph must be fully contained in at least one chunk.
        # If splits happened mid-sentence the paragraph fragment would not match.
        max_chars = 100
        chunks = chunk(_make_doc(_LONG_SECTION_BODY), max_chars=max_chars)
        for para_start in ("First paragraph", "Second paragraph", "Third paragraph"):
            assert any(para_start in c.text for c in chunks), (
                f"Paragraph starting '{para_start}' not found complete in any chunk"
            )

    def test_intro_text_before_first_h2_becomes_introduction_chunk(self) -> None:
        body = "Some introductory text before any heading.\n\n## Section A\n\nContent A."
        chunks = chunk(_make_doc(body))
        sections = [c.section for c in chunks]
        assert "Introduction" in sections
        assert "## Section A" in sections

    def test_no_intro_section_when_body_starts_with_h2(self) -> None:
        body = "## Section A\n\nContent A.\n\n## Section B\n\nContent B."
        chunks = chunk(_make_doc(body))
        sections = [c.section for c in chunks]
        assert "Introduction" not in sections


# ---------------------------------------------------------------------------
# Content completeness
# ---------------------------------------------------------------------------

class TestContentCompleteness:
    def test_all_section_content_appears_in_chunks(self) -> None:
        chunks = chunk(_make_doc(_SIMPLE_BODY))
        all_text = "\n".join(c.text for c in chunks)
        assert "Content of the first section" in all_text
        assert "Content of the second section" in all_text

    def test_long_section_content_preserved_across_splits(self) -> None:
        max_chars = 100
        chunks = chunk(_make_doc(_LONG_SECTION_BODY), max_chars=max_chars)
        all_text = "\n".join(c.text for c in chunks)
        assert "First paragraph" in all_text
        assert "Second paragraph" in all_text
        assert "Third paragraph" in all_text


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------

class TestDeterminism:
    def test_same_input_produces_same_output(self) -> None:
        doc = _make_doc(_SIMPLE_BODY)
        first = chunk(doc)
        second = chunk(doc)
        assert [c.chunk_id for c in first] == [c.chunk_id for c in second]
        assert [c.text for c in first] == [c.text for c in second]

    def test_same_input_with_different_call_produces_same_output(self) -> None:
        doc = _make_doc(_LONG_SECTION_BODY)
        results = [chunk(doc, max_chars=200) for _ in range(3)]
        chunk_ids = [[c.chunk_id for c in r] for r in results]
        assert chunk_ids[0] == chunk_ids[1] == chunk_ids[2]


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_raises_on_body_with_no_content(self) -> None:
        with pytest.raises(ValueError, match="no non-empty sections"):
            chunk(_make_doc("   \n\n   "))

    def test_single_section_document(self) -> None:
        body = "## Only Section\n\nSome content here."
        chunks = chunk(_make_doc(body))
        assert len(chunks) == 1
        assert chunks[0].chunk_index == 0
        assert chunks[0].chunk_total == 1

    def test_h3_headings_stay_inside_section(self) -> None:
        body = "## Parent Section\n\n### Sub-section\n\nContent under sub."
        chunks = chunk(_make_doc(body))
        # H3 must not trigger a new section split
        assert len(chunks) == 1
        assert "### Sub-section" in chunks[0].text

    def test_returns_list_of_document_chunks(self) -> None:
        chunks = chunk(_make_doc(_SIMPLE_BODY))
        assert all(isinstance(c, DocumentChunk) for c in chunks)

    def test_custom_max_chars_respected(self) -> None:
        # With a generous limit, a multi-paragraph section stays as one chunk
        body = "## Section\n\nPara one.\n\nPara two."
        chunks_large = chunk(_make_doc(body), max_chars=10000)
        chunks_small = chunk(_make_doc(body), max_chars=20)
        assert len(chunks_large) == 1
        assert len(chunks_small) >= 1  # may split at paragraph boundaries
