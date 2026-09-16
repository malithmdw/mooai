"""Tests for the ingestion pipeline (discover + ingest).

Integration tests use the actual data/knowledge/ corpus.  Unit tests use
temp directories with controlled fixture files.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.retrieval.ingestion.pipeline import discover, ingest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_MINIMAL_MD = """\
---
document_id: {doc_id}
title: {title}
department: Engineering
document_type: architecture_document
access_level: INTERNAL
created_date: "2024-01-01"
allowed_roles: ["ENGINEER", "ANALYST"]
---

## Section One

Content of section one about payments.

## Section Two

Content of section two about certificates.
"""

_MINIMAL_INCIDENT = {
    "document_id": "INC-PIPE-001",
    "title": "Pipeline Test Incident",
    "department": "Payments Engineering",
    "document_type": "incident_report",
    "access_level": "INTERNAL",
    "created_date": "2024-02-01",
    "allowed_roles": ["ENGINEER", "ANALYST", "ADMINISTRATOR"],
    "severity": "P2",
    "status": "Resolved",
    "duration_minutes": 30,
    "affected_systems": ["payment-gateway"],
    "content": {
        "summary": "A brief test incident summary for the pipeline tests.",
        "timeline": [{"time": "09:00 UTC", "event": "Alert fires"}],
        "root_cause": "Root cause for pipeline test incident.",
        "contributing_factors": ["Factor one", "Factor two"],
        "impact": {"failed_transactions": 500},
        "remediation": {
            "immediate": ["Immediate step"],
            "short_term": ["Short-term step"],
            "long_term": ["Long-term step"],
        },
        "related_incidents": [],
        "related_documents": ["ARCH-003"],
    },
}


def _write_md(directory: Path, doc_id: str, title: str) -> Path:
    path = directory / f"{doc_id}.md"
    path.write_text(_MINIMAL_MD.format(doc_id=doc_id, title=title), encoding="utf-8")
    return path


def _write_incident(directory: Path, data: dict[str, object] | None = None) -> Path:
    payload = dict(_MINIMAL_INCIDENT) if data is None else data
    doc_id = str(payload.get("document_id", "INC-PIPE-001"))
    path = directory / f"{doc_id}.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# discover()
# ---------------------------------------------------------------------------

class TestDiscover:
    def test_finds_markdown_files(self, tmp_path: Path) -> None:
        _write_md(tmp_path, "DOC-001", "Doc One")
        paths = discover(tmp_path)
        assert any(p.suffix == ".md" for p in paths)

    def test_finds_json_files(self, tmp_path: Path) -> None:
        _write_incident(tmp_path)
        paths = discover(tmp_path)
        assert any(p.suffix == ".json" for p in paths)

    def test_ignores_gitkeep(self, tmp_path: Path) -> None:
        (tmp_path / ".gitkeep").touch()
        paths = discover(tmp_path)
        assert not any(p.name == ".gitkeep" for p in paths)

    def test_ignores_unknown_extensions(self, tmp_path: Path) -> None:
        (tmp_path / "notes.txt").write_text("some notes")
        paths = discover(tmp_path)
        assert not any(p.suffix == ".txt" for p in paths)

    def test_returns_sorted_alphabetical_order(self, tmp_path: Path) -> None:
        _write_md(tmp_path, "ZZZ-999", "Last Doc")
        _write_md(tmp_path, "AAA-001", "First Doc")
        paths = discover(tmp_path)
        names = [p.stem for p in paths]
        assert names == sorted(names)

    def test_discovers_nested_files(self, tmp_path: Path) -> None:
        subdir = tmp_path / "incidents"
        subdir.mkdir()
        _write_incident(subdir)
        paths = discover(tmp_path)
        assert len(paths) >= 1

    def test_empty_directory_returns_empty_list(self, tmp_path: Path) -> None:
        assert discover(tmp_path) == []

    def test_deterministic_across_calls(self, tmp_path: Path) -> None:
        _write_md(tmp_path, "DOC-A", "A")
        _write_md(tmp_path, "DOC-B", "B")
        first = [str(p) for p in discover(tmp_path)]
        second = [str(p) for p in discover(tmp_path)]
        assert first == second


# ---------------------------------------------------------------------------
# ingest() — document attribution
# ---------------------------------------------------------------------------

class TestIngestAttribution:
    def test_chunks_have_correct_document_id(self, tmp_path: Path) -> None:
        _write_md(tmp_path, "ARCH-TST-001", "Attribution Test")
        chunks = ingest(tmp_path)
        assert all(c.document_id == "ARCH-TST-001" for c in chunks)

    def test_chunks_have_correct_title(self, tmp_path: Path) -> None:
        _write_md(tmp_path, "ARCH-TST-002", "My Custom Title")
        chunks = ingest(tmp_path)
        assert all(c.title == "My Custom Title" for c in chunks)

    def test_chunk_id_contains_document_id(self, tmp_path: Path) -> None:
        _write_md(tmp_path, "ARCH-TST-003", "Chunk ID Test")
        chunks = ingest(tmp_path)
        assert all("ARCH-TST-003" in c.chunk_id for c in chunks)

    def test_metadata_preserved_in_chunks(self, tmp_path: Path) -> None:
        from src.models.enums import AccessLevel, Role

        _write_md(tmp_path, "ARCH-TST-004", "Meta Test")
        chunks = ingest(tmp_path)
        assert all(c.metadata.access_level is AccessLevel.INTERNAL for c in chunks)
        assert all(Role.ENGINEER in c.metadata.allowed_roles for c in chunks)


# ---------------------------------------------------------------------------
# ingest() — ordering and completeness
# ---------------------------------------------------------------------------

class TestIngestOrdering:
    def test_chunks_sorted_by_document_id_then_index(self, tmp_path: Path) -> None:
        _write_md(tmp_path, "ZZZ-002", "Z Doc")
        _write_md(tmp_path, "AAA-001", "A Doc")
        chunks = ingest(tmp_path)
        doc_ids = [c.document_id for c in chunks]
        # Chunks for AAA-001 should come before ZZZ-002
        aaa_idx = next(i for i, c in enumerate(chunks) if c.document_id == "AAA-001")
        zzz_idx = next(i for i, c in enumerate(chunks) if c.document_id == "ZZZ-002")
        assert aaa_idx < zzz_idx

    def test_chunk_indices_per_document_are_sequential(self, tmp_path: Path) -> None:
        _write_md(tmp_path, "DOC-SEQ-001", "Sequential Test")
        chunks = ingest(tmp_path)
        doc_chunks = [c for c in chunks if c.document_id == "DOC-SEQ-001"]
        indices = [c.chunk_index for c in doc_chunks]
        assert indices == list(range(len(indices)))

    def test_chunk_total_consistent_within_document(self, tmp_path: Path) -> None:
        _write_md(tmp_path, "DOC-TOT-001", "Total Test")
        chunks = ingest(tmp_path)
        doc_chunks = [c for c in chunks if c.document_id == "DOC-TOT-001"]
        totals = {c.chunk_total for c in doc_chunks}
        assert len(totals) == 1
        assert totals.pop() == len(doc_chunks)


# ---------------------------------------------------------------------------
# ingest() — multi-document corpus
# ---------------------------------------------------------------------------

class TestIngestMultiDocument:
    def test_ingests_both_md_and_json(self, tmp_path: Path) -> None:
        _write_md(tmp_path, "ARCH-MULTI-001", "Arch Doc")
        _write_incident(tmp_path)
        chunks = ingest(tmp_path)
        doc_ids = {c.document_id for c in chunks}
        assert "ARCH-MULTI-001" in doc_ids
        assert "INC-PIPE-001" in doc_ids

    def test_parse_error_does_not_abort_pipeline(self, tmp_path: Path) -> None:
        # Write a valid doc and an invalid JSON file
        _write_md(tmp_path, "VALID-001", "Valid")
        bad = tmp_path / "bad.json"
        bad.write_text("{ not valid json }", encoding="utf-8")
        # Should not raise; valid doc still produces chunks
        chunks = ingest(tmp_path)
        doc_ids = {c.document_id for c in chunks}
        assert "VALID-001" in doc_ids

    def test_returns_empty_list_for_empty_directory(self, tmp_path: Path) -> None:
        assert ingest(tmp_path) == []


# ---------------------------------------------------------------------------
# Integration: actual data/knowledge/ corpus
# ---------------------------------------------------------------------------

KNOWLEDGE_DIR = Path(__file__).parents[3] / "data" / "knowledge"


@pytest.mark.skipif(
    not KNOWLEDGE_DIR.exists(),
    reason="data/knowledge/ not present",
)
class TestCorpusIntegration:
    def test_discovers_at_least_60_files(self) -> None:
        paths = discover(KNOWLEDGE_DIR)
        assert len(paths) >= 60, f"Expected ≥60 files, found {len(paths)}"

    def test_ingests_all_files_without_errors(self) -> None:
        chunks = ingest(KNOWLEDGE_DIR)
        # Every file should produce at least one chunk
        doc_ids = {c.document_id for c in chunks}
        assert len(doc_ids) >= 60

    def test_all_chunks_have_non_empty_text(self) -> None:
        chunks = ingest(KNOWLEDGE_DIR)
        assert all(c.text.strip() for c in chunks)

    def test_all_chunks_have_non_empty_section(self) -> None:
        chunks = ingest(KNOWLEDGE_DIR)
        assert all(c.section for c in chunks)

    def test_chunk_ids_are_globally_unique(self) -> None:
        chunks = ingest(KNOWLEDGE_DIR)
        ids = [c.chunk_id for c in chunks]
        assert len(ids) == len(set(ids)), "Duplicate chunk IDs detected"

    def test_incident_chunks_contain_root_cause(self) -> None:
        chunks = ingest(KNOWLEDGE_DIR)
        inc_chunks = [c for c in chunks if c.document_id.startswith("INC-")]
        root_cause_chunks = [c for c in inc_chunks if "Root Cause" in c.section]
        # At least some incidents should have a Root Cause section chunk
        assert root_cause_chunks, "No Root Cause section chunks found in incidents"

    def test_restricted_documents_carry_correct_roles(self) -> None:
        from src.models.enums import AccessLevel, Role

        chunks = ingest(KNOWLEDGE_DIR)
        restricted = [c for c in chunks if c.metadata.access_level is AccessLevel.RESTRICTED]
        assert restricted, "Expected at least some RESTRICTED chunks"
        for c in restricted:
            roles = set(c.metadata.allowed_roles)
            # RESTRICTED docs must not allow the general ENGINEER role
            assert Role.ENGINEER not in roles, (
                f"RESTRICTED doc {c.document_id} incorrectly allows ENGINEER role"
            )

    def test_payment_incident_chunks_reference_known_systems(self) -> None:
        chunks = ingest(KNOWLEDGE_DIR)
        payment_inc_chunks = [
            c for c in chunks
            if c.document_id.startswith("INC-") and "payment" in c.text.lower()
        ]
        assert payment_inc_chunks, "No payment-related incident chunks found"
