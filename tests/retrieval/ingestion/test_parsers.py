"""Tests for the Markdown and JSON parsers.

Each parser test uses either inline strings (via tmp_path) or fixture files
from data/knowledge/ — no network, no external services.
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pytest

from src.models.enums import AccessLevel, Role
from src.retrieval.ingestion.parsers.markdown import _parse_frontmatter, parse_markdown
from src.retrieval.ingestion.parsers.json_incident import parse_json_incident


# ---------------------------------------------------------------------------
# Markdown frontmatter parser (_parse_frontmatter)
# ---------------------------------------------------------------------------

_VALID_FRONTMATTER = """\
---
document_id: ARCH-001
title: Core Banking Ledger — Architecture Overview
department: Core Banking
document_type: architecture_document
access_level: INTERNAL
created_date: "2023-06-01"
allowed_roles: ["ENGINEER", "ANALYST", "ADMINISTRATOR"]
---

# Core Banking Ledger — Architecture Overview

## 1. Purpose

This document describes the architecture.
"""


class TestParseFrontmatter:
    def test_extracts_string_field(self) -> None:
        fields, _ = _parse_frontmatter(_VALID_FRONTMATTER)
        assert fields["document_id"] == "ARCH-001"

    def test_extracts_title_with_special_characters(self) -> None:
        fields, _ = _parse_frontmatter(_VALID_FRONTMATTER)
        assert fields["title"] == "Core Banking Ledger — Architecture Overview"

    def test_extracts_quoted_date(self) -> None:
        fields, _ = _parse_frontmatter(_VALID_FRONTMATTER)
        assert fields["created_date"] == "2023-06-01"

    def test_extracts_json_array_roles(self) -> None:
        fields, _ = _parse_frontmatter(_VALID_FRONTMATTER)
        assert fields["allowed_roles"] == ["ENGINEER", "ANALYST", "ADMINISTRATOR"]

    def test_body_starts_after_closing_delimiter(self) -> None:
        _, body = _parse_frontmatter(_VALID_FRONTMATTER)
        assert body.startswith("# Core Banking Ledger")

    def test_body_does_not_contain_frontmatter(self) -> None:
        _, body = _parse_frontmatter(_VALID_FRONTMATTER)
        assert "document_id:" not in body
        assert "---" not in body

    def test_raises_on_missing_frontmatter(self) -> None:
        with pytest.raises(ValueError, match="frontmatter"):
            _parse_frontmatter("# Just a heading\n\nNo frontmatter here.")

    def test_raises_on_unclosed_frontmatter(self) -> None:
        with pytest.raises(ValueError, match="frontmatter"):
            _parse_frontmatter("---\ndocument_id: X\n# No closing delimiter")


# ---------------------------------------------------------------------------
# parse_markdown — full integration with DocumentMetadata
# ---------------------------------------------------------------------------

_MINIMAL_MD = """\
---
document_id: TEST-MD-001
title: Minimal Test Document
department: Engineering
document_type: test
access_level: INTERNAL
created_date: "2024-01-01"
allowed_roles: ["ENGINEER"]
---

## Section One

Content of section one.
"""


class TestParseMarkdown:
    def test_returns_source_document(self, tmp_path: Path) -> None:
        from src.retrieval.ingestion.models import SourceDocument

        f = tmp_path / "test.md"
        f.write_text(_MINIMAL_MD, encoding="utf-8")
        doc = parse_markdown(f)
        assert isinstance(doc, SourceDocument)

    def test_metadata_document_id(self, tmp_path: Path) -> None:
        f = tmp_path / "test.md"
        f.write_text(_MINIMAL_MD, encoding="utf-8")
        doc = parse_markdown(f)
        assert doc.metadata.document_id == "TEST-MD-001"

    def test_metadata_title(self, tmp_path: Path) -> None:
        f = tmp_path / "test.md"
        f.write_text(_MINIMAL_MD, encoding="utf-8")
        doc = parse_markdown(f)
        assert doc.metadata.title == "Minimal Test Document"

    def test_metadata_access_level(self, tmp_path: Path) -> None:
        f = tmp_path / "test.md"
        f.write_text(_MINIMAL_MD, encoding="utf-8")
        doc = parse_markdown(f)
        assert doc.metadata.access_level is AccessLevel.INTERNAL

    def test_metadata_allowed_roles(self, tmp_path: Path) -> None:
        f = tmp_path / "test.md"
        f.write_text(_MINIMAL_MD, encoding="utf-8")
        doc = parse_markdown(f)
        assert Role.ENGINEER in doc.metadata.allowed_roles

    def test_metadata_created_date(self, tmp_path: Path) -> None:
        f = tmp_path / "test.md"
        f.write_text(_MINIMAL_MD, encoding="utf-8")
        doc = parse_markdown(f)
        assert doc.metadata.created_date == date(2024, 1, 1)

    def test_body_excludes_frontmatter(self, tmp_path: Path) -> None:
        f = tmp_path / "test.md"
        f.write_text(_MINIMAL_MD, encoding="utf-8")
        doc = parse_markdown(f)
        assert "document_id:" not in doc.body
        assert "---" not in doc.body

    def test_body_contains_content(self, tmp_path: Path) -> None:
        f = tmp_path / "test.md"
        f.write_text(_MINIMAL_MD, encoding="utf-8")
        doc = parse_markdown(f)
        assert "Section One" in doc.body
        assert "Content of section one" in doc.body

    def test_restricted_access_level(self, tmp_path: Path) -> None:
        restricted = _MINIMAL_MD.replace("access_level: INTERNAL", "access_level: RESTRICTED")
        restricted = restricted.replace(
            'allowed_roles: ["ENGINEER"]',
            'allowed_roles: ["ANALYST", "ADMINISTRATOR"]',
        )
        f = tmp_path / "restricted.md"
        f.write_text(restricted, encoding="utf-8")
        doc = parse_markdown(f)
        assert doc.metadata.access_level is AccessLevel.RESTRICTED
        assert Role.ANALYST in doc.metadata.allowed_roles
        assert Role.ENGINEER not in doc.metadata.allowed_roles

    def test_file_path_preserved(self, tmp_path: Path) -> None:
        f = tmp_path / "test.md"
        f.write_text(_MINIMAL_MD, encoding="utf-8")
        doc = parse_markdown(f)
        assert doc.file_path == f

    def test_raises_on_empty_body(self, tmp_path: Path) -> None:
        no_body = _MINIMAL_MD.split("## Section One")[0]  # strip content
        f = tmp_path / "empty.md"
        f.write_text(no_body, encoding="utf-8")
        with pytest.raises(ValueError, match="empty"):
            parse_markdown(f)


# ---------------------------------------------------------------------------
# parse_json_incident — full integration
# ---------------------------------------------------------------------------

def _make_incident(tmp_path: Path, **overrides: object) -> Path:
    base: dict[str, object] = {
        "document_id": "INC-TEST-001",
        "title": "Test Payment Failure",
        "department": "Payments Engineering",
        "document_type": "incident_report",
        "access_level": "INTERNAL",
        "created_date": "2024-03-01",
        "allowed_roles": ["ENGINEER", "ANALYST", "ADMINISTRATOR"],
        "severity": "P1",
        "status": "Resolved",
        "duration_minutes": 60,
        "affected_systems": ["payment-gateway", "fps-connector"],
        "content": {
            "summary": "A payment gateway failure caused 10,000 failed transactions.",
            "timeline": [
                {"time": "10:00 UTC", "event": "Alert fires"},
                {"time": "10:05 UTC", "event": "On-call paged"},
                {"time": "10:30 UTC", "event": "Resolved"},
            ],
            "root_cause": "Certificate expiry on FPS connector.",
            "contributing_factors": [
                "No retry in cert-renewal job",
                "No expiry alert configured",
            ],
            "impact": {
                "failed_transactions": 10000,
                "revenue_loss_gbp": 150000,
                "regulatory_notification_required": False,
            },
            "remediation": {
                "immediate": ["Deploy new certificate"],
                "short_term": ["Add cert-renewal retry", "Add expiry alert"],
                "long_term": ["Migrate to ACME automatic renewal"],
            },
            "related_incidents": ["INC-2024-001"],
            "related_documents": ["ARCH-003", "POL-005"],
        },
    }
    base.update(overrides)
    path = tmp_path / "INC-TEST-001.json"
    path.write_text(json.dumps(base), encoding="utf-8")
    return path


class TestParseJsonIncident:
    def test_returns_source_document(self, tmp_path: Path) -> None:
        from src.retrieval.ingestion.models import SourceDocument

        doc = parse_json_incident(_make_incident(tmp_path))
        assert isinstance(doc, SourceDocument)

    def test_metadata_document_id(self, tmp_path: Path) -> None:
        doc = parse_json_incident(_make_incident(tmp_path))
        assert doc.metadata.document_id == "INC-TEST-001"

    def test_metadata_title(self, tmp_path: Path) -> None:
        doc = parse_json_incident(_make_incident(tmp_path))
        assert doc.metadata.title == "Test Payment Failure"

    def test_metadata_access_level(self, tmp_path: Path) -> None:
        doc = parse_json_incident(_make_incident(tmp_path))
        assert doc.metadata.access_level is AccessLevel.INTERNAL

    def test_metadata_created_date(self, tmp_path: Path) -> None:
        doc = parse_json_incident(_make_incident(tmp_path))
        assert doc.metadata.created_date == date(2024, 3, 1)

    def test_metadata_allowed_roles(self, tmp_path: Path) -> None:
        doc = parse_json_incident(_make_incident(tmp_path))
        roles = doc.metadata.allowed_roles
        assert Role.ENGINEER in roles
        assert Role.ANALYST in roles

    def test_body_contains_summary(self, tmp_path: Path) -> None:
        doc = parse_json_incident(_make_incident(tmp_path))
        assert "payment gateway failure" in doc.body

    def test_body_contains_root_cause(self, tmp_path: Path) -> None:
        doc = parse_json_incident(_make_incident(tmp_path))
        assert "Certificate expiry" in doc.body

    def test_body_contains_timeline_events(self, tmp_path: Path) -> None:
        doc = parse_json_incident(_make_incident(tmp_path))
        assert "10:00 UTC" in doc.body
        assert "Alert fires" in doc.body

    def test_body_contains_contributing_factors(self, tmp_path: Path) -> None:
        doc = parse_json_incident(_make_incident(tmp_path))
        assert "No retry in cert-renewal job" in doc.body

    def test_body_contains_remediation(self, tmp_path: Path) -> None:
        doc = parse_json_incident(_make_incident(tmp_path))
        assert "Deploy new certificate" in doc.body

    def test_body_contains_related_documents(self, tmp_path: Path) -> None:
        doc = parse_json_incident(_make_incident(tmp_path))
        assert "ARCH-003" in doc.body

    def test_restricted_incident_parses(self, tmp_path: Path) -> None:
        path = _make_incident(
            tmp_path,
            access_level="RESTRICTED",
            allowed_roles=["ANALYST", "ADMINISTRATOR"],
        )
        doc = parse_json_incident(path)
        assert doc.metadata.access_level is AccessLevel.RESTRICTED
        assert Role.ANALYST in doc.metadata.allowed_roles
        assert Role.ENGINEER not in doc.metadata.allowed_roles

    def test_raises_on_missing_content(self, tmp_path: Path) -> None:
        path = tmp_path / "bad.json"
        path.write_text(
            json.dumps({
                "document_id": "INC-BAD-001",
                "title": "Bad",
                "department": "Eng",
                "document_type": "incident_report",
                "access_level": "INTERNAL",
                "created_date": "2024-01-01",
                "allowed_roles": ["ANALYST"],
            }),
            encoding="utf-8",
        )
        with pytest.raises((ValueError, KeyError)):
            parse_json_incident(path)
