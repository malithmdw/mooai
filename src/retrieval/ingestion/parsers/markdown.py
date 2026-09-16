"""Markdown document parser.

Handles the YAML-style frontmatter used by all Markdown knowledge documents:

    ---
    document_id: ARCH-001
    title: Core Banking Ledger — Architecture Overview
    department: Core Banking
    document_type: architecture_document
    access_level: INTERNAL
    created_date: "2023-06-01"
    allowed_roles: ["ENGINEER", "ANALYST", "ADMINISTRATOR"]
    ---

    # Document body starts here...

Frontmatter format rules (deterministic subset of YAML our generator uses):
- Each metadata field is on its own line: ``key: value``
- Array values are JSON arrays on a single line: ``["a", "b"]``
- String values may be bare or double-quoted
- The body is everything after the closing ``---``
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from src.models.documents import DocumentMetadata
from src.retrieval.ingestion.models import SourceDocument

_FRONTMATTER_RE = re.compile(r"\A---\r?\n(.*?)\r?\n---\r?\n", re.DOTALL)


def _parse_frontmatter(text: str) -> tuple[dict[str, object], str]:
    """Extract and parse YAML-style frontmatter from *text*.

    Returns ``(fields, body)`` where *body* is the text after the closing
    ``---`` delimiter.  Raises ``ValueError`` if frontmatter is absent or
    malformed.

    Supported value forms:
    - ``key: bare value``
    - ``key: "quoted value"``
    - ``key: ["json", "array"]``
    """
    match = _FRONTMATTER_RE.match(text)
    if not match:
        raise ValueError("Missing or malformed YAML frontmatter (expected --- … ---)")

    fields: dict[str, object] = {}
    for line in match.group(1).splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if ": " not in line:
            continue
        key, _, raw = line.partition(": ")
        key = key.strip()
        raw = raw.strip()

        if raw.startswith("["):
            fields[key] = json.loads(raw)
        elif raw.startswith('"') and raw.endswith('"') and len(raw) >= 2:
            fields[key] = raw[1:-1]
        else:
            fields[key] = raw

    body = text[match.end() :]
    return fields, body


def parse_markdown(path: Path) -> SourceDocument:
    """Parse a Markdown knowledge document and return a `SourceDocument`.

    The file must have a valid frontmatter block containing all fields
    required by `DocumentMetadata`.  The body (text after frontmatter)
    is returned as-is — chunking is a separate step.
    """
    text = path.read_text(encoding="utf-8")
    fields, body = _parse_frontmatter(text)

    metadata = DocumentMetadata.model_validate(fields)
    body = body.strip()
    if not body:
        raise ValueError(f"Document body is empty after stripping frontmatter: {path}")

    return SourceDocument(file_path=path, metadata=metadata, body=body)
