"""JSON incident-report parser.

Incident reports use a flat JSON structure where metadata fields sit at the
top level and operational content lives under a ``"content"`` object:

    {
      "document_id": "INC-2024-001",
      "title": "...",
      ...                          ← DocumentMetadata fields
      "severity": "P1",
      "affected_systems": [...],
      "content": {
        "summary": "...",
        "timeline": [...],
        "root_cause": "...",
        "contributing_factors": [...],
        "impact": {...},
        "remediation": {...},
        "related_incidents": [...],
        "related_documents": [...]
      }
    }

The parser converts the ``content`` object and selected top-level fields to
a Markdown-like body text so that the single Markdown chunker can be applied
to all document types uniformly.  The rendering is deterministic: same JSON
input always produces the same body text.
"""

from __future__ import annotations

import json
from pathlib import Path

from src.models.documents import DocumentMetadata
from src.retrieval.ingestion.models import SourceDocument

_METADATA_KEYS = frozenset(
    {"document_id", "title", "department", "document_type", "access_level",
     "created_date", "allowed_roles"}
)


def _str(value: object) -> str:
    return str(value)


def _render_body(raw: dict[str, object]) -> str:
    """Render an incident JSON dict to a Markdown-style body text.

    Sections rendered (in order):
    1. Incident Overview  — severity, status, duration, affected systems
    2. Summary
    3. Timeline
    4. Root Cause
    5. Contributing Factors
    6. Impact
    7. Remediation (Immediate / Short-term / Long-term)
    8. Related Documents
    """
    lines: list[str] = []

    # --- Incident Overview ---------------------------------------------------
    overview: list[str] = []
    for field, label in (
        ("severity", "Severity"),
        ("status", "Status"),
        ("duration_minutes", "Duration (minutes)"),
    ):
        if (val := raw.get(field)) is not None:
            overview.append(f"- **{label}**: {val}")
    if affected := raw.get("affected_systems"):
        if isinstance(affected, list):
            overview.append(f"- **Affected Systems**: {', '.join(_str(s) for s in affected)}")
    if overview:
        lines.append("## Incident Overview")
        lines.extend(overview)
        lines.append("")

    # --- Content sections ----------------------------------------------------
    content_raw = raw.get("content")
    if not isinstance(content_raw, dict):
        raise ValueError("Missing or invalid 'content' object in incident JSON")
    content: dict[str, object] = content_raw

    # Summary
    if summary := content.get("summary"):
        lines.append("## Summary")
        lines.append(_str(summary))
        lines.append("")

    # Timeline
    timeline_raw = content.get("timeline")
    if isinstance(timeline_raw, list) and timeline_raw:
        lines.append("## Timeline")
        for event in timeline_raw:
            if isinstance(event, dict):
                time = _str(event.get("time", ""))
                evt = _str(event.get("event", ""))
                lines.append(f"- **{time}**: {evt}")
        lines.append("")

    # Root Cause
    if root_cause := content.get("root_cause"):
        lines.append("## Root Cause")
        lines.append(_str(root_cause))
        lines.append("")

    # Contributing Factors
    factors_raw = content.get("contributing_factors")
    if isinstance(factors_raw, list) and factors_raw:
        lines.append("## Contributing Factors")
        for factor in factors_raw:
            lines.append(f"- {_str(factor)}")
        lines.append("")

    # Impact
    impact_raw = content.get("impact")
    if isinstance(impact_raw, dict) and impact_raw:
        lines.append("## Impact")
        for key, val in impact_raw.items():
            label = key.replace("_", " ").title()
            lines.append(f"- **{label}**: {val}")
        lines.append("")

    # Remediation
    remediation_raw = content.get("remediation")
    if isinstance(remediation_raw, dict) and remediation_raw:
        lines.append("## Remediation")
        for phase, heading in (
            ("immediate", "Immediate"),
            ("short_term", "Short-term"),
            ("long_term", "Long-term"),
        ):
            steps_raw = remediation_raw.get(phase)
            if isinstance(steps_raw, list) and steps_raw:
                lines.append(f"### {heading}")
                for step in steps_raw:
                    lines.append(f"- {_str(step)}")
                lines.append("")

    # Related documents (searchable cross-references)
    related_raw = content.get("related_documents")
    if isinstance(related_raw, list) and related_raw:
        lines.append("## Related Documents")
        lines.append(", ".join(_str(r) for r in related_raw))
        lines.append("")

    return "\n".join(lines).strip()


def parse_json_incident(path: Path) -> SourceDocument:
    """Parse a JSON incident report and return a `SourceDocument`.

    The ``content`` object is rendered to Markdown-style text so that the
    shared chunker can process all document types uniformly.
    """
    raw: dict[str, object] = json.loads(path.read_text(encoding="utf-8"))

    metadata_fields = {k: raw[k] for k in _METADATA_KEYS if k in raw}
    metadata = DocumentMetadata.model_validate(metadata_fields)

    body = _render_body(raw)
    if not body:
        raise ValueError(f"Rendered body is empty for incident: {path}")

    return SourceDocument(file_path=path, metadata=metadata, body=body)
