"""Guardrails: output validation with retry and LangSmith tracing.

Public surface
--------------
- ``CitationGuardrail``    — Validates every citation against retrieved evidence.
- ``SAFE_FAILURE_RESPONSE`` — Canned message returned when all validation attempts fail.
"""

from src.agents.guardrails.citation_validation import SAFE_FAILURE_RESPONSE, CitationGuardrail

__all__ = [
    "CitationGuardrail",
    "SAFE_FAILURE_RESPONSE",
]
