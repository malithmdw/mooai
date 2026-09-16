"""Guardrails: input, output, and citation validation.

Public surface
--------------
Input / document guards
- ``InputGuard``              — Pattern-based validator for user message text.
- ``DocumentInjectionGuard``  — Scans retrieved document text for injected instructions.
- ``InputValidationResult``   — Result of ``InputGuard.validate()``.
- ``DocumentScanResult``      — Per-document result of ``DocumentInjectionGuard.scan()``.
- ``ThreatType``              — StrEnum classifying detected injection categories.
- ``MAX_MESSAGE_LENGTH``      — Character limit enforced by ``InputGuard``.
- ``INJECTION_BLOCKED_RESPONSE`` — Generic refusal returned when input is blocked.

Output guard
- ``OutputGuard``             — Scans LLM-generated answers for leakage / jailbreak indicators.
- ``OutputValidationResult``  — Result of ``OutputGuard.validate()``.

Citation guardrail
- ``CitationGuardrail``       — Validates every citation against retrieved evidence.
- ``SAFE_FAILURE_RESPONSE``   — Canned message returned when all validation attempts fail.
"""

from src.agents.guardrails.citation_validation import SAFE_FAILURE_RESPONSE, CitationGuardrail
from src.agents.guardrails.input_guard import (
    INJECTION_BLOCKED_RESPONSE,
    MAX_MESSAGE_LENGTH,
    DocumentInjectionGuard,
    DocumentScanResult,
    InputGuard,
    InputValidationResult,
    ThreatType,
)
from src.agents.guardrails.output_guard import OutputGuard, OutputValidationResult

__all__ = [
    # Input / document guards
    "InputGuard",
    "DocumentInjectionGuard",
    "InputValidationResult",
    "DocumentScanResult",
    "ThreatType",
    "MAX_MESSAGE_LENGTH",
    "INJECTION_BLOCKED_RESPONSE",
    # Output guard
    "OutputGuard",
    "OutputValidationResult",
    # Citation guardrail
    "CitationGuardrail",
    "SAFE_FAILURE_RESPONSE",
]
