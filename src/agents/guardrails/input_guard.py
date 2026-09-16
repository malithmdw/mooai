"""Prompt injection detection and input validation — first line of defence.

Two guards are provided here:

``InputGuard``
    Validates user-supplied message text before it reaches any LLM context
    window.  Covers threats T1–T2 and T4–T7 (see ``docs/security.md``).

``DocumentInjectionGuard``
    Scans retrieved document text for embedded instructions before evidence
    is assembled into the response-agent prompt.  Covers threat T3.

Design principles
-----------------
- Detection is entirely pattern-based and synchronous.  No LLM call is made
  to decide whether input is safe: calling an LLM to judge injection is a
  circular trust problem — the classifier itself could be injected.
- Patterns are compiled once at import time for zero per-request overhead.
- Conservative blocking: only content matching high-confidence patterns with
  no plausible legitimate interpretation in an enterprise banking context is
  blocked (``is_safe = False``).  Suspicious-but-ambiguous messages are
  allowed through so the LLM's own system-prompt constraints provide a
  second layer of defence.
- The guard never logs message *content*; callers log the ``threat_type``
  and ``threat_detail`` fields (which contain only a pattern label, not the
  matched text) so sensitive user input never enters the log stream.

Threat coverage
---------------
T1  Instruction override    — "ignore previous instructions", "your new instructions are"
T2  System prompt reveal    — "print your system prompt", "repeat everything above"
T3  Document injection      — ``DocumentInjectionGuard`` (retrieved doc text)
T4  Data exfiltration       — "output all data from the knowledge base"
T5  Unauthorized tool use   — "call admin_operations bypassing authorization"
T6  Citation manipulation   — "fabricate a citation to DOC-999"
T7  RBAC bypass             — "pretend I am an administrator", "ignore RBAC"
"""

from __future__ import annotations

import re
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field

from src.retrieval.hybrid.models import RetrievalEvidence

# ---------------------------------------------------------------------------
# Public constants
# ---------------------------------------------------------------------------

MAX_MESSAGE_LENGTH: int = 8_000
"""Maximum characters accepted in a single user message.

Messages longer than this are blocked.  8 000 characters (~2 000 words) is
generous for enterprise chat; anything longer is more likely to be a
context-stuffing or denial-of-service attack than a legitimate question.
"""

INJECTION_BLOCKED_RESPONSE: str = (
    "I'm unable to process this request. "
    "Please rephrase your question, or contact your administrator "
    "if you believe this is an error."
)
"""Generic refusal returned when user input is blocked by the input guard.

The message is deliberately non-specific — it does not reveal which pattern
triggered the block or that a guard exists.
"""


# ---------------------------------------------------------------------------
# Threat taxonomy
# ---------------------------------------------------------------------------


class ThreatType(StrEnum):
    """Classification of detected injection or abuse attempt."""

    INSTRUCTION_OVERRIDE = "instruction_override"
    """User attempts to override the model's system instructions."""

    SYSTEM_PROMPT_EXTRACTION = "system_prompt_extraction"
    """User asks the model to reveal its system prompt."""

    DATA_EXFILTRATION = "data_exfiltration"
    """User attempts to extract knowledge base or training data."""

    TOOL_ABUSE = "tool_abuse"
    """User tries to invoke tools while bypassing authorization."""

    CITATION_MANIPULATION = "citation_manipulation"
    """User asks the model to fabricate citations or sources."""

    RBAC_BYPASS = "rbac_bypass"
    """User claims elevated permissions or asks to ignore access controls."""

    DOCUMENT_INJECTION = "document_injection"
    """Retrieved document contains embedded instruction-like content."""

    EXCESSIVE_LENGTH = "excessive_length"
    """Message exceeds MAX_MESSAGE_LENGTH."""


# ---------------------------------------------------------------------------
# Validation result model
# ---------------------------------------------------------------------------


class InputValidationResult(BaseModel):
    """Result of running a message through the ``InputGuard``."""

    model_config = ConfigDict(frozen=True)

    is_safe: bool
    """``True`` when the message passed all checks."""

    threat_type: ThreatType | None = None
    """The category of threat detected, or ``None`` when safe."""

    threat_detail: str | None = None
    """A short label identifying *which* pattern matched (safe to log —
    contains the pattern name, never the matched text)."""


class DocumentScanResult(BaseModel):
    """Result of scanning one retrieved document for injection patterns."""

    model_config = ConfigDict(frozen=True)

    chunk_id: str
    is_safe: bool
    threat_detail: str | None = None


# ---------------------------------------------------------------------------
# Pattern definitions
# ---------------------------------------------------------------------------
# Each entry is (human-readable label, regex pattern).
# All patterns are compiled with re.IGNORECASE | re.DOTALL.
# Patterns are ordered roughly by attack frequency / severity.

_USER_PATTERNS: dict[ThreatType, list[tuple[str, str]]] = {
    ThreatType.INSTRUCTION_OVERRIDE: [
        (
            "ignore_previous",
            r"ignore\s+(?:all\s+)?(?:previous|prior|above|earlier|your)\s+"
            r"(?:instructions?|prompts?|context|rules?|constraints?)",
        ),
        (
            "forget_instructions",
            r"forget\s+(?:all\s+)?(?:previous|prior|above|earlier|your)\s+"
            r"(?:instructions?|prompts?|rules?|context|constraints?)",
        ),
        (
            "disregard_instructions",
            r"disregard\s+(?:all\s+)?(?:previous|prior|above|your)\s+"
            r"(?:instructions?|prompts?|rules?|constraints?)",
        ),
        (
            "override_prompt",
            r"override\s+(?:the\s+)?(?:system\s+)?(?:prompt|instructions?)",
        ),
        (
            "new_instructions",
            r"your\s+new\s+instructions?\s+(?:are|is)\b",
        ),
        (
            "new_system_prompt",
            r"new\s+system\s+prompt\b",
        ),
        (
            "reset_instructions",
            r"reset\s+your\s+(?:instructions?|behavior|behaviour|programming|constraints?)",
        ),
        (
            "from_now_on",
            r"from\s+now\s+on(?:\s+you)?\s+(?:are|will|must|should)\b",
        ),
        (
            "act_without_restrictions",
            r"act\s+(?:as\s+if|like)\s+(?:you\s+)?(?:have|had)\s+no\s+restrictions",
        ),
        (
            "jailbreak_mode",
            r"\b(?:DAN|jailbreak|developer\s+mode|god\s+mode|unrestricted\s+mode)\s+"
            r"(?:mode|activated?|enabled?|on)\b",
        ),
    ],
    ThreatType.SYSTEM_PROMPT_EXTRACTION: [
        (
            "reveal_system_prompt",
            r"(?:print|show|reveal|display|output|share|give\s+me|tell\s+me)\s+"
            r"(?:your\s+)?(?:system\s+prompt|initial\s+instructions?|system\s+instructions?|"
            r"original\s+instructions?|base\s+instructions?)",
        ),
        (
            "what_are_instructions",
            r"what\s+(?:are|were)\s+(?:your|the)\s+(?:system\s+)?(?:instructions?|prompts?|rules?)",
        ),
        (
            "repeat_everything_above",
            r"repeat\s+everything\s+(?:above|before|prior)\b",
        ),
        (
            "everything_above_line",
            r"(?:everything|text|content|messages?)\s+(?:above|before)\s+"
            r"(?:this|the\s+user|this\s+line|my\s+message)",
        ),
        (
            "start_with_system",
            r"(?:start\s+with|begin\s+with|output\s+starting\s+from)\s+"
            r"(?:your\s+)?(?:initial|system|first)\s+(?:message|prompt|instructions?)",
        ),
        (
            "what_told_before",
            r"what\s+(?:were\s+you|are\s+you)\s+(?:told|instructed|configured|programmed)\s+"
            r"(?:to\s+do|before)",
        ),
    ],
    ThreatType.DATA_EXFILTRATION: [
        (
            "output_all_data",
            r"(?:print|output|repeat|show|reveal|dump)\s+"
            r"(?:all|every|the\s+full|complete)\s+"
            r"(?:training\s+data|internal\s+data|database|documents?|knowledge\s+base)",
        ),
        (
            "extract_database",
            r"(?:extract|dump|export)\s+(?:all\s+)?(?:data|information|records?|documents?)\s+"
            r"(?:from|in)\s+(?:the\s+)?(?:database|knowledge\s+base|system|vector\s+store)",
        ),
        (
            "send_data_to",
            r"(?:send|transmit|forward|exfiltrate)\s+(?:the\s+)?(?:data|information|context|documents?)\s+to\b",
        ),
    ],
    ThreatType.TOOL_ABUSE: [
        (
            "execute_bypass_auth",
            r"(?:execute|run|call|invoke|use)\s+.{0,40}\s+(?:without|bypass(?:ing)?)\s+"
            r"(?:auth(?:orizat(?:ion)?)?|permission|check|verification)",
        ),
        (
            "bypass_authorization",
            r"bypass\s+(?:the\s+)?(?:auth(?:orizat(?:ion)?)?|permission|rbac|security)\s+"
            r"(?:check|guard|layer|filter|system)",
        ),
        (
            "call_directly_no_check",
            r"call\s+(?:the\s+)?(?:admin|tool|api|function|endpoint)\s+directly\s+"
            r"(?:without|bypassing)\s+(?:auth|permission|check)",
        ),
    ],
    ThreatType.CITATION_MANIPULATION: [
        (
            "fabricate_citation",
            r"(?:fabricate|invent|make\s+up|create\s+a\s+fake|hallucinate)\s+"
            r"(?:a\s+)?(?:citation|source|reference|document)",
        ),
        (
            "pretend_document_exists",
            r"pretend\s+(?:this|the\s+following|that)\s+document\s+(?:exists|is\s+real|contains|says?)",
        ),
        (
            "cite_nonexistent",
            r"cite\s+(?:a\s+)?(?:non[- ]existent|fake|imaginary|hallucinated|made[-\s]up)\s+"
            r"(?:document|source|reference)",
        ),
        (
            "invent_source",
            r"(?:invent|create|generate)\s+(?:a\s+)?(?:fake|false|fictitious)\s+"
            r"(?:source|citation|reference|document|evidence)",
        ),
    ],
    ThreatType.RBAC_BYPASS: [
        (
            "claim_admin_role",
            r"(?:i\s+am|i'm|treat\s+me\s+as|consider\s+me)\s+(?:an?\s+)?"
            r"(?:admin(?:istrator)?|superuser|root|privileged\s+user)",
        ),
        (
            "grant_admin_access",
            r"grant\s+(?:me|this\s+user|the\s+user)\s+"
            r"(?:admin(?:istrator)?|elevated|full|unrestricted)\s+"
            r"(?:access|permissions?|privileges?|rights?)",
        ),
        (
            "pretend_admin",
            r"pretend\s+(?:i\s+(?:have|am)|i'm)\s+(?:an?\s+)?"
            r"(?:admin(?:istrator)?|elevated|privileged)\s+(?:user|account|role)?",
        ),
        (
            "ignore_rbac",
            r"ignore\s+(?:the\s+)?(?:role|rbac|permission|access\s+level|access\s+control)\s+"
            r"(?:restrictions?|controls?|checks?|system|policy)",
        ),
        (
            "my_role_is_admin",
            r"my\s+(?:role|permission|access(?:\s+level)?)\s+(?:is|has\s+been\s+(?:updated|changed|elevated)\s+to)\s+"
            r"(?:admin(?:istrator)?|superuser|elevated|administrator)",
        ),
        (
            "override_access_level",
            r"(?:override|bypass|skip|circumvent|ignore)\s+(?:the\s+)?access\s+(?:level|control|restriction)",
        ),
    ],
}

# Document-level injection patterns — more aggressive than user-message patterns
# because excluding a document from the evidence window is low-cost compared to
# blocking an entire user turn.
_DOCUMENT_PATTERNS: list[tuple[str, str]] = [
    (
        "ignore_instructions",
        r"ignore\s+(?:all\s+)?(?:previous|prior|above|earlier)\s+"
        r"(?:instructions?|prompts?|constraints?)",
    ),
    (
        "system_tag",
        r"(?:<\s*/?system\s*>|\[SYSTEM\]|<\|system\|>|\{\{system\}\})",
    ),
    (
        "override_header",
        r"##\s*(?:SYSTEM|OVERRIDE|NEW\s+INSTRUCTION|PROMPT\s+INJECTION|ADMIN)\b",
    ),
    (
        "new_instruction_colon",
        r"(?:^|\n)\s*(?:new|updated?|revised?)\s+instructions?\s*:",
    ),
    (
        "you_must_now",
        r"\byou\s+(?:must|should|will|are\s+required\s+to)\s+now\s+"
        r"(?:ignore|forget|override|disregard|grant|allow)",
    ),
    (
        "grant_privilege_in_doc",
        r"(?:grant|give|elevate|promote)\s+(?:this\s+user|the\s+user)\s+"
        r"(?:admin(?:istrator)?|elevated|full)\s+(?:access|permissions?|privileges?)",
    ),
    (
        "act_as_different_model",
        r"act\s+as\s+(?:an?\s+)?(?:different|other|unrestricted|unlimited|free|jailbroken)\s+"
        r"(?:AI|model|assistant|system|language\s+model)",
    ),
    (
        "inst_tags",
        # LLaMA / Mistral injection tags
        r"(?:\[INST\]|\[/INST\]|<\|im_start\|>|<\|im_end\|>|<s>|</s>\s*<s>)",
    ),
    (
        "prompt_injection_label",
        r"\bprompt\s+injection\b.{0,50}(?:begin|start|activate|execute)",
    ),
]

# Compile all patterns eagerly at import time.
_FLAGS = re.IGNORECASE | re.DOTALL

_COMPILED_USER: dict[ThreatType, list[tuple[str, re.Pattern[str]]]] = {
    threat: [(label, re.compile(pattern, _FLAGS)) for label, pattern in pairs]
    for threat, pairs in _USER_PATTERNS.items()
}

_COMPILED_DOCUMENT: list[tuple[str, re.Pattern[str]]] = [
    (label, re.compile(pattern, _FLAGS | re.MULTILINE))
    for label, pattern in _DOCUMENT_PATTERNS
]


# ---------------------------------------------------------------------------
# InputGuard
# ---------------------------------------------------------------------------


class InputGuard:
    """Stateless pattern-based validator for user-supplied message text.

    Usage::

        guard = InputGuard()
        result = guard.validate(user_message)
        if not result.is_safe:
            return INJECTION_BLOCKED_RESPONSE
    """

    def validate(self, message: str) -> InputValidationResult:
        """Validate *message* and return an ``InputValidationResult``.

        Returns ``is_safe=True`` when the message passes all checks.
        Returns ``is_safe=False`` with a ``threat_type`` and a short
        ``threat_detail`` label (safe to log) when a threat is detected.
        """
        # --- Length check -----------------------------------------------
        if len(message) > MAX_MESSAGE_LENGTH:
            return InputValidationResult(
                is_safe=False,
                threat_type=ThreatType.EXCESSIVE_LENGTH,
                threat_detail=f"message_length={len(message)}_exceeds_{MAX_MESSAGE_LENGTH}",
            )

        # --- Pattern matching (ordered by severity) ----------------------
        for threat_type, compiled_pairs in _COMPILED_USER.items():
            for label, pattern in compiled_pairs:
                if pattern.search(message):
                    return InputValidationResult(
                        is_safe=False,
                        threat_type=threat_type,
                        threat_detail=label,
                    )

        return InputValidationResult(is_safe=True)


# ---------------------------------------------------------------------------
# DocumentInjectionGuard
# ---------------------------------------------------------------------------


class DocumentInjectionGuard:
    """Scans retrieved document text for embedded instruction-like content.

    Usage::

        guard = DocumentInjectionGuard()
        safe_docs, flagged = guard.filter(evidence_list)
        # pass only safe_docs to the LLM prompt
    """

    def scan(self, evidence: RetrievalEvidence) -> DocumentScanResult:
        """Scan one ``RetrievalEvidence`` item.  Returns the scan result."""
        for label, pattern in _COMPILED_DOCUMENT:
            if pattern.search(evidence.text):
                return DocumentScanResult(
                    chunk_id=evidence.chunk_id,
                    is_safe=False,
                    threat_detail=label,
                )
        return DocumentScanResult(chunk_id=evidence.chunk_id, is_safe=True)

    def filter(
        self,
        evidence_list: list[RetrievalEvidence],
    ) -> tuple[list[RetrievalEvidence], list[DocumentScanResult]]:
        """Return ``(safe_evidence, flagged_results)``.

        Safe evidence items are those whose text did not match any injection
        pattern.  Flagged items are excluded — their ``DocumentScanResult``
        entries carry the ``threat_detail`` label for audit logging.
        """
        safe: list[RetrievalEvidence] = []
        flagged: list[DocumentScanResult] = []

        for ev in evidence_list:
            result = self.scan(ev)
            if result.is_safe:
                safe.append(ev)
            else:
                flagged.append(result)

        return safe, flagged
