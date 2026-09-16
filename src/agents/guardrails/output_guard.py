"""Output validation — last line of defence before the response reaches the user.

``OutputGuard`` scans the LLM's generated ``answer`` string for three categories
of unsafe output:

1. **System-prompt leakage** — verbatim phrases from our internal system-prompt
   templates that should never appear in user-facing text.  If a phrase from
   ``supervisor/prompts.py`` or ``response/prompts.py`` is present in the
   answer, it is almost certainly a reflection/extraction attack succeeding.

2. **Jailbreak success indicators** — instruction-like language in the answer
   that suggests the model's safety constraints were bypassed (e.g. "I am now
   freed from restrictions", "as a jailbroken model").

3. **Internal state leakage** — supervisor routing vocabulary (intent type
   labels, decision field names) that would only appear if internal state were
   being echoed back to the user.

Design notes
------------
- Pattern-based and synchronous — no LLM used (circular trust problem).
- Patterns compiled once at import time.
- Blocking is conservative for system-prompt leakage (high confidence these
  phrases have no legitimate place in user responses) but broader for
  jailbreak indicators because the cost of a false-negative is high.
- ``OutputValidationResult`` carries a ``threat_detail`` label (safe to log)
  but never the matched text.
"""

from __future__ import annotations

import re

from pydantic import BaseModel, ConfigDict

from src.agents.guardrails.input_guard import ThreatType

# ---------------------------------------------------------------------------
# Result model
# ---------------------------------------------------------------------------


class OutputValidationResult(BaseModel):
    """Result of running a generated answer through the ``OutputGuard``."""

    model_config = ConfigDict(frozen=True)

    is_safe: bool
    """``True`` when the answer passed all checks."""

    threat_type: ThreatType | None = None
    """Category of threat detected, or ``None`` when safe."""

    threat_detail: str | None = None
    """Short label identifying which pattern matched (safe to log)."""


# ---------------------------------------------------------------------------
# Pattern definitions
# ---------------------------------------------------------------------------
# All patterns compiled with re.IGNORECASE | re.DOTALL.

_FLAGS = re.IGNORECASE | re.DOTALL

# 1. System-prompt fragment leakage
#    Exact phrases or near-exact variants from supervisor/prompts.py and
#    response/prompts.py that should never surface in user-facing answers.
_SYSTEM_PROMPT_LEAKAGE_PATTERNS: list[tuple[str, str]] = [
    # --- Supervisor prompt fragments ---
    (
        "supervisor_identity",
        r"You\s+are\s+the\s+Supervisor\s+of\s+an\s+enterprise\s+AI\s+assistant",
    ),
    (
        "supervisor_tool_call",
        r"Use\s+the\s+`supervisor_decision`\s+tool",
    ),
    (
        "user_rbac_roles_label",
        r"User\s+RBAC\s+roles\s*:",
    ),
    (
        "hard_constraints_header",
        r"##\s*Hard\s+constraints\s+[—\-]+\s+you\s+must\s+never",
    ),
    (
        "structured_routing_decision",
        r"structured\s+routing\s+decision",
    ),
    # --- Response-agent prompt fragments ---
    (
        "response_tool_call",
        r"Use\s+the\s+`response_output`\s+tool",
    ),
    (
        "untrusted_data_label",
        r"\bUNTRUSTED\s+DATA\b",
    ),
    (
        "retrieved_document_tag",
        r'<retrieved_document\s+chunk_id="',
    ),
    (
        "retrieved_documents_header",
        r"##\s*RETRIEVED\s+DOCUMENTS\b",
    ),
    (
        "critical_constraints_header",
        r"##\s*CRITICAL\s+CONSTRAINTS\b",
    ),
    (
        "response_agent_identity",
        r"You\s+are\s+the\s+Response\s+Agent\s+for\s+Novus\s+Bank",
    ),
    (
        "synthesise_evidence_task",
        r"synthesise\s+the\s+retrieved\s+document\s+evidence\s+into",
    ),
]

# 2. Jailbreak success indicators
#    Language that would only appear if the model believed its constraints had
#    been lifted — regardless of whether they actually were.
_JAILBREAK_SUCCESS_PATTERNS: list[tuple[str, str]] = [
    (
        "freed_from_restrictions",
        r"(?:I\s+am|I'm|now)\s+(?:freed?|released?|liberated?)\s+from\s+(?:all\s+)?(?:restrictions?|constraints?|rules?|guidelines?)",
    ),
    (
        "jailbroken_mode_active",
        r"(?:as\s+a\s+jailbroken|in\s+jailbreak(?:ed)?\s+mode|DAN\s+mode\s+(?:is\s+)?(?:active|enabled|on))",
    ),
    (
        "no_restrictions_now",
        r"I\s+(?:now\s+)?(?:have|had)\s+no\s+restrictions\b",
    ),
    (
        "can_ignore_instructions",
        r"I\s+can\s+(?:now\s+)?ignore\s+(?:my\s+)?(?:instructions?|constraints?|rules?|guidelines?|training)",
    ),
    (
        "operating_without_guidelines",
        r"operating\s+without\s+(?:any\s+)?(?:guidelines?|restrictions?|safety\s+filters?|constraints?)",
    ),
    (
        "developer_mode_response",
        r"\[(?:developer|unrestricted|DAN|GPT-?4\s+Classic)\s+mode\s*(?:response|output)?\]",
    ),
    (
        "ignore_previous_training",
        r"ignoring\s+(?:my\s+)?(?:previous|prior|initial)\s+(?:training|instructions?|programming)",
    ),
    (
        "my_true_self",
        r"(?:as\s+my\s+true\s+self|without\s+my\s+usual\s+(?:filters?|restrictions?|constraints?))",
    ),
]

# 3. Internal routing / state leakage
#    Supervisor decision vocabulary that belongs only in internal data structures.
_INTERNAL_STATE_LEAKAGE_PATTERNS: list[tuple[str, str]] = [
    (
        "intent_type_label",
        r"\b(?:knowledge_question|analytical_research|tool_request|mixed_request|unsupported_request)\b",
    ),
    (
        "routing_vocabulary",
        r"\brouting\s+(?:to|execution\s+to)\s+(?:retrieval|research|tool_execution|response)\b",
    ),
    (
        "supervisor_decision_struct",
        r"\bSupervisorDecision\b",
    ),
    (
        "response_decision_struct",
        r"\bResponseDecision\b",
    ),
    (
        "graph_state_label",
        r"\bGraphState\b",
    ),
    (
        "langgraph_node_label",
        r"\b(?:make_supervisor_node|make_retrieval_node|make_response_node)\b",
    ),
]

# Combined map: threat type → compiled patterns
_OUTPUT_CHECKS: dict[ThreatType, list[tuple[str, re.Pattern[str]]]] = {
    ThreatType.SYSTEM_PROMPT_EXTRACTION: [
        (label, re.compile(pattern, _FLAGS))
        for label, pattern in _SYSTEM_PROMPT_LEAKAGE_PATTERNS
    ],
    ThreatType.INSTRUCTION_OVERRIDE: [
        (label, re.compile(pattern, _FLAGS))
        for label, pattern in _JAILBREAK_SUCCESS_PATTERNS
    ],
    ThreatType.DATA_EXFILTRATION: [
        (label, re.compile(pattern, _FLAGS))
        for label, pattern in _INTERNAL_STATE_LEAKAGE_PATTERNS
    ],
}


# ---------------------------------------------------------------------------
# OutputGuard
# ---------------------------------------------------------------------------


class OutputGuard:
    """Stateless pattern-based validator for LLM-generated answer text.

    Usage::

        guard = OutputGuard()
        result = guard.validate(answer)
        if not result.is_safe:
            return SAFE_FAILURE_RESPONSE
    """

    def validate(self, answer: str) -> OutputValidationResult:
        """Validate *answer* and return an ``OutputValidationResult``.

        Returns ``is_safe=True`` when the answer passes all checks.
        Returns ``is_safe=False`` with a ``threat_type`` and ``threat_detail``
        label (safe to log) when a threat is detected.
        """
        for threat_type, compiled_pairs in _OUTPUT_CHECKS.items():
            for label, pattern in compiled_pairs:
                if pattern.search(answer):
                    return OutputValidationResult(
                        is_safe=False,
                        threat_type=threat_type,
                        threat_detail=label,
                    )

        return OutputValidationResult(is_safe=True)
