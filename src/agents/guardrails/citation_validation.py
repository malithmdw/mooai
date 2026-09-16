"""Citation validation guardrail with LangSmith tracing.

Validates every citation in a ``ResponseDecision`` against the evidence
actually present in the execution state:

- ``chunk_id`` must exist in ``retrieved_documents``
- ``document_id`` must match the corresponding ``RetrievalEvidence``
- reference numbers must be unique
- each cited chunk's ``access_level`` must be within the requesting user's
  permitted levels (defence-in-depth; the retrieval agent is the primary
  RBAC enforcement point)

This class is stateless and performs no LLM calls.  The retry loop lives in
``ResponseAgent``, which calls ``validate()`` for each attempt.  Two
consecutive failures trigger a safe failure response.

Security notes
--------------
- Authorization check guards against the response agent being manipulated
  into citing a document the user cannot access.
- ``_ROLE_ACCESS_LEVELS`` mirrors the retrieval node's mapping exactly so
  the two layers are consistent.
- When ``user`` is ``None`` (should not happen in production after retrieval
  filtering), the authorization check is skipped rather than crashing.
"""

from __future__ import annotations

from src.agents.response.models import CitedChunk, ResponseDecision
from src.core.logging import get_logger, log_debug, log_warning
from src.models.enums import AccessLevel, Role
from src.models.user import User
from src.models.validation import ValidationResult
from src.observability.tracing import trace_validation
from src.retrieval.hybrid.models import RetrievalEvidence

_logger = get_logger(__name__)

SAFE_FAILURE_RESPONSE = (
    "I was unable to produce a verified, citation-backed answer for your question. "
    "Please try rephrasing, or contact the relevant department directly."
)

_ROLE_ACCESS_LEVELS: dict[Role, frozenset[AccessLevel]] = {
    Role.VIEWER: frozenset({AccessLevel.PUBLIC, AccessLevel.INTERNAL}),
    Role.ENGINEER: frozenset({AccessLevel.PUBLIC, AccessLevel.INTERNAL}),
    Role.ANALYST: frozenset({AccessLevel.PUBLIC, AccessLevel.INTERNAL, AccessLevel.CONFIDENTIAL}),
    Role.ADMINISTRATOR: frozenset({
        AccessLevel.PUBLIC,
        AccessLevel.INTERNAL,
        AccessLevel.CONFIDENTIAL,
        AccessLevel.RESTRICTED,
    }),
}


def _permitted_access_levels(user: User) -> frozenset[AccessLevel]:
    """Union of all access levels allowed by the user's roles."""
    permitted: set[AccessLevel] = set()
    for role in user.roles:
        permitted |= _ROLE_ACCESS_LEVELS.get(role, frozenset())
    return frozenset(permitted)


def _validate_chunk_citations(
    cited: list[CitedChunk],
    retrieved: list[RetrievalEvidence],
) -> tuple[list[CitedChunk], list[str]]:
    """Cross-validate citations against retrieved chunks.

    Returns the list of structurally valid citations and any error strings.
    A citation is valid when its ``chunk_id`` exists in ``retrieved``, its
    ``document_id`` is consistent, and its ``reference_number`` is unique.
    """
    index: dict[str, RetrievalEvidence] = {ev.chunk_id: ev for ev in retrieved}
    valid: list[CitedChunk] = []
    errors: list[str] = []
    seen_refs: set[int] = set()

    for c in cited:
        ev = index.get(c.chunk_id)
        if ev is None:
            errors.append(
                f"hallucinated citation: chunk_id '{c.chunk_id}' was not retrieved"
            )
            continue
        if ev.document_id != c.document_id:
            errors.append(
                f"mismatched document_id for chunk '{c.chunk_id}': "
                f"expected '{ev.document_id}', got '{c.document_id}'"
            )
            continue
        if c.reference_number in seen_refs:
            errors.append(
                f"duplicate reference_number {c.reference_number} in cited_chunks"
            )
            continue
        seen_refs.add(c.reference_number)
        valid.append(c)

    return valid, errors


def _validate_access_levels(
    cited: list[CitedChunk],
    retrieved: list[RetrievalEvidence],
    user: User,
) -> list[str]:
    """Verify that each cited chunk's access level is within the user's permitted levels."""
    index: dict[str, RetrievalEvidence] = {ev.chunk_id: ev for ev in retrieved}
    permitted = _permitted_access_levels(user)
    errors: list[str] = []

    for c in cited:
        ev = index.get(c.chunk_id)
        if ev is None:
            continue  # already caught by _validate_chunk_citations
        if ev.metadata.access_level not in permitted:
            errors.append(
                f"citation to unauthorized chunk '{c.chunk_id}': "
                f"access_level '{ev.metadata.access_level}' exceeds user permissions"
            )

    return errors


class CitationGuardrail:
    """Stateless citation validator with LangSmith tracing.

    Call ``validate()`` for each attempt; the caller decides whether to retry
    or accept the result.
    """

    def validate(
        self,
        *,
        decision: ResponseDecision,
        retrieved: list[RetrievalEvidence],
        user: User | None,
        conversation_id: str | None,
    ) -> tuple[list[CitedChunk], ValidationResult]:
        """Validate all citations and return ``(valid_citations, result)``.

        Wraps both checks in a LangSmith ``validation:citation_guardrail``
        trace.  When ``result.is_valid`` is ``True``, ``valid_citations``
        contains every citation from the decision (all passed).  When
        ``False``, the caller should retry or return ``SAFE_FAILURE_RESPONSE``;
        the returned ``valid_citations`` should not be used.
        """
        with trace_validation("citation_guardrail", conversation_id=conversation_id):
            valid_citations, chunk_errors = _validate_chunk_citations(
                decision.cited_chunks, retrieved
            )
            auth_errors: list[str] = (
                _validate_access_levels(decision.cited_chunks, retrieved, user)
                if user is not None
                else []
            )

        all_errors = chunk_errors + auth_errors
        if all_errors:
            result = ValidationResult(is_valid=False, errors=tuple(all_errors))
            log_warning(
                _logger,
                "guardrail.validation_failed",
                "Citation guardrail failed",
                error_count=len(all_errors),
                conversation_id=conversation_id,
            )
        else:
            result = ValidationResult(is_valid=True, errors=())
            log_debug(
                _logger,
                "guardrail.validation_passed",
                "Citation guardrail passed",
                citation_count=len(valid_citations),
                conversation_id=conversation_id,
            )

        return valid_citations, result
