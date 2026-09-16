"""Response Agent LangGraph node.

Responsibilities
----------------
1. Short-circuit with a canned refusal when intent is ``unsupported_request``
   or no evidence is available — avoids token waste and LLM hallucination risk.
2. Call the Anthropic API with a forced ``response_output`` tool to obtain a
   structured answer grounded in the retrieved evidence.
3. Validate every citation: any ``chunk_id`` not present in
   ``GraphState.retrieved_documents`` is a hallucinated citation that is
   stripped before the answer reaches the user.  Stripped citations are
   recorded in ``GraphState.validation_results``.
4. Convert validated citations to ``Evidence`` + ``Citation`` objects and
   write them to the graph state.
5. Distinguish retrieved facts from model interpretation via prompt constraints
   (the LLM is instructed to mark each statement's source).

Security principles
-------------------
- Retrieved document text passed to the LLM is clearly labelled as UNTRUSTED
  DATA in the system prompt; it is structurally isolated inside XML delimiters.
- Citation cross-validation ensures the model cannot claim provenance for
  content that was never retrieved.
- ``reasoning_summary`` is logged for observability but never placed in the
  graph state or shown to the user.
- Document text is never placed in ``Evidence.excerpt`` directly; only the
  LLM-generated excerpt (derived from retrieved text) is stored.
"""

from __future__ import annotations

from collections.abc import Callable, Coroutine
from typing import Any

import anthropic
from pydantic import ValidationError

from src.agents.guardrails.citation_validation import SAFE_FAILURE_RESPONSE, CitationGuardrail
from src.agents.response.models import CitedChunk, ConfidenceLevel, ResponseDecision
from src.agents.response.prompts import (
    build_messages,
    build_system_prompt,
    no_evidence_response,
    unsupported_request_response,
)
from src.agents.state import GraphState
from src.core.config import get_settings
from src.core.logging import get_logger, log_debug, log_error, log_info, log_warning
from src.models.enums import AgentState
from src.models.evidence import Citation, Evidence
from src.models.validation import ValidationResult
from src.retrieval.hybrid.models import RetrievalEvidence

_logger = get_logger(__name__)
_guardrail = CitationGuardrail()

# Maximum evidence items forwarded to the LLM.  Higher-ranked items are
# preferred; excess items are silently discarded to keep the prompt bounded.
MAX_EVIDENCE_IN_PROMPT: int = 5

# ---------------------------------------------------------------------------
# Anthropic tool definition
# ---------------------------------------------------------------------------

_RESPONSE_TOOL: dict[str, Any] = {
    "name": "response_output",
    "description": (
        "Record the evidence-grounded response for the user's query. "
        "Call this tool exactly once with your complete, validated answer."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "answer": {
                "type": "string",
                "description": (
                    "The direct answer to the user's question. "
                    "Use [N] inline markers to reference cited documents."
                ),
            },
            "cited_chunks": {
                "type": "array",
                "description": "Citations to retrieved document chunks backing the answer.",
                "items": {
                    "type": "object",
                    "properties": {
                        "chunk_id": {
                            "type": "string",
                            "description": (
                                "Exact chunk_id from the retrieved_document header. "
                                "Do NOT invent or modify this value."
                            ),
                        },
                        "document_id": {
                            "type": "string",
                            "description": "Exact document_id from the retrieved_document header.",
                        },
                        "excerpt": {
                            "type": "string",
                            "description": "The specific passage excerpt supporting this claim.",
                        },
                        "reference_number": {
                            "type": "integer",
                            "minimum": 1,
                            "description": "The [N] number used for this citation in the answer.",
                        },
                    },
                    "required": ["chunk_id", "document_id", "excerpt", "reference_number"],
                },
            },
            "confidence": {
                "type": "string",
                "enum": [c.value for c in ConfidenceLevel],
                "description": "How well the available evidence supports this answer.",
            },
            "limitations": {
                "type": "string",
                "description": (
                    "What this answer cannot address, gaps in evidence, or uncertainty "
                    "the user should be aware of. Required even when confidence is high."
                ),
            },
            "reasoning_summary": {
                "type": "string",
                "description": (
                    "High-level explanation of how the answer was derived. "
                    "Internal observability only — not shown to the user."
                ),
            },
        },
        "required": [
            "answer",
            "cited_chunks",
            "confidence",
            "limitations",
            "reasoning_summary",
        ],
    },
}


# ---------------------------------------------------------------------------
# Citation validation
# ---------------------------------------------------------------------------


def _validate_citations(
    cited: list[CitedChunk],
    retrieved: list[RetrievalEvidence],
) -> tuple[list[CitedChunk], ValidationResult]:
    """Cross-validate LLM citations against the actually retrieved chunks.

    Returns the list of valid citations (chunk_id present in *retrieved* and
    document_id consistent) plus a ``ValidationResult`` summarising the check.

    Any citation whose ``chunk_id`` is not in *retrieved*, or whose
    ``document_id`` does not match the corresponding ``RetrievalEvidence``,
    is treated as a hallucinated citation and excluded from the valid list.
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

    if errors:
        return valid, ValidationResult(is_valid=False, errors=tuple(errors))
    return valid, ValidationResult(is_valid=True, errors=())


# ---------------------------------------------------------------------------
# Evidence / Citation conversion
# ---------------------------------------------------------------------------


def _to_evidence_and_citations(
    valid_citations: list[CitedChunk],
    retrieved_index: dict[str, RetrievalEvidence],
) -> tuple[list[Evidence], list[Citation]]:
    """Convert validated ``CitedChunk`` entries to ``Evidence`` + ``Citation`` pairs."""
    evidence_list: list[Evidence] = []
    citation_list: list[Citation] = []

    for cited in valid_citations:
        ev = retrieved_index[cited.chunk_id]
        evidence = Evidence(
            evidence_id=cited.chunk_id,
            document_id=cited.document_id,
            excerpt=cited.excerpt,
            relevance_score=min(1.0, ev.final_score),
        )
        citation = Citation(
            evidence_id=cited.chunk_id,
            reference_number=cited.reference_number,
        )
        evidence_list.append(evidence)
        citation_list.append(citation)

    return evidence_list, citation_list


# ---------------------------------------------------------------------------
# Response Agent
# ---------------------------------------------------------------------------


class ResponseAgent:
    """Produces an evidence-grounded, citation-validated answer.

    Construct via ``make_response_node`` rather than directly.
    """

    def __init__(
        self,
        *,
        client: anthropic.AsyncAnthropic,
        model: str,
        max_evidence: int = MAX_EVIDENCE_IN_PROMPT,
    ) -> None:
        self._client = client
        self._model = model
        self._max_evidence = max_evidence

    async def run(self, state: GraphState) -> dict[str, Any]:
        """Execute the response step and return a partial GraphState update.

        Never raises: all exceptions are captured and returned as ``errors``
        entries so the graph can continue gracefully.
        """
        budget = state["budget"]
        conv_id = state["conversation_id"]
        intent = state["intent"] or ""

        # --- Short-circuit: unsupported request intent ----------------------
        if intent == "unsupported_request":
            log_info(
                _logger,
                "response.unsupported_request",
                "Returning refusal for unsupported_request intent",
                conversation_id=conv_id,
            )
            return {
                "response": unsupported_request_response(),
                "current_agent": AgentState.RESPONSE,
                "current_node": "response",
                "budget": budget,
            }

        # --- Short-circuit: no retrieved evidence ---------------------------
        retrieved = state["retrieved_documents"]
        if not retrieved:
            log_info(
                _logger,
                "response.no_evidence",
                "No retrieved documents — returning no-evidence response",
                conversation_id=conv_id,
            )
            return {
                "response": no_evidence_response(),
                "current_agent": AgentState.RESPONSE,
                "current_node": "response",
                "budget": budget,
            }

        # --- Budget check ---------------------------------------------------
        if budget.is_exhausted:
            log_warning(
                _logger,
                "response.budget_exhausted",
                "Response Agent skipped — token budget exhausted",
                conversation_id=conv_id,
            )
            return {
                "response": no_evidence_response(),
                "current_agent": AgentState.RESPONSE,
                "current_node": "response",
                "errors": ["response agent skipped: token budget exhausted"],
                "budget": budget,
            }

        # --- Select top evidence for the prompt -----------------------------
        top_evidence = sorted(retrieved, key=lambda e: -e.final_score)[: self._max_evidence]

        system_prompt = build_system_prompt(top_evidence)
        messages = build_messages(list(state["messages"]))

        log_debug(
            _logger,
            "response.call_start",
            "Calling Anthropic for evidence-grounded response",
            model=self._model,
            evidence_count=len(top_evidence),
            conversation_id=conv_id,
        )

        # --- LLM call -------------------------------------------------------
        try:
            api_response = await self._client.messages.create(
                model=self._model,
                max_tokens=1024,
                system=system_prompt,
                tools=[_RESPONSE_TOOL],  # type: ignore[list-item]
                tool_choice={"type": "tool", "name": "response_output"},
                messages=messages,  # type: ignore[arg-type]
            )
        except Exception as exc:
            log_error(
                _logger,
                "response.llm_error",
                "Anthropic call failed in response agent",
                error=exc,
                conversation_id=conv_id,
            )
            return {
                "response": no_evidence_response(),
                "current_agent": AgentState.RESPONSE,
                "current_node": "response",
                "errors": [f"response LLM call failed: {exc}"],
                "budget": budget,
            }

        tokens_used = api_response.usage.input_tokens + api_response.usage.output_tokens
        new_budget = budget.consume(tokens=tokens_used)

        # --- Extract tool block ---------------------------------------------
        tool_block = next(
            (b for b in api_response.content if getattr(b, "type", None) == "tool_use"),
            None,
        )
        if tool_block is None:
            log_error(
                _logger,
                "response.no_tool_block",
                "Response agent LLM returned no tool-use block",
                conversation_id=conv_id,
            )
            return {
                "response": no_evidence_response(),
                "current_agent": AgentState.RESPONSE,
                "current_node": "response",
                "errors": ["response LLM returned no tool-use block"],
                "budget": new_budget,
            }

        # --- Parse & validate Pydantic model --------------------------------
        try:
            decision = ResponseDecision.model_validate(tool_block.input)
        except ValidationError as exc:
            log_error(
                _logger,
                "response.validation_error",
                "ResponseDecision failed Pydantic validation",
                error=exc,
                conversation_id=conv_id,
            )
            return {
                "response": no_evidence_response(),
                "current_agent": AgentState.RESPONSE,
                "current_node": "response",
                "errors": [f"response decision validation failed: {exc}"],
                "budget": new_budget,
            }

        # Log reasoning summary internally — never expose to user or state.
        log_debug(
            _logger,
            "response.reasoning_summary",
            "Response reasoning summary (internal)",
            reasoning_summary=decision.reasoning_summary,
            confidence=decision.confidence,
            conversation_id=conv_id,
        )

        # --- Citation guardrail (attempt 1) ---------------------------------
        valid_citations, validation_result = _guardrail.validate(
            decision=decision,
            retrieved=retrieved,
            user=state["user"],
            conversation_id=conv_id,
        )

        if not validation_result.is_valid:
            log_warning(
                _logger,
                "response.guardrail_retry",
                "Citation guardrail failed on first attempt; regenerating",
                error_count=len(validation_result.errors),
                conversation_id=conv_id,
            )

            # --- Regeneration call ------------------------------------------
            try:
                regen_api_response = await self._client.messages.create(
                    model=self._model,
                    max_tokens=1024,
                    system=system_prompt,
                    tools=[_RESPONSE_TOOL],  # type: ignore[list-item]
                    tool_choice={"type": "tool", "name": "response_output"},
                    messages=messages,  # type: ignore[arg-type]
                )
            except Exception as exc:
                log_error(
                    _logger,
                    "response.regeneration_error",
                    "Anthropic call failed during regeneration",
                    error=exc,
                    conversation_id=conv_id,
                )
                return {
                    "response": SAFE_FAILURE_RESPONSE,
                    "current_agent": AgentState.RESPONSE,
                    "current_node": "response",
                    "errors": [f"response regeneration failed: {exc}"],
                    "validation_results": [validation_result],
                    "budget": new_budget,
                }

            regen_tokens = (
                regen_api_response.usage.input_tokens
                + regen_api_response.usage.output_tokens
            )
            new_budget = new_budget.consume(tokens=regen_tokens)

            regen_tool_block = next(
                (
                    b
                    for b in regen_api_response.content
                    if getattr(b, "type", None) == "tool_use"
                ),
                None,
            )
            if regen_tool_block is None:
                log_error(
                    _logger,
                    "response.regeneration_no_tool_block",
                    "Regeneration returned no tool-use block",
                    conversation_id=conv_id,
                )
                return {
                    "response": SAFE_FAILURE_RESPONSE,
                    "current_agent": AgentState.RESPONSE,
                    "current_node": "response",
                    "errors": ["response regeneration returned no tool-use block"],
                    "validation_results": [validation_result],
                    "budget": new_budget,
                }

            try:
                regen_decision = ResponseDecision.model_validate(regen_tool_block.input)
            except ValidationError as exc:
                log_error(
                    _logger,
                    "response.regeneration_parse_error",
                    "Regenerated ResponseDecision failed Pydantic validation",
                    error=exc,
                    conversation_id=conv_id,
                )
                return {
                    "response": SAFE_FAILURE_RESPONSE,
                    "current_agent": AgentState.RESPONSE,
                    "current_node": "response",
                    "errors": [f"response regeneration parse failed: {exc}"],
                    "validation_results": [validation_result],
                    "budget": new_budget,
                }

            log_debug(
                _logger,
                "response.regeneration_reasoning_summary",
                "Regeneration reasoning summary (internal)",
                reasoning_summary=regen_decision.reasoning_summary,
                confidence=regen_decision.confidence,
                conversation_id=conv_id,
            )

            # --- Citation guardrail (attempt 2) -----------------------------
            valid_citations, regen_validation_result = _guardrail.validate(
                decision=regen_decision,
                retrieved=retrieved,
                user=state["user"],
                conversation_id=conv_id,
            )

            if not regen_validation_result.is_valid:
                log_warning(
                    _logger,
                    "response.guardrail_safe_failure",
                    "Citation guardrail failed after regeneration; returning safe failure",
                    conversation_id=conv_id,
                )
                return {
                    "response": SAFE_FAILURE_RESPONSE,
                    "current_agent": AgentState.RESPONSE,
                    "current_node": "response",
                    "errors": ["citation guardrail failed after regeneration"],
                    "validation_results": [validation_result, regen_validation_result],
                    "budget": new_budget,
                }

            decision = regen_decision
            validation_result = regen_validation_result

        # --- Convert to Evidence / Citation objects -------------------------
        retrieved_index: dict[str, RetrievalEvidence] = {
            ev.chunk_id: ev for ev in retrieved
        }
        evidence_list, citation_list = _to_evidence_and_citations(
            valid_citations, retrieved_index
        )

        log_info(
            _logger,
            "response.completed",
            "Response assembled",
            confidence=decision.confidence,
            evidence_count=len(evidence_list),
            citation_count=len(citation_list),
            citations_valid=validation_result.is_valid,
            conversation_id=conv_id,
        )

        update: dict[str, Any] = {
            "response": decision.answer,
            "evidence": evidence_list,
            "citations": citation_list,
            "validation_results": [validation_result],
            "current_agent": AgentState.RESPONSE,
            "current_node": "response",
            "budget": new_budget,
        }

        return update


# ---------------------------------------------------------------------------
# Public factory
# ---------------------------------------------------------------------------


def make_response_node(
    *,
    client: anthropic.AsyncAnthropic,
    model: str | None = None,
    max_evidence: int = MAX_EVIDENCE_IN_PROMPT,
) -> Callable[[GraphState], Coroutine[Any, Any, dict[str, Any]]]:
    """Return a LangGraph-compatible async callable for the response node.

    Parameters
    ----------
    client:
        An ``anthropic.AsyncAnthropic`` instance.  Injected for testability.
    model:
        Model identifier.  Defaults to ``Settings.model_name``.
    max_evidence:
        Maximum number of retrieved evidence items forwarded to the LLM.
    """
    resolved_model = model or get_settings().model_name
    agent = ResponseAgent(client=client, model=resolved_model, max_evidence=max_evidence)
    return agent.run
