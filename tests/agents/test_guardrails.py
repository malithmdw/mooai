"""Tests for the citation validation guardrail.

Coverage
--------
CitationGuardrail.validate():
  - valid citations pass through untouched
  - fabricated chunk_id rejected (hallucinated document ID)
  - mismatched document_id rejected (implicit fabricated-title coverage)
  - citations to non-retrieved documents rejected
  - citations to documents with an access_level the user cannot see rejected
  - duplicate reference numbers rejected
  - empty cited_chunks always valid
  - mixed batch: valid citations returned alongside errors when is_valid=False
  - trace_validation is called for each validate() invocation

ResponseAgent integration (retry and safe-failure paths):
  - retry on first guardrail failure: regeneration passes → valid response returned
  - safe failure after two guardrail failures (retry also invalid)
  - safe failure when regeneration call raises an exception
  - safe failure when regeneration returns no tool-use block
  - safe failure when regeneration ResponseDecision parse fails
  - budget consumed for both the first call and the regeneration call
  - both ValidationResults recorded when safe failure triggered
"""

from __future__ import annotations

from datetime import date
from unittest.mock import AsyncMock, MagicMock, patch

from src.agents.budget import ExecutionBudget
from src.agents.guardrails.citation_validation import (
    SAFE_FAILURE_RESPONSE,
    CitationGuardrail,
    _validate_access_levels,
    _validate_chunk_citations,
)
from src.agents.response.models import CitedChunk, ConfidenceLevel, ResponseDecision
from src.agents.response.node import make_response_node
from src.agents.state import GraphState, initial_state
from src.models.chat import Message
from src.models.documents import DocumentMetadata
from src.models.enums import AccessLevel, MessageRole, Role
from src.models.user import User
from src.retrieval.hybrid.models import RetrievalEvidence, RetrievalSource


# ---------------------------------------------------------------------------
# Shared test helpers
# ---------------------------------------------------------------------------


def _user(roles: tuple[Role, ...] = (Role.ANALYST,)) -> User:
    return User(
        user_id="user-0001",
        username="analyst01",
        display_name="Alice",
        roles=roles,
    )


def _meta(
    document_id: str = "DOC-001",
    access_level: AccessLevel = AccessLevel.INTERNAL,
    allowed_roles: tuple[Role, ...] = (Role.ANALYST,),
) -> DocumentMetadata:
    return DocumentMetadata(
        document_id=document_id,
        title="Policy Document",
        department="Risk",
        document_type="policy",
        access_level=access_level,
        created_date=date(2024, 1, 1),
        allowed_roles=allowed_roles,
    )


def _evidence(
    chunk_id: str = "DOC-001-chunk-0000",
    document_id: str = "DOC-001",
    access_level: AccessLevel = AccessLevel.INTERNAL,
    final_score: float = 0.8,
    rank: int = 1,
) -> RetrievalEvidence:
    return RetrievalEvidence(
        chunk_id=chunk_id,
        document_id=document_id,
        title="Policy Document",
        text="The loan default rate for Q3 2024 was 2.3 percent.",
        metadata=_meta(document_id, access_level),
        source=RetrievalSource.BOTH,
        dense_score=0.8,
        sparse_score=0.7,
        final_score=final_score,
        rank=rank,
    )


def _cited(
    chunk_id: str = "DOC-001-chunk-0000",
    document_id: str = "DOC-001",
    reference_number: int = 1,
) -> CitedChunk:
    return CitedChunk(
        chunk_id=chunk_id,
        document_id=document_id,
        excerpt="The loan default rate for Q3 2024 was 2.3 percent.",
        reference_number=reference_number,
    )


def _decision(
    chunk_id: str = "DOC-001-chunk-0000",
    document_id: str = "DOC-001",
) -> ResponseDecision:
    return ResponseDecision(
        answer=f"The default rate was 2.3%. [1]",
        cited_chunks=[_cited(chunk_id, document_id)],
        confidence=ConfidenceLevel.HIGH,
        limitations="Q3 2024 only.",
        reasoning_summary="Direct match found.",
    )


def _message(
    content: str = "What is the loan default rate?",
    role: MessageRole = MessageRole.USER,
    index: int = 1,
) -> Message:
    return Message(
        message_id=f"msg-{index:04d}",
        conversation_id="conv-0001",
        role=role,
        content=content,
    )


def _state(
    content: str = "What is the loan default rate?",
    *,
    retrieved: list[RetrievalEvidence] | None = None,
    intent: str | None = "knowledge_question",
    user: User | None = None,
    budget: ExecutionBudget | None = None,
) -> GraphState:
    state = initial_state(
        user=user or _user(), conversation_id="conv-0001", budget=budget
    )
    state["messages"] = [_message(content)]
    state["intent"] = intent
    state["retrieved_documents"] = retrieved if retrieved is not None else [_evidence()]
    return state


def _tool_block(tool_input: dict) -> MagicMock:  # type: ignore[type-arg]
    block = MagicMock()
    block.type = "tool_use"
    block.name = "response_output"
    block.input = tool_input
    return block


def _mock_api_response(
    tool_input: dict,  # type: ignore[type-arg]
    *,
    input_tokens: int = 300,
    output_tokens: int = 150,
) -> MagicMock:  # type: ignore[type-arg]
    usage = MagicMock()
    usage.input_tokens = input_tokens
    usage.output_tokens = output_tokens
    resp = MagicMock()
    resp.content = [_tool_block(tool_input)]
    resp.usage = usage
    return resp


def _valid_tool_input(
    chunk_id: str = "DOC-001-chunk-0000",
    document_id: str = "DOC-001",
) -> dict:  # type: ignore[type-arg]
    return {
        "answer": "The loan default rate was 2.3 percent. [1]",
        "cited_chunks": [
            {
                "chunk_id": chunk_id,
                "document_id": document_id,
                "excerpt": "The loan default rate for Q3 2024 was 2.3 percent.",
                "reference_number": 1,
            }
        ],
        "confidence": "high",
        "limitations": "Q3 2024 only; other periods may differ.",
        "reasoning_summary": "Direct match found in policy document.",
    }


def _hallucinated_tool_input() -> dict:  # type: ignore[type-arg]
    return {
        "answer": "The rate is 5%. [1]",
        "cited_chunks": [
            {
                "chunk_id": "FAKE-chunk-9999",
                "document_id": "DOC-FAKE",
                "excerpt": "hallucinated excerpt",
                "reference_number": 1,
            }
        ],
        "confidence": "high",
        "limitations": "None.",
        "reasoning_summary": "Using fabricated source.",
    }


# ---------------------------------------------------------------------------
# _validate_chunk_citations unit tests
# ---------------------------------------------------------------------------


class TestValidateChunkCitations:
    def test_valid_citation_passes(self) -> None:
        retrieved = [_evidence("DOC-001-chunk-0000", "DOC-001")]
        cited = [_cited("DOC-001-chunk-0000", "DOC-001")]
        valid, errors = _validate_chunk_citations(cited, retrieved)
        assert len(valid) == 1
        assert errors == []

    def test_fabricated_chunk_id_rejected(self) -> None:
        retrieved = [_evidence("DOC-001-chunk-0000", "DOC-001")]
        cited = [_cited("FAKE-chunk-9999", "DOC-001")]
        valid, errors = _validate_chunk_citations(cited, retrieved)
        assert valid == []
        assert any("hallucinated" in e for e in errors)
        assert any("FAKE-chunk-9999" in e for e in errors)

    def test_mismatched_document_id_rejected(self) -> None:
        retrieved = [_evidence("DOC-001-chunk-0000", "DOC-001")]
        cited = [_cited("DOC-001-chunk-0000", "DOC-WRONG")]
        valid, errors = _validate_chunk_citations(cited, retrieved)
        assert valid == []
        assert any("mismatched" in e for e in errors)

    def test_duplicate_reference_number_rejected(self) -> None:
        retrieved = [
            _evidence("DOC-001-chunk-0000", "DOC-001"),
            _evidence("DOC-002-chunk-0000", "DOC-002"),
        ]
        cited = [
            _cited("DOC-001-chunk-0000", "DOC-001", reference_number=1),
            _cited("DOC-002-chunk-0000", "DOC-002", reference_number=1),  # dupe
        ]
        valid, errors = _validate_chunk_citations(cited, retrieved)
        assert len(valid) == 1
        assert any("duplicate" in e for e in errors)

    def test_empty_cited_returns_no_errors(self) -> None:
        valid, errors = _validate_chunk_citations([], [_evidence()])
        assert valid == []
        assert errors == []

    def test_mixed_valid_and_invalid(self) -> None:
        retrieved = [_evidence("DOC-001-chunk-0000", "DOC-001")]
        cited = [
            _cited("DOC-001-chunk-0000", "DOC-001", reference_number=1),
            _cited("INVENTED-chunk", "DOC-001", reference_number=2),
        ]
        valid, errors = _validate_chunk_citations(cited, retrieved)
        assert len(valid) == 1
        assert valid[0].chunk_id == "DOC-001-chunk-0000"
        assert len(errors) == 1


# ---------------------------------------------------------------------------
# _validate_access_levels unit tests
# ---------------------------------------------------------------------------


class TestValidateAccessLevels:
    def test_permitted_level_passes(self) -> None:
        user = _user(roles=(Role.ANALYST,))
        retrieved = [_evidence(access_level=AccessLevel.CONFIDENTIAL)]
        cited = [_cited()]
        errors = _validate_access_levels(cited, retrieved, user)
        assert errors == []

    def test_unauthorized_level_rejected(self) -> None:
        user = _user(roles=(Role.VIEWER,))
        retrieved = [_evidence(access_level=AccessLevel.RESTRICTED)]
        cited = [_cited()]
        errors = _validate_access_levels(cited, retrieved, user)
        assert len(errors) == 1
        assert "unauthorized" in errors[0]

    def test_viewer_cannot_access_confidential(self) -> None:
        user = _user(roles=(Role.VIEWER,))
        retrieved = [_evidence(access_level=AccessLevel.CONFIDENTIAL)]
        cited = [_cited()]
        errors = _validate_access_levels(cited, retrieved, user)
        assert len(errors) == 1

    def test_administrator_can_access_restricted(self) -> None:
        user = _user(roles=(Role.ADMINISTRATOR,))
        retrieved = [_evidence(access_level=AccessLevel.RESTRICTED)]
        cited = [_cited()]
        errors = _validate_access_levels(cited, retrieved, user)
        assert errors == []

    def test_unknown_chunk_id_skipped(self) -> None:
        user = _user(roles=(Role.VIEWER,))
        # cited chunk_id not in retrieved — already caught by chunk validation
        cited = [_cited("NONEXISTENT-chunk", "DOC-001")]
        errors = _validate_access_levels(cited, [], user)
        assert errors == []


# ---------------------------------------------------------------------------
# CitationGuardrail.validate() tests
# ---------------------------------------------------------------------------


class TestCitationGuardrailValidate:
    def _guardrail(self) -> CitationGuardrail:
        return CitationGuardrail()

    def test_valid_citations_pass(self) -> None:
        decision = _decision()
        retrieved = [_evidence()]
        valid, result = self._guardrail().validate(
            decision=decision,
            retrieved=retrieved,
            user=_user(),
            conversation_id="conv-0001",
        )
        assert result.is_valid
        assert len(valid) == 1

    def test_fabricated_chunk_id_rejected(self) -> None:
        decision = _decision(chunk_id="FAKE-9999", document_id="DOC-FAKE")
        retrieved = [_evidence()]
        _, result = self._guardrail().validate(
            decision=decision,
            retrieved=retrieved,
            user=_user(),
            conversation_id="conv-0001",
        )
        assert not result.is_valid
        assert any("hallucinated" in e for e in result.errors)

    def test_fabricated_document_id_rejected(self) -> None:
        """Covers fabricated-title scenario: chunk exists but document_id is wrong."""
        retrieved = [_evidence("DOC-001-chunk-0000", "DOC-001")]
        decision = _decision(chunk_id="DOC-001-chunk-0000", document_id="DOC-FABRICATED")
        _, result = self._guardrail().validate(
            decision=decision,
            retrieved=retrieved,
            user=_user(),
            conversation_id="conv-0001",
        )
        assert not result.is_valid
        assert any("mismatched" in e for e in result.errors)

    def test_non_retrieved_document_rejected(self) -> None:
        retrieved = [_evidence("DOC-001-chunk-0000", "DOC-001")]
        decision = _decision(chunk_id="DOC-002-chunk-0000", document_id="DOC-002")
        _, result = self._guardrail().validate(
            decision=decision,
            retrieved=retrieved,
            user=_user(),
            conversation_id="conv-0001",
        )
        assert not result.is_valid

    def test_unauthorized_access_level_rejected(self) -> None:
        user = _user(roles=(Role.VIEWER,))
        retrieved = [_evidence(access_level=AccessLevel.RESTRICTED)]
        decision = _decision()
        _, result = self._guardrail().validate(
            decision=decision,
            retrieved=retrieved,
            user=user,
            conversation_id="conv-0001",
        )
        assert not result.is_valid
        assert any("unauthorized" in e for e in result.errors)

    def test_empty_cited_chunks_is_valid(self) -> None:
        decision = ResponseDecision(
            answer="No specific citations.",
            cited_chunks=[],
            confidence=ConfidenceLevel.LOW,
            limitations="No supporting documents found.",
            reasoning_summary="No evidence available.",
        )
        _, result = self._guardrail().validate(
            decision=decision,
            retrieved=[_evidence()],
            user=_user(),
            conversation_id="conv-0001",
        )
        assert result.is_valid

    def test_none_user_skips_auth_check(self) -> None:
        """When user is None, auth check is skipped rather than crashing."""
        retrieved = [_evidence(access_level=AccessLevel.RESTRICTED)]
        decision = _decision()
        valid, result = self._guardrail().validate(
            decision=decision,
            retrieved=retrieved,
            user=None,
            conversation_id="conv-0001",
        )
        assert result.is_valid
        assert len(valid) == 1

    def test_validation_result_errors_contain_chunk_id(self) -> None:
        retrieved = [_evidence()]
        decision = _decision(chunk_id="HALLUCINATED-chunk", document_id="DOC-X")
        _, result = self._guardrail().validate(
            decision=decision,
            retrieved=retrieved,
            user=_user(),
            conversation_id="conv-0001",
        )
        assert any("HALLUCINATED-chunk" in e for e in result.errors)

    def test_trace_validation_called(self) -> None:
        """trace_validation context manager is invoked for each validate() call."""
        with patch(
            "src.agents.guardrails.citation_validation.trace_validation"
        ) as mock_trace:
            # MagicMock supports __enter__/__exit__ natively
            self._guardrail().validate(
                decision=_decision(),
                retrieved=[_evidence()],
                user=_user(),
                conversation_id="conv-0001",
            )
        mock_trace.assert_called_once_with(
            "citation_guardrail", conversation_id="conv-0001"
        )


# ---------------------------------------------------------------------------
# ResponseAgent retry and safe-failure integration tests
# ---------------------------------------------------------------------------


class TestResponseAgentGuardrailRetry:
    """Integration tests for the retry / safe-failure paths inside ResponseAgent."""

    def _mock_client_sequence(self, *tool_inputs: dict) -> MagicMock:  # type: ignore[type-arg]
        """Create a mock client that returns each tool_input in sequence."""
        responses = [_mock_api_response(inp) for inp in tool_inputs]
        client = MagicMock()
        client.messages.create = AsyncMock(side_effect=responses)
        return client

    async def test_retry_succeeds_on_regeneration(self) -> None:
        """First response has hallucinated citations; regeneration is clean."""
        state = _state()
        client = self._mock_client_sequence(
            _hallucinated_tool_input(),  # attempt 1: invalid → triggers retry
            _valid_tool_input(),          # attempt 2: valid → accepted
        )
        node = make_response_node(client=client, model="test")
        update = await node(state)

        assert update["response"] == "The loan default rate was 2.3 percent. [1]"
        assert len(update.get("citations", [])) == 1
        assert update["citations"][0].evidence_id == "DOC-001-chunk-0000"
        assert update.get("validation_results", [{}])[-1].is_valid  # type: ignore[union-attr]
        assert client.messages.create.call_count == 2

    async def test_safe_failure_after_two_guardrail_failures(self) -> None:
        """Both attempts return hallucinated citations → safe failure response."""
        state = _state()
        client = self._mock_client_sequence(
            _hallucinated_tool_input(),  # attempt 1
            _hallucinated_tool_input(),  # attempt 2 (regeneration)
        )
        node = make_response_node(client=client, model="test")
        update = await node(state)

        assert update["response"] == SAFE_FAILURE_RESPONSE
        assert client.messages.create.call_count == 2
        assert any("guardrail failed after regeneration" in e for e in update.get("errors", []))

    async def test_safe_failure_when_regeneration_raises(self) -> None:
        """Regeneration call raises an exception → safe failure."""
        state = _state()
        client = MagicMock()
        client.messages.create = AsyncMock(
            side_effect=[
                _mock_api_response(_hallucinated_tool_input()),
                RuntimeError("network error"),
            ]
        )
        node = make_response_node(client=client, model="test")
        update = await node(state)

        assert update["response"] == SAFE_FAILURE_RESPONSE
        assert any("regeneration failed" in e for e in update.get("errors", []))

    async def test_safe_failure_when_regeneration_no_tool_block(self) -> None:
        """Regeneration returns a response with no tool-use block → safe failure."""
        empty_resp = MagicMock()
        empty_resp.content = []
        usage = MagicMock()
        usage.input_tokens = 100
        usage.output_tokens = 50
        empty_resp.usage = usage

        state = _state()
        client = MagicMock()
        client.messages.create = AsyncMock(
            side_effect=[
                _mock_api_response(_hallucinated_tool_input()),
                empty_resp,
            ]
        )
        node = make_response_node(client=client, model="test")
        update = await node(state)

        assert update["response"] == SAFE_FAILURE_RESPONSE
        assert any("no tool-use block" in e for e in update.get("errors", []))

    async def test_safe_failure_when_regeneration_parse_fails(self) -> None:
        """Regeneration tool block has invalid schema → safe failure."""
        bad_input = {"answer": "", "confidence": "invalid", "limitations": ""}
        state = _state()
        client = MagicMock()
        client.messages.create = AsyncMock(
            side_effect=[
                _mock_api_response(_hallucinated_tool_input()),
                _mock_api_response(bad_input),
            ]
        )
        node = make_response_node(client=client, model="test")
        update = await node(state)

        assert update["response"] == SAFE_FAILURE_RESPONSE
        assert any("regeneration parse failed" in e for e in update.get("errors", []))

    async def test_budget_consumed_for_both_calls(self) -> None:
        """Tokens from both the first call and the retry call are tracked."""
        state = _state()
        client = self._mock_client_sequence(
            _hallucinated_tool_input(),
            _hallucinated_tool_input(),
        )
        node = make_response_node(client=client, model="test")
        update = await node(state)

        # 2 calls × (300 input + 150 output) = 900 tokens consumed
        assert update["budget"].used_tokens == 900

    async def test_both_validation_results_recorded_on_safe_failure(self) -> None:
        """When safe failure is triggered, both failed validation_results are in state."""
        state = _state()
        client = self._mock_client_sequence(
            _hallucinated_tool_input(),
            _hallucinated_tool_input(),
        )
        node = make_response_node(client=client, model="test")
        update = await node(state)

        validation_results = update.get("validation_results", [])
        assert len(validation_results) == 2
        assert all(not vr.is_valid for vr in validation_results)

    async def test_no_retry_on_valid_first_response(self) -> None:
        """No retry is attempted when the first response passes guardrail."""
        state = _state()
        client = MagicMock()
        client.messages.create = AsyncMock(
            return_value=_mock_api_response(_valid_tool_input())
        )
        node = make_response_node(client=client, model="test")
        await node(state)

        assert client.messages.create.call_count == 1
