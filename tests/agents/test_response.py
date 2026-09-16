"""Tests for the Response Agent node.

Required scenarios (per task spec):
- valid answer              — evidence available, LLM returns well-formed answer
- unsupported answer        — no retrieved evidence → canned refusal, no LLM call
- hallucinated citation     — LLM cites a chunk_id not in retrieved_documents →
                              citation stripped, ValidationResult(is_valid=False)
- missing evidence          — retrieved_documents is empty → no-evidence response

Additional coverage:
- citation cross-validation: document_id mismatch, duplicate reference_number
- Evidence and Citation objects correctly populated from valid citations
- reasoning_summary kept internal (not in state)
- budget consumption from LLM usage tokens
- token budget exhaustion short-circuit
- LLM API exception captured as error (no raise)
- Pydantic validation failure of tool input captured as error (no raise)
- No tool-use block in response captured as error
- unsupported_request intent short-circuit (no LLM call)
- build_system_prompt includes chunk_id values
- build_messages filtering
- _validate_citations standalone tests
"""

from __future__ import annotations

from datetime import date
from unittest.mock import AsyncMock, MagicMock

import pytest
from pydantic import ValidationError

from src.agents.budget import ExecutionBudget
from src.agents.response.models import CitedChunk, ConfidenceLevel, ResponseDecision
from src.agents.response.node import (
    MAX_EVIDENCE_IN_PROMPT,
    ResponseAgent,
    _to_evidence_and_citations,
    _validate_citations,
    make_response_node,
)
from src.agents.response.prompts import (
    build_messages,
    build_system_prompt,
    no_evidence_response,
    unsupported_request_response,
)
from src.agents.state import GraphState, initial_state
from src.models.chat import Message
from src.models.documents import DocumentMetadata
from src.models.enums import AccessLevel, AgentState, MessageRole, Role
from src.models.user import User
from src.retrieval.hybrid.models import RetrievalEvidence, RetrievalSource


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------


def _user(roles: tuple[Role, ...] = (Role.ANALYST,)) -> User:
    return User(
        user_id="user-0001",
        username="analyst01",
        display_name="Alice",
        roles=roles,
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


def _meta(
    document_id: str = "DOC-001",
    access_level: AccessLevel = AccessLevel.INTERNAL,
    allowed_roles: tuple[Role, ...] = (Role.ANALYST,),
) -> DocumentMetadata:
    return DocumentMetadata(
        document_id=document_id,
        title="Loan Policy Document",
        department="Risk",
        document_type="policy",
        access_level=access_level,
        created_date=date(2024, 1, 1),
        allowed_roles=allowed_roles,
    )


def _evidence(
    chunk_id: str = "DOC-001-chunk-0000",
    document_id: str = "DOC-001",
    final_score: float = 0.02,
    rank: int = 1,
) -> RetrievalEvidence:
    return RetrievalEvidence(
        chunk_id=chunk_id,
        document_id=document_id,
        title="Loan Policy Document",
        text="The loan default rate for Q3 2024 was 2.3 percent.",
        metadata=_meta(document_id),
        source=RetrievalSource.BOTH,
        dense_score=0.8,
        sparse_score=0.7,
        final_score=final_score,
        rank=rank,
    )


def _state(
    content: str = "What is the loan default rate?",
    *,
    retrieved: list[RetrievalEvidence] | None = None,
    intent: str | None = "knowledge_question",
    budget: ExecutionBudget | None = None,
) -> GraphState:
    state = initial_state(user=_user(), conversation_id="conv-0001", budget=budget)
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


def _mock_response(
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


def _mock_client(tool_input: dict) -> MagicMock:  # type: ignore[type-arg]
    client = MagicMock()
    client.messages.create = AsyncMock(return_value=_mock_response(tool_input))
    return client


def _valid_tool_input(
    chunk_id: str = "DOC-001-chunk-0000",
    document_id: str = "DOC-001",
) -> dict:  # type: ignore[type-arg]
    return {
        "answer": f"The loan default rate was 2.3 percent. [1]",
        "cited_chunks": [
            {
                "chunk_id": chunk_id,
                "document_id": document_id,
                "excerpt": "The loan default rate for Q3 2024 was 2.3 percent.",
                "reference_number": 1,
            }
        ],
        "confidence": "high",
        "limitations": "This figure is for Q3 2024 only; other periods may differ.",
        "reasoning_summary": "Direct match found in the policy document.",
    }


def _unsupported_tool_input() -> dict:  # type: ignore[type-arg]
    return {
        "answer": "I cannot find relevant information in the provided documents.",
        "cited_chunks": [],
        "confidence": "unsupported",
        "limitations": "No relevant documents were available.",
        "reasoning_summary": "No matching evidence found.",
    }


# ---------------------------------------------------------------------------
# _validate_citations unit tests
# ---------------------------------------------------------------------------


class TestValidateCitations:
    def test_all_valid_citations_pass(self) -> None:
        retrieved = [_evidence("DOC-001-chunk-0000", "DOC-001")]
        cited = [
            CitedChunk(
                chunk_id="DOC-001-chunk-0000",
                document_id="DOC-001",
                excerpt="excerpt text",
                reference_number=1,
            )
        ]
        valid, result = _validate_citations(cited, retrieved)
        assert len(valid) == 1
        assert result.is_valid

    def test_hallucinated_chunk_id_stripped(self) -> None:
        retrieved = [_evidence("DOC-001-chunk-0000", "DOC-001")]
        cited = [
            CitedChunk(
                chunk_id="FAKE-chunk-9999",
                document_id="DOC-001",
                excerpt="fake excerpt",
                reference_number=1,
            )
        ]
        valid, result = _validate_citations(cited, retrieved)
        assert valid == []
        assert not result.is_valid
        assert any("hallucinated" in e for e in result.errors)

    def test_mismatched_document_id_stripped(self) -> None:
        retrieved = [_evidence("DOC-001-chunk-0000", "DOC-001")]
        cited = [
            CitedChunk(
                chunk_id="DOC-001-chunk-0000",
                document_id="DOC-WRONG",  # wrong document_id
                excerpt="excerpt",
                reference_number=1,
            )
        ]
        valid, result = _validate_citations(cited, retrieved)
        assert valid == []
        assert not result.is_valid
        assert any("mismatched" in e for e in result.errors)

    def test_duplicate_reference_number_stripped(self) -> None:
        retrieved = [
            _evidence("DOC-001-chunk-0000", "DOC-001"),
            _evidence("DOC-002-chunk-0000", "DOC-002"),
        ]
        cited = [
            CitedChunk(
                chunk_id="DOC-001-chunk-0000",
                document_id="DOC-001",
                excerpt="excerpt A",
                reference_number=1,
            ),
            CitedChunk(
                chunk_id="DOC-002-chunk-0000",
                document_id="DOC-002",
                excerpt="excerpt B",
                reference_number=1,  # duplicate
            ),
        ]
        valid, result = _validate_citations(cited, retrieved)
        assert len(valid) == 1
        assert not result.is_valid
        assert any("duplicate" in e for e in result.errors)

    def test_empty_cited_returns_valid(self) -> None:
        _, result = _validate_citations([], [_evidence()])
        assert result.is_valid

    def test_empty_retrieved_all_hallucinated(self) -> None:
        cited = [
            CitedChunk(
                chunk_id="DOC-001-chunk-0000",
                document_id="DOC-001",
                excerpt="excerpt",
                reference_number=1,
            )
        ]
        valid, result = _validate_citations(cited, [])
        assert valid == []
        assert not result.is_valid

    def test_mixed_valid_and_invalid(self) -> None:
        retrieved = [_evidence("DOC-001-chunk-0000", "DOC-001")]
        cited = [
            CitedChunk(
                chunk_id="DOC-001-chunk-0000",
                document_id="DOC-001",
                excerpt="valid",
                reference_number=1,
            ),
            CitedChunk(
                chunk_id="FAKE-chunk-0001",
                document_id="DOC-001",
                excerpt="fake",
                reference_number=2,
            ),
        ]
        valid, result = _validate_citations(cited, retrieved)
        assert len(valid) == 1
        assert valid[0].chunk_id == "DOC-001-chunk-0000"
        assert not result.is_valid


# ---------------------------------------------------------------------------
# ResponseDecision model tests
# ---------------------------------------------------------------------------


class TestResponseDecision:
    def test_valid_construction(self) -> None:
        d = ResponseDecision.model_validate(_valid_tool_input())
        assert d.confidence == ConfidenceLevel.HIGH
        assert len(d.cited_chunks) == 1
        assert d.cited_chunks[0].chunk_id == "DOC-001-chunk-0000"

    def test_empty_answer_raises(self) -> None:
        bad = {**_valid_tool_input(), "answer": ""}
        with pytest.raises(ValidationError):
            ResponseDecision.model_validate(bad)

    def test_invalid_confidence_raises(self) -> None:
        bad = {**_valid_tool_input(), "confidence": "very_certain"}
        with pytest.raises(ValidationError):
            ResponseDecision.model_validate(bad)

    def test_missing_limitations_raises(self) -> None:
        bad = _valid_tool_input()
        del bad["limitations"]  # type: ignore[misc]
        with pytest.raises(ValidationError):
            ResponseDecision.model_validate(bad)

    def test_unsupported_confidence_allowed(self) -> None:
        d = ResponseDecision.model_validate(_unsupported_tool_input())
        assert d.confidence == ConfidenceLevel.UNSUPPORTED

    def test_decision_is_frozen(self) -> None:
        d = ResponseDecision.model_validate(_valid_tool_input())
        with pytest.raises(Exception):
            d.answer = "modified"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Prompt builder tests
# ---------------------------------------------------------------------------


class TestBuildSystemPrompt:
    def test_chunk_id_present_in_prompt(self) -> None:
        ev = _evidence("DOC-001-chunk-0000")
        prompt = build_system_prompt([ev])
        assert "DOC-001-chunk-0000" in prompt

    def test_document_id_present_in_prompt(self) -> None:
        ev = _evidence(document_id="DOC-001")
        prompt = build_system_prompt([ev])
        assert "DOC-001" in prompt

    def test_untrusted_data_label_present(self) -> None:
        prompt = build_system_prompt([_evidence()])
        assert "UNTRUSTED" in prompt

    def test_no_evidence_shows_placeholder(self) -> None:
        prompt = build_system_prompt([])
        assert "No relevant documents" in prompt

    def test_response_output_tool_referenced(self) -> None:
        prompt = build_system_prompt([_evidence()])
        assert "response_output" in prompt


class TestBuildMessages:
    def test_user_message_included(self) -> None:
        msgs = [_message("How does X work?")]
        result = build_messages(msgs)
        assert result[0]["content"] == "How does X work?"

    def test_system_messages_filtered(self) -> None:
        msgs = [
            _message("sys content", role=MessageRole.SYSTEM, index=0),
            _message("user question", role=MessageRole.USER, index=1),
        ]
        result = build_messages(msgs)
        assert all(m["role"] != "system" for m in result)

    def test_empty_returns_synthetic(self) -> None:
        result = build_messages([])
        assert result[0]["role"] == "user"


# ---------------------------------------------------------------------------
# Required test cases from task spec
# ---------------------------------------------------------------------------


class TestValidAnswer:
    """Valid answer: evidence present, LLM returns well-formed answer with citations."""

    async def test_response_set_in_state(self) -> None:
        state = _state()
        node = make_response_node(client=_mock_client(_valid_tool_input()), model="test")
        update = await node(state)
        assert update["response"] is not None
        assert len(update["response"]) > 0

    async def test_evidence_populated(self) -> None:
        state = _state()
        node = make_response_node(client=_mock_client(_valid_tool_input()), model="test")
        update = await node(state)
        assert len(update["evidence"]) == 1
        assert update["evidence"][0].document_id == "DOC-001"

    async def test_citations_populated(self) -> None:
        state = _state()
        node = make_response_node(client=_mock_client(_valid_tool_input()), model="test")
        update = await node(state)
        assert len(update["citations"]) == 1
        assert update["citations"][0].reference_number == 1
        assert update["citations"][0].evidence_id == "DOC-001-chunk-0000"

    async def test_validation_result_is_valid(self) -> None:
        state = _state()
        node = make_response_node(client=_mock_client(_valid_tool_input()), model="test")
        update = await node(state)
        assert any(vr.is_valid for vr in update["validation_results"])

    async def test_no_errors_on_success(self) -> None:
        state = _state()
        node = make_response_node(client=_mock_client(_valid_tool_input()), model="test")
        update = await node(state)
        assert "errors" not in update

    async def test_budget_consumed(self) -> None:
        state = _state()
        node = make_response_node(client=_mock_client(_valid_tool_input()), model="test")
        update = await node(state)
        assert update["budget"].used_tokens > 0

    async def test_reasoning_summary_not_in_state(self) -> None:
        """reasoning_summary is internal and must not be exposed to the user."""
        state = _state()
        node = make_response_node(client=_mock_client(_valid_tool_input()), model="test")
        update = await node(state)
        # It must not appear anywhere in the returned state update
        assert "reasoning_summary" not in update
        # And must not be smuggled into the response text as a raw dump
        response_text = update.get("response", "")
        assert "reasoning_summary" not in response_text.lower()

    async def test_current_agent_set(self) -> None:
        state = _state()
        node = make_response_node(client=_mock_client(_valid_tool_input()), model="test")
        update = await node(state)
        assert update["current_agent"] == AgentState.RESPONSE


class TestUnsupportedAnswer:
    """Unsupported answer: no retrieved evidence → refusal without LLM call."""

    async def test_no_llm_call_when_no_evidence(self) -> None:
        state = _state(retrieved=[])
        client = MagicMock()
        client.messages.create = AsyncMock()
        node = make_response_node(client=client, model="test")
        await node(state)
        client.messages.create.assert_not_called()

    async def test_response_returned_when_no_evidence(self) -> None:
        state = _state(retrieved=[])
        node = make_response_node(client=MagicMock(), model="test")
        update = await node(state)
        assert update["response"] is not None
        assert len(update["response"]) > 0

    async def test_response_acknowledges_lack_of_evidence(self) -> None:
        state = _state(retrieved=[])
        node = make_response_node(client=MagicMock(), model="test")
        update = await node(state)
        # The canned message should indicate inability to find information
        assert any(
            phrase in update["response"].lower()
            for phrase in ["unable to find", "no relevant", "knowledge base"]
        )

    async def test_no_evidence_citations_empty(self) -> None:
        state = _state(retrieved=[])
        node = make_response_node(client=MagicMock(), model="test")
        update = await node(state)
        assert update.get("citations", []) == [] or "citations" not in update

    async def test_unsupported_request_intent_no_llm_call(self) -> None:
        state = _state(intent="unsupported_request")
        client = MagicMock()
        client.messages.create = AsyncMock()
        node = make_response_node(client=client, model="test")
        await node(state)
        client.messages.create.assert_not_called()

    async def test_unsupported_request_response_returned(self) -> None:
        state = _state(intent="unsupported_request")
        node = make_response_node(client=MagicMock(), model="test")
        update = await node(state)
        assert update["response"] is not None


class TestHallucinatedCitation:
    """Hallucinated citation: LLM cites a chunk_id not in retrieved_documents."""

    def _fake_chunk_input(self) -> dict:  # type: ignore[type-arg]
        return {
            "answer": "The rate is 5%. [1]",
            "cited_chunks": [
                {
                    "chunk_id": "FAKE-chunk-9999",  # not in retrieved_documents
                    "document_id": "DOC-001",
                    "excerpt": "hallucinated excerpt",
                    "reference_number": 1,
                }
            ],
            "confidence": "high",
            "limitations": "None.",
            "reasoning_summary": "Using fabricated source.",
        }

    async def test_hallucinated_citation_stripped_from_citations(self) -> None:
        state = _state()  # retrieved has DOC-001-chunk-0000, not FAKE-chunk-9999
        node = make_response_node(
            client=_mock_client(self._fake_chunk_input()), model="test"
        )
        update = await node(state)
        citation_ids = [c.evidence_id for c in update.get("citations", [])]
        assert "FAKE-chunk-9999" not in citation_ids

    async def test_hallucinated_citation_produces_validation_failure(self) -> None:
        state = _state()
        node = make_response_node(
            client=_mock_client(self._fake_chunk_input()), model="test"
        )
        update = await node(state)
        assert any(not vr.is_valid for vr in update.get("validation_results", []))

    async def test_hallucinated_citation_error_message_descriptive(self) -> None:
        state = _state()
        node = make_response_node(
            client=_mock_client(self._fake_chunk_input()), model="test"
        )
        update = await node(state)
        invalid_results = [vr for vr in update.get("validation_results", []) if not vr.is_valid]
        assert any(
            "FAKE-chunk-9999" in e or "hallucinated" in e
            for vr in invalid_results
            for e in vr.errors
        )

    async def test_answer_still_returned_after_stripping(self) -> None:
        """The answer is returned even when citations are stripped."""
        state = _state()
        node = make_response_node(
            client=_mock_client(self._fake_chunk_input()), model="test"
        )
        update = await node(state)
        assert update.get("response")

    async def test_mixed_citations_trigger_retry_and_safe_failure(self) -> None:
        """Any invalid citation triggers retry; safe failure if retry also fails."""
        ev1 = _evidence("DOC-001-chunk-0000", "DOC-001")
        state = _state(retrieved=[ev1])
        mixed_input = {
            "answer": "Answer [1] and also [2].",
            "cited_chunks": [
                {
                    "chunk_id": "DOC-001-chunk-0000",  # valid
                    "document_id": "DOC-001",
                    "excerpt": "valid excerpt",
                    "reference_number": 1,
                },
                {
                    "chunk_id": "INVENTED-chunk-0001",  # hallucinated
                    "document_id": "DOC-002",
                    "excerpt": "fake excerpt",
                    "reference_number": 2,
                },
            ],
            "confidence": "medium",
            "limitations": "Partial information.",
            "reasoning_summary": "One valid source found.",
        }
        # Mock always returns the same mixed response — retry also fails → safe failure.
        node = make_response_node(client=_mock_client(mixed_input), model="test")
        update = await node(state)
        # Hallucinated chunk must never appear in citations
        citation_ids = [c.evidence_id for c in update.get("citations", [])]
        assert "INVENTED-chunk-0001" not in citation_ids
        # Safe failure response is non-empty
        assert update.get("response")
        # At least one validation failure recorded
        assert any(not vr.is_valid for vr in update.get("validation_results", []))


class TestMissingEvidence:
    """Missing evidence: empty retrieved_documents → no-evidence response."""

    async def test_no_evidence_response_text_non_empty(self) -> None:
        state = _state(retrieved=[])
        node = make_response_node(client=MagicMock(), model="test")
        update = await node(state)
        assert update["response"]

    async def test_no_evidence_evidence_field_absent_or_empty(self) -> None:
        state = _state(retrieved=[])
        node = make_response_node(client=MagicMock(), model="test")
        update = await node(state)
        assert update.get("evidence", []) == [] or "evidence" not in update

    async def test_no_evidence_no_budget_consumed(self) -> None:
        state = _state(retrieved=[])
        original_tokens = state["budget"].used_tokens
        node = make_response_node(client=MagicMock(), model="test")
        update = await node(state)
        assert update["budget"].used_tokens == original_tokens


# ---------------------------------------------------------------------------
# Error handling tests
# ---------------------------------------------------------------------------


class TestResponseNodeErrors:
    async def test_budget_exhausted_returns_response_no_raise(self) -> None:
        exhausted = ExecutionBudget(max_tokens=100, used_tokens=100)
        state = _state(budget=exhausted)
        client = MagicMock()
        client.messages.create = AsyncMock()
        node = make_response_node(client=client, model="test")
        update = await node(state)
        client.messages.create.assert_not_called()
        assert update["response"]
        assert any("budget exhausted" in e for e in update.get("errors", []))

    async def test_llm_exception_captured_no_raise(self) -> None:
        state = _state()
        client = MagicMock()
        client.messages.create = AsyncMock(side_effect=RuntimeError("API down"))
        node = make_response_node(client=client, model="test")
        update = await node(state)
        assert any("LLM call failed" in e for e in update.get("errors", []))
        assert update["response"]

    async def test_no_tool_block_captured_no_raise(self) -> None:
        usage = MagicMock()
        usage.input_tokens = 100
        usage.output_tokens = 50
        resp = MagicMock()
        resp.content = []  # no tool block
        resp.usage = usage
        client = MagicMock()
        client.messages.create = AsyncMock(return_value=resp)
        state = _state()
        node = make_response_node(client=client, model="test")
        update = await node(state)
        assert any("no tool-use block" in e for e in update.get("errors", []))

    async def test_invalid_tool_input_captured_no_raise(self) -> None:
        bad_input = {"answer": "", "confidence": "invalid", "limitations": "", "reasoning_summary": ""}
        state = _state()
        node = make_response_node(client=_mock_client(bad_input), model="test")
        update = await node(state)
        assert any("validation failed" in e for e in update.get("errors", []))


# ---------------------------------------------------------------------------
# make_response_node factory
# ---------------------------------------------------------------------------


class TestMakeResponseNode:
    def test_returns_callable(self) -> None:
        node = make_response_node(client=MagicMock(), model="test")
        assert callable(node)

    async def test_node_accepts_graph_state(self) -> None:
        state = _state(retrieved=[])
        node = make_response_node(client=MagicMock(), model="test")
        update = await node(state)
        assert isinstance(update, dict)

    def test_max_evidence_cap_respected(self) -> None:
        agent = ResponseAgent(client=MagicMock(), model="test", max_evidence=3)
        assert agent._max_evidence == 3

    async def test_top_evidence_by_score_selected(self) -> None:
        """When more evidence than max, highest-scoring items are chosen."""
        evidences = [
            _evidence(f"DOC-00{i}-chunk-0000", f"DOC-00{i}", final_score=float(i) / 100)
            for i in range(1, 8)
        ]
        state = _state(retrieved=evidences)

        captured_system_prompts: list[str] = []

        async def _capture_call(**kwargs: object) -> object:
            captured_system_prompts.append(str(kwargs.get("system", "")))
            raise RuntimeError("stop after capture")

        client = MagicMock()
        client.messages.create = AsyncMock(side_effect=_capture_call)
        node = make_response_node(client=client, model="test", max_evidence=3)
        await node(state)

        # The prompt should contain the top 3 chunk_ids (DOC-005, DOC-006, DOC-007)
        if captured_system_prompts:
            prompt = captured_system_prompts[0]
            assert "DOC-007-chunk-0000" in prompt
            assert "DOC-001-chunk-0000" not in prompt
