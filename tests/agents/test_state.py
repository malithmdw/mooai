"""Tests for GraphState and ExecutionBudget.

Tests cover:
- Initial state construction and default values
- Field presence and type annotation inspection (correct reducers wired)
- State transitions: append-reducer fields accumulate; replace fields overwrite
- ExecutionBudget: consume(), is_exhausted, immutability
- Serialization: full state round-trips through JSON
"""

from __future__ import annotations

import json
import operator
from datetime import UTC, date, datetime
from typing import Annotated, get_args, get_origin, get_type_hints

import pytest

from src.agents.budget import ExecutionBudget
from src.agents.state import GraphState, initial_state
from src.models.chat import Message
from src.models.documents import DocumentMetadata
from src.models.enums import AccessLevel, AgentState, MessageRole, ResearchStatus, Role
from src.models.evidence import Citation, Evidence
from src.models.research import ResearchResult, ResearchTask
from src.models.tools import ToolCall, ToolResult
from src.models.user import User
from src.models.validation import ValidationResult
from src.retrieval.hybrid.models import RetrievalEvidence, RetrievalSource


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------


def _user(roles: tuple[Role, ...] = (Role.ENGINEER,)) -> User:
    return User(
        user_id="user-0001",
        username="engineer01",
        display_name="Alice Engineer",
        roles=roles,
    )


def _message(index: int = 0, role: MessageRole = MessageRole.USER) -> Message:
    return Message(
        message_id=f"msg-{index:04d}",
        conversation_id="conv-0001",
        role=role,
        content=f"Message {index}",
        created_at=datetime(2024, 1, 1, 0, index, tzinfo=UTC),
    )


def _doc_meta(doc_id: str = "DOC-001") -> DocumentMetadata:
    return DocumentMetadata(
        document_id=doc_id,
        title="Test Doc",
        department="Engineering",
        document_type="runbook",
        access_level=AccessLevel.INTERNAL,
        created_date=date(2024, 1, 1),
        allowed_roles=(Role.ENGINEER,),
    )


def _evidence_item(ev_id: str = "ev-0001") -> Evidence:
    return Evidence(
        evidence_id=ev_id,
        document_id="DOC-001",
        excerpt="Relevant excerpt.",
        relevance_score=0.9,
    )


def _retrieval_evidence(chunk_id: str = "DOC-001-chunk-0000") -> RetrievalEvidence:
    return RetrievalEvidence(
        chunk_id=chunk_id,
        document_id="DOC-001",
        title="Test Doc",
        text="Some content.",
        metadata=_doc_meta(),
        source=RetrievalSource.DENSE,
        dense_score=0.9,
        sparse_score=None,
        final_score=0.016,
        rank=1,
    )


def _tool_call(call_id: str = "call-0001") -> ToolCall:
    return ToolCall(tool_call_id=call_id, tool_name="search_tool")


def _tool_result(call_id: str = "call-0001") -> ToolResult:
    return ToolResult(tool_call_id=call_id, success=True, output="result data")


# ---------------------------------------------------------------------------
# Helper: apply a partial state update using the annotated reducers.
#
# This mirrors what LangGraph does internally and lets tests verify state
# transitions without needing a compiled StateGraph.
# ---------------------------------------------------------------------------


def _apply_update(state: GraphState, update: dict) -> GraphState:
    """Return a new state with ``update`` merged using annotated reducers."""
    hints = get_type_hints(GraphState, include_extras=True)
    result = dict(state)
    for key, value in update.items():
        hint = hints[key]
        if get_origin(hint) is Annotated:
            args = get_args(hint)
            reducer = args[1]
            result[key] = reducer(state[key], value)
        else:
            result[key] = value
    return GraphState(**result)  # type: ignore[typeddict-item]


# ---------------------------------------------------------------------------
# Initial state
# ---------------------------------------------------------------------------


class TestInitialState:
    def test_returns_graph_state_with_all_keys(self) -> None:
        state = initial_state()
        hints = get_type_hints(GraphState, include_extras=True)
        for key in hints:
            assert key in state, f"key '{key}' missing from initial_state()"

    def test_no_user_sets_empty_user_role(self) -> None:
        assert initial_state()["user_role"] == []

    def test_user_populates_user_role(self) -> None:
        user = _user(roles=(Role.ENGINEER, Role.ANALYST))
        state = initial_state(user=user)
        assert set(state["user_role"]) == {Role.ENGINEER, Role.ANALYST}

    def test_user_stored_in_state(self) -> None:
        user = _user()
        assert initial_state(user=user)["user"] is user

    def test_conversation_id_stored(self) -> None:
        state = initial_state(conversation_id="conv-0001")
        assert state["conversation_id"] == "conv-0001"

    def test_all_list_fields_empty(self) -> None:
        state = initial_state()
        list_fields = [
            "messages", "retrieval_queries", "retrieved_documents", "evidence",
            "research_tasks", "research_results", "tool_calls", "tool_results",
            "memory_updates", "validation_results", "citations", "errors",
            "task_plan",
        ]
        for field in list_fields:
            assert state[field] == [], f"field '{field}' should be empty list"

    def test_optional_fields_are_none(self) -> None:
        state = initial_state()
        for field in ("user", "conversation_id", "intent", "current_agent", "current_node", "response"):
            assert state[field] is None, f"field '{field}' should be None"

    def test_default_budget_created(self) -> None:
        state = initial_state()
        assert isinstance(state["budget"], ExecutionBudget)

    def test_custom_budget_used(self) -> None:
        budget = ExecutionBudget(max_tokens=5_000)
        state = initial_state(budget=budget)
        assert state["budget"].max_tokens == 5_000


# ---------------------------------------------------------------------------
# Reducer annotations — verify the schema has correct reducers
# ---------------------------------------------------------------------------


class TestReducerAnnotations:
    """Verify that accumulate fields carry operator.add as their reducer."""

    _ACCUMULATE_FIELDS = {
        "messages",
        "retrieval_queries",
        "retrieved_documents",
        "evidence",
        "research_results",
        "tool_calls",
        "tool_results",
        "memory_updates",
        "validation_results",
        "citations",
        "errors",
    }

    _REPLACE_FIELDS = {
        "user",
        "conversation_id",
        "user_role",
        "intent",
        "task_plan",
        "research_tasks",
        "current_agent",
        "current_node",
        "response",
        "budget",
    }

    def test_accumulate_fields_have_add_reducer(self) -> None:
        hints = get_type_hints(GraphState, include_extras=True)
        for field in self._ACCUMULATE_FIELDS:
            hint = hints[field]
            assert get_origin(hint) is Annotated, (
                f"field '{field}' should be Annotated"
            )
            reducer = get_args(hint)[1]
            assert reducer is operator.add, (
                f"field '{field}' should have operator.add reducer, got {reducer}"
            )

    def test_replace_fields_are_not_annotated(self) -> None:
        hints = get_type_hints(GraphState, include_extras=True)
        for field in self._REPLACE_FIELDS:
            hint = hints[field]
            assert get_origin(hint) is not Annotated, (
                f"field '{field}' should NOT be Annotated (replace semantics)"
            )

    def test_all_fields_accounted_for(self) -> None:
        hints = get_type_hints(GraphState, include_extras=True)
        all_declared = set(hints.keys())
        covered = self._ACCUMULATE_FIELDS | self._REPLACE_FIELDS
        assert all_declared == covered, (
            f"Uncategorised fields: {all_declared - covered}"
        )


# ---------------------------------------------------------------------------
# State transitions — accumulate fields
# ---------------------------------------------------------------------------


class TestAccumulateTransitions:
    def test_messages_append(self) -> None:
        state = initial_state()
        msg1 = _message(0)
        msg2 = _message(1)
        state = _apply_update(state, {"messages": [msg1]})
        state = _apply_update(state, {"messages": [msg2]})
        assert state["messages"] == [msg1, msg2]

    def test_errors_append_across_nodes(self) -> None:
        state = initial_state()
        state = _apply_update(state, {"errors": ["retrieval timeout"]})
        state = _apply_update(state, {"errors": ["tool call failed"]})
        assert state["errors"] == ["retrieval timeout", "tool call failed"]

    def test_retrieval_queries_append(self) -> None:
        state = initial_state()
        state = _apply_update(state, {"retrieval_queries": ["query A"]})
        state = _apply_update(state, {"retrieval_queries": ["query B"]})
        assert state["retrieval_queries"] == ["query A", "query B"]

    def test_retrieved_documents_append(self) -> None:
        state = initial_state()
        doc1 = _retrieval_evidence("DOC-001-chunk-0000")
        doc2 = _retrieval_evidence("DOC-002-chunk-0000")
        state = _apply_update(state, {"retrieved_documents": [doc1]})
        state = _apply_update(state, {"retrieved_documents": [doc2]})
        assert len(state["retrieved_documents"]) == 2

    def test_evidence_accumulates(self) -> None:
        state = initial_state()
        ev1 = _evidence_item("ev-0001")
        ev2 = _evidence_item("ev-0002")
        state = _apply_update(state, {"evidence": [ev1]})
        state = _apply_update(state, {"evidence": [ev2]})
        assert {e.evidence_id for e in state["evidence"]} == {"ev-0001", "ev-0002"}

    def test_tool_calls_accumulate(self) -> None:
        state = initial_state()
        tc1 = _tool_call("call-0001")
        tc2 = _tool_call("call-0002")
        state = _apply_update(state, {"tool_calls": [tc1]})
        state = _apply_update(state, {"tool_calls": [tc2]})
        assert len(state["tool_calls"]) == 2

    def test_tool_results_accumulate(self) -> None:
        state = initial_state()
        tr1 = _tool_result("call-0001")
        tr2 = _tool_result("call-0002")
        state = _apply_update(state, {"tool_results": [tr1, tr2]})
        assert len(state["tool_results"]) == 2

    def test_memory_updates_append(self) -> None:
        state = initial_state()
        state = _apply_update(state, {"memory_updates": ["saved message msg-0001"]})
        state = _apply_update(state, {"memory_updates": ["saved summary sum-0001"]})
        assert len(state["memory_updates"]) == 2

    def test_citations_accumulate(self) -> None:
        state = initial_state()
        c1 = Citation(evidence_id="ev-0001", reference_number=1)
        c2 = Citation(evidence_id="ev-0002", reference_number=2)
        state = _apply_update(state, {"citations": [c1]})
        state = _apply_update(state, {"citations": [c2]})
        assert len(state["citations"]) == 2

    def test_validation_results_accumulate(self) -> None:
        state = initial_state()
        vr = ValidationResult(is_valid=True)
        state = _apply_update(state, {"validation_results": [vr]})
        assert state["validation_results"] == [vr]

    def test_research_results_accumulate(self) -> None:
        state = initial_state()
        rr = ResearchResult(task_id="task-0001", summary="Findings")
        state = _apply_update(state, {"research_results": [rr]})
        assert state["research_results"] == [rr]

    def test_empty_update_changes_nothing(self) -> None:
        state = initial_state()
        updated = _apply_update(state, {"messages": []})
        assert updated["messages"] == []


# ---------------------------------------------------------------------------
# State transitions — replace fields
# ---------------------------------------------------------------------------


class TestReplaceTransitions:
    def test_intent_replaced(self) -> None:
        state = initial_state()
        state = _apply_update(state, {"intent": "search for FPS runbooks"})
        assert state["intent"] == "search for FPS runbooks"
        state = _apply_update(state, {"intent": "summarize findings"})
        assert state["intent"] == "summarize findings"

    def test_task_plan_replaced_atomically(self) -> None:
        state = initial_state()
        state = _apply_update(state, {"task_plan": ["step 1", "step 2"]})
        state = _apply_update(state, {"task_plan": ["revised step 1"]})
        assert state["task_plan"] == ["revised step 1"]

    def test_current_agent_replaced(self) -> None:
        state = initial_state()
        state = _apply_update(state, {"current_agent": AgentState.RETRIEVAL})
        assert state["current_agent"] == AgentState.RETRIEVAL
        state = _apply_update(state, {"current_agent": AgentState.RESEARCH})
        assert state["current_agent"] == AgentState.RESEARCH

    def test_current_node_replaced(self) -> None:
        state = initial_state()
        state = _apply_update(state, {"current_node": "retrieval_node"})
        state = _apply_update(state, {"current_node": "research_node"})
        assert state["current_node"] == "research_node"

    def test_response_replaced(self) -> None:
        state = initial_state()
        state = _apply_update(state, {"response": "Initial draft."})
        state = _apply_update(state, {"response": "Revised answer."})
        assert state["response"] == "Revised answer."

    def test_budget_replaced_on_consume(self) -> None:
        state = initial_state()
        original_budget = state["budget"]
        updated_budget = original_budget.consume(tokens=500, tool_calls=1)
        state = _apply_update(state, {"budget": updated_budget})
        assert state["budget"].used_tokens == 500
        assert state["budget"].used_tool_calls == 1
        assert original_budget.used_tokens == 0  # original unchanged

    def test_research_tasks_replaced(self) -> None:
        state = initial_state()
        task = ResearchTask(
            task_id="task-0001",
            conversation_id="conv-0001",
            objective="Find FPS docs",
        )
        state = _apply_update(state, {"research_tasks": [task]})
        assert len(state["research_tasks"]) == 1

        task_updated = ResearchTask(
            task_id="task-0001",
            conversation_id="conv-0001",
            objective="Find FPS docs",
            status=ResearchStatus.COMPLETED,
        )
        state = _apply_update(state, {"research_tasks": [task_updated]})
        assert state["research_tasks"][0].status == ResearchStatus.COMPLETED
        assert len(state["research_tasks"]) == 1  # replaced, not appended


# ---------------------------------------------------------------------------
# Multi-field updates — simultaneous partial state updates
# ---------------------------------------------------------------------------


class TestMultiFieldUpdates:
    def test_retrieval_node_output(self) -> None:
        """Simulate a retrieval node updating multiple fields at once."""
        state = initial_state()
        query = "FPS error handling"
        doc = _retrieval_evidence()
        budget_after = state["budget"].consume(retrieval_results=1)

        state = _apply_update(state, {
            "retrieval_queries": [query],
            "retrieved_documents": [doc],
            "current_agent": AgentState.RETRIEVAL,
            "budget": budget_after,
        })

        assert state["retrieval_queries"] == [query]
        assert len(state["retrieved_documents"]) == 1
        assert state["current_agent"] == AgentState.RETRIEVAL
        assert state["budget"].used_retrieval_results == 1

    def test_supervisor_planning_output(self) -> None:
        """Simulate the supervisor node setting intent and task plan."""
        state = initial_state()
        state = _apply_update(state, {
            "intent": "explain HMAC authentication flow",
            "task_plan": ["retrieve auth docs", "synthesise answer"],
            "current_agent": AgentState.SUPERVISOR,
            "current_node": "supervisor_node",
        })

        assert state["intent"] == "explain HMAC authentication flow"
        assert state["task_plan"] == ["retrieve auth docs", "synthesise answer"]

    def test_errors_do_not_clear_existing_state(self) -> None:
        """Errors accumulate; other fields are untouched."""
        state = initial_state()
        msg = _message(0)
        state = _apply_update(state, {"messages": [msg]})
        state = _apply_update(state, {"errors": ["retrieval failed"]})
        assert state["messages"] == [msg]
        assert state["errors"] == ["retrieval failed"]


# ---------------------------------------------------------------------------
# ExecutionBudget
# ---------------------------------------------------------------------------


class TestExecutionBudget:
    def test_default_limits(self) -> None:
        b = ExecutionBudget()
        assert b.max_tokens == 10_000
        assert b.max_tool_calls == 10
        assert b.max_retrieval_results == 20
        assert b.max_research_depth == 3

    def test_default_usage_is_zero(self) -> None:
        b = ExecutionBudget()
        assert b.used_tokens == 0
        assert b.used_tool_calls == 0
        assert b.used_retrieval_results == 0
        assert b.used_research_steps == 0

    def test_consume_returns_new_instance(self) -> None:
        b = ExecutionBudget()
        b2 = b.consume(tokens=100)
        assert b2 is not b

    def test_consume_does_not_mutate_original(self) -> None:
        b = ExecutionBudget()
        b.consume(tokens=500, tool_calls=3)
        assert b.used_tokens == 0
        assert b.used_tool_calls == 0

    def test_consume_accumulates_tokens(self) -> None:
        b = ExecutionBudget().consume(tokens=300).consume(tokens=200)
        assert b.used_tokens == 500

    def test_consume_accumulates_tool_calls(self) -> None:
        b = ExecutionBudget().consume(tool_calls=2).consume(tool_calls=3)
        assert b.used_tool_calls == 5

    def test_consume_accumulates_retrieval_results(self) -> None:
        b = ExecutionBudget().consume(retrieval_results=10)
        assert b.used_retrieval_results == 10

    def test_consume_accumulates_research_steps(self) -> None:
        b = ExecutionBudget().consume(research_steps=2)
        assert b.used_research_steps == 2

    def test_remaining_tokens_decreases_on_consume(self) -> None:
        b = ExecutionBudget(max_tokens=1_000).consume(tokens=400)
        assert b.remaining_tokens == 600

    def test_remaining_tokens_floors_at_zero(self) -> None:
        b = ExecutionBudget(max_tokens=100).consume(tokens=200)
        assert b.remaining_tokens == 0

    def test_remaining_tool_calls_decreases(self) -> None:
        b = ExecutionBudget(max_tool_calls=5).consume(tool_calls=3)
        assert b.remaining_tool_calls == 2

    def test_is_not_exhausted_at_start(self) -> None:
        assert not ExecutionBudget().is_exhausted

    def test_is_token_exhausted_when_at_limit(self) -> None:
        b = ExecutionBudget(max_tokens=100).consume(tokens=100)
        assert b.is_token_exhausted
        assert b.is_exhausted

    def test_is_tool_exhausted_when_at_limit(self) -> None:
        b = ExecutionBudget(max_tool_calls=3).consume(tool_calls=3)
        assert b.is_tool_exhausted
        assert b.is_exhausted

    def test_one_exhausted_budget_triggers_is_exhausted(self) -> None:
        b = ExecutionBudget(max_tokens=1_000, max_tool_calls=2).consume(tool_calls=2)
        assert b.is_exhausted
        assert not b.is_token_exhausted  # tokens still available

    def test_custom_limits(self) -> None:
        b = ExecutionBudget(max_tokens=500, max_tool_calls=5)
        assert b.max_tokens == 500
        assert b.max_tool_calls == 5

    def test_negative_usage_raises(self) -> None:
        with pytest.raises(Exception):
            ExecutionBudget(used_tokens=-1)


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------


class TestSerialization:
    def _json_roundtrip(self, obj: object) -> str:
        """Serialize to JSON; fails the test if any value is not serializable."""
        return json.dumps(
            obj,
            default=lambda x: (
                x.model_dump() if hasattr(x, "model_dump")
                else x.value if hasattr(x, "value")  # StrEnum
                else str(x)
            ),
        )

    def test_initial_state_is_json_serializable(self) -> None:
        state = initial_state()
        serialized = self._json_roundtrip(state)
        assert isinstance(serialized, str)
        assert len(serialized) > 0

    def test_state_with_user_is_serializable(self) -> None:
        state = initial_state(user=_user(), conversation_id="conv-0001")
        self._json_roundtrip(state)

    def test_state_with_messages_is_serializable(self) -> None:
        state = initial_state()
        state = _apply_update(state, {"messages": [_message(0), _message(1)]})
        self._json_roundtrip(state)

    def test_state_with_retrieved_documents_is_serializable(self) -> None:
        state = initial_state()
        state = _apply_update(state, {"retrieved_documents": [_retrieval_evidence()]})
        self._json_roundtrip(state)

    def test_state_with_research_tasks_is_serializable(self) -> None:
        task = ResearchTask(
            task_id="task-0001",
            conversation_id="conv-0001",
            objective="Research FPS auth",
        )
        state = initial_state()
        state = _apply_update(state, {"research_tasks": [task]})
        self._json_roundtrip(state)

    def test_state_with_tool_calls_and_results_is_serializable(self) -> None:
        state = initial_state()
        tc = _tool_call()
        tr = _tool_result()
        state = _apply_update(state, {"tool_calls": [tc], "tool_results": [tr]})
        self._json_roundtrip(state)

    def test_budget_model_dump_round_trips(self) -> None:
        b = ExecutionBudget(max_tokens=5_000).consume(tokens=1_000, tool_calls=2)
        data = b.model_dump()
        b2 = ExecutionBudget(**data)
        assert b2.used_tokens == b.used_tokens
        assert b2.used_tool_calls == b.used_tool_calls
        assert b2.max_tokens == b.max_tokens

    def test_fully_populated_state_is_serializable(self) -> None:
        """Smoke test: a state with every field populated serializes without error."""
        user = _user()
        state = initial_state(user=user, conversation_id="conv-0001")
        state = _apply_update(state, {
            "messages": [_message(0), _message(1, MessageRole.ASSISTANT)],
            "intent": "explain authentication",
            "task_plan": ["retrieve", "synthesise"],
            "retrieval_queries": ["authentication flow"],
            "retrieved_documents": [_retrieval_evidence()],
            "evidence": [_evidence_item()],
            "tool_calls": [_tool_call()],
            "tool_results": [_tool_result()],
            "memory_updates": ["saved message"],
            "validation_results": [ValidationResult(is_valid=True)],
            "current_agent": AgentState.RETRIEVAL,
            "current_node": "retrieval_node",
            "response": "Authentication uses HMAC-SHA256.",
            "citations": [Citation(evidence_id="ev-0001", reference_number=1)],
            "errors": [],
            "budget": state["budget"].consume(tokens=300, retrieval_results=1),
        })
        self._json_roundtrip(state)
