"""Tests for the Retrieval Agent node.

Covers:
- Query generation: primary query from user message, plan-derived sub-queries
  for analytical_research / mixed_request, de-duplication, max-query cap
- Access-level derivation from RBAC roles
- Successful retrieval: state update, RBAC parameters forwarded, budget consumed
- Deduplication: same chunk from multiple queries kept once (highest score)
- Insufficient evidence detection
- Budget exhaustion short-circuit
- Retriever exception captured as error (no raise)
- No user messages returns error (no raise)
- AgentEvent emission (verified via caplog)
"""

from __future__ import annotations

from datetime import date
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.agents.budget import ExecutionBudget
from src.agents.retrieval.node import (
    DEFAULT_TOP_K,
    INSUFFICIENT_EVIDENCE_THRESHOLD,
    RetrievalAgent,
    _access_levels_for_roles,
    make_retrieval_node,
)
from src.agents.retrieval.queries import generate_queries
from src.agents.state import GraphState, initial_state
from src.models.chat import Message
from src.models.documents import DocumentMetadata
from src.models.enums import AccessLevel, AgentState, MessageRole, Role
from src.models.user import User
from src.retrieval.hybrid.models import RetrievalEvidence, RetrievalSource


# ---------------------------------------------------------------------------
# Test fixtures / helpers
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
    allowed_roles: tuple[Role, ...] = (Role.ANALYST, Role.ENGINEER),
) -> DocumentMetadata:
    return DocumentMetadata(
        document_id=document_id,
        title="Test Document",
        department="Finance",
        document_type="report",
        access_level=access_level,
        created_date=date(2024, 1, 1),
        allowed_roles=allowed_roles,
    )


def _evidence(
    chunk_id: str = "DOC-001-chunk-0000",
    document_id: str = "DOC-001",
    final_score: float = 0.9,
    rank: int = 1,
    access_level: AccessLevel = AccessLevel.INTERNAL,
    allowed_roles: tuple[Role, ...] = (Role.ANALYST, Role.ENGINEER),
) -> RetrievalEvidence:
    return RetrievalEvidence(
        chunk_id=chunk_id,
        document_id=document_id,
        title="Test Document",
        # text is untrusted data — use a neutral placeholder in tests
        text="document content placeholder",
        metadata=_meta(document_id, access_level, allowed_roles),
        source=RetrievalSource.BOTH,
        dense_score=0.8,
        sparse_score=0.7,
        final_score=final_score,
        rank=rank,
    )


def _state(
    content: str = "What is the loan default rate?",
    *,
    roles: tuple[Role, ...] = (Role.ANALYST,),
    intent: str | None = "knowledge_question",
    task_plan: list[str] | None = None,
    budget: ExecutionBudget | None = None,
) -> GraphState:
    state = initial_state(
        user=_user(roles),
        conversation_id="conv-0001",
        budget=budget,
    )
    state["messages"] = [_message(content)]
    state["intent"] = intent
    state["task_plan"] = task_plan or []
    return state


def _mock_retriever(results: list[RetrievalEvidence] | None = None) -> MagicMock:
    retriever = MagicMock()
    retriever.search = AsyncMock(return_value=results or [_evidence()])
    return retriever


# ---------------------------------------------------------------------------
# generate_queries tests
# ---------------------------------------------------------------------------


class TestGenerateQueries:
    def test_user_message_is_primary_query(self) -> None:
        state = _state("What is the default rate?")
        assert generate_queries(state) == ["What is the default rate?"]

    def test_no_user_messages_returns_empty(self) -> None:
        state = _state()
        state["messages"] = [_message("system note", role=MessageRole.SYSTEM)]
        assert generate_queries(state) == []

    def test_empty_messages_returns_empty(self) -> None:
        state = _state()
        state["messages"] = []
        assert generate_queries(state) == []

    def test_knowledge_question_only_one_query(self) -> None:
        state = _state(
            "What is the default rate?",
            intent="knowledge_question",
            task_plan=["Retrieve docs", "Summarise", "Respond"],
        )
        assert generate_queries(state) == ["What is the default rate?"]

    def test_analytical_research_adds_plan_queries(self) -> None:
        state = _state(
            "Analyse default trends",
            intent="analytical_research",
            task_plan=["Break down sub-questions", "Research each"],
        )
        result = generate_queries(state)
        assert result[0] == "Analyse default trends"
        assert len(result) > 1

    def test_mixed_request_adds_plan_queries(self) -> None:
        state = _state(
            "Give me a report",
            intent="mixed_request",
            task_plan=["Retrieve loan data", "Run analytics"],
        )
        result = generate_queries(state)
        assert len(result) >= 2

    def test_queries_capped_at_max(self) -> None:
        state = _state(
            "Primary question",
            intent="analytical_research",
            task_plan=["step1", "step2", "step3", "step4", "step5"],
        )
        result = generate_queries(state)
        assert len(result) <= 3

    def test_plan_step_identical_to_primary_skipped(self) -> None:
        state = _state(
            "Primary question",
            intent="analytical_research",
            task_plan=["Primary question", "Different step"],
        )
        result = generate_queries(state)
        assert result.count("Primary question") == 1

    def test_case_insensitive_dedup(self) -> None:
        state = _state(
            "Primary Question",
            intent="analytical_research",
            task_plan=["primary question"],
        )
        result = generate_queries(state)
        assert len(result) == 1

    def test_uses_last_user_message(self) -> None:
        state = _state()
        state["messages"] = [
            _message("First question", index=1),
            _message("Second question", index=2),
        ]
        result = generate_queries(state)
        assert result[0] == "Second question"


# ---------------------------------------------------------------------------
# _access_levels_for_roles tests
# ---------------------------------------------------------------------------


class TestAccessLevelsForRoles:
    def test_viewer_gets_public_and_internal(self) -> None:
        levels = set(_access_levels_for_roles([Role.VIEWER]))
        assert AccessLevel.PUBLIC in levels
        assert AccessLevel.INTERNAL in levels
        assert AccessLevel.CONFIDENTIAL not in levels
        assert AccessLevel.RESTRICTED not in levels

    def test_engineer_gets_public_and_internal(self) -> None:
        levels = set(_access_levels_for_roles([Role.ENGINEER]))
        assert AccessLevel.PUBLIC in levels
        assert AccessLevel.INTERNAL in levels
        assert AccessLevel.CONFIDENTIAL not in levels

    def test_analyst_gets_confidential(self) -> None:
        levels = set(_access_levels_for_roles([Role.ANALYST]))
        assert AccessLevel.CONFIDENTIAL in levels
        assert AccessLevel.RESTRICTED not in levels

    def test_administrator_gets_all_levels(self) -> None:
        levels = set(_access_levels_for_roles([Role.ADMINISTRATOR]))
        assert levels == set(AccessLevel)

    def test_empty_roles_returns_empty(self) -> None:
        assert _access_levels_for_roles([]) == []

    def test_multi_role_union(self) -> None:
        levels = set(_access_levels_for_roles([Role.VIEWER, Role.ANALYST]))
        assert AccessLevel.CONFIDENTIAL in levels
        assert AccessLevel.PUBLIC in levels

    def test_unknown_role_contributes_nothing(self) -> None:
        # ENGINEER is defined; ensure it doesn't grant CONFIDENTIAL
        levels = set(_access_levels_for_roles([Role.ENGINEER]))
        assert AccessLevel.CONFIDENTIAL not in levels


# ---------------------------------------------------------------------------
# RetrievalAgent node — success paths
# ---------------------------------------------------------------------------


class TestRetrievalNodeSuccess:
    async def test_retrieved_documents_in_state(self) -> None:
        ev = _evidence()
        retriever = _mock_retriever([ev])
        state = _state()
        node = make_retrieval_node(retriever=retriever)
        update = await node(state)

        assert ev in update["retrieved_documents"]

    async def test_queries_in_state(self) -> None:
        retriever = _mock_retriever()
        state = _state("What is the default rate?")
        node = make_retrieval_node(retriever=retriever)
        update = await node(state)

        assert "What is the default rate?" in update["retrieval_queries"]

    async def test_rbac_roles_forwarded_to_retriever(self) -> None:
        retriever = _mock_retriever()
        state = _state(roles=(Role.ANALYST,))
        node = make_retrieval_node(retriever=retriever)
        await node(state)

        call_kwargs = retriever.search.call_args.kwargs
        assert Role.ANALYST in call_kwargs["roles"]

    async def test_access_levels_forwarded_to_retriever(self) -> None:
        retriever = _mock_retriever()
        state = _state(roles=(Role.ANALYST,))
        node = make_retrieval_node(retriever=retriever)
        await node(state)

        call_kwargs = retriever.search.call_args.kwargs
        levels = call_kwargs.get("access_levels") or []
        assert AccessLevel.CONFIDENTIAL in levels

    async def test_viewer_access_levels_exclude_confidential(self) -> None:
        retriever = _mock_retriever()
        state = _state(roles=(Role.VIEWER,))
        node = make_retrieval_node(retriever=retriever)
        await node(state)

        call_kwargs = retriever.search.call_args.kwargs
        levels = call_kwargs.get("access_levels") or []
        assert AccessLevel.CONFIDENTIAL not in levels
        assert AccessLevel.RESTRICTED not in levels

    async def test_budget_consumed_by_result_count(self) -> None:
        results = [_evidence(f"DOC-00{i}-chunk-0000", f"DOC-00{i}") for i in range(5)]
        retriever = _mock_retriever(results)
        state = _state()
        node = make_retrieval_node(retriever=retriever)
        update = await node(state)

        assert update["budget"].used_retrieval_results == 5

    async def test_current_agent_set(self) -> None:
        retriever = _mock_retriever()
        node = make_retrieval_node(retriever=retriever)
        update = await node(_state())
        assert update["current_agent"] == AgentState.RETRIEVAL

    async def test_current_node_set(self) -> None:
        retriever = _mock_retriever()
        node = make_retrieval_node(retriever=retriever)
        update = await node(_state())
        assert update["current_node"] == "retrieval"

    async def test_no_errors_when_results_returned(self) -> None:
        retriever = _mock_retriever([_evidence()])
        node = make_retrieval_node(retriever=retriever, min_evidence=1)
        update = await node(_state())
        assert "errors" not in update

    async def test_multiple_queries_for_analytical_research(self) -> None:
        retriever = _mock_retriever()
        state = _state(
            "Analyse default trends",
            intent="analytical_research",
            task_plan=["Research sub-question A", "Research sub-question B"],
        )
        node = make_retrieval_node(retriever=retriever)
        await node(state)

        assert retriever.search.call_count > 1

    async def test_deduplication_keeps_highest_score(self) -> None:
        low = _evidence("DOC-001-chunk-0000", "DOC-001", final_score=0.5)
        high = _evidence("DOC-001-chunk-0000", "DOC-001", final_score=0.9)

        call_count = 0

        async def _side_effect(*_args: object, **_kwargs: object) -> list[RetrievalEvidence]:
            nonlocal call_count
            call_count += 1
            return [low] if call_count == 1 else [high]

        retriever = MagicMock()
        retriever.search = AsyncMock(side_effect=_side_effect)

        state = _state(
            "question",
            intent="analytical_research",
            task_plan=["step one"],
        )
        node = make_retrieval_node(retriever=retriever)
        update = await node(state)

        unique = [e for e in update["retrieved_documents"] if e.chunk_id == "DOC-001-chunk-0000"]
        assert len(unique) == 1
        assert unique[0].final_score == 0.9

    async def test_results_sorted_by_score_descending(self) -> None:
        results = [
            _evidence("DOC-001-chunk-0000", "DOC-001", final_score=0.5, rank=2),
            _evidence("DOC-002-chunk-0000", "DOC-002", final_score=0.9, rank=1),
        ]
        retriever = _mock_retriever(results)
        state = _state()
        node = make_retrieval_node(retriever=retriever)
        update = await node(state)

        scores = [e.final_score for e in update["retrieved_documents"]]
        assert scores == sorted(scores, reverse=True)


# ---------------------------------------------------------------------------
# RetrievalAgent node — error / edge-case paths
# ---------------------------------------------------------------------------


class TestRetrievalNodeErrors:
    async def test_budget_exhausted_skips_retriever(self) -> None:
        exhausted = ExecutionBudget(max_retrieval_results=5, used_retrieval_results=5)
        retriever = _mock_retriever()
        state = _state(budget=exhausted)
        node = make_retrieval_node(retriever=retriever)
        update = await node(state)

        retriever.search.assert_not_called()
        assert any("budget exhausted" in e for e in update["errors"])

    async def test_budget_exhausted_does_not_raise(self) -> None:
        exhausted = ExecutionBudget(max_retrieval_results=5, used_retrieval_results=5)
        retriever = _mock_retriever()
        state = _state(budget=exhausted)
        node = make_retrieval_node(retriever=retriever)
        update = await node(state)
        assert isinstance(update, dict)

    async def test_no_user_messages_returns_error(self) -> None:
        retriever = _mock_retriever()
        state = _state()
        state["messages"] = []
        node = make_retrieval_node(retriever=retriever)
        update = await node(state)

        retriever.search.assert_not_called()
        assert any("no queries" in e for e in update["errors"])

    async def test_retriever_exception_captured_as_error(self) -> None:
        retriever = MagicMock()
        retriever.search = AsyncMock(side_effect=RuntimeError("Pinecone timeout"))
        state = _state()
        node = make_retrieval_node(retriever=retriever)
        update = await node(state)

        assert any("search failed" in e for e in update["errors"])
        assert any("Pinecone timeout" in e for e in update["errors"])

    async def test_retriever_exception_does_not_raise(self) -> None:
        retriever = MagicMock()
        retriever.search = AsyncMock(side_effect=Exception("catastrophic"))
        state = _state()
        node = make_retrieval_node(retriever=retriever)
        update = await node(state)
        assert isinstance(update, dict)

    async def test_insufficient_evidence_adds_error(self) -> None:
        retriever = _mock_retriever([])  # empty results
        state = _state()
        node = make_retrieval_node(retriever=retriever, min_evidence=1)
        update = await node(state)

        assert any("insufficient evidence" in e for e in update["errors"])

    async def test_sufficient_evidence_no_error(self) -> None:
        retriever = _mock_retriever([_evidence()])
        state = _state()
        node = make_retrieval_node(retriever=retriever, min_evidence=1)
        update = await node(state)

        assert "errors" not in update

    async def test_budget_preserved_on_budget_exhaustion(self) -> None:
        exhausted = ExecutionBudget(max_retrieval_results=5, used_retrieval_results=5)
        retriever = _mock_retriever()
        state = _state(budget=exhausted)
        node = make_retrieval_node(retriever=retriever)
        update = await node(state)

        assert update["budget"] is exhausted


# ---------------------------------------------------------------------------
# AgentEvent emission (via structured logging)
# ---------------------------------------------------------------------------


class TestRetrievalEvents:
    async def test_query_generation_event_logged(self, caplog: pytest.LogCaptureFixture) -> None:
        retriever = _mock_retriever()
        state = _state()
        node = make_retrieval_node(retriever=retriever)

        with caplog.at_level("INFO", logger="src.agents.retrieval.node"):
            await node(state)

        assert any("query_generation" in msg for msg in caplog.messages)

    async def test_dense_search_event_logged(self, caplog: pytest.LogCaptureFixture) -> None:
        retriever = _mock_retriever()
        state = _state()
        node = make_retrieval_node(retriever=retriever)

        with caplog.at_level("INFO", logger="src.agents.retrieval.node"):
            await node(state)

        assert any("dense_search" in msg for msg in caplog.messages)

    async def test_sparse_search_event_logged(self, caplog: pytest.LogCaptureFixture) -> None:
        retriever = _mock_retriever()
        state = _state()
        node = make_retrieval_node(retriever=retriever)

        with caplog.at_level("INFO", logger="src.agents.retrieval.node"):
            await node(state)

        assert any("sparse_search" in msg for msg in caplog.messages)

    async def test_ranking_event_logged(self, caplog: pytest.LogCaptureFixture) -> None:
        retriever = _mock_retriever()
        state = _state()
        node = make_retrieval_node(retriever=retriever)

        with caplog.at_level("INFO", logger="src.agents.retrieval.node"):
            await node(state)

        assert any("ranking" in msg for msg in caplog.messages)

    async def test_filtering_event_logged(self, caplog: pytest.LogCaptureFixture) -> None:
        retriever = _mock_retriever()
        state = _state()
        node = make_retrieval_node(retriever=retriever)

        with caplog.at_level("INFO", logger="src.agents.retrieval.node"):
            await node(state)

        assert any("filtering" in msg for msg in caplog.messages)

    async def test_final_evidence_event_logged(self, caplog: pytest.LogCaptureFixture) -> None:
        retriever = _mock_retriever()
        state = _state()
        node = make_retrieval_node(retriever=retriever)

        with caplog.at_level("INFO", logger="src.agents.retrieval.node"):
            await node(state)

        assert any("final_evidence" in msg for msg in caplog.messages)


# ---------------------------------------------------------------------------
# make_retrieval_node factory
# ---------------------------------------------------------------------------


class TestMakeRetrievalNode:
    def test_returns_callable(self) -> None:
        node = make_retrieval_node(retriever=_mock_retriever())
        assert callable(node)

    async def test_node_accepts_graph_state(self) -> None:
        node = make_retrieval_node(retriever=_mock_retriever())
        update = await node(_state())
        assert isinstance(update, dict)

    def test_custom_top_k_respected(self) -> None:
        agent = RetrievalAgent(retriever=_mock_retriever(), top_k=5)
        assert agent._top_k == 5

    def test_custom_min_evidence_respected(self) -> None:
        agent = RetrievalAgent(retriever=_mock_retriever(), min_evidence=3)
        assert agent._min_evidence == 3
