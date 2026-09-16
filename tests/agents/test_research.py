"""Tests for the Research Agent — recursive, RLM-style multi-step research.

Coverage
--------
Pure helpers:
  - _partition_into_batches: count/char-size bounded batching
  - _to_evidence: truncation and score clamping
  - _merge_evidence: dedup, ranking, cap
  - _detect_contradictions: grouping, normalization, no false positives
  - _build_search_plan: dedup, cap, root question always first
  - _extract_root_question: last user message, None when absent

ResearchAgent.run() end to end (mocked Anthropic client + HybridRetriever):
  - simple single-batch, single-sub-question run (no recursion needed)
  - genuine recursion: multi-batch evidence forces child tasks + aggregation
  - depth budget forces truncation instead of further recursion
  - max_total_tasks caps semantic fan-out (defense in depth)
  - no evidence found -> short-circuit result
  - question-analysis LLM failure -> captured as error, no exception
  - final-synthesis failure -> partial state preserved, captured as error
  - budget-exhausted / no-question short-circuits -> no LLM calls at all
  - cross-branch contradiction detection (two sub-questions disagree)
  - AgentEvents: research plan, task creation, recursion depth, batch
    processing, aggregation (via caplog)

A worked example using payment incident reports (INC-2024-001 / INC-2024-002
style data) lives in TestPaymentIncidentExample and mirrors docs/rlm.md.
"""

from __future__ import annotations

from datetime import date
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.agents.budget import ExecutionBudget
from src.agents.research.models import Claim, QuestionAnalysis
from src.agents.research.node import (
    ResearchAgent,
    _build_search_plan,
    _detect_contradictions,
    _extract_root_question,
    _merge_evidence,
    _partition_into_batches,
    _RunContext,
    _to_evidence,
    make_research_node,
)
from src.agents.state import GraphState, initial_state
from src.models.chat import Message
from src.models.documents import DocumentMetadata
from src.models.enums import AccessLevel, MessageRole, ResearchStatus, Role
from src.models.research import ResearchResult, ResearchTask
from src.models.user import User
from src.retrieval.hybrid.models import RetrievalEvidence, RetrievalSource
from src.retrieval.hybrid.retriever import HybridRetriever

# ---------------------------------------------------------------------------
# Shared test helpers
# ---------------------------------------------------------------------------


def _user(roles: tuple[Role, ...] = (Role.ANALYST,)) -> User:
    return User(user_id="user-0001", username="analyst01", display_name="Alice", roles=roles)


def _message(content: str, *, index: int = 1, role: MessageRole = MessageRole.USER) -> Message:
    return Message(
        message_id=f"msg-{index:04d}", conversation_id="conv-0001", role=role, content=content
    )


def _meta(document_id: str = "INC-2024-001") -> DocumentMetadata:
    return DocumentMetadata(
        document_id=document_id,
        title="Incident Report",
        department="Payments Engineering",
        document_type="incident_report",
        access_level=AccessLevel.INTERNAL,
        created_date=date(2024, 1, 15),
        allowed_roles=(Role.ANALYST, Role.ADMINISTRATOR),
    )


def _evidence(
    chunk_id: str,
    *,
    document_id: str = "INC-2024-001",
    text: str = "A connection pool exhaustion caused cascading timeouts.",
    final_score: float = 0.8,
    rank: int = 1,
) -> RetrievalEvidence:
    return RetrievalEvidence(
        chunk_id=chunk_id,
        document_id=document_id,
        title="Payment Gateway Timeout Cascade",
        text=text,
        metadata=_meta(document_id),
        source=RetrievalSource.BOTH,
        dense_score=0.8,
        sparse_score=0.7,
        final_score=final_score,
        rank=rank,
    )


def _state(
    question: str = "What caused the payment gateway incidents?", **kwargs: object
) -> GraphState:
    state = initial_state(user=_user(), conversation_id="conv-0001", **kwargs)  # type: ignore[arg-type]
    state["messages"] = [_message(question)]
    return state


def _tool_block(tool_name: str, tool_input: dict) -> MagicMock:  # type: ignore[type-arg]
    block = MagicMock()
    block.type = "tool_use"
    block.name = tool_name
    block.input = tool_input
    return block


def _mock_response(
    tool_name: str,
    tool_input: dict,  # type: ignore[type-arg]
    *,
    input_tokens: int = 100,
    output_tokens: int = 50,
) -> MagicMock:  # type: ignore[type-arg]
    usage = MagicMock()
    usage.input_tokens = input_tokens
    usage.output_tokens = output_tokens
    resp = MagicMock()
    resp.content = [_tool_block(tool_name, tool_input)]
    resp.usage = usage
    return resp


def _no_tool_block_response() -> MagicMock:  # type: ignore[type-arg]
    usage = MagicMock()
    usage.input_tokens = 10
    usage.output_tokens = 5
    resp = MagicMock()
    resp.content = []
    resp.usage = usage
    return resp


def _mock_client(responses: list) -> MagicMock:  # type: ignore[type-arg]
    client = MagicMock()
    client.messages.create = AsyncMock(side_effect=responses)
    return client


def _mock_retriever(pool: list[RetrievalEvidence]) -> MagicMock:  # type: ignore[type-arg]
    retriever = MagicMock(spec=HybridRetriever)
    retriever.search = AsyncMock(return_value=pool)
    return retriever


def _qa_input(sub_questions: list[str], key_entities: list[str] | None = None) -> dict:
    return {
        "sub_questions": sub_questions,
        "key_entities": key_entities or [],
        "scope_notes": "scope",
    }


def _claim_input(
    subject: str, attribute: str, value: str, chunk_ids: list[str] | None = None
) -> dict:
    return {
        "subject": subject,
        "attribute": attribute,
        "value": value,
        "chunk_ids": chunk_ids or [],
    }


def _finding_input(
    summary: str, claims: list[dict] | None = None, confidence: str = "medium"
) -> dict:
    return {"summary": summary, "claims": claims or [], "confidence": confidence}


def _synthesis_input(
    summary: str = "Overall answer.",
    key_findings: list[str] | None = None,
    limitations: str = "None known.",
    confidence: str = "medium",
) -> dict:
    return {
        "summary": summary,
        "key_findings": key_findings or ["finding 1"],
        "limitations": limitations,
        "confidence": confidence,
    }


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------


class TestPartitionIntoBatches:
    def test_empty_input_returns_no_batches(self) -> None:
        assert _partition_into_batches([], max_chunks=3, max_chars=1000) == []

    def test_fewer_than_max_chunks_is_one_batch(self) -> None:
        chunks = [_evidence(f"c{i}", text="x") for i in range(2)]
        batches = _partition_into_batches(chunks, max_chunks=3, max_chars=1000)
        assert len(batches) == 1
        assert len(batches[0]) == 2

    def test_exact_multiple_splits_evenly(self) -> None:
        chunks = [_evidence(f"c{i}", text="x") for i in range(6)]
        batches = _partition_into_batches(chunks, max_chunks=2, max_chars=1000)
        assert [len(b) for b in batches] == [2, 2, 2]

    def test_char_budget_forces_new_batch_before_chunk_count(self) -> None:
        chunks = [_evidence("c0", text="a" * 600), _evidence("c1", text="b" * 600)]
        batches = _partition_into_batches(chunks, max_chunks=10, max_chars=1000)
        assert len(batches) == 2
        assert len(batches[0]) == 1
        assert len(batches[1]) == 1

    def test_single_oversized_chunk_is_its_own_batch(self) -> None:
        chunks = [_evidence("c0", text="a" * 5000)]
        batches = _partition_into_batches(chunks, max_chunks=3, max_chars=1000)
        assert len(batches) == 1
        assert len(batches[0]) == 1

    def test_all_chunks_are_preserved_across_batches(self) -> None:
        chunks = [_evidence(f"c{i}", text="x" * 100) for i in range(7)]
        batches = _partition_into_batches(chunks, max_chunks=3, max_chars=1000)
        flattened = [c.chunk_id for batch in batches for c in batch]
        assert flattened == [c.chunk_id for c in chunks]


class TestToEvidence:
    def test_truncates_long_text(self) -> None:
        chunk = _evidence("c0", text="a" * 500)
        evidence = _to_evidence([chunk], max_chars=100)
        assert len(evidence[0].excerpt) <= 102  # 100 chars + ellipsis marker
        assert evidence[0].excerpt.endswith("…")

    def test_preserves_short_text_unchanged(self) -> None:
        chunk = _evidence("c0", text="short text")
        evidence = _to_evidence([chunk], max_chars=100)
        assert evidence[0].excerpt == "short text"

    def test_clamps_score_above_one(self) -> None:
        chunk = _evidence("c0", final_score=1.5)
        evidence = _to_evidence([chunk])
        assert evidence[0].relevance_score == 1.0

    def test_preserves_ids(self) -> None:
        chunk = _evidence("c0", document_id="INC-2024-001")
        evidence = _to_evidence([chunk])
        assert evidence[0].evidence_id == "c0"
        assert evidence[0].document_id == "INC-2024-001"


class TestMergeEvidence:
    def test_dedupes_keeping_higher_score(self) -> None:
        t1 = ResearchTask(task_id="t1", question="q", status=ResearchStatus.COMPLETED)
        t1.result = ResearchResult(
            task_id="t1", summary="s", evidence=_to_evidence([_evidence("c0", final_score=0.5)])
        )
        t2 = ResearchTask(task_id="t2", question="q", status=ResearchStatus.COMPLETED)
        t2.result = ResearchResult(
            task_id="t2", summary="s", evidence=_to_evidence([_evidence("c0", final_score=0.9)])
        )
        merged = _merge_evidence([t1, t2])
        assert len(merged) == 1
        assert merged[0].relevance_score == 0.9

    def test_caps_at_max_items(self) -> None:
        tasks = []
        for i in range(10):
            t = ResearchTask(task_id=f"t{i}", question="q", status=ResearchStatus.COMPLETED)
            t.result = ResearchResult(
                task_id=f"t{i}",
                summary="s",
                evidence=_to_evidence([_evidence(f"c{i}", final_score=0.1 * i)]),
            )
            tasks.append(t)
        merged = _merge_evidence(tasks, max_items=3)
        assert len(merged) == 3
        # highest scores kept
        assert [round(e.relevance_score, 2) for e in merged] == [0.9, 0.8, 0.7]

    def test_tasks_without_result_are_skipped(self) -> None:
        t = ResearchTask(task_id="t1", question="q")  # result is None
        assert _merge_evidence([t]) == ()


class TestDetectContradictions:
    def test_no_claims_no_contradictions(self) -> None:
        assert _detect_contradictions([]) == []

    def test_single_claim_no_contradiction(self) -> None:
        claims = [Claim(subject="INC-1", attribute="root_cause", value="timeout", chunk_ids=["c0"])]
        assert _detect_contradictions(claims) == []

    def test_two_claims_same_value_no_contradiction(self) -> None:
        claims = [
            Claim(subject="INC-1", attribute="root_cause", value="Timeout", chunk_ids=["c0"]),
            Claim(subject="inc-1", attribute="Root_Cause", value="  timeout  ", chunk_ids=["c1"]),
        ]
        assert _detect_contradictions(claims) == []

    def test_two_claims_different_value_is_contradiction(self) -> None:
        claims = [
            Claim(
                subject="INC-2024-001",
                attribute="root_cause",
                value="connection pool exhaustion",
                chunk_ids=["c0"],
            ),
            Claim(
                subject="INC-2024-001",
                attribute="root_cause",
                value="certificate sync failure",
                chunk_ids=["c1"],
            ),
        ]
        contradictions = _detect_contradictions(claims)
        assert len(contradictions) == 1
        c = contradictions[0]
        assert c.subject == "INC-2024-001"
        assert c.attribute == "root_cause"
        assert set(c.conflicting_values) == {
            "connection pool exhaustion",
            "certificate sync failure",
        }
        assert set(c.chunk_ids) == {"c0", "c1"}

    def test_different_attribute_is_not_a_contradiction(self) -> None:
        claims = [
            Claim(subject="INC-1", attribute="root_cause", value="A", chunk_ids=["c0"]),
            Claim(subject="INC-1", attribute="impact", value="B", chunk_ids=["c1"]),
        ]
        assert _detect_contradictions(claims) == []

    def test_different_subject_is_not_a_contradiction(self) -> None:
        claims = [
            Claim(subject="INC-1", attribute="root_cause", value="A", chunk_ids=["c0"]),
            Claim(subject="INC-2", attribute="root_cause", value="B", chunk_ids=["c1"]),
        ]
        assert _detect_contradictions(claims) == []

    def test_three_way_conflict_reports_all_values(self) -> None:
        claims = [
            Claim(subject="INC-1", attribute="severity", value="P1", chunk_ids=["c0"]),
            Claim(subject="INC-1", attribute="severity", value="P2", chunk_ids=["c1"]),
            Claim(subject="INC-1", attribute="severity", value="P3", chunk_ids=["c2"]),
        ]
        contradictions = _detect_contradictions(claims)
        assert len(contradictions) == 1
        assert set(contradictions[0].conflicting_values) == {"P1", "P2", "P3"}

    def test_description_property_is_readable(self) -> None:
        claims = [
            Claim(subject="INC-1", attribute="root_cause", value="A", chunk_ids=["c0"]),
            Claim(subject="INC-1", attribute="root_cause", value="B", chunk_ids=["c1"]),
        ]
        c = _detect_contradictions(claims)[0]
        assert "INC-1" in c.description
        assert "root_cause" in c.description


class TestBuildSearchPlan:
    def test_root_question_always_first(self) -> None:
        analysis = QuestionAnalysis(sub_questions=["sub 1"], key_entities=[], scope_notes="n")
        plan = _build_search_plan("root question", analysis)
        assert plan[0] == "root question"

    def test_sub_questions_appended(self) -> None:
        analysis = QuestionAnalysis(
            sub_questions=["sub 1", "sub 2"], key_entities=[], scope_notes="n"
        )
        plan = _build_search_plan("root question", analysis)
        assert plan == ["root question", "sub 1", "sub 2"]

    def test_duplicates_deduped_case_insensitively(self) -> None:
        analysis = QuestionAnalysis(
            sub_questions=["Root Question", "sub 1"], key_entities=[], scope_notes="n"
        )
        plan = _build_search_plan("root question", analysis)
        assert plan == ["root question", "sub 1"]

    def test_capped_at_max_queries(self) -> None:
        analysis = QuestionAnalysis(
            sub_questions=["a", "b", "c", "d", "e"], key_entities=[], scope_notes="n"
        )
        plan = _build_search_plan("root", analysis, max_queries=3)
        assert len(plan) == 3


class TestExtractRootQuestion:
    def test_returns_last_user_message(self) -> None:
        state = _state("What happened?")
        assert _extract_root_question(state) == "What happened?"

    def test_returns_none_when_no_messages(self) -> None:
        state = initial_state(user=_user(), conversation_id="c1")
        assert _extract_root_question(state) is None

    def test_ignores_trailing_assistant_message(self) -> None:
        state = initial_state(user=_user(), conversation_id="c1")
        state["messages"] = [
            _message("real question", index=1),
            _message("an answer", index=2, role=MessageRole.ASSISTANT),
        ]
        assert _extract_root_question(state) == "real question"


# ---------------------------------------------------------------------------
# ResearchAgent.run() — end to end with mocked client + retriever
# ---------------------------------------------------------------------------


class TestSimpleRun:
    """Single sub-question, evidence fits in one batch — no recursion needed."""

    async def test_completes_without_recursion(self) -> None:
        pool = [_evidence("c0", final_score=0.9), _evidence("c1", final_score=0.8)]
        client = _mock_client(
            [
                _mock_response("question_analysis", _qa_input(["Q1"])),
                _mock_response(
                    "research_finding",
                    _finding_input(
                        "Batch summary.",
                        claims=[_claim_input("INC-2024-001", "root_cause", "timeout", ["c0"])],
                    ),
                ),
                _mock_response("final_synthesis", _synthesis_input()),
            ]
        )
        agent = ResearchAgent(client=client, model="test", retriever=_mock_retriever(pool))

        result = await agent.run(_state())

        assert "errors" not in result
        tasks = {t.task_id: t for t in result["research_tasks"]}
        assert set(tasks) == {"root", "root-sq1"}
        assert tasks["root"].depth == 0
        assert tasks["root"].parent_task_id is None
        assert tasks["root"].status == ResearchStatus.COMPLETED
        assert tasks["root-sq1"].depth == 1
        assert tasks["root-sq1"].parent_task_id == "root"
        assert tasks["root-sq1"].status == ResearchStatus.COMPLETED
        assert len(result["research_results"]) == 2
        assert result["budget"].used_research_steps == 1
        assert result["budget"].used_tokens > 0

    async def test_document_ids_never_contain_full_document_text(self) -> None:
        """Only chunk ids flow through document_ids — never document content."""
        pool = [_evidence("c0")]
        client = _mock_client(
            [
                _mock_response("question_analysis", _qa_input(["Q1"])),
                _mock_response("research_finding", _finding_input("s")),
                _mock_response("final_synthesis", _synthesis_input()),
            ]
        )
        agent = ResearchAgent(client=client, model="test", retriever=_mock_retriever(pool))

        result = await agent.run(_state())

        for task in result["research_tasks"]:
            for doc_id in task.document_ids:
                assert doc_id == "c0"  # a chunk id, not document text
            if task.result is not None:
                for ev in task.result.evidence:
                    assert len(ev.excerpt) <= 402  # bounded excerpt, not full text


class TestRecursiveRun:
    """Evidence spanning multiple batches forces real recursion + aggregation."""

    async def test_multi_batch_evidence_creates_child_tasks(self) -> None:
        pool = [_evidence(f"c{i}", final_score=0.9 - i * 0.1) for i in range(3)]
        client = _mock_client(
            [
                _mock_response("question_analysis", _qa_input(["Q1"])),
                _mock_response("research_finding", _finding_input("leaf 0")),
                _mock_response("research_finding", _finding_input("leaf 1")),
                _mock_response("research_finding", _finding_input("leaf 2")),
                _mock_response("research_finding", _finding_input("aggregate")),
                _mock_response("final_synthesis", _synthesis_input()),
            ]
        )
        agent = ResearchAgent(
            client=client, model="test", retriever=_mock_retriever(pool), max_chunks_per_batch=1
        )

        result = await agent.run(_state())

        tasks = {t.task_id: t for t in result["research_tasks"]}
        assert set(tasks) == {"root", "root-sq1", "root-sq1-b0", "root-sq1-b1", "root-sq1-b2"}
        assert tasks["root-sq1-b0"].depth == 2
        assert tasks["root-sq1-b0"].parent_task_id == "root-sq1"
        assert all(t.status == ResearchStatus.COMPLETED for t in tasks.values())
        # 4 recursive `_process_task` calls: root-sq1 + its 3 batch children
        assert result["budget"].used_research_steps == 4
        assert client.messages.create.call_count == 6

    async def test_depth_budget_forces_truncation_instead_of_recursion(self) -> None:
        """At the depth cap, a multi-batch task is truncated to one leaf call
        rather than spawning children — this is the "prevent infinite
        recursion" mechanism."""
        pool = [_evidence(f"c{i}", final_score=0.9 - i * 0.1) for i in range(3)]
        client = _mock_client(
            [
                _mock_response("question_analysis", _qa_input(["Q1"])),
                _mock_response("research_finding", _finding_input("truncated leaf")),
                _mock_response("final_synthesis", _synthesis_input()),
            ]
        )
        agent = ResearchAgent(
            client=client, model="test", retriever=_mock_retriever(pool), max_chunks_per_batch=1
        )
        budget = ExecutionBudget(max_research_depth=1)  # root-sq1 is already at depth 1

        result = await agent.run(_state(budget=budget))

        tasks = {t.task_id: t for t in result["research_tasks"]}
        assert set(tasks) == {"root", "root-sq1"}  # no batch children created
        assert "truncated" in (tasks["root-sq1"].result.summary or "")
        assert result["budget"].used_research_steps == 1
        assert client.messages.create.call_count == 3

    async def test_max_total_tasks_caps_semantic_fan_out(self) -> None:
        """Defense-in-depth cap: even with many sub-questions, task creation
        stops once the hard total-task limit is reached."""
        pool = [_evidence("c0")]
        client = _mock_client(
            [
                _mock_response("question_analysis", _qa_input(["Q1", "Q2", "Q3", "Q4"])),
                _mock_response("research_finding", _finding_input("sq1 leaf")),
                _mock_response("research_finding", _finding_input("sq2 leaf")),
                _mock_response("final_synthesis", _synthesis_input()),
            ]
        )
        agent = ResearchAgent(
            client=client, model="test", retriever=_mock_retriever(pool), max_total_tasks=3
        )

        result = await agent.run(_state())

        tasks = {t.task_id: t for t in result["research_tasks"]}
        assert set(tasks) == {"root", "root-sq1", "root-sq2"}
        assert len(result["research_tasks"]) <= 3


class TestContradictionDetectionEndToEnd:
    async def test_conflicting_claims_across_sub_questions_are_detected(self) -> None:
        pool = [_evidence("c0")]
        client = _mock_client(
            [
                _mock_response(
                    "question_analysis",
                    _qa_input(["What was the root cause?", "What was the impact?"]),
                ),
                _mock_response(
                    "research_finding",
                    _finding_input(
                        "Early note blamed the connection pool.",
                        claims=[
                            _claim_input(
                                "INC-2024-001", "root_cause", "connection pool exhaustion", ["c0"]
                            )
                        ],
                    ),
                ),
                _mock_response(
                    "research_finding",
                    _finding_input(
                        "Finalized RCA blamed certificate sync.",
                        claims=[
                            _claim_input(
                                "INC-2024-001",
                                "root_cause",
                                "certificate sync misconfiguration",
                                ["c0"],
                            )
                        ],
                    ),
                ),
                _mock_response("final_synthesis", _synthesis_input()),
            ]
        )
        agent = ResearchAgent(client=client, model="test", retriever=_mock_retriever(pool))

        result = await agent.run(_state("What caused INC-2024-001?"))

        root = next(t for t in result["research_tasks"] if t.task_id == "root")
        assert root.result is not None
        assert len(root.result.contradictions) == 1
        assert "INC-2024-001" in root.result.contradictions[0]
        assert "root_cause" in root.result.contradictions[0]


class TestFailureModes:
    async def test_budget_exhausted_skips_with_no_llm_calls(self) -> None:
        client = _mock_client([])
        agent = ResearchAgent(client=client, model="test", retriever=_mock_retriever([]))
        exhausted = ExecutionBudget(max_tokens=10, used_tokens=10)

        result = await agent.run(_state(budget=exhausted))

        assert result["errors"] == ["research skipped: execution budget exhausted"]
        client.messages.create.assert_not_called()

    async def test_no_question_skips_with_no_llm_calls(self) -> None:
        client = _mock_client([])
        agent = ResearchAgent(client=client, model="test", retriever=_mock_retriever([]))
        state = initial_state(user=_user(), conversation_id="c1")  # no messages

        result = await agent.run(state)

        assert result["errors"] == ["research skipped: no user question found"]
        client.messages.create.assert_not_called()

    async def test_question_analysis_failure_is_captured(self) -> None:
        client = MagicMock()
        client.messages.create = AsyncMock(side_effect=RuntimeError("API down"))
        agent = ResearchAgent(client=client, model="test", retriever=_mock_retriever([]))

        result = await agent.run(_state())

        assert result["errors"] == ["research question analysis failed"]
        root = result["research_tasks"][0]
        assert root.status == ResearchStatus.FAILED

    async def test_no_evidence_found_short_circuits(self) -> None:
        client = _mock_client([_mock_response("question_analysis", _qa_input(["Q1"]))])
        agent = ResearchAgent(client=client, model="test", retriever=_mock_retriever([]))

        result = await agent.run(_state())

        assert "errors" not in result
        root = result["research_tasks"][0]
        assert root.status == ResearchStatus.COMPLETED
        assert "No relevant documents" in root.result.summary
        assert client.messages.create.call_count == 1

    async def test_final_synthesis_failure_preserves_partial_state(self) -> None:
        pool = [_evidence("c0")]
        client = _mock_client(
            [
                _mock_response("question_analysis", _qa_input(["Q1"])),
                _mock_response("research_finding", _finding_input("leaf summary")),
                _no_tool_block_response(),  # final synthesis: no tool_use block
            ]
        )
        agent = ResearchAgent(client=client, model="test", retriever=_mock_retriever(pool))

        result = await agent.run(_state())

        assert result["errors"] == ["research final synthesis failed"]
        tasks = {t.task_id: t for t in result["research_tasks"]}
        assert tasks["root"].status == ResearchStatus.FAILED
        assert tasks["root-sq1"].status == ResearchStatus.COMPLETED  # partial work preserved


class TestProcessTaskDirect:
    async def test_task_with_no_matching_evidence_completes_without_llm_call(self) -> None:
        client = _mock_client([])
        agent = ResearchAgent(client=client, model="test", retriever=_mock_retriever([]))
        task = ResearchTask(task_id="orphan", question="q", document_ids=("missing-chunk",))

        ctx = _RunContext(
            conversation_id="c1", budget=ExecutionBudget(), max_depth=3, max_total_tasks=50
        )
        resolved = await agent._process_task(task, {}, ctx)

        assert resolved.status == ResearchStatus.COMPLETED
        assert "No evidence" in (resolved.result.summary or "")
        client.messages.create.assert_not_called()


class TestAgentEventLogging:
    async def test_emits_expected_event_types(self, caplog: pytest.LogCaptureFixture) -> None:
        pool = [_evidence(f"c{i}", final_score=0.9 - i * 0.1) for i in range(3)]
        client = _mock_client(
            [
                _mock_response("question_analysis", _qa_input(["Q1"])),
                _mock_response("research_finding", _finding_input("leaf 0")),
                _mock_response("research_finding", _finding_input("leaf 1")),
                _mock_response("research_finding", _finding_input("leaf 2")),
                _mock_response("research_finding", _finding_input("aggregate")),
                _mock_response("final_synthesis", _synthesis_input()),
            ]
        )
        agent = ResearchAgent(
            client=client, model="test", retriever=_mock_retriever(pool), max_chunks_per_batch=1
        )

        with caplog.at_level("INFO", logger="src.agents.research.node"):
            await agent.run(_state())

        for expected in (
            "research_plan",
            "task_created",
            "recursion_step",
            "batch_processing",
            "aggregation",
            "document_discovery",
            "final_synthesis",
        ):
            assert expected in caplog.messages, f"missing AgentEvent {expected!r} in log messages"


# ---------------------------------------------------------------------------
# make_research_node factory
# ---------------------------------------------------------------------------


class TestMakeResearchNode:
    async def test_returns_bound_run_callable(self) -> None:
        client = _mock_client([])
        node = make_research_node(client=client, retriever=_mock_retriever([]), model="test")
        exhausted = ExecutionBudget(max_tokens=10, used_tokens=10)

        result = await node(_state(budget=exhausted))

        assert result["errors"] == ["research skipped: execution budget exhausted"]


# ---------------------------------------------------------------------------
# Worked example: payment incident reports (mirrors docs/rlm.md)
# ---------------------------------------------------------------------------


class TestPaymentIncidentExample:
    """Two payment incidents (INC-2024-001, INC-2024-002 style reports) plus
    an early on-call note about INC-2024-001 that disagrees with the
    finalized root-cause analysis — the same worked example documented in
    docs/rlm.md.
    """

    def _incident_pool(self) -> list[RetrievalEvidence]:
        return [
            _evidence(
                "INC-2024-001-early-note",
                document_id="INC-2024-001-notes",
                text=(
                    "On-call notes 14:07 UTC: visa-processor connection pool at 100% "
                    "utilisation; looks like the pool size (50 connections) was simply "
                    "too small for peak load."
                ),
                final_score=0.75,
            ),
            _evidence(
                "INC-2024-001-final-rca",
                document_id="INC-2024-001",
                text=(
                    "Root cause: the Kubernetes ExternalSecret operator failed to sync "
                    "the rotated PostgreSQL TLS certificate to the visa-processor "
                    "namespace due to a misconfigured sync interval, exhausting the "
                    "connection pool as pods rejected the expired certificate."
                ),
                final_score=0.92,
            ),
            _evidence(
                "INC-2024-002-summary",
                document_id="INC-2024-002",
                text=(
                    "A retry-storm in the SWIFT message broker caused 38 outbound "
                    "MT103 wire transfers to be submitted twice during a DR failover "
                    "because the DR node lacked access to the shared idempotency store."
                ),
                final_score=0.7,
            ),
        ]

    async def test_flags_contradiction_between_early_note_and_final_rca(self) -> None:
        pool = self._incident_pool()
        client = _mock_client(
            [
                _mock_response(
                    "question_analysis",
                    _qa_input(
                        [
                            "What was the root cause of INC-2024-001?",
                            "What other payment incidents occurred recently?",
                        ],
                        key_entities=["INC-2024-001", "INC-2024-002"],
                    ),
                ),
                _mock_response(
                    "research_finding",
                    _finding_input(
                        "Early on-call notes attributed the outage to pool sizing.",
                        claims=[
                            _claim_input(
                                "INC-2024-001",
                                "root_cause",
                                "connection pool sized too small",
                                ["INC-2024-001-early-note"],
                            )
                        ],
                        confidence="low",
                    ),
                ),
                _mock_response(
                    "research_finding",
                    _finding_input(
                        "The finalized RCA for INC-2024-001 blames certificate sync, not pool "
                        "sizing; INC-2024-002 was a separate SWIFT duplication incident.",
                        claims=[
                            _claim_input(
                                "INC-2024-001",
                                "root_cause",
                                "certificate sync misconfiguration",
                                ["INC-2024-001-final-rca"],
                            ),
                            _claim_input(
                                "INC-2024-002",
                                "root_cause",
                                "DR node lacked idempotency store access",
                                ["INC-2024-002-summary"],
                            ),
                        ],
                        confidence="high",
                    ),
                ),
                _mock_response(
                    "final_synthesis",
                    _synthesis_input(
                        summary="Two payment incidents were examined; one has conflicting RCAs.",
                        key_findings=[
                            "INC-2024-001's early notes and final RCA disagree on root cause.",
                            "INC-2024-002 was caused by a DR idempotency gap.",
                        ],
                        limitations=(
                            "Final RCA for INC-2024-001 is more authoritative than on-call notes."
                        ),
                    ),
                ),
            ]
        )
        agent = ResearchAgent(client=client, model="test", retriever=_mock_retriever(pool))

        result = await agent.run(_state("What caused our recent payment incidents?"))

        # Explicit state: a full task tree was created and is inspectable.
        tasks = {t.task_id: t for t in result["research_tasks"]}
        assert tasks["root"].question == "What caused our recent payment incidents?"
        assert tasks["root-sq1"].depth == 1
        assert tasks["root-sq1"].parent_task_id == "root"

        # Contradiction surfaced without ever concatenating the full pool
        # into one prompt — each sub-agent call only saw its own batch.
        root = tasks["root"]
        assert root.result is not None
        assert len(root.result.contradictions) == 1
        assert "INC-2024-001" in root.result.contradictions[0]

        # Evidence attached to the final result stays a bounded excerpt set,
        # never full incident reports.
        assert len(root.result.evidence) <= 5
        for ev in root.result.evidence:
            assert len(ev.excerpt) <= 402
