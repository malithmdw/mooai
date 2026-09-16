"""Tests for `src.models.research.ResearchTask` and `ResearchResult`."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.models.enums import ResearchStatus
from src.models.evidence import Evidence
from src.models.research import ResearchResult, ResearchTask


def test_task_defaults_to_pending() -> None:
    task = ResearchTask(task_id="task-1", question="Find the policy")
    assert task.status is ResearchStatus.PENDING


def test_task_defaults_are_root_level() -> None:
    task = ResearchTask(task_id="task-1", question="Find the policy")
    assert task.parent_task_id is None
    assert task.depth == 0
    assert task.document_ids == ()
    assert task.result is None


def test_task_status_is_mutable_and_validated() -> None:
    task = ResearchTask(task_id="task-1", question="Find the policy")

    task.status = ResearchStatus.IN_PROGRESS

    assert task.status is ResearchStatus.IN_PROGRESS


def test_task_rejects_invalid_status_on_assignment() -> None:
    task = ResearchTask(task_id="task-1", question="Find the policy")

    with pytest.raises(ValidationError):
        task.status = "SOMEDAY"  # type: ignore[assignment]


def test_task_rejects_empty_question() -> None:
    with pytest.raises(ValidationError, match="question"):
        ResearchTask(task_id="task-1", question="   ")


def test_task_rejects_negative_depth() -> None:
    with pytest.raises(ValidationError):
        ResearchTask(task_id="task-1", question="Find the policy", depth=-1)


def test_child_task_records_parent_and_depth() -> None:
    child = ResearchTask(
        task_id="task-1-b0",
        question="Find the policy",
        document_ids=("chunk-1", "chunk-2"),
        parent_task_id="task-1",
        depth=1,
    )
    assert child.parent_task_id == "task-1"
    assert child.depth == 1
    assert child.document_ids == ("chunk-1", "chunk-2")


def test_task_result_is_mutable_and_validated() -> None:
    task = ResearchTask(task_id="task-1", question="Find the policy")
    result = ResearchResult(task_id="task-1", summary="Found the policy.")

    task.result = result
    task.status = ResearchStatus.COMPLETED

    assert task.result is result
    assert task.status is ResearchStatus.COMPLETED


def test_valid_research_result_is_immutable() -> None:
    evidence = Evidence(
        evidence_id="ev-1", document_id="doc-1", excerpt="Relevant excerpt.", relevance_score=0.8
    )
    result = ResearchResult(task_id="task-1", summary="Found the policy.", evidence=(evidence,))

    with pytest.raises(ValidationError):
        result.summary = "changed"  # type: ignore[misc]


def test_research_result_defaults_to_no_contradictions() -> None:
    result = ResearchResult(task_id="task-1", summary="Found the policy.")
    assert result.contradictions == ()


def test_research_result_can_carry_contradictions() -> None:
    result = ResearchResult(
        task_id="task-1",
        summary="Found the policy.",
        contradictions=("Conflicting root_cause reported for INC-2024-001.",),
    )
    assert result.contradictions == ("Conflicting root_cause reported for INC-2024-001.",)
