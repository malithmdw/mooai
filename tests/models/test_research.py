"""Tests for `src.models.research.ResearchTask` and `ResearchResult`."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.models.enums import ResearchStatus
from src.models.evidence import Evidence
from src.models.research import ResearchResult, ResearchTask


def test_task_defaults_to_pending() -> None:
    task = ResearchTask(task_id="task-1", conversation_id="conv-1", objective="Find the policy")
    assert task.status is ResearchStatus.PENDING


def test_task_status_is_mutable_and_validated() -> None:
    task = ResearchTask(task_id="task-1", conversation_id="conv-1", objective="Find the policy")

    task.status = ResearchStatus.IN_PROGRESS

    assert task.status is ResearchStatus.IN_PROGRESS


def test_task_rejects_invalid_status_on_assignment() -> None:
    task = ResearchTask(task_id="task-1", conversation_id="conv-1", objective="Find the policy")

    with pytest.raises(ValidationError):
        task.status = "SOMEDAY"  # type: ignore[assignment]


def test_task_rejects_empty_objective() -> None:
    with pytest.raises(ValidationError, match="objective"):
        ResearchTask(task_id="task-1", conversation_id="conv-1", objective="   ")


def test_valid_research_result_is_immutable() -> None:
    evidence = Evidence(
        evidence_id="ev-1", document_id="doc-1", excerpt="Relevant excerpt.", relevance_score=0.8
    )
    result = ResearchResult(task_id="task-1", summary="Found the policy.", evidence=(evidence,))

    with pytest.raises(ValidationError):
        result.summary = "changed"  # type: ignore[misc]
