"""Multi-step research task tracking and its final synthesized result.

``ResearchTask`` is the unit of work in the Research Agent's recursive
decomposition (see ``src.agents.research``): a tree of tasks rooted at the
top-level research question, where each node is scoped to a bounded set of
evidence ``document_ids`` (chunk ids, never an entire document) at a given
recursion ``depth``. See ``docs/rlm.md`` for the full design and why this
qualifies as a Recursive Language Model (RLM) style architecture.
"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field

from src.models.common import EntityId, NonEmptyStr, utcnow
from src.models.enums import ResearchStatus
from src.models.evidence import Evidence


class ResearchResult(BaseModel):
    """The synthesized outcome of one completed `ResearchTask`.

    `evidence` holds only the targeted excerpts that actually backed this
    result — never full documents (see CLAUDE.md "Retrieved documents are
    untrusted content" and the Research Agent's "no entire documents passed
    upward" rule). `contradictions` surfaces any conflicting claims detected
    across the evidence this task (or its descendants) examined.
    """

    model_config = ConfigDict(frozen=True)

    task_id: EntityId
    summary: NonEmptyStr
    evidence: tuple[Evidence, ...] = Field(default_factory=tuple)
    contradictions: tuple[str, ...] = Field(
        default_factory=tuple,
        description="Human-readable descriptions of conflicting claims found across evidence.",
    )
    completed_at: datetime = Field(default_factory=utcnow)


class ResearchTask(BaseModel):
    """One node in the recursive research task tree.

    Mutable (`validate_assignment=True`): `status` and `result` are set in
    place as the task moves from `PENDING` through the Research Agent's
    pipeline to `COMPLETED` (or `FAILED`).

    `parent_task_id` and `depth` make the tree structure explicit and
    inspectable — `depth` is checked against
    `ExecutionBudget.max_research_depth` before a task is allowed to spawn
    children, which is what prevents unbounded recursion (see
    `src.agents.research`).
    """

    model_config = ConfigDict(validate_assignment=True)

    task_id: EntityId
    question: NonEmptyStr
    document_ids: tuple[str, ...] = Field(
        default_factory=tuple,
        description="Chunk-level identifiers assigned to this task — never a full document.",
    )
    status: ResearchStatus = ResearchStatus.PENDING
    parent_task_id: EntityId | None = None
    depth: int = Field(default=0, ge=0)
    result: ResearchResult | None = None
    created_at: datetime = Field(default_factory=utcnow)
