"""Research Agent LangGraph node — recursive, RLM-style multi-step research.

Demonstrates that the system can reason over a document collection too
large for one context window by recursively decomposing a research
question into a tree of bounded ``ResearchTask``s, never concatenating more
than a few small evidence chunks into any single LLM call. See
``docs/rlm.md`` for the full design rationale and why this qualifies as a
Recursive Language Model (RLM) style architecture rather than repeated flat
LLM calls.

Pipeline stages
----------------
1. Research question analysis  — ``_analyze_question`` (LLM).
2. Search-plan generation      — ``_build_search_plan`` (deterministic).
3. Document discovery          — ``_discover_documents`` (HybridRetriever).
4. Task decomposition          — ``run`` splits the root into one child
   ``ResearchTask`` per sub-question (semantic decomposition).
5. Partitioning into batches   — ``_partition_into_batches`` (mechanical,
   context-window-bounded decomposition, applied inside every task).
6. Recursive/sub-agent analysis — ``_process_task`` calling itself: a task
   whose evidence does not fit in one batch spawns one child task per
   batch, at ``depth + 1``, and recurses into each (this is the actual
   recursion — not a flat loop over a growing prompt).
7. Intermediate result aggregation — ``_aggregate``, invoked by
   ``_process_task`` after its children resolve; reused at every level of
   the tree.
8. Contradiction detection     — ``_detect_contradictions`` (deterministic,
   over the structured ``Claim``s collected at every level).
9. Final evidence synthesis    — ``_synthesize_final`` (one LLM call at the
   root, over condensed findings only).

Depth/recursion budget
-----------------------
A task may only spawn children while ``task.depth < budget.max_research_depth``
(``ExecutionBudget.max_research_depth``, part of ``GraphState``). At the
depth limit — or once ``max_total_tasks`` tasks have been created, a second,
independent cap — a task with too much evidence to fit one batch is forced
to analyze only its highest-scoring batch directly, noting the truncation,
rather than recursing further. Both are hard stops: recursion cannot run
away regardless of corpus size or LLM behaviour.

Security
--------
- RBAC roles are forwarded to every ``HybridRetriever.search`` call, exactly
  as in ``src.agents.retrieval`` — the Research Agent never bypasses
  authorization to reach documents.
- No stage ever places raw document/chunk text in a parent task's result —
  only a synthesized summary, structured ``Claim``s, and short bounded
  excerpts (see ``_to_evidence``). A parent task's LLM call only ever sees
  its children's condensed findings, never the underlying evidence chunks.
"""

from __future__ import annotations

from collections.abc import Callable, Coroutine, Sequence
from dataclasses import dataclass, field
from typing import Any, TypeVar
from uuid import uuid4

import anthropic
from pydantic import BaseModel, ValidationError

from src.agents.budget import ExecutionBudget
from src.agents.research.models import (
    Claim,
    Contradiction,
    FinalSynthesis,
    QuestionAnalysis,
    ResearchFinding,
)
from src.agents.research.prompts import (
    build_aggregation_prompt,
    build_batch_analysis_prompt,
    build_final_synthesis_prompt,
    build_question_analysis_prompt,
)
from src.agents.state import GraphState
from src.core.config import get_settings
from src.core.logging import get_logger, log_error, log_info, log_warning
from src.models.agent_events import AgentEvent
from src.models.enums import AgentState, MessageRole, ResearchStatus, Role
from src.models.evidence import Evidence
from src.models.research import ResearchResult, ResearchTask
from src.retrieval.hybrid.models import RetrievalEvidence
from src.retrieval.hybrid.retriever import HybridRetriever

_logger = get_logger(__name__)

_M = TypeVar("_M", bound=BaseModel)

# --- Tunable limits — every one is a bound that prevents unbounded growth --
MAX_SUB_QUESTIONS: int = 4
"""Cap on Stage 4's semantic fan-out — how many sibling tasks the root splits into."""

MAX_CHUNKS_PER_BATCH: int = 3
"""Cap on evidence chunks per batch — keeps every sub-agent call's context small."""

MAX_CHARS_PER_BATCH: int = 3_000
"""Secondary cap on total evidence characters per batch, independent of chunk count."""

MAX_TOTAL_TASKS: int = 50
"""Hard cap on ResearchTask objects created in one run — defense in depth
against runaway recursion, independent of the depth budget."""

MAX_SEARCH_QUERIES: int = 5
"""Cap on Stage 2's search plan size."""

DEFAULT_TOP_K_PER_QUERY: int = 8
"""Results requested per search-plan query during document discovery."""

MAX_EVIDENCE_POOL: int = 24
"""Cap on the deduplicated evidence pool discovered in Stage 3."""

MAX_EVIDENCE_PER_RESULT: int = 5
"""Cap on how many `Evidence` items any single `ResearchResult` carries."""


# ---------------------------------------------------------------------------
# Anthropic tool definitions for structured stage output
# ---------------------------------------------------------------------------

_QUESTION_ANALYSIS_TOOL: dict[str, Any] = {
    "name": "question_analysis",
    "description": (
        "Record the structured breakdown of the research question. Call this tool exactly once."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "sub_questions": {
                "type": "array",
                "items": {"type": "string"},
                "minItems": 1,
                "maxItems": 6,
                "description": "Focused sub-questions covering the research question.",
            },
            "key_entities": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Named entities (IDs, systems, dates) central to the question.",
            },
            "scope_notes": {
                "type": "string",
                "description": "Scope/ambiguity notes for the search plan.",
            },
        },
        "required": ["sub_questions", "key_entities", "scope_notes"],
    },
}

_CLAIM_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "subject": {
            "type": "string",
            "description": "What the claim is about, e.g. an incident ID.",
        },
        "attribute": {"type": "string", "description": "The kind of fact, e.g. 'root_cause'."},
        "value": {"type": "string", "description": "The asserted answer for that attribute."},
        "chunk_ids": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Evidence chunk_ids this claim is drawn from.",
        },
    },
    "required": ["subject", "attribute", "value", "chunk_ids"],
}

_RESEARCH_FINDING_TOOL: dict[str, Any] = {
    "name": "research_finding",
    "description": (
        "Record a condensed finding derived from the evidence or summaries provided. "
        "Call this tool exactly once."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "summary": {"type": "string", "description": "Condensed synthesis, no raw quoting."},
            "claims": {
                "type": "array",
                "items": _CLAIM_SCHEMA,
                "description": "Distinct structured factual assertions extracted.",
            },
            "confidence": {
                "type": "string",
                "enum": ["high", "medium", "low", "unsupported"],
                "description": "How well the input supports this finding.",
            },
        },
        "required": ["summary", "claims", "confidence"],
    },
}

_FINAL_SYNTHESIS_TOOL: dict[str, Any] = {
    "name": "final_synthesis",
    "description": (
        "Record the final, corpus-wide synthesized answer. Call this tool exactly once."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "summary": {"type": "string"},
            "key_findings": {
                "type": "array",
                "items": {"type": "string"},
                "minItems": 1,
            },
            "limitations": {"type": "string"},
            "confidence": {
                "type": "string",
                "enum": ["high", "medium", "low", "unsupported"],
            },
        },
        "required": ["summary", "key_findings", "limitations", "confidence"],
    },
}


# ---------------------------------------------------------------------------
# Per-run mutable context (never stored on the agent instance — see
# ResearchAgent docstring on concurrency safety)
# ---------------------------------------------------------------------------


@dataclass
class _RunContext:
    """Bookkeeping threaded through one `ResearchAgent.run` call.

    Created fresh per call and passed by reference through the recursion —
    never stored as an attribute on `ResearchAgent` itself, so concurrent
    `run()` calls for different conversations never share state.
    """

    conversation_id: str | None
    budget: ExecutionBudget
    max_depth: int
    max_total_tasks: int
    tasks: list[ResearchTask] = field(default_factory=list)
    claims: list[Claim] = field(default_factory=list)

    @property
    def task_limit_reached(self) -> bool:
        return len(self.tasks) >= self.max_total_tasks


def _emit(ctx: _RunContext, message: str, metadata: dict[str, Any]) -> AgentEvent:
    """Create and log one Research-stage `AgentEvent`.

    Only primitive JSON-safe values are recorded — see
    `src.agents.retrieval.node._emit` for the identical discipline; document
    text and raw claim values are never included. Mirrors the retrieval
    node's pattern: the log line is the event's sink (`GraphState` has no
    event-list field), so the constructed `AgentEvent` is returned mainly
    for callers (e.g. tests) that want to inspect it directly.
    """
    safe_meta: dict[str, Any] = {
        k: v
        for k, v in metadata.items()
        if isinstance(v, (str, int, float, bool, type(None), list))
    }
    event = AgentEvent(
        event_id=str(uuid4()),
        conversation_id=ctx.conversation_id or "no-conversation",
        state=AgentState.RESEARCH,
        message=message,
        metadata=safe_meta,
    )
    log_info(
        _logger,
        f"agent_event.research.{message}",
        message,
        event_id=event.event_id,
        agent_state=event.state.value,
        **safe_meta,
    )
    return event


# ---------------------------------------------------------------------------
# Pure helpers — partitioning, evidence conversion, contradiction detection
# ---------------------------------------------------------------------------


def _partition_into_batches(
    chunks: Sequence[RetrievalEvidence],
    *,
    max_chunks: int,
    max_chars: int,
) -> list[list[RetrievalEvidence]]:
    """Partition *chunks* into batches bounded by both count and character size.

    This is the mechanism that keeps any single sub-agent call's context
    bounded regardless of how large the overall evidence pool is — see
    ``docs/rlm.md`` "why this is RLM-style, not repeated LLM calls".
    """
    batches: list[list[RetrievalEvidence]] = []
    current: list[RetrievalEvidence] = []
    current_chars = 0

    for chunk in chunks:
        chunk_chars = len(chunk.text)
        would_overflow = current and (
            len(current) >= max_chunks or current_chars + chunk_chars > max_chars
        )
        if would_overflow:
            batches.append(current)
            current = []
            current_chars = 0
        current.append(chunk)
        current_chars += chunk_chars

    if current:
        batches.append(current)

    return batches


def _to_evidence(
    chunks: Sequence[RetrievalEvidence], *, max_chars: int = 400
) -> tuple[Evidence, ...]:
    """Convert evidence chunks to bounded `Evidence` excerpts.

    Never the full chunk — truncated to `max_chars` — and never a whole
    document; chunks are already document fragments. See CLAUDE.md
    "Retrieved documents are untrusted content".
    """
    result: list[Evidence] = []
    for chunk in chunks:
        text = chunk.text[:max_chars]
        if len(chunk.text) > max_chars:
            text += " …"
        result.append(
            Evidence(
                evidence_id=chunk.chunk_id,
                document_id=chunk.document_id,
                excerpt=text,
                relevance_score=min(1.0, max(0.0, chunk.final_score)),
            )
        )
    return tuple(result)


def _merge_evidence(
    children: Sequence[ResearchTask], *, max_items: int = MAX_EVIDENCE_PER_RESULT
) -> tuple[Evidence, ...]:
    """Combine children's already-bounded evidence into one bounded set.

    Deduplicates by `evidence_id`, keeps the highest-scoring instance, and
    caps the result — so a parent task's evidence never grows with the size
    of its subtree.
    """
    best: dict[str, Evidence] = {}
    for child in children:
        if child.result is None:
            continue
        for ev in child.result.evidence:
            existing = best.get(ev.evidence_id)
            if existing is None or ev.relevance_score > existing.relevance_score:
                best[ev.evidence_id] = ev
    ranked = sorted(best.values(), key=lambda e: -e.relevance_score)
    return tuple(ranked[:max_items])


def _detect_contradictions(claims: Sequence[Claim]) -> list[Contradiction]:
    """Stage 8 — deterministic contradiction detection over structured claims.

    Groups claims by normalized `(subject, attribute)`. A group with more
    than one distinct normalized `value` is a contradiction: the same fact
    about the same subject was asserted differently by evidence examined in
    different (possibly sibling) sub-agent calls. Runs over `Claim`s only —
    never raw evidence text — so it stays cheap regardless of corpus size.
    """
    subject_display: dict[str, str] = {}
    groups: dict[tuple[str, str], dict[str, tuple[str, set[str]]]] = {}

    for claim in claims:
        subject_norm = claim.subject.strip().lower()
        subject_display.setdefault(subject_norm, claim.subject.strip())

        key = (subject_norm, claim.attribute.strip().lower())
        value_norm = claim.value.strip().lower()
        bucket = groups.setdefault(key, {})
        display_value, chunk_ids = bucket.get(value_norm, (claim.value.strip(), set()))
        chunk_ids.update(claim.chunk_ids)
        bucket[value_norm] = (display_value, chunk_ids)

    contradictions: list[Contradiction] = []
    for (subject_norm, attribute), values in groups.items():
        if len(values) < 2:
            continue
        conflicting_values = tuple(v[0] for v in values.values())
        all_chunk_ids = tuple(sorted({cid for _, cids in values.values() for cid in cids}))
        contradictions.append(
            Contradiction(
                subject=subject_display[subject_norm],
                attribute=attribute,
                conflicting_values=conflicting_values,
                chunk_ids=all_chunk_ids,
            )
        )
    return contradictions


def _extract_root_question(state: GraphState) -> str | None:
    """The most recent user message is the root research question."""
    for message in reversed(list(state["messages"])):
        if message.role == MessageRole.USER:
            content = message.content.strip()
            return content or None
    return None


def _build_search_plan(
    question: str, analysis: QuestionAnalysis, *, max_queries: int = MAX_SEARCH_QUERIES
) -> list[str]:
    """Stage 2 — deterministic search-plan generation.

    The root question is always the primary query; each sub-question from
    Stage 1 is added as an additional query, deduplicated case-insensitively
    and capped at `max_queries` — mirrors
    `src.agents.retrieval.queries.generate_queries`.
    """
    primary = question.strip()
    queries: list[str] = [primary]
    seen_lower: set[str] = {primary.lower()}

    for sub_question in analysis.sub_questions:
        sub_question = sub_question.strip()
        if sub_question and sub_question.lower() not in seen_lower:
            queries.append(sub_question)
            seen_lower.add(sub_question.lower())
        if len(queries) >= max_queries:
            break

    return queries


# ---------------------------------------------------------------------------
# Research Agent
# ---------------------------------------------------------------------------


class ResearchAgent:
    """Recursive research over a bounded evidence pool — see module docstring.

    Stateless across calls beyond its injected dependencies: `run()` builds
    a fresh `_RunContext` every time, so one instance is safe to reuse
    concurrently across conversations. Construct via `make_research_node`.
    """

    def __init__(
        self,
        *,
        client: anthropic.AsyncAnthropic,
        model: str,
        retriever: HybridRetriever,
        max_chunks_per_batch: int = MAX_CHUNKS_PER_BATCH,
        max_chars_per_batch: int = MAX_CHARS_PER_BATCH,
        max_total_tasks: int = MAX_TOTAL_TASKS,
        max_sub_questions: int = MAX_SUB_QUESTIONS,
        top_k_per_query: int = DEFAULT_TOP_K_PER_QUERY,
        max_evidence_pool: int = MAX_EVIDENCE_POOL,
    ) -> None:
        if max_chunks_per_batch < 1:
            raise ValueError("max_chunks_per_batch must be >= 1")
        if max_total_tasks < 1:
            raise ValueError("max_total_tasks must be >= 1")

        self._client = client
        self._model = model
        self._retriever = retriever
        self._max_chunks_per_batch = max_chunks_per_batch
        self._max_chars_per_batch = max_chars_per_batch
        self._max_total_tasks = max_total_tasks
        self._max_sub_questions = max_sub_questions
        self._top_k_per_query = top_k_per_query
        self._max_evidence_pool = max_evidence_pool

    # ------------------------------------------------------------------
    # Shared structured-call helper (stages 1, 6, 7, 9 all use this)
    # ------------------------------------------------------------------

    async def _call_structured(
        self,
        *,
        system_prompt: str,
        user_content: str,
        tool: dict[str, Any],
        tool_name: str,
        model_cls: type[_M],
        ctx: _RunContext,
        max_tokens: int = 1024,
    ) -> _M | None:
        """Forced tool-call, budget consumption, and structured parse.

        Returns `None` (and logs) on any failure — API error, missing
        tool-use block, or schema validation failure — so callers degrade
        gracefully rather than raising, matching every other agent node.
        """
        try:
            response = await self._client.messages.create(  # type: ignore[call-overload]
                model=self._model,
                max_tokens=max_tokens,
                system=system_prompt,
                tools=[tool],
                tool_choice={"type": "tool", "name": tool_name},
                messages=[{"role": "user", "content": user_content}],
            )
        except Exception as exc:
            log_error(
                _logger,
                "research.llm_error",
                "Anthropic call failed in research agent",
                error=exc,
                tool_name=tool_name,
                conversation_id=ctx.conversation_id,
            )
            return None

        ctx.budget = ctx.budget.consume(
            tokens=response.usage.input_tokens + response.usage.output_tokens
        )

        tool_block = next(
            (b for b in response.content if getattr(b, "type", None) == "tool_use"), None
        )
        if tool_block is None:
            log_error(
                _logger,
                "research.no_tool_block",
                "Research LLM response contained no tool-use block",
                tool_name=tool_name,
                conversation_id=ctx.conversation_id,
            )
            return None

        try:
            return model_cls.model_validate(tool_block.input)
        except ValidationError as exc:
            log_error(
                _logger,
                "research.validation_error",
                "Research structured output failed Pydantic validation",
                error=exc,
                tool_name=tool_name,
                conversation_id=ctx.conversation_id,
            )
            return None

    # ------------------------------------------------------------------
    # Stage 1: research question analysis
    # ------------------------------------------------------------------

    async def _analyze_question(self, question: str, ctx: _RunContext) -> QuestionAnalysis | None:
        return await self._call_structured(
            system_prompt=build_question_analysis_prompt(),
            user_content=question,
            tool=_QUESTION_ANALYSIS_TOOL,
            tool_name="question_analysis",
            model_cls=QuestionAnalysis,
            ctx=ctx,
            max_tokens=512,
        )

    # ------------------------------------------------------------------
    # Stage 3: document discovery
    # ------------------------------------------------------------------

    async def _discover_documents(
        self, queries: list[str], roles: list[Role], ctx: _RunContext
    ) -> list[RetrievalEvidence]:
        """Run every search-plan query and merge results into one pool.

        Deduplicated by `chunk_id` (highest score kept), capped at
        `max_evidence_pool` — this pool is the largest amount of evidence
        ever held in memory at once, and it is never sent to the LLM in a
        single call; only bounded batches of it are (see
        `_partition_into_batches`).
        """
        best: dict[str, RetrievalEvidence] = {}
        for query_index, query in enumerate(queries):
            try:
                results = await self._retriever.search(
                    query, top_k=self._top_k_per_query, roles=roles
                )
            except Exception as exc:
                log_error(
                    _logger,
                    "research.discovery_error",
                    "Document discovery search failed",
                    error=exc,
                    query_index=query_index,
                    conversation_id=ctx.conversation_id,
                )
                continue

            ctx.budget = ctx.budget.consume(retrieval_results=len(results))
            _emit(
                ctx,
                "document_discovery",
                {"query_index": query_index, "result_count": len(results)},
            )

            for ev in results:
                existing = best.get(ev.chunk_id)
                if existing is None or ev.final_score > existing.final_score:
                    best[ev.chunk_id] = ev

        pool = sorted(best.values(), key=lambda e: -e.final_score)[: self._max_evidence_pool]
        return pool

    # ------------------------------------------------------------------
    # Stage 6: recursive / sub-agent batch analysis
    # ------------------------------------------------------------------

    async def _analyze_batch(
        self,
        task: ResearchTask,
        batch: list[RetrievalEvidence],
        *,
        batch_index: int,
        ctx: _RunContext,
    ) -> ResearchFinding | None:
        _emit(
            ctx,
            "batch_analysis",
            {
                "task_id": task.task_id,
                "batch_index": batch_index,
                "chunk_count": len(batch),
                "depth": task.depth,
            },
        )
        return await self._call_structured(
            system_prompt=build_batch_analysis_prompt(task.question, batch),
            user_content=f"Analyze the evidence batch above and answer: {task.question}",
            tool=_RESEARCH_FINDING_TOOL,
            tool_name="research_finding",
            model_cls=ResearchFinding,
            ctx=ctx,
        )

    # ------------------------------------------------------------------
    # Stage 7: intermediate result aggregation
    # ------------------------------------------------------------------

    async def _aggregate(
        self, task: ResearchTask, children: list[ResearchTask], ctx: _RunContext
    ) -> ResearchResult | None:
        summaries = [
            (child.task_id, child.result.summary) for child in children if child.result is not None
        ]
        if not summaries:
            return ResearchResult(
                task_id=task.task_id, summary="No sub-task produced a usable finding."
            )

        finding = await self._call_structured(
            system_prompt=build_aggregation_prompt(task.question, summaries),
            user_content=(
                f"Combine the sub-task summaries above into one answer for: {task.question}"
            ),
            tool=_RESEARCH_FINDING_TOOL,
            tool_name="research_finding",
            model_cls=ResearchFinding,
            ctx=ctx,
        )
        if finding is None:
            return None

        ctx.claims.extend(finding.claims)
        return ResearchResult(
            task_id=task.task_id,
            summary=finding.summary,
            evidence=_merge_evidence(children),
        )

    # ------------------------------------------------------------------
    # Stages 5-7: recursive task resolution (the literal recursion)
    # ------------------------------------------------------------------

    async def _process_task(
        self,
        task: ResearchTask,
        evidence_index: dict[str, RetrievalEvidence],
        ctx: _RunContext,
    ) -> ResearchTask:
        """Resolve *task* in place, recursing into child tasks as needed.

        Base case: the task's evidence fits in one batch, or the depth/task
        budget has been reached — resolve with a single direct sub-agent
        analysis call (Stage 6).

        Recursive case: the evidence spans multiple batches — spawn one
        child `ResearchTask` per batch at `depth + 1` (Stage 5), recurse
        into each via this same method (Stage 6, applied per leaf), then
        combine their results (Stage 7).
        """
        task.status = ResearchStatus.IN_PROGRESS
        ctx.budget = ctx.budget.consume(research_steps=1)

        _emit(
            ctx,
            "recursion_step",
            {
                "task_id": task.task_id,
                "parent_task_id": task.parent_task_id,
                "depth": task.depth,
                "max_depth": ctx.max_depth,
                "document_count": len(task.document_ids),
            },
        )

        chunks = sorted(
            (evidence_index[cid] for cid in task.document_ids if cid in evidence_index),
            key=lambda e: -e.final_score,
        )

        if not chunks:
            task.status = ResearchStatus.COMPLETED
            task.result = ResearchResult(
                task_id=task.task_id,
                summary="No evidence was available for this sub-question.",
            )
            return task

        batches = _partition_into_batches(
            chunks, max_chunks=self._max_chunks_per_batch, max_chars=self._max_chars_per_batch
        )
        _emit(
            ctx,
            "batch_processing",
            {"task_id": task.task_id, "batch_count": len(batches), "chunk_count": len(chunks)},
        )

        at_depth_limit = task.depth >= ctx.max_depth
        at_task_limit = ctx.task_limit_reached

        # --- Base case: single batch, or budget forces a leaf -----------
        if len(batches) <= 1 or at_depth_limit or at_task_limit:
            chosen = batches[0] if batches else []
            finding = await self._analyze_batch(task, chosen, batch_index=0, ctx=ctx)

            if finding is None:
                task.status = ResearchStatus.FAILED
                return task

            ctx.claims.extend(finding.claims)
            summary = finding.summary
            if len(batches) > 1:
                summary += " (evidence truncated: recursion/task budget reached)"
                log_warning(
                    _logger,
                    "research.truncated",
                    "Task evidence truncated to one batch by depth/task budget",
                    task_id=task.task_id,
                    depth=task.depth,
                    batch_count=len(batches),
                    conversation_id=ctx.conversation_id,
                )

            task.status = ResearchStatus.COMPLETED
            task.result = ResearchResult(
                task_id=task.task_id, summary=summary, evidence=_to_evidence(chosen)
            )
            return task

        # --- Recursive case: one child task per batch --------------------
        child_tasks: list[ResearchTask] = []
        for batch_index, batch in enumerate(batches):
            if ctx.task_limit_reached:
                log_warning(
                    _logger,
                    "research.task_limit_reached",
                    "Max total task count reached; remaining batches skipped",
                    task_id=task.task_id,
                    max_total_tasks=ctx.max_total_tasks,
                    conversation_id=ctx.conversation_id,
                )
                break

            child = ResearchTask(
                task_id=f"{task.task_id}-b{batch_index}",
                question=task.question,
                document_ids=tuple(c.chunk_id for c in batch),
                parent_task_id=task.task_id,
                depth=task.depth + 1,
            )
            ctx.tasks.append(child)
            _emit(
                ctx,
                "task_created",
                {
                    "task_id": child.task_id,
                    "parent_task_id": task.task_id,
                    "depth": child.depth,
                    "document_count": len(child.document_ids),
                },
            )

            resolved_child = await self._process_task(child, evidence_index, ctx)  # recursion
            child_tasks.append(resolved_child)

        aggregated = await self._aggregate(task, child_tasks, ctx)
        task.status = ResearchStatus.COMPLETED if aggregated is not None else ResearchStatus.FAILED
        task.result = aggregated
        _emit(
            ctx,
            "aggregation",
            {
                "task_id": task.task_id,
                "child_count": len(child_tasks),
                "status": task.status.value,
            },
        )
        return task

    # ------------------------------------------------------------------
    # Stage 9: final evidence synthesis
    # ------------------------------------------------------------------

    async def _synthesize_final(
        self,
        question: str,
        sub_tasks: list[ResearchTask],
        contradictions: list[Contradiction],
        ctx: _RunContext,
    ) -> ResearchResult | None:
        findings = [(t.task_id, t.result.summary) for t in sub_tasks if t.result is not None]
        descriptions = [c.description for c in contradictions]

        synthesis = await self._call_structured(
            system_prompt=build_final_synthesis_prompt(question, findings, descriptions),
            user_content=f"Synthesize the final answer for: {question}",
            tool=_FINAL_SYNTHESIS_TOOL,
            tool_name="final_synthesis",
            model_cls=FinalSynthesis,
            ctx=ctx,
            max_tokens=1024,
        )
        if synthesis is None:
            return None

        key_findings_block = "\n".join(f"- {f}" for f in synthesis.key_findings)
        summary = f"{synthesis.summary}\n\nKey findings:\n{key_findings_block}"
        if synthesis.limitations:
            summary += f"\n\nLimitations: {synthesis.limitations}"

        return ResearchResult(
            task_id="root",
            summary=summary,
            evidence=_merge_evidence(sub_tasks),
            contradictions=tuple(descriptions),
        )

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    async def run(self, state: GraphState) -> dict[str, Any]:
        """Execute the full recursive research pipeline.

        Never raises: every failure mode is captured as a `errors` entry so
        the graph can continue or terminate gracefully, matching every
        other agent node in this codebase.
        """
        budget = state["budget"]
        conv_id = state["conversation_id"]
        roles = state["user_role"]

        if budget.is_exhausted:
            log_warning(
                _logger,
                "research.budget_exhausted",
                "Research Agent skipped — execution budget already exhausted",
                conversation_id=conv_id,
            )
            return {
                "current_agent": AgentState.RESEARCH,
                "current_node": "research",
                "errors": ["research skipped: execution budget exhausted"],
                "budget": budget,
            }

        question = _extract_root_question(state)
        if question is None:
            log_warning(
                _logger,
                "research.no_question",
                "No user question found in state — research skipped",
                conversation_id=conv_id,
            )
            return {
                "current_agent": AgentState.RESEARCH,
                "current_node": "research",
                "errors": ["research skipped: no user question found"],
                "budget": budget,
            }

        ctx = _RunContext(
            conversation_id=conv_id,
            budget=budget,
            max_depth=budget.max_research_depth,
            max_total_tasks=self._max_total_tasks,
        )

        root = ResearchTask(task_id="root", question=question, depth=0)
        root.status = ResearchStatus.IN_PROGRESS
        ctx.tasks.append(root)

        # --- Stage 1: research question analysis --------------------------
        analysis = await self._analyze_question(question, ctx)
        if analysis is None:
            root.status = ResearchStatus.FAILED
            return {
                "research_tasks": ctx.tasks,
                "current_agent": AgentState.RESEARCH,
                "current_node": "research",
                "errors": ["research question analysis failed"],
                "budget": ctx.budget,
            }

        # --- Stage 2: search-plan generation -------------------------------
        queries = _build_search_plan(question, analysis)

        _emit(
            ctx,
            "research_plan",
            {
                "sub_question_count": len(analysis.sub_questions),
                "query_count": len(queries),
                "key_entity_count": len(analysis.key_entities),
            },
        )

        # --- Stage 3: document discovery -----------------------------------
        evidence_pool = await self._discover_documents(queries, roles, ctx)
        if not evidence_pool:
            root.status = ResearchStatus.COMPLETED
            root.result = ResearchResult(
                task_id=root.task_id,
                summary="No relevant documents were found for this research question.",
            )
            log_info(
                _logger,
                "research.no_evidence",
                "Research Agent found no evidence — returning empty result",
                conversation_id=conv_id,
            )
            return {
                "research_tasks": ctx.tasks,
                "research_results": [root.result],
                "current_agent": AgentState.RESEARCH,
                "current_node": "research",
                "budget": ctx.budget,
            }

        evidence_index = {ev.chunk_id: ev for ev in evidence_pool}
        root.document_ids = tuple(evidence_index)

        # --- Stage 4: task decomposition (semantic, by sub-question) ------
        sub_questions = analysis.sub_questions[: self._max_sub_questions]
        pool_ids = tuple(evidence_index)  # already sorted by score (see _discover_documents)

        children: list[ResearchTask] = []
        for index, sub_question in enumerate(sub_questions, start=1):
            if ctx.task_limit_reached:
                break
            child = ResearchTask(
                task_id=f"root-sq{index}",
                question=sub_question,
                document_ids=pool_ids,
                parent_task_id=root.task_id,
                depth=1,
            )
            ctx.tasks.append(child)
            children.append(child)
            _emit(
                ctx,
                "task_created",
                {
                    "task_id": child.task_id,
                    "parent_task_id": root.task_id,
                    "depth": child.depth,
                    "document_count": len(child.document_ids),
                },
            )

        # --- Stages 5-7: recursive resolution of each sub-question --------
        resolved: list[ResearchTask] = []
        for child in children:
            resolved.append(await self._process_task(child, evidence_index, ctx))

        # --- Stage 8: contradiction detection ------------------------------
        contradictions = _detect_contradictions(ctx.claims)
        for c in contradictions:
            _emit(
                ctx,
                "contradiction_detected",
                {
                    "subject": c.subject,
                    "attribute": c.attribute,
                    "conflicting_value_count": len(c.conflicting_values),
                },
            )

        # --- Stage 9: final evidence synthesis ------------------------------
        final = await self._synthesize_final(question, resolved, contradictions, ctx)
        root.status = ResearchStatus.COMPLETED if final is not None else ResearchStatus.FAILED
        root.result = final

        _emit(
            ctx,
            "final_synthesis",
            {
                "task_id": root.task_id,
                "sub_task_count": len(resolved),
                "contradiction_count": len(contradictions),
                "status": root.status.value,
                "total_tasks": len(ctx.tasks),
            },
        )

        log_info(
            _logger,
            "research.completed",
            "Research Agent completed",
            total_tasks=len(ctx.tasks),
            contradiction_count=len(contradictions),
            status=root.status.value,
            conversation_id=conv_id,
        )

        results = [t.result for t in ctx.tasks if t.result is not None]
        update: dict[str, Any] = {
            "research_tasks": ctx.tasks,
            "research_results": results,
            "current_agent": AgentState.RESEARCH,
            "current_node": "research",
            "budget": ctx.budget,
        }
        if final is None:
            update["errors"] = ["research final synthesis failed"]
        return update


# ---------------------------------------------------------------------------
# Public factory
# ---------------------------------------------------------------------------


def make_research_node(
    *,
    client: anthropic.AsyncAnthropic,
    retriever: HybridRetriever,
    model: str | None = None,
) -> Callable[[GraphState], Coroutine[Any, Any, dict[str, Any]]]:
    """Return a LangGraph-compatible async callable for the research node.

    Parameters
    ----------
    client:
        An ``anthropic.AsyncAnthropic`` instance. Injected for testability.
    retriever:
        An initialised ``HybridRetriever`` — shared with the retrieval node;
        document discovery uses the same RBAC-filtered search path.
    model:
        Model identifier. Defaults to ``Settings.model_name``.
    """
    resolved_model = model or get_settings().model_name
    agent = ResearchAgent(client=client, model=resolved_model, retriever=retriever)
    return agent.run
