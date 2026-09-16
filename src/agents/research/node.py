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

import re
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
    IncidentAnalysisResult,
    IncidentFinalAnswer,
    IncidentRootCauseFinding,
    QuestionAnalysis,
    RecurringRootCause,
    ResearchFinding,
)
from src.agents.research.prompts import (
    build_aggregation_prompt,
    build_batch_analysis_prompt,
    build_final_synthesis_prompt,
    build_incident_batch_prompt,
    build_incident_final_answer_prompt,
    build_question_analysis_prompt,
)
from src.agents.state import GraphState
from src.core.config import get_settings
from src.core.logging import get_logger, log_error, log_info, log_warning
from src.models.agent_events import AgentEvent
from src.models.enums import AccessLevel, AgentState, MessageRole, ResearchStatus, Role
from src.models.evidence import Evidence
from src.models.research import ResearchResult, ResearchTask
from src.retrieval.hybrid.filtering import apply_access_filter
from src.retrieval.hybrid.models import RetrievalEvidence
from src.retrieval.hybrid.retriever import HybridRetriever
from src.security.authorization import AuthorizationPolicy, Permission
from src.tools.analysis import AnalysisOperation, AnalysisRequest, run_analysis

_logger = get_logger(__name__)
_policy = AuthorizationPolicy()

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

RECURRING_THRESHOLD: int = 2
"""Minimum distinct incidents citing the same root cause to call it "recurring"."""

# Role -> permitted access levels, mirroring `src.agents.retrieval.node`'s
# mapping (and `src.agents.guardrails.citation_validation`'s). Kept as a
# small local copy rather than importing another module's private mapping —
# see `_filter_by_authorization`, the incident workflow's explicit,
# separately-audited re-check of what the general pipeline's retriever call
# already enforces once (defense in depth, CLAUDE.md).
_ROLE_ACCESS_LEVELS: dict[Role, frozenset[AccessLevel]] = {
    Role.VIEWER: frozenset({AccessLevel.PUBLIC, AccessLevel.INTERNAL}),
    Role.ENGINEER: frozenset({AccessLevel.PUBLIC, AccessLevel.INTERNAL}),
    Role.ANALYST: frozenset({AccessLevel.PUBLIC, AccessLevel.INTERNAL, AccessLevel.CONFIDENTIAL}),
    Role.ADMINISTRATOR: frozenset(AccessLevel),
}


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

_ROOT_CAUSE_ASSERTION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "incident_id": {
            "type": "string",
            "description": "The incident's document/subject ID.",
        },
        "cause": {"type": "string", "description": "The asserted root cause."},
        "chunk_ids": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Evidence chunk_ids this assertion is drawn from.",
        },
    },
    "required": ["incident_id", "cause", "chunk_ids"],
}

_INCIDENT_BATCH_TOOL: dict[str, Any] = {
    "name": "incident_root_causes",
    "description": (
        "Record whether this batch concerns a payment outage/failure and any root "
        "causes found. Call this tool exactly once."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "relevant": {
                "type": "boolean",
                "description": "True if this batch concerns a payment outage/failure incident.",
            },
            "summary": {"type": "string"},
            "root_causes": {
                "type": "array",
                "items": _ROOT_CAUSE_ASSERTION_SCHEMA,
            },
        },
        "required": ["relevant", "summary", "root_causes"],
    },
}

_INCIDENT_FINAL_ANSWER_TOOL: dict[str, Any] = {
    "name": "incident_final_answer",
    "description": (
        "Record the final narrative summary and limitations. Call this tool exactly "
        "once. Do NOT invent, round, or restate any count not given to you verbatim."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "summary": {"type": "string"},
            "limitations": {"type": "string"},
        },
        "required": ["summary", "limitations"],
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
# Incident root-cause analysis workflow — pure helpers
# ---------------------------------------------------------------------------


def _access_levels_for_roles(roles: Sequence[Role]) -> list[AccessLevel]:
    """Union of access levels permitted by *roles* — see `_ROLE_ACCESS_LEVELS`."""
    permitted: set[AccessLevel] = set()
    for role in roles:
        permitted.update(_ROLE_ACCESS_LEVELS.get(role, frozenset()))
    return list(permitted)


def _filter_by_authorization(
    evidence: Sequence[RetrievalEvidence], *, roles: Sequence[Role]
) -> tuple[list[RetrievalEvidence], list[RetrievalEvidence]]:
    """Stage 2 — explicit, separately-audited authorization re-check.

    `HybridRetriever.search` already RBAC-filters by role (twice — see its
    own docstring); this applies the same authoritative filter again, this
    time also constrained by access level, as its own visible pipeline
    stage with its own `AgentEvent` — defense in depth, and it makes the
    authorization boundary inspectable in this workflow's trace, not just
    enforced silently inside the retriever.

    Returns `(authorized, rejected)`.
    """
    access_levels = _access_levels_for_roles(roles)
    authorized = apply_access_filter(list(evidence), roles=roles, access_levels=access_levels)
    authorized_ids = {ev.chunk_id for ev in authorized}
    rejected = [ev for ev in evidence if ev.chunk_id not in authorized_ids]
    return authorized, rejected


def _verify_claims(
    claims: Sequence[Claim], authorized_index: dict[str, RetrievalEvidence]
) -> tuple[list[Claim], int]:
    """Stage 8 — drop any claim not grounded in the authorized evidence pool.

    A claim is verified only if every `chunk_id` it cites is present in
    *authorized_index* — i.e. it was both actually retrieved and passed
    authorization. This is what stops a hallucinated or unauthorized
    "root cause" from ever reaching the counting step, let alone the user.
    Runs before counting (not literally last in the pipeline) so that
    Stage 6's counts are only ever built from verified data — see
    `ResearchAgent.analyze_incident_root_causes`.

    Returns `(verified_claims, unverified_count)`.
    """
    verified: list[Claim] = []
    unverified = 0
    for claim in claims:
        if claim.chunk_ids and all(cid in authorized_index for cid in claim.chunk_ids):
            verified.append(claim)
        else:
            unverified += 1
    return verified, unverified


def _select_analytics_role(roles: Sequence[Role]) -> Role | None:
    """First held role authorized for `Permission.ANALYTICS`, or `None`.

    The Python Analysis Tool is invoked on the requesting user's behalf —
    see CLAUDE.md "Agents must never bypass application authorization" —
    so if no held role carries analytics permission, the caller must skip
    the tool rather than call it anyway.
    """
    for role in roles:
        if _policy.is_allowed(role, Permission.ANALYTICS):
            return role
    return None


_NUMBER_PATTERN = re.compile(r"\d+(?:\.\d+)?%?")


def _numbers_in(text: str) -> set[str]:
    """Every standalone digit sequence appearing in *text*.

    Used both to find numbers to validate and to harvest "contextually
    legitimate" numbers (a year mentioned in the question, digits embedded
    in an incident/document ID) so they are never mistaken for an invented
    statistic — see `_unsupported_numbers`.
    """
    return set(_NUMBER_PATTERN.findall(text))


def _unsupported_numbers(text: str, allowed: set[str]) -> list[str]:
    """Stage 9 guard — standalone numbers in *text* not present in *allowed*.

    See CLAUDE.md "LLM output must be validated before use" and this
    workflow's "do not allow unsupported statistical claims" requirement:
    every number the model states must be traceable to a number it was
    actually given (see `ResearchAgent._generate_incident_final_answer`,
    which builds *allowed* from verified counts plus every number that
    legitimately appears in the question or a known incident/document ID —
    e.g. "2025" or the "001" in "INC-2025-001" — so restating those is
    never flagged as an invented statistic).
    """
    found = _numbers_in(text)
    return sorted(found - allowed)


def _build_fallback_summary(
    recurring: Sequence[RecurringRootCause], supporting_document_ids: Sequence[str]
) -> str:
    """Deterministic, template-only summary used when the LLM's narrative
    cannot be trusted (no tool-use block, or it stated an unverified
    number) — mirrors `SAFE_FAILURE_RESPONSE` in
    `src.agents.guardrails.citation_validation`: safe, verified content
    only, never LLM prose.
    """
    if not recurring:
        return (
            f"Reviewed {len(supporting_document_ids)} supporting document(s); "
            "no root cause recurred across more than one incident in the evidence examined."
        )
    lines = [f"Reviewed {len(supporting_document_ids)} supporting document(s)."]
    lines.append("Recurring root causes:")
    for cause in recurring:
        incidents = ", ".join(cause.incident_ids)
        lines.append(f"- {cause.root_cause}: {cause.count} incident(s) ({incidents})")
    return "\n".join(lines)


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
        self,
        queries: list[str],
        roles: Sequence[Role],
        ctx: _RunContext,
        *,
        document_type: str | None = None,
    ) -> list[RetrievalEvidence]:
        """Run every search-plan query and merge results into one pool.

        Deduplicated by `chunk_id` (highest score kept), capped at
        `max_evidence_pool` — this pool is the largest amount of evidence
        ever held in memory at once, and it is never sent to the LLM in a
        single call; only bounded batches of it are (see
        `_partition_into_batches`). `document_type` narrows discovery to one
        document type (e.g. `"incident_report"`) when the caller already
        knows the corpus of interest — see
        `ResearchAgent.analyze_incident_root_causes`.
        """
        best: dict[str, RetrievalEvidence] = {}
        for query_index, query in enumerate(queries):
            try:
                results = await self._retriever.search(
                    query,
                    top_k=self._top_k_per_query,
                    roles=roles,
                    document_type=document_type,
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

    # ==================================================================
    # Incident root-cause analysis workflow
    # ==================================================================
    #
    # Extends the RLM implementation for the demonstration query: "Summarize
    # all payment outages related to payment failures during 2025 and
    # identify recurring root causes." See docs/rlm.md for the full
    # walkthrough. Numbered stages below match the workflow's own spec, not
    # the general pipeline's nine stages above (they overlap but are not
    # identical — this workflow adds an explicit authorization re-check,
    # Python-Analysis-Tool-backed recurrence counting, and an evidence
    # verification gate the general pipeline does not have).

    # ------------------------------------------------------------------
    # Stage 4/5: batch-level root-cause extraction
    # ------------------------------------------------------------------

    async def _extract_root_causes_batch(
        self, question: str, batch: list[RetrievalEvidence], *, batch_index: int, ctx: _RunContext
    ) -> IncidentRootCauseFinding | None:
        _emit(
            ctx,
            "batch_analysis",
            {"batch_index": batch_index, "chunk_count": len(batch)},
        )
        return await self._call_structured(
            system_prompt=build_incident_batch_prompt(question, batch),
            user_content=f"Extract payment-outage root causes for: {question}",
            tool=_INCIDENT_BATCH_TOOL,
            tool_name="incident_root_causes",
            model_cls=IncidentRootCauseFinding,
            ctx=ctx,
        )

    # ------------------------------------------------------------------
    # Stages 3-7 (recursive): partition, analyze batches, aggregate
    # ------------------------------------------------------------------

    async def _process_incident_task(
        self,
        task: ResearchTask,
        question: str,
        evidence_index: dict[str, RetrievalEvidence],
        ctx: _RunContext,
    ) -> ResearchTask:
        """Recursive counterpart to `_process_task`, specialized for
        root-cause extraction: same partition/recurse/aggregate structure
        and the same depth/task budget guards (so this workflow is bounded
        exactly like the general pipeline — see the module docstring), but
        leaf analysis calls `_extract_root_causes_batch` instead of the
        generic `_analyze_batch`, and every extracted root cause is
        converted to a `Claim(attribute="root_cause", ...)` and pushed onto
        `ctx.claims` for Stage 6 (recurring-pattern counting) and Stage 8
        (evidence verification) to consume afterward.
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
                task_id=task.task_id, summary="No evidence was available for this batch."
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

        if len(batches) <= 1 or at_depth_limit or at_task_limit:
            chosen = batches[0] if batches else []
            finding = await self._extract_root_causes_batch(
                question, chosen, batch_index=0, ctx=ctx
            )
            if finding is None:
                task.status = ResearchStatus.FAILED
                return task

            for assertion in finding.root_causes:
                ctx.claims.append(
                    Claim(
                        subject=assertion.incident_id,
                        attribute="root_cause",
                        value=assertion.cause,
                        chunk_ids=assertion.chunk_ids,
                    )
                )

            summary = (
                finding.summary
                if finding.relevant
                else f"Not relevant to payment outages: {finding.summary}"
            )
            if len(batches) > 1:
                summary += " (evidence truncated: recursion/task budget reached)"

            task.status = ResearchStatus.COMPLETED
            task.result = ResearchResult(
                task_id=task.task_id, summary=summary, evidence=_to_evidence(chosen)
            )
            return task

        child_tasks: list[ResearchTask] = []
        for batch_index, batch in enumerate(batches):
            if ctx.task_limit_reached:
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
            resolved_child = await self._process_incident_task(
                child, question, evidence_index, ctx
            )  # recursion
            child_tasks.append(resolved_child)

        aggregated = await self._aggregate(task, child_tasks, ctx)
        task.status = ResearchStatus.COMPLETED if aggregated is not None else ResearchStatus.FAILED
        task.result = aggregated
        _emit(
            ctx,
            "aggregation",
            {"task_id": task.task_id, "child_count": len(child_tasks), "status": task.status.value},
        )
        return task

    # ------------------------------------------------------------------
    # Stage 6: identify recurring patterns (Python Analysis Tool)
    # ------------------------------------------------------------------

    async def _identify_recurring_patterns(
        self, claims: Sequence[Claim], roles: Sequence[Role], ctx: _RunContext
    ) -> tuple[list[RecurringRootCause], str | None]:
        """Deterministic frequency counting over VERIFIED root-cause claims,
        via `src.tools.analysis.run_analysis` — never an LLM guess. One
        claim per incident is counted (first one seen), so a single
        incident can never itself inflate a "recurring" count.

        Returns `(recurring_causes, skip_reason)`; `skip_reason` is set
        (and `recurring_causes` empty) when there is nothing to count or
        the caller's role lacks `Permission.ANALYTICS` — the workflow
        degrades gracefully rather than bypassing authorization to run the
        tool anyway.
        """
        root_cause_claims = [c for c in claims if c.attribute.strip().lower() == "root_cause"]
        if not root_cause_claims:
            return [], None

        per_incident_norm: dict[str, str] = {}
        display_by_norm: dict[str, str] = {}
        for claim in root_cause_claims:
            incident_id = claim.subject.strip()
            cause_norm = claim.value.strip().lower()
            per_incident_norm.setdefault(incident_id, cause_norm)
            display_by_norm.setdefault(cause_norm, claim.value.strip())

        role = _select_analytics_role(roles)
        if role is None:
            log_warning(
                _logger,
                "research.analytics_permission_denied",
                "Recurring-pattern analysis skipped — no held role has analytics permission",
                conversation_id=ctx.conversation_id,
            )
            return [], (
                "Recurring-pattern analysis was skipped: no held role has analytics permission."
            )

        request = AnalysisRequest(
            operation=AnalysisOperation.COUNT_BY,
            data=[
                {"incident_id": iid, "root_cause": cause}
                for iid, cause in per_incident_norm.items()
            ],
            parameters={"field": "root_cause"},
        )
        try:
            analysis_result = await run_analysis(request, role=role)
        except Exception as exc:
            log_error(
                _logger,
                "research.analytics_error",
                "Python Analysis Tool call failed during recurring-pattern detection",
                error=exc,
                conversation_id=ctx.conversation_id,
            )
            return [], "Recurring-pattern analysis failed and was skipped."

        ctx.budget = ctx.budget.consume(tool_calls=1)
        counts: dict[str, int] = analysis_result.result["counts"]

        recurring: list[RecurringRootCause] = []
        for cause_norm, count in counts.items():
            if count < RECURRING_THRESHOLD:
                continue
            incident_ids = tuple(
                sorted(iid for iid, c in per_incident_norm.items() if c == cause_norm)
            )
            recurring.append(
                RecurringRootCause(
                    root_cause=display_by_norm[cause_norm],
                    count=count,
                    incident_ids=incident_ids,
                )
            )

        recurring.sort(key=lambda r: (-r.count, r.root_cause))
        _emit(
            ctx,
            "recurring_patterns",
            {
                "recurring_cause_count": len(recurring),
                "incidents_considered": len(per_incident_norm),
            },
        )
        return recurring, None

    # ------------------------------------------------------------------
    # Stage 9: generate final answer
    # ------------------------------------------------------------------

    async def _generate_incident_final_answer(
        self,
        question: str,
        aggregated_summary: str,
        recurring: list[RecurringRootCause],
        supporting_document_ids: tuple[str, ...],
        unverified_count: int,
        analytics_skip_reason: str | None,
        ctx: _RunContext,
    ) -> IncidentAnalysisResult:
        recurring_lines = [
            f"{r.root_cause}: {r.count} incidents ({', '.join(r.incident_ids)})" for r in recurring
        ]
        facts: list[str] = [f"Supporting documents reviewed: {len(supporting_document_ids)}"]
        if unverified_count:
            facts.append(f"Unverified claims excluded: {unverified_count}")
        if analytics_skip_reason:
            facts.append(analytics_skip_reason)

        # Every number the LLM is permitted to state — its own summary is
        # rejected below if it states any number outside this set. Includes
        # both verified statistics AND numbers that legitimately appear in
        # the question or a known ID (a year, a case number) so restating
        # those is never mistaken for an invented statistic — see
        # `_unsupported_numbers`.
        allowed_numbers = {str(r.count) for r in recurring}
        allowed_numbers |= {str(len(r.incident_ids)) for r in recurring}
        allowed_numbers |= {str(len(supporting_document_ids)), str(len(recurring))}
        allowed_numbers |= {str(unverified_count)} if unverified_count else set()
        allowed_numbers |= _numbers_in(question)
        for cause in recurring:
            for incident_id in cause.incident_ids:
                allowed_numbers |= _numbers_in(incident_id)
        for document_id in supporting_document_ids:
            allowed_numbers |= _numbers_in(document_id)

        synthesis = await self._call_structured(
            system_prompt=build_incident_final_answer_prompt(
                question, aggregated_summary, recurring_lines, list(supporting_document_ids), facts
            ),
            user_content=f"Write the final narrative summary for: {question}",
            tool=_INCIDENT_FINAL_ANSWER_TOOL,
            tool_name="incident_final_answer",
            model_cls=IncidentFinalAnswer,
            ctx=ctx,
        )

        extra_limitations: list[str] = [
            "Document date ranges (e.g. a stated year) were not independently "
            "verified against document metadata; results rely on retrieval "
            "relevance and the evidence text itself."
        ]
        if analytics_skip_reason:
            extra_limitations.append(analytics_skip_reason)
        if unverified_count:
            extra_limitations.append(
                f"{unverified_count} extracted claim(s) could not be verified against "
                "authorized evidence and were excluded from all counts."
            )
        if not recurring:
            extra_limitations.append(
                "No root cause recurred across more than one incident in the evidence examined."
            )

        if synthesis is None:
            summary = _build_fallback_summary(recurring, supporting_document_ids)
            limitations = " ".join(extra_limitations) or "Final synthesis could not be generated."
        else:
            bad_numbers = _unsupported_numbers(synthesis.summary, allowed_numbers)
            if bad_numbers:
                log_warning(
                    _logger,
                    "research.unsupported_statistic_blocked",
                    "Final summary stated unverified figures; replaced with a deterministic one",
                    bad_numbers=bad_numbers,
                    conversation_id=ctx.conversation_id,
                )
                summary = _build_fallback_summary(recurring, supporting_document_ids)
                extra_limitations.append(
                    f"The narrative summary was replaced because it stated figures "
                    f"({', '.join(bad_numbers)}) not present in the verified data."
                )
            else:
                summary = synthesis.summary

            limitations = synthesis.limitations
            if extra_limitations:
                limitations = f"{limitations} {' '.join(extra_limitations)}".strip()

        return IncidentAnalysisResult(
            summary=summary,
            recurring_root_causes=tuple(recurring),
            supporting_document_ids=supporting_document_ids,
            limitations=limitations,
            unverified_claim_count=unverified_count,
        )

    # ------------------------------------------------------------------
    # Public entry point for the demonstration workflow
    # ------------------------------------------------------------------

    async def analyze_incident_root_causes(
        self,
        *,
        question: str,
        roles: Sequence[Role],
        budget: ExecutionBudget,
        conversation_id: str | None = None,
        document_type: str | None = "incident_report",
    ) -> IncidentAnalysisResult:
        """Recursive incident root-cause analysis — the RLM demonstration
        workflow (see docs/rlm.md): discover -> filter by authorization ->
        partition -> analyze batches independently -> extract root causes
        -> identify recurring patterns -> aggregate -> verify evidence ->
        generate final answer.

        Unlike `run()`, this does not read from a `GraphState` — it takes
        exactly what it needs (question, roles, budget) so it can be
        invoked directly for this specific query, e.g. from a script or a
        test, without constructing a full chat turn.
        """
        ctx = _RunContext(
            conversation_id=conversation_id,
            budget=budget,
            max_depth=budget.max_research_depth,
            max_total_tasks=self._max_total_tasks,
        )

        if budget.is_exhausted:
            log_warning(
                _logger,
                "research.budget_exhausted",
                "Incident analysis skipped — execution budget already exhausted",
                conversation_id=conversation_id,
            )
            return IncidentAnalysisResult(
                summary="Analysis could not run: execution budget already exhausted.",
                limitations="No analysis was performed because the execution budget was exhausted.",
            )

        # --- Stage 1: discover relevant documents --------------------------
        pool = await self._discover_documents([question], roles, ctx, document_type=document_type)
        if not pool:
            return IncidentAnalysisResult(
                summary="No relevant documents were found for this query.",
                limitations="The knowledge base returned no matching documents.",
            )

        # --- Stage 2: filter by authorization -------------------------------
        authorized, rejected = _filter_by_authorization(pool, roles=roles)
        _emit(
            ctx,
            "authorization_filter",
            {"authorized_count": len(authorized), "rejected_count": len(rejected)},
        )
        if not authorized:
            return IncidentAnalysisResult(
                summary="No authorized evidence was found for this query.",
                limitations=(
                    f"{len(rejected)} document(s) were found but excluded because the "
                    "requesting user's role does not permit access to them."
                ),
            )

        authorized_index = {ev.chunk_id: ev for ev in authorized}

        # --- Stages 3-7: partition, analyze independently, aggregate -------
        root_task = ResearchTask(
            task_id="incident-root", question=question, document_ids=tuple(authorized_index)
        )
        ctx.tasks.append(root_task)
        resolved_root = await self._process_incident_task(
            root_task, question, authorized_index, ctx
        )
        aggregated_summary = (
            resolved_root.result.summary
            if resolved_root.result is not None
            else "No summary could be produced from the retrieved evidence."
        )

        # --- Stage 8 (run before counting, so counts are always built from
        # verified data only — see `_verify_claims` docstring) -------------
        verified_claims, unverified_count = _verify_claims(ctx.claims, authorized_index)
        _emit(
            ctx,
            "verify_evidence",
            {"verified_count": len(verified_claims), "unverified_count": unverified_count},
        )

        # --- Stage 6: identify recurring patterns ---------------------------
        recurring, analytics_skip_reason = await self._identify_recurring_patterns(
            verified_claims, roles, ctx
        )

        supporting_document_ids = tuple(
            sorted(
                {
                    authorized_index[cid].document_id
                    for claim in verified_claims
                    for cid in claim.chunk_ids
                    if cid in authorized_index
                }
            )
        )

        # --- Stage 9: generate final answer ----------------------------------
        result = await self._generate_incident_final_answer(
            question,
            aggregated_summary,
            recurring,
            supporting_document_ids,
            unverified_count,
            analytics_skip_reason,
            ctx,
        )

        log_info(
            _logger,
            "research.incident_analysis_completed",
            "Incident root-cause analysis completed",
            total_tasks=len(ctx.tasks),
            recurring_cause_count=len(recurring),
            supporting_document_count=len(supporting_document_ids),
            unverified_claim_count=unverified_count,
            conversation_id=conversation_id,
        )

        return result


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
