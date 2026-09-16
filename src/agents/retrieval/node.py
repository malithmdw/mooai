"""Retrieval Agent LangGraph node.

Responsibilities
----------------
1. Generate search queries from the GraphState.
2. Call ``HybridRetriever.search`` with RBAC-derived access filters.
3. Verify authoritative post-retrieval access filtering (done inside the
   retriever — see ``src.retrieval.hybrid.filtering``).
4. Deduplicate results across multiple queries.
5. Detect insufficient evidence.
6. Emit ``AgentEvent`` records for each observable pipeline step.

Security principles
-------------------
- Retrieved document text is UNTRUSTED DATA.  It must never appear in log
  records, AgentEvent metadata, or any execution context.  Only safe
  metadata (counts, scores, chunk_ids, lengths) is recorded.
- RBAC roles and access levels are derived from ``GraphState.user_role``
  and are **always** passed to the retriever.  The retrieval agent must
  never call the retriever without authorization parameters.
- Document content is passed as opaque ``RetrievalEvidence`` objects; it is
  the downstream agent's responsibility to handle it as untrusted data.
"""

from __future__ import annotations

from collections.abc import Callable, Coroutine
from typing import Any
from uuid import uuid4

from src.agents.retrieval.queries import generate_queries
from src.agents.state import GraphState
from src.core.logging import get_logger, log_debug, log_error, log_info, log_warning
from src.models.agent_events import AgentEvent
from src.models.enums import AccessLevel, AgentState, Role
from src.retrieval.hybrid.models import RetrievalEvidence
from src.retrieval.hybrid.retriever import HybridRetriever

_logger = get_logger(__name__)

DEFAULT_TOP_K: int = 10
INSUFFICIENT_EVIDENCE_THRESHOLD: int = 1

# ---------------------------------------------------------------------------
# Role → permitted access-level mapping
# ---------------------------------------------------------------------------
# Access levels a role may retrieve.  Used to derive the `access_levels`
# parameter forwarded to HybridRetriever.search — a second layer of
# authoritative access control beyond the `allowed_roles` document field.

_ROLE_ACCESS_LEVELS: dict[Role, frozenset[AccessLevel]] = {
    Role.VIEWER: frozenset({AccessLevel.PUBLIC, AccessLevel.INTERNAL}),
    Role.ENGINEER: frozenset({AccessLevel.PUBLIC, AccessLevel.INTERNAL}),
    Role.ANALYST: frozenset(
        {AccessLevel.PUBLIC, AccessLevel.INTERNAL, AccessLevel.CONFIDENTIAL}
    ),
    Role.ADMINISTRATOR: frozenset(AccessLevel),
}


def _access_levels_for_roles(roles: list[Role]) -> list[AccessLevel]:
    """Return the union of access levels permitted by *roles*.

    Returns an empty list when *roles* is empty — callers then get no
    results (secure default: deny when role is unknown).
    """
    permitted: set[AccessLevel] = set()
    for role in roles:
        permitted.update(_ROLE_ACCESS_LEVELS.get(role, frozenset()))
    return list(permitted)


# ---------------------------------------------------------------------------
# AgentEvent helper
# ---------------------------------------------------------------------------


def _emit(
    conversation_id: str | None,
    message: str,
    metadata: dict[str, Any],
) -> AgentEvent:
    """Create and log one retrieval-phase ``AgentEvent``.

    Only primitive JSON-safe values (str, int, float, bool, None) are
    recorded in metadata — document text and raw user queries are excluded
    to prevent leaking untrusted content into the observability layer.
    """
    safe_meta: dict[str, Any] = {
        k: v
        for k, v in metadata.items()
        if isinstance(v, (str, int, float, bool, type(None), list))
    }
    event = AgentEvent(
        event_id=str(uuid4()),
        conversation_id=conversation_id or "no-conversation",
        state=AgentState.RETRIEVAL,
        message=message,
        metadata=safe_meta,
    )
    log_info(
        _logger,
        f"agent_event.retrieval.{message.lower().replace(' ', '_')}",
        message,
        event_id=event.event_id,
        agent_state=event.state.value,
        **safe_meta,
    )
    return event


# ---------------------------------------------------------------------------
# Retrieval Agent
# ---------------------------------------------------------------------------


class RetrievalAgent:
    """Orchestrates query generation, hybrid retrieval, and evidence assembly.

    Inject a ``HybridRetriever`` rather than constructing one here so the
    retriever's external dependencies (Pinecone, BM25 corpus) are managed
    by the application layer, not this node.
    """

    def __init__(
        self,
        *,
        retriever: HybridRetriever,
        top_k: int = DEFAULT_TOP_K,
        min_evidence: int = INSUFFICIENT_EVIDENCE_THRESHOLD,
    ) -> None:
        self._retriever = retriever
        self._top_k = top_k
        self._min_evidence = min_evidence

    async def run(self, state: GraphState) -> dict[str, Any]:
        """Execute the retrieval step and return a partial GraphState update.

        Never raises: all exceptions are captured and returned as ``errors``
        entries so the graph can continue or terminate gracefully.
        """
        budget = state["budget"]
        conv_id = state["conversation_id"]

        if budget.used_retrieval_results >= budget.max_retrieval_results:
            log_warning(
                _logger,
                "retrieval.budget_exhausted",
                "Retrieval skipped — retrieval budget already exhausted",
                conversation_id=conv_id,
                used=budget.used_retrieval_results,
                max=budget.max_retrieval_results,
            )
            return {
                "current_agent": AgentState.RETRIEVAL,
                "current_node": "retrieval",
                "errors": ["retrieval skipped: retrieval budget exhausted"],
                "budget": budget,
            }

        # 1. Generate search queries -------------------------------------------
        queries = generate_queries(state)

        _emit(
            conv_id,
            "query_generation",
            {
                "query_count": len(queries),
                "intent": state["intent"],
                "task_plan_steps": len(state["task_plan"]),
            },
        )

        if not queries:
            log_warning(
                _logger,
                "retrieval.no_queries",
                "No retrieval queries generated — no user messages in state",
                conversation_id=conv_id,
            )
            return {
                "retrieval_queries": [],
                "current_agent": AgentState.RETRIEVAL,
                "current_node": "retrieval",
                "errors": ["retrieval skipped: no queries could be generated"],
                "budget": budget,
            }

        # 2. Derive RBAC filters -----------------------------------------------
        user_roles = state["user_role"]
        access_levels = _access_levels_for_roles(user_roles)

        log_debug(
            _logger,
            "retrieval.rbac_filters",
            "Retrieval RBAC filters derived",
            role_count=len(user_roles),
            access_level_count=len(access_levels),
            conversation_id=conv_id,
        )

        # 3. Execute retrieval for each query ----------------------------------
        # Per-query accumulator: chunk_id → highest-scoring RetrievalEvidence
        best: dict[str, RetrievalEvidence] = {}
        updated_budget = budget

        for query_idx, query in enumerate(queries):
            # Emit dense + sparse search events (both run in parallel inside
            # the retriever — we emit them before the call to record intent).
            _emit(
                conv_id,
                "dense_search",
                {
                    "query_index": query_idx,
                    "query_length": len(query),
                    "top_k": self._top_k,
                },
            )
            _emit(
                conv_id,
                "sparse_search",
                {
                    "query_index": query_idx,
                    "query_length": len(query),
                    "top_k": self._top_k,
                },
            )

            try:
                results = await self._retriever.search(
                    query,
                    top_k=self._top_k,
                    roles=user_roles,
                    access_levels=access_levels if access_levels else None,
                )
            except Exception as exc:
                log_error(
                    _logger,
                    "retrieval.search_error",
                    "HybridRetriever.search failed",
                    error=exc,
                    query_index=query_idx,
                    conversation_id=conv_id,
                )
                return {
                    "retrieval_queries": queries,
                    "current_agent": AgentState.RETRIEVAL,
                    "current_node": "retrieval",
                    "errors": [f"retrieval search failed: {exc}"],
                    "budget": updated_budget,
                }

            top_score: float | None = results[0].final_score if results else None

            _emit(
                conv_id,
                "ranking",
                {
                    "query_index": query_idx,
                    "result_count": len(results),
                    "top_score": top_score,
                },
            )

            # RBAC authorisation was applied twice inside the retriever:
            # once at query-time and once post-RRF.  Emit a filtering event
            # confirming this happened so it is visible in the trace.
            _emit(
                conv_id,
                "filtering",
                {
                    "query_index": query_idx,
                    "result_count": len(results),
                    "roles_applied": len(user_roles),
                    "access_levels_applied": len(access_levels),
                },
            )

            # Consume retrieval budget.
            updated_budget = updated_budget.consume(retrieval_results=len(results))

            # Accumulate: keep the highest-scoring instance per chunk_id.
            for ev in results:
                existing = best.get(ev.chunk_id)
                if existing is None or ev.final_score > existing.final_score:
                    best[ev.chunk_id] = ev

        # 4. Sort and cap -------------------------------------------------------
        final: list[RetrievalEvidence] = sorted(
            best.values(), key=lambda e: -e.final_score
        )[: self._top_k]

        # 5. Detect insufficient evidence -------------------------------------
        has_sufficient = len(final) >= self._min_evidence

        _emit(
            conv_id,
            "final_evidence",
            {
                "unique_results": len(final),
                "query_count": len(queries),
                "sufficient": has_sufficient,
                "chunk_ids": [e.chunk_id for e in final[:10]],
            },
        )

        log_info(
            _logger,
            "retrieval.completed",
            "Retrieval completed",
            query_count=len(queries),
            unique_results=len(final),
            sufficient=has_sufficient,
            conversation_id=conv_id,
        )

        update: dict[str, Any] = {
            "retrieval_queries": queries,
            "retrieved_documents": final,
            "current_agent": AgentState.RETRIEVAL,
            "current_node": "retrieval",
            "budget": updated_budget,
        }

        if not has_sufficient:
            update["errors"] = [
                f"insufficient evidence: retrieved {len(final)} result(s), "
                f"minimum required is {self._min_evidence}"
            ]

        return update


# ---------------------------------------------------------------------------
# Public factory
# ---------------------------------------------------------------------------


def make_retrieval_node(
    *,
    retriever: HybridRetriever,
    top_k: int = DEFAULT_TOP_K,
    min_evidence: int = INSUFFICIENT_EVIDENCE_THRESHOLD,
) -> Callable[[GraphState], Coroutine[Any, Any, dict[str, Any]]]:
    """Return a LangGraph-compatible async callable for the retrieval node.

    Parameters
    ----------
    retriever:
        An initialised ``HybridRetriever``.  Injected so callers control the
        retriever's lifecycle; tests can pass a mock.
    top_k:
        Maximum number of evidence items to return per run.
    min_evidence:
        Minimum number of results required; fewer triggers an ``errors``
        entry for the downstream supervisor to handle.
    """
    agent = RetrievalAgent(retriever=retriever, top_k=top_k, min_evidence=min_evidence)
    return agent.run
