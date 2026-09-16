"""Retrieval Agent — hybrid search, RBAC filtering, and evidence assembly.

Wraps ``HybridRetriever`` in a LangGraph node that generates search queries,
enforces authorization, deduplicates results, and emits ``AgentEvent``
records for every observable pipeline step.

Public surface
--------------
- ``RetrievalAgent``      — Agent class (prefer ``make_retrieval_node``).
- ``make_retrieval_node`` — Factory returning a LangGraph node callable.
- ``generate_queries``    — Query-generation helper (exposed for testing).
- ``DEFAULT_TOP_K``       — Default number of evidence items to return.
- ``INSUFFICIENT_EVIDENCE_THRESHOLD`` — Minimum results before an error is
  added to the graph state.
"""

from src.agents.retrieval.node import (
    DEFAULT_TOP_K,
    INSUFFICIENT_EVIDENCE_THRESHOLD,
    RetrievalAgent,
    make_retrieval_node,
)
from src.agents.retrieval.queries import generate_queries

__all__ = [
    "RetrievalAgent",
    "make_retrieval_node",
    "generate_queries",
    "DEFAULT_TOP_K",
    "INSUFFICIENT_EVIDENCE_THRESHOLD",
]
