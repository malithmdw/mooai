"""Authoritative access-control filter for hybrid retrieval results.

This module is the final RBAC gate before evidence reaches the agent layer.
It sits AFTER Reciprocal Rank Fusion so it operates on the merged, ranked
result list — not on individual retriever outputs.

Why post-RRF rather than only at query time?
--------------------------------------------
Both the Pinecone query-time filter and the BM25 corpus filter apply RBAC
during retrieval for efficiency (they shrink the candidate pool early).
However those filters are best-effort optimisations:
- Pinecone's metadata filter could theoretically be misconfigured.
- The BM25 corpus might be rebuilt from a checkpoint that predates an
  access-level change.

This post-RRF filter is the authoritative enforcement point — it is always
executed, unconditionally, before any result is returned to the caller.
Access filtering here cannot be skipped by a misconfigured upstream layer.

After filtering, ranks are re-assigned starting at 1 so the ``rank`` field
on ``RetrievalEvidence`` always reflects the final position in the list
delivered to the agent.
"""

from __future__ import annotations

from collections.abc import Sequence

from src.models.enums import AccessLevel, Role
from src.retrieval.hybrid.models import RetrievalEvidence


def apply_access_filter(
    evidence: list[RetrievalEvidence],
    *,
    roles: Sequence[Role] | None = None,
    access_levels: Sequence[AccessLevel] | None = None,
    department: str | None = None,
    document_type: str | None = None,
) -> list[RetrievalEvidence]:
    """Filter *evidence* and re-rank the survivors.

    Every predicate that is not ``None`` must be satisfied for a result to
    pass.  Predicates that are ``None`` are unconstrained — all results pass
    that predicate.

    Access predicates (roles, access_levels) check the ``metadata`` field
    on each ``RetrievalEvidence`` item, mirroring the RBAC logic in
    ``src.retrieval.indexing.metadata.build_access_filter`` and
    ``src.retrieval.bm25.corpus._passes_filter``.

    After filtering, ``rank`` is reassigned starting at 1 so there are no
    gaps in the sequence delivered to callers.

    Parameters
    ----------
    evidence:
        RRF-ranked list of ``RetrievalEvidence`` items.
    roles:
        If given, only items whose ``allowed_roles`` overlap with this set
        are kept.  An empty sequence matches nothing.
    access_levels:
        If given, only items whose ``access_level`` is in this set are kept.
    department:
        If given, only items from this department (exact match) are kept.
    document_type:
        If given, only items of this document type (exact match) are kept.
    """
    role_set: frozenset[str] | None = (
        frozenset(r.value for r in roles) if roles is not None else None
    )
    level_set: frozenset[str] | None = (
        frozenset(lv.value for lv in access_levels) if access_levels is not None else None
    )

    survivors: list[RetrievalEvidence] = []
    for ev in evidence:
        if role_set is not None:
            doc_roles = {r.value for r in ev.metadata.allowed_roles}
            if not role_set.intersection(doc_roles):
                continue
        if level_set is not None and ev.metadata.access_level.value not in level_set:
            continue
        if department is not None and ev.metadata.department != department:
            continue
        if document_type is not None and ev.metadata.document_type != document_type:
            continue
        survivors.append(ev)

    # Re-assign ranks so the final list is always 1-based with no gaps.
    return [ev.model_copy(update={"rank": i + 1}) for i, ev in enumerate(survivors)]
