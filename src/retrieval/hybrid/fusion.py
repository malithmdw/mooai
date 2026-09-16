"""Reciprocal Rank Fusion for hybrid dense + sparse retrieval.

The algorithm is implemented explicitly here — not hidden behind a
generic "merge" abstraction — so it can be audited, tested in isolation,
and reasoned about without reading downstream code.

Algorithm (Cormack, Clarke, Buettcher 2009)
-------------------------------------------
Given ranked lists L_dense and L_sparse, the RRF score for document d is:

    RRF(d) = Σ_{L ∈ {dense, sparse}}  1 / (k + rank_L(d))

where k = 60 (the standard damping constant that prevents very high ranks
from dominating) and rank_L(d) is the 1-based position of d in list L.
Documents absent from a list contribute 0 from that list.

Score normalisation
-------------------
Raw dense scores (cosine similarity, 0–1) and raw sparse scores (BM25, 0–∞)
are on incomparable scales.  Before storing them on ``RetrievalEvidence``,
each list is independently min-max normalised to [0, 1] so the stored
``dense_score`` and ``sparse_score`` fields are human-readable and comparable
across queries.  Normalisation does NOT affect RRF ranking — only ranks matter
for the RRF formula.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

from src.retrieval.hybrid.models import RetrievalEvidence, RetrievalSource
from src.retrieval.ingestion.models import DocumentChunk

RRF_K: Final[int] = 60


# ---------------------------------------------------------------------------
# Typed hit types — one per retriever
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DenseHit:
    """One result from Pinecone dense vector search.

    ``score`` is the raw cosine similarity returned by Pinecone (0–1).
    ``rank`` is the 1-based position within the dense result list.
    """

    chunk: DocumentChunk
    score: float   # cosine similarity, 0–1
    rank: int      # 1-based position in the dense result list


@dataclass(frozen=True)
class SparseHit:
    """One result from BM25 sparse keyword search.

    ``score`` is the raw BM25 score (0–∞, higher is better).
    ``rank`` is the 1-based position within the sparse result list.
    """

    chunk: DocumentChunk
    score: float   # BM25 score, 0–∞
    rank: int      # 1-based position in the sparse result list


# ---------------------------------------------------------------------------
# Score normalisation
# ---------------------------------------------------------------------------


def normalize_scores(scores: list[float]) -> list[float]:
    """Min-max normalise *scores* to the range [0, 1].

    When all values are equal (including a single-item list), returns a
    list of 1.0 so no result is penalised for being the only candidate.

    This is used to make ``dense_score`` and ``sparse_score`` on
    ``RetrievalEvidence`` human-readable; it has no effect on RRF ranking.
    """
    if not scores:
        return []
    lo = min(scores)
    hi = max(scores)
    if hi == lo:
        return [1.0] * len(scores)
    span = hi - lo
    return [(s - lo) / span for s in scores]


# ---------------------------------------------------------------------------
# Reciprocal Rank Fusion
# ---------------------------------------------------------------------------


def reciprocal_rank_fusion(
    dense: list[DenseHit],
    sparse: list[SparseHit],
    *,
    k: int = RRF_K,
    top_k: int = 10,
) -> list[RetrievalEvidence]:
    """Merge *dense* and *sparse* results using Reciprocal Rank Fusion.

    Steps
    -----
    1. Normalise dense scores and sparse scores independently to [0, 1].
    2. For each unique chunk_id, accumulate its RRF score:
           rrf_score += 1 / (k + rank)  for each retriever that returned it.
    3. Sort merged candidates by ``rrf_score`` descending.
       Tie-break: ``chunk_id`` ascending (deterministic across identical queries).
    4. Take the top *top_k* and emit as ``RetrievalEvidence``.

    The ``source`` field on each result records which retriever(s) contributed:
    ``DENSE``, ``SPARSE``, or ``BOTH``.

    Parameters
    ----------
    dense:
        Results from Pinecone, ordered by cosine similarity descending.
        Rank is inferred from list position if not set in the caller.
    sparse:
        Results from BM25, ordered by BM25 score descending.
    k:
        RRF damping constant (default 60).  Lower values amplify the
        advantage of top-ranked documents; higher values flatten it.
    top_k:
        Maximum number of ``RetrievalEvidence`` items to return.
    """
    # Step 1 — normalise raw scores for storage on RetrievalEvidence.
    dense_norm_scores: list[float] = normalize_scores([h.score for h in dense])
    sparse_norm_scores: list[float] = normalize_scores([h.score for h in sparse])

    # Step 2 — accumulate RRF scores.
    # Each dict maps chunk_id → value for its retriever.
    rrf: dict[str, float] = {}           # chunk_id → accumulated RRF score
    norm_dense: dict[str, float] = {}    # chunk_id → normalised dense score
    norm_sparse: dict[str, float] = {}   # chunk_id → normalised sparse score
    chunks: dict[str, DocumentChunk] = {}  # chunk_id → DocumentChunk

    for hit, norm in zip(dense, dense_norm_scores):
        cid = hit.chunk.chunk_id
        rrf[cid] = rrf.get(cid, 0.0) + 1.0 / (k + hit.rank)
        norm_dense[cid] = norm
        chunks[cid] = hit.chunk

    for hit, norm in zip(sparse, sparse_norm_scores):
        cid = hit.chunk.chunk_id
        rrf[cid] = rrf.get(cid, 0.0) + 1.0 / (k + hit.rank)
        norm_sparse[cid] = norm
        chunks[cid] = hit.chunk

    # Step 3 — sort by RRF score descending; tie-break by chunk_id ascending.
    ordered = sorted(rrf.items(), key=lambda item: (-item[1], item[0]))[:top_k]

    # Step 4 — build RetrievalEvidence with full provenance.
    results: list[RetrievalEvidence] = []
    for rank_pos, (cid, rrf_score) in enumerate(ordered):
        in_dense = cid in norm_dense
        in_sparse = cid in norm_sparse

        if in_dense and in_sparse:
            source = RetrievalSource.BOTH
        elif in_dense:
            source = RetrievalSource.DENSE
        else:
            source = RetrievalSource.SPARSE

        chunk = chunks[cid]
        results.append(
            RetrievalEvidence(
                chunk_id=cid,
                document_id=chunk.document_id,
                title=chunk.title,
                text=chunk.text,
                metadata=chunk.metadata,
                source=source,
                dense_score=norm_dense.get(cid),
                sparse_score=norm_sparse.get(cid),
                final_score=rrf_score,
                rank=rank_pos + 1,
            )
        )

    return results
