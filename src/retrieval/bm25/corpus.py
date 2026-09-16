"""BM25 corpus: in-memory sparse index over DocumentChunk objects.

The corpus wraps ``rank_bm25.BM25Okapi`` (Okapi BM25 algorithm) and adds
RBAC-aware metadata filtering consistent with the Pinecone access-control
filter in ``src.retrieval.indexing.metadata``.

Usage
-----
    from src.retrieval.bm25 import BM25Corpus, BM25Result
    from src.models.enums import Role

    corpus = BM25Corpus(chunks)
    results = corpus.search("payment timeout ERR-429", roles=[Role.ENGINEER])
"""

from __future__ import annotations

import logging
from collections.abc import Sequence

from pydantic import BaseModel, ConfigDict, Field
from rank_bm25 import BM25Okapi

from src.core.logging import get_logger, log_debug, log_info
from src.models.enums import AccessLevel, Role
from src.retrieval.bm25.tokenizer import tokenize
from src.retrieval.ingestion.models import DocumentChunk

logger: logging.Logger = get_logger(__name__)

_DEFAULT_K1: float = 1.5
_DEFAULT_B: float = 0.75


class BM25Result(BaseModel):
    """One document chunk returned by a BM25 search, with its score and rank."""

    model_config = ConfigDict(frozen=True)

    chunk: DocumentChunk
    score: float = Field(ge=0.0)
    rank: int = Field(ge=1)


class BM25Corpus:
    """In-memory BM25 index over a fixed set of ``DocumentChunk`` objects.

    The corpus is immutable once constructed — rebuild it whenever the
    underlying documents change.  This mirrors the full-rebuild model used
    by ``PineconeIndexService``.

    Parameters
    ----------
    chunks:
        All chunks to include in the index.
    k1:
        BM25 term-frequency saturation parameter (default 1.5).  Higher
        values let term frequency keep contributing to the score for longer.
    b:
        BM25 document-length normalisation parameter (default 0.75).  1.0
        fully normalises by document length; 0.0 applies no normalisation.
    """

    def __init__(
        self,
        chunks: Sequence[DocumentChunk],
        *,
        k1: float = _DEFAULT_K1,
        b: float = _DEFAULT_B,
    ) -> None:
        self._chunks: list[DocumentChunk] = list(chunks)
        tokenized = [tokenize(c.text) for c in self._chunks]
        self._bm25: BM25Okapi = BM25Okapi(tokenized, k1=k1, b=b)
        log_info(
            logger,
            "bm25.corpus.built",
            "BM25 corpus built",
            chunk_count=len(self._chunks),
            k1=k1,
            b=b,
        )

    @property
    def chunk_count(self) -> int:
        """Total number of chunks in this corpus."""
        return len(self._chunks)

    def search(
        self,
        query: str,
        *,
        top_k: int = 10,
        roles: Sequence[Role] | None = None,
        access_levels: Sequence[AccessLevel] | None = None,
        department: str | None = None,
        document_type: str | None = None,
    ) -> list[BM25Result]:
        """Return the top-*k* BM25-scored chunks matching *query*.

        Filtering is applied before ranking: chunks that fail the RBAC or
        metadata predicates are excluded regardless of their BM25 score.
        This mirrors the behaviour of Pinecone's ``filter`` parameter.

        Tie-breaking: equal scores are broken by ``chunk_id`` ascending so
        the output is deterministic across identical queries, regardless of
        corpus insertion order.

        Parameters
        ----------
        query:
            Raw search string — tokenized internally by the same tokenizer
            used to build the corpus.
        top_k:
            Maximum number of results to return (after filtering).
        roles:
            If provided, only chunks whose ``allowed_roles`` overlap with
            this set are returned.  An empty sequence matches nothing.
        access_levels:
            If provided, only chunks whose ``access_level`` is in this list
            are returned.
        department:
            If provided, only chunks from this department (exact match).
        document_type:
            If provided, only chunks of this document type (exact match).
        """
        query_tokens = tokenize(query)
        log_debug(
            logger,
            "bm25.search.start",
            "BM25 search started",
            query=query,
            query_tokens=query_tokens,
            top_k=top_k,
        )

        if not query_tokens or not self._chunks:
            return []

        scores: list[float] = self._bm25.get_scores(query_tokens).tolist()

        role_set: frozenset[str] | None = (
            frozenset(r.value for r in roles) if roles is not None else None
        )
        level_set: frozenset[str] | None = (
            frozenset(lv.value for lv in access_levels) if access_levels is not None else None
        )

        # Collect positive-scoring, filter-passing candidates as
        # (score, chunk_id, corpus_index) so we can reconstruct chunks later.
        candidates: list[tuple[float, str, int]] = []
        for i, (chunk, score) in enumerate(zip(self._chunks, scores)):
            if score <= 0.0:
                continue
            if not _passes_filter(chunk, role_set, level_set, department, document_type):
                continue
            candidates.append((score, chunk.chunk_id, i))

        # Primary sort: highest score first.
        # Tie-break: chunk_id ascending → deterministic across runs.
        candidates.sort(key=lambda t: (-t[0], t[1]))
        candidates = candidates[:top_k]

        results = [
            BM25Result(chunk=self._chunks[idx], score=score, rank=rank + 1)
            for rank, (score, _chunk_id, idx) in enumerate(candidates)
        ]

        log_info(
            logger,
            "bm25.search.completed",
            "BM25 search completed",
            query=query,
            result_count=len(results),
            top_k=top_k,
        )
        return results


def _passes_filter(
    chunk: DocumentChunk,
    role_set: frozenset[str] | None,
    level_set: frozenset[str] | None,
    department: str | None,
    document_type: str | None,
) -> bool:
    """Return True if *chunk* satisfies all active filter predicates."""
    if role_set is not None:
        chunk_roles = {r.value for r in chunk.metadata.allowed_roles}
        if not role_set.intersection(chunk_roles):
            return False
    if level_set is not None and chunk.metadata.access_level.value not in level_set:
        return False
    if department is not None and chunk.metadata.department != department:
        return False
    if document_type is not None and chunk.metadata.document_type != document_type:
        return False
    return True
