"""Unit tests for BM25Corpus.

Covers: basic search, empty-corpus/empty-query edge cases, RBAC role
filtering, access-level filtering, department/document-type filtering,
result ordering, top-k limiting, determinism, and — critically — scenarios
where BM25 finds results that a dense (embedding) search would typically
miss or rank poorly.
"""

from __future__ import annotations

from datetime import date

import pytest

from src.models.documents import DocumentMetadata
from src.models.enums import AccessLevel, Role
from src.retrieval.bm25.corpus import BM25Corpus, BM25Result
from src.retrieval.ingestion.models import DocumentChunk


# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------


def _make_meta(
    *,
    document_id: str = "DOC-001",
    department: str = "Engineering",
    document_type: str = "architecture_document",
    access_level: AccessLevel = AccessLevel.INTERNAL,
    allowed_roles: tuple[Role, ...] = (Role.ENGINEER,),
) -> DocumentMetadata:
    return DocumentMetadata(
        document_id=document_id,
        title="Test Document",
        department=department,
        document_type=document_type,
        access_level=access_level,
        created_date=date(2024, 1, 1),
        allowed_roles=allowed_roles,
    )


def _make_chunk(
    *,
    chunk_id: str = "DOC-001-chunk-0000",
    document_id: str = "DOC-001",
    text: str = "sample text content",
    chunk_index: int = 0,
    chunk_total: int = 1,
    metadata: DocumentMetadata | None = None,
) -> DocumentChunk:
    return DocumentChunk(
        chunk_id=chunk_id,
        document_id=document_id,
        title="Test Document",
        section="## Overview",
        chunk_index=chunk_index,
        chunk_total=chunk_total,
        text=text,
        metadata=metadata or _make_meta(document_id=document_id),
    )


# ---------------------------------------------------------------------------
# BM25Result model
# ---------------------------------------------------------------------------


class TestBM25Result:
    def test_is_frozen(self) -> None:
        chunk = _make_chunk()
        result = BM25Result(chunk=chunk, score=1.5, rank=1)
        with pytest.raises(Exception):
            result.score = 99.0  # type: ignore[misc]

    def test_score_ge_zero(self) -> None:
        chunk = _make_chunk()
        with pytest.raises(Exception):
            BM25Result(chunk=chunk, score=-0.1, rank=1)

    def test_rank_ge_one(self) -> None:
        chunk = _make_chunk()
        with pytest.raises(Exception):
            BM25Result(chunk=chunk, score=1.0, rank=0)

    def test_valid_result(self) -> None:
        chunk = _make_chunk()
        result = BM25Result(chunk=chunk, score=0.0, rank=1)
        assert result.chunk is chunk
        assert result.score == 0.0
        assert result.rank == 1


# ---------------------------------------------------------------------------
# BM25Corpus edge cases
# ---------------------------------------------------------------------------


class TestBM25CorpusEdgeCases:
    def test_empty_corpus_returns_empty_list(self) -> None:
        corpus = BM25Corpus([])
        assert corpus.search("payment") == []

    def test_empty_query_returns_empty_list(self) -> None:
        chunk = _make_chunk(text="payment processing system")
        corpus = BM25Corpus([chunk])
        assert corpus.search("") == []

    def test_all_stopword_query_returns_empty_list(self) -> None:
        chunk = _make_chunk(text="payment processing system")
        corpus = BM25Corpus([chunk])
        assert corpus.search("the and or") == []

    def test_chunk_count_property(self) -> None:
        chunks = [_make_chunk(chunk_id=f"DOC-{i:03d}-chunk-0000", document_id=f"DOC-{i:03d}") for i in range(5)]
        corpus = BM25Corpus(chunks)
        assert corpus.chunk_count == 5

    def test_non_matching_query_returns_empty_list(self) -> None:
        chunk = _make_chunk(text="payment processing timeout")
        corpus = BM25Corpus([chunk])
        results = corpus.search("xylophone")
        assert results == []


# ---------------------------------------------------------------------------
# Basic search behaviour
# ---------------------------------------------------------------------------


class TestBM25CorpusSearch:
    def test_matching_chunk_returned(self) -> None:
        chunk = _make_chunk(text="payment processing timeout FPS gateway")
        corpus = BM25Corpus([chunk])
        results = corpus.search("payment timeout")
        assert len(results) == 1
        assert results[0].chunk.chunk_id == chunk.chunk_id

    def test_result_has_positive_score(self) -> None:
        chunk = _make_chunk(text="FPS payment gateway timeout error")
        corpus = BM25Corpus([chunk])
        results = corpus.search("payment gateway")
        assert results[0].score > 0.0

    def test_result_rank_starts_at_one(self) -> None:
        chunk = _make_chunk(text="core banking ledger service")
        corpus = BM25Corpus([chunk])
        results = corpus.search("banking ledger")
        assert results[0].rank == 1

    def test_multiple_results_ranked_by_score_descending(self) -> None:
        # chunk_a has the query term twice; chunk_b once — TF advantage
        chunk_a = _make_chunk(
            chunk_id="DOC-001-chunk-0000",
            document_id="DOC-001",
            text="timeout timeout network issue",
        )
        chunk_b = _make_chunk(
            chunk_id="DOC-002-chunk-0000",
            document_id="DOC-002",
            text="timeout network connectivity",
        )
        corpus = BM25Corpus([chunk_a, chunk_b])
        results = corpus.search("timeout")
        assert len(results) == 2
        assert results[0].score >= results[1].score

    def test_ranks_are_sequential(self) -> None:
        chunks = [
            _make_chunk(
                chunk_id=f"DOC-{i:03d}-chunk-0000",
                document_id=f"DOC-{i:03d}",
                text="payment processing error timeout gateway",
            )
            for i in range(3)
        ]
        corpus = BM25Corpus(chunks)
        results = corpus.search("payment error")
        for i, r in enumerate(results):
            assert r.rank == i + 1

    def test_top_k_limits_results(self) -> None:
        chunks = [
            _make_chunk(
                chunk_id=f"DOC-{i:03d}-chunk-0000",
                document_id=f"DOC-{i:03d}",
                text="FPS payment gateway timeout certificate error",
            )
            for i in range(10)
        ]
        corpus = BM25Corpus(chunks)
        results = corpus.search("payment timeout", top_k=3)
        assert len(results) <= 3

    def test_results_are_deterministic(self) -> None:
        chunks = [
            _make_chunk(
                chunk_id=f"DOC-{i:03d}-chunk-0000",
                document_id=f"DOC-{i:03d}",
                text="payment processing timeout FPS gateway error",
            )
            for i in range(5)
        ]
        corpus = BM25Corpus(chunks)
        first = corpus.search("payment timeout")
        second = corpus.search("payment timeout")
        assert [r.chunk.chunk_id for r in first] == [r.chunk.chunk_id for r in second]


# ---------------------------------------------------------------------------
# RBAC role filtering
# ---------------------------------------------------------------------------


class TestRoleFiltering:
    def test_role_filter_excludes_unauthorized_chunks(self) -> None:
        chunk_eng = _make_chunk(
            chunk_id="DOC-001-chunk-0000",
            document_id="DOC-001",
            text="payment gateway architecture design",
            metadata=_make_meta(document_id="DOC-001", allowed_roles=(Role.ENGINEER,)),
        )
        chunk_ana = _make_chunk(
            chunk_id="DOC-002-chunk-0000",
            document_id="DOC-002",
            text="payment gateway analytics report",
            metadata=_make_meta(document_id="DOC-002", allowed_roles=(Role.ANALYST,)),
        )
        corpus = BM25Corpus([chunk_eng, chunk_ana])

        analyst_results = corpus.search("payment gateway", roles=[Role.ANALYST])
        ids = [r.chunk.chunk_id for r in analyst_results]
        assert "DOC-002-chunk-0000" in ids
        assert "DOC-001-chunk-0000" not in ids

    def test_role_filter_returns_nothing_when_no_match(self) -> None:
        chunk = _make_chunk(
            text="restricted certificate signing authority",
            metadata=_make_meta(allowed_roles=(Role.ADMINISTRATOR,)),
        )
        corpus = BM25Corpus([chunk])
        results = corpus.search("certificate", roles=[Role.VIEWER])
        assert results == []

    def test_multi_role_document_accessible_to_any_permitted_role(self) -> None:
        chunk = _make_chunk(
            text="payment processing core ledger",
            metadata=_make_meta(allowed_roles=(Role.ENGINEER, Role.ANALYST)),
        )
        corpus = BM25Corpus([chunk])
        assert len(corpus.search("payment", roles=[Role.ENGINEER])) == 1
        assert len(corpus.search("payment", roles=[Role.ANALYST])) == 1

    def test_no_role_filter_returns_all_matching_chunks(self) -> None:
        chunks = [
            _make_chunk(
                chunk_id="DOC-001-chunk-0000",
                document_id="DOC-001",
                text="payment gateway timeout",
                metadata=_make_meta(document_id="DOC-001", allowed_roles=(Role.ENGINEER,)),
            ),
            _make_chunk(
                chunk_id="DOC-002-chunk-0000",
                document_id="DOC-002",
                text="payment gateway timeout",
                metadata=_make_meta(document_id="DOC-002", allowed_roles=(Role.ANALYST,)),
            ),
        ]
        corpus = BM25Corpus(chunks)
        results = corpus.search("payment gateway", roles=None)
        assert len(results) == 2


# ---------------------------------------------------------------------------
# Access-level filtering
# ---------------------------------------------------------------------------


class TestAccessLevelFiltering:
    def test_access_level_filter_excludes_wrong_level(self) -> None:
        chunk_internal = _make_chunk(
            chunk_id="DOC-001-chunk-0000",
            document_id="DOC-001",
            text="payment network architecture",
            metadata=_make_meta(document_id="DOC-001", access_level=AccessLevel.INTERNAL),
        )
        chunk_confidential = _make_chunk(
            chunk_id="DOC-002-chunk-0000",
            document_id="DOC-002",
            text="payment network secrets",
            metadata=_make_meta(document_id="DOC-002", access_level=AccessLevel.CONFIDENTIAL),
        )
        corpus = BM25Corpus([chunk_internal, chunk_confidential])
        results = corpus.search("payment network", access_levels=[AccessLevel.INTERNAL])
        ids = [r.chunk.chunk_id for r in results]
        assert "DOC-001-chunk-0000" in ids
        assert "DOC-002-chunk-0000" not in ids

    def test_multiple_access_levels_allowed(self) -> None:
        chunks = [
            _make_chunk(
                chunk_id=f"DOC-{lv.value}-chunk-0000",
                document_id=f"DOC-{lv.value}",
                text="payment processing timeout error",
                metadata=_make_meta(document_id=f"DOC-{lv.value}", access_level=lv),
            )
            for lv in [AccessLevel.PUBLIC, AccessLevel.INTERNAL, AccessLevel.CONFIDENTIAL]
        ]
        corpus = BM25Corpus(chunks)
        results = corpus.search(
            "payment timeout",
            access_levels=[AccessLevel.PUBLIC, AccessLevel.INTERNAL],
        )
        returned_ids = {r.chunk.chunk_id for r in results}
        assert "DOC-PUBLIC-chunk-0000" in returned_ids
        assert "DOC-INTERNAL-chunk-0000" in returned_ids
        assert "DOC-CONFIDENTIAL-chunk-0000" not in returned_ids


# ---------------------------------------------------------------------------
# Department and document_type filtering
# ---------------------------------------------------------------------------


class TestMetadataFiltering:
    def test_department_filter(self) -> None:
        chunk_core = _make_chunk(
            chunk_id="DOC-001-chunk-0000",
            document_id="DOC-001",
            text="payment gateway timeout error processing",
            metadata=_make_meta(document_id="DOC-001", department="Core Banking"),
        )
        chunk_fps = _make_chunk(
            chunk_id="DOC-002-chunk-0000",
            document_id="DOC-002",
            text="payment gateway timeout error routing",
            metadata=_make_meta(document_id="DOC-002", department="FPS"),
        )
        corpus = BM25Corpus([chunk_core, chunk_fps])
        results = corpus.search("payment gateway timeout", department="Core Banking")
        ids = [r.chunk.chunk_id for r in results]
        assert "DOC-001-chunk-0000" in ids
        assert "DOC-002-chunk-0000" not in ids

    def test_document_type_filter(self) -> None:
        chunk_runbook = _make_chunk(
            chunk_id="RB-001-chunk-0000",
            document_id="RB-001",
            text="payment incident runbook steps restart procedure",
            metadata=_make_meta(document_id="RB-001", document_type="runbook"),
        )
        chunk_arch = _make_chunk(
            chunk_id="ARCH-001-chunk-0000",
            document_id="ARCH-001",
            text="payment architecture design components overview",
            metadata=_make_meta(document_id="ARCH-001", document_type="architecture_document"),
        )
        corpus = BM25Corpus([chunk_runbook, chunk_arch])
        results = corpus.search("payment", document_type="runbook")
        ids = [r.chunk.chunk_id for r in results]
        assert "RB-001-chunk-0000" in ids
        assert "ARCH-001-chunk-0000" not in ids

    def test_combined_role_and_department_filter(self) -> None:
        chunk = _make_chunk(
            chunk_id="DOC-001-chunk-0000",
            document_id="DOC-001",
            text="payment timeout certificate expiry FPS",
            metadata=_make_meta(
                document_id="DOC-001",
                department="FPS",
                allowed_roles=(Role.ENGINEER,),
            ),
        )
        corpus = BM25Corpus([chunk])
        # Correct role + correct department → match
        assert len(corpus.search("payment", roles=[Role.ENGINEER], department="FPS")) == 1
        # Wrong role → no match
        assert corpus.search("payment", roles=[Role.ANALYST], department="FPS") == []
        # Wrong department → no match
        assert corpus.search("payment", roles=[Role.ENGINEER], department="Core Banking") == []


# ---------------------------------------------------------------------------
# BM25 advantages over dense (embedding-based) retrieval
# ---------------------------------------------------------------------------


class TestBM25AdvantagesOverDenseSearch:
    """Demonstrate retrieval scenarios where BM25 outperforms dense search.

    Dense (vector) retrieval excels at semantic similarity and paraphrasing
    but struggles with:

    1. Exact error codes and numeric identifiers — a query for "ERR-429" may
       not align well with an embedding for "rate limit exceeded" even though
       they co-occur in incident reports.  BM25 matches the exact token.

    2. Rare technical terms (high IDF) — "idempotency" appears far less often
       than "payment" in a banking corpus, so its IDF weight is much higher.
       A dense model trained on general text may under-weight it.

    3. Full query-term coverage — BM25 rewards chunks that contain *all*
       query tokens, even rare ones like algorithm names ("sha256", "hmac").

    4. Technical algorithm identifiers — "HMAC-SHA256" tokenises to two rare
       tokens; BM25 finds the exact document while a dense model might return
       semantically adjacent but less specific chunks.
    """

    def test_exact_error_code_ranked_above_generic_description(self) -> None:
        """BM25 advantage: exact token match for numeric error codes."""
        chunk_specific = _make_chunk(
            chunk_id="INC-001-chunk-0000",
            document_id="INC-001",
            text=(
                "The FPS service returned ERR-429 rate limit error code during "
                "peak payment processing hours."
            ),
        )
        chunk_generic = _make_chunk(
            chunk_id="INC-002-chunk-0000",
            document_id="INC-002",
            text=(
                "The payment service encountered an error during processing and "
                "failed to complete the transaction."
            ),
        )
        corpus = BM25Corpus([chunk_specific, chunk_generic])
        results = corpus.search("ERR-429")
        assert results, "Expected at least one result"
        assert results[0].chunk.chunk_id == "INC-001-chunk-0000"

    def test_rare_technical_term_ranked_above_common_term(self) -> None:
        """BM25 advantage: rare term 'idempotency' has high IDF; beats common 'payment'."""
        chunk_rare_term = _make_chunk(
            chunk_id="ARCH-001-chunk-0000",
            document_id="ARCH-001",
            text=(
                "The idempotency key ensures each payment request is processed "
                "exactly once, preventing duplicate transactions."
            ),
        )
        # chunk_common mentions "payment" many times but never "idempotency"
        chunk_common_term = _make_chunk(
            chunk_id="ARCH-002-chunk-0000",
            document_id="ARCH-002",
            text=(
                "Payment systems route payment requests through the payment gateway "
                "to the payment processor for settlement."
            ),
        )
        corpus = BM25Corpus([chunk_rare_term, chunk_common_term])
        results = corpus.search("idempotency")
        assert results, "Expected at least one result"
        assert results[0].chunk.chunk_id == "ARCH-001-chunk-0000"

    def test_chunk_with_all_query_terms_ranked_higher(self) -> None:
        """BM25 advantage: full query-term coverage beats partial matches."""
        chunk_full_match = _make_chunk(
            chunk_id="INC-010-chunk-0000",
            document_id="INC-010",
            text=(
                "Certificate expiry caused the FPS gateway to reject connections, "
                "resulting in widespread payment failures."
            ),
        )
        # chunk_partial contains some but not all terms from the query
        chunk_partial = _make_chunk(
            chunk_id="INC-011-chunk-0000",
            document_id="INC-011",
            text="The gateway experienced high latency during peak hours.",
        )
        corpus = BM25Corpus([chunk_full_match, chunk_partial])
        results = corpus.search("certificate expiry FPS gateway")
        assert results, "Expected at least one result"
        assert results[0].chunk.chunk_id == "INC-010-chunk-0000"

    def test_cryptographic_algorithm_identifier_matched(self) -> None:
        """BM25 advantage: identifier 'HMAC-SHA256' matched by exact tokens."""
        chunk_hmac = _make_chunk(
            chunk_id="ARCH-005-chunk-0000",
            document_id="ARCH-005",
            text=(
                "Message integrity is verified using HMAC-SHA256 signing before "
                "forwarding to the payment processor."
            ),
        )
        chunk_vague = _make_chunk(
            chunk_id="ARCH-006-chunk-0000",
            document_id="ARCH-006",
            text=(
                "Message integrity is verified using a cryptographic signature "
                "scheme before forwarding to the payment processor."
            ),
        )
        corpus = BM25Corpus([chunk_hmac, chunk_vague])
        results = corpus.search("HMAC-SHA256")
        assert results, "Expected at least one result"
        assert results[0].chunk.chunk_id == "ARCH-005-chunk-0000"

    def test_high_tf_term_scores_higher_when_idf_equal(self) -> None:
        """BM25 TF component: repeated term in one chunk gives it higher score."""
        chunk_high_tf = _make_chunk(
            chunk_id="DOC-001-chunk-0000",
            document_id="DOC-001",
            text="timeout timeout timeout detected in FPS payment service",
        )
        chunk_low_tf = _make_chunk(
            chunk_id="DOC-002-chunk-0000",
            document_id="DOC-002",
            text="timeout detected in FPS payment service gateway routing",
        )
        corpus = BM25Corpus([chunk_high_tf, chunk_low_tf])
        results = corpus.search("timeout")
        assert len(results) == 2
        assert results[0].chunk.chunk_id == "DOC-001-chunk-0000"
