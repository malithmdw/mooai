"""Tests for the incident root-cause analysis workflow
(``ResearchAgent.analyze_incident_root_causes``) — the RLM extension for the
demonstration query: "Summarize all payment outages related to payment
failures during 2025 and identify recurring root causes."

Coverage
--------
Pure helpers:
  - _filter_by_authorization: explicit re-check drops unauthorized evidence
  - _verify_claims: drops claims not grounded in the authorized pool
  - _select_analytics_role: RBAC-respecting role selection for the tool call
  - _unsupported_numbers: flags numbers not in the verified set
  - _build_fallback_summary: deterministic, template-only summary

End to end, using a synthetic 2025 payment-incident dataset:
  - full pipeline: discover -> filter by authorization -> partition ->
    analyze batches independently -> extract root causes -> identify
    recurring patterns (Python Analysis Tool) -> aggregate -> verify
    evidence -> generate final answer
  - genuine recursion: multi-batch evidence creates child tasks + aggregation
  - evidence verification drops a hallucinated claim (bad chunk_id) before
    it can affect counts or supporting documents
  - authorization filtering excludes a RESTRICTED document an ANALYST
    cannot see, before any LLM call ever sees it
  - RBAC: a role without analytics permission gets no recurring-cause
    analysis, with an honest limitation explaining why (no bypass)
  - unsupported statistical claims: a final-answer narrative stating a
    number not present in the verified data is rejected and replaced with
    a deterministic, data-only summary
  - budget-exhausted / no-evidence / no-authorized-evidence short-circuits
"""

from __future__ import annotations

from datetime import date
from unittest.mock import AsyncMock, MagicMock

from src.agents.budget import ExecutionBudget
from src.agents.research.models import Claim, RecurringRootCause
from src.agents.research.node import (
    ResearchAgent,
    _build_fallback_summary,
    _filter_by_authorization,
    _select_analytics_role,
    _unsupported_numbers,
    _verify_claims,
)
from src.models.documents import DocumentMetadata
from src.models.enums import AccessLevel, Role
from src.retrieval.hybrid.models import RetrievalEvidence, RetrievalSource
from src.retrieval.hybrid.retriever import HybridRetriever

# ---------------------------------------------------------------------------
# Shared test helpers
# ---------------------------------------------------------------------------


def _meta(
    document_id: str,
    *,
    access_level: AccessLevel = AccessLevel.INTERNAL,
    allowed_roles: tuple[Role, ...] = (Role.ANALYST, Role.ADMINISTRATOR),
    document_type: str = "incident_report",
) -> DocumentMetadata:
    return DocumentMetadata(
        document_id=document_id,
        title=document_id,
        department="Payments Engineering",
        document_type=document_type,
        access_level=access_level,
        created_date=date(2025, 3, 1),
        allowed_roles=allowed_roles,
    )


def _evidence(
    chunk_id: str,
    document_id: str,
    text: str,
    *,
    access_level: AccessLevel = AccessLevel.INTERNAL,
    allowed_roles: tuple[Role, ...] = (Role.ANALYST, Role.ADMINISTRATOR),
    final_score: float = 0.8,
) -> RetrievalEvidence:
    return RetrievalEvidence(
        chunk_id=chunk_id,
        document_id=document_id,
        title=document_id,
        text=text,
        metadata=_meta(document_id, access_level=access_level, allowed_roles=allowed_roles),
        source=RetrievalSource.BOTH,
        dense_score=0.8,
        sparse_score=0.7,
        final_score=final_score,
        rank=1,
    )


def _tool_block(tool_name: str, tool_input: dict) -> MagicMock:  # type: ignore[type-arg]
    block = MagicMock()
    block.type = "tool_use"
    block.name = tool_name
    block.input = tool_input
    return block


def _mock_response(
    tool_name: str, tool_input: dict, *, input_tokens: int = 100, output_tokens: int = 50
) -> MagicMock:  # type: ignore[type-arg]
    usage = MagicMock()
    usage.input_tokens = input_tokens
    usage.output_tokens = output_tokens
    resp = MagicMock()
    resp.content = [_tool_block(tool_name, tool_input)]
    resp.usage = usage
    return resp


def _mock_client(responses: list) -> MagicMock:  # type: ignore[type-arg]
    client = MagicMock()
    client.messages.create = AsyncMock(side_effect=responses)
    return client


def _mock_retriever(pool: list[RetrievalEvidence]) -> MagicMock:  # type: ignore[type-arg]
    retriever = MagicMock(spec=HybridRetriever)
    retriever.search = AsyncMock(return_value=pool)
    return retriever


def _root_cause(incident_id: str, cause: str, chunk_ids: list[str]) -> dict:
    return {"incident_id": incident_id, "cause": cause, "chunk_ids": chunk_ids}


def _incident_finding_input(relevant: bool, summary: str, root_causes: list[dict]) -> dict:
    return {"relevant": relevant, "summary": summary, "root_causes": root_causes}


def _research_finding_input(summary: str, claims: list[dict] | None = None) -> dict:
    return {"summary": summary, "claims": claims or [], "confidence": "medium"}


def _final_answer_input(summary: str, limitations: str = "None known.") -> dict:
    return {"summary": summary, "limitations": limitations}


# ---------------------------------------------------------------------------
# Synthetic 2025 payment-incident dataset
# ---------------------------------------------------------------------------

QUESTION = (
    "Summarize all payment outages related to payment failures during 2025 "
    "and identify recurring root causes."
)

_C1 = _evidence(
    "c1",
    "INC-2025-001",
    "Payment gateway outage: Visa debit processing failed due to connection "
    "pool exhaustion on the visa-processor adapter during peak load.",
)
_C2 = _evidence(
    "c2",
    "INC-2025-002",
    "A second payment gateway outage in Q2 2025 was traced to connection "
    "pool exhaustion after a certificate rotation left stale connections open.",
)
_C3 = _evidence(
    "c3",
    "INC-2025-003",
    "SWIFT wire transfer duplication occurred because the DR node lacked "
    "access to the shared idempotency store during a failover drill.",
)
_C4 = _evidence(
    "c4",
    "HR-POLICY-2025-001",
    "This document describes the 2025 remote-work policy for engineering staff.",
    access_level=AccessLevel.INTERNAL,
)
_C5_RESTRICTED = _evidence(
    "c5",
    "INC-2025-999",
    "A RESTRICTED post-mortem for a payment fraud incident under regulatory review.",
    access_level=AccessLevel.RESTRICTED,
    allowed_roles=(Role.ADMINISTRATOR,),
)


def _full_pool() -> list[RetrievalEvidence]:
    return [_C1, _C2, _C3, _C4, _C5_RESTRICTED]


def _agent(
    client: MagicMock,  # type: ignore[type-arg]
    retriever: MagicMock,  # type: ignore[type-arg]
    **overrides: object,
) -> ResearchAgent:
    overrides.setdefault("max_chunks_per_batch", 2)
    return ResearchAgent(client=client, model="test", retriever=retriever, **overrides)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------


class TestFilterByAuthorization:
    def test_excludes_restricted_document_for_analyst(self) -> None:
        authorized, rejected = _filter_by_authorization(_full_pool(), roles=[Role.ANALYST])
        assert {e.chunk_id for e in authorized} == {"c1", "c2", "c3", "c4"}
        assert {e.chunk_id for e in rejected} == {"c5"}

    def test_administrator_sees_everything(self) -> None:
        authorized, rejected = _filter_by_authorization(_full_pool(), roles=[Role.ADMINISTRATOR])
        assert len(authorized) == 5
        assert rejected == []

    def test_no_roles_authorizes_nothing(self) -> None:
        authorized, rejected = _filter_by_authorization(_full_pool(), roles=[])
        assert authorized == []
        assert len(rejected) == 5


class TestVerifyClaims:
    def test_claim_with_authorized_chunk_is_verified(self) -> None:
        authorized_index = {"c1": _C1}
        claims = [
            Claim(subject="INC-2025-001", attribute="root_cause", value="x", chunk_ids=["c1"])
        ]
        verified, unverified = _verify_claims(claims, authorized_index)
        assert len(verified) == 1
        assert unverified == 0

    def test_claim_with_unknown_chunk_is_dropped(self) -> None:
        authorized_index = {"c1": _C1}
        claims = [
            Claim(subject="INC-X", attribute="root_cause", value="x", chunk_ids=["does-not-exist"])
        ]
        verified, unverified = _verify_claims(claims, authorized_index)
        assert verified == []
        assert unverified == 1

    def test_claim_with_no_chunk_ids_is_dropped(self) -> None:
        claims = [Claim(subject="INC-X", attribute="root_cause", value="x", chunk_ids=[])]
        verified, unverified = _verify_claims(claims, {})
        assert verified == []
        assert unverified == 1

    def test_mixed_claims_partially_verified(self) -> None:
        authorized_index = {"c1": _C1}
        claims = [
            Claim(subject="INC-1", attribute="root_cause", value="a", chunk_ids=["c1"]),
            Claim(subject="INC-2", attribute="root_cause", value="b", chunk_ids=["missing"]),
        ]
        verified, unverified = _verify_claims(claims, authorized_index)
        assert len(verified) == 1
        assert unverified == 1


class TestSelectAnalyticsRole:
    def test_analyst_is_selected(self) -> None:
        assert _select_analytics_role([Role.ANALYST]) == Role.ANALYST

    def test_administrator_is_selected(self) -> None:
        assert _select_analytics_role([Role.ADMINISTRATOR]) == Role.ADMINISTRATOR

    def test_viewer_alone_returns_none(self) -> None:
        assert _select_analytics_role([Role.VIEWER]) is None

    def test_first_permitted_role_wins(self) -> None:
        assert _select_analytics_role([Role.VIEWER, Role.ANALYST]) == Role.ANALYST


class TestUnsupportedNumbers:
    def test_no_numbers_is_fine(self) -> None:
        assert _unsupported_numbers("No statistics mentioned here.", set()) == []

    def test_allowed_number_passes(self) -> None:
        assert _unsupported_numbers("We found 2 recurring causes.", {"2"}) == []

    def test_disallowed_number_is_flagged(self) -> None:
        assert _unsupported_numbers("This affected 500 customers.", {"2"}) == ["500"]

    def test_multiple_disallowed_numbers_all_flagged(self) -> None:
        bad = _unsupported_numbers("3 incidents, 7 customers.", set())
        assert set(bad) == {"3", "7"}


class TestBuildFallbackSummary:
    def test_no_recurring_causes(self) -> None:
        summary = _build_fallback_summary([], ["INC-1"])
        assert "no root cause recurred" in summary.lower()
        assert "1 supporting document" in summary

    def test_with_recurring_causes(self) -> None:
        recurring = [
            RecurringRootCause(
                root_cause="connection pool exhaustion",
                count=2,
                incident_ids=("INC-2025-001", "INC-2025-002"),
            )
        ]
        summary = _build_fallback_summary(recurring, ["INC-2025-001", "INC-2025-002"])
        assert "connection pool exhaustion: 2 incident(s)" in summary
        assert "INC-2025-001" in summary and "INC-2025-002" in summary


# ---------------------------------------------------------------------------
# End-to-end: analyze_incident_root_causes
# ---------------------------------------------------------------------------


class TestIncidentAnalysisEndToEnd:
    async def test_full_pipeline_identifies_recurring_root_cause(self) -> None:
        client = _mock_client(
            [
                # batch b0: c1, c2 (the recurring "connection pool exhaustion")
                _mock_response(
                    "incident_root_causes",
                    _incident_finding_input(
                        True,
                        "Two outages traced to connection pool exhaustion.",
                        [
                            _root_cause("INC-2025-001", "connection pool exhaustion", ["c1"]),
                            _root_cause("INC-2025-002", "Connection Pool Exhaustion", ["c2"]),
                        ],
                    ),
                ),
                # batch b1: c3, c4 — c3 relevant with a hallucinated chunk_id,
                # c4 is an unrelated HR policy document.
                _mock_response(
                    "incident_root_causes",
                    _incident_finding_input(
                        True,
                        "One SWIFT duplication incident; one unrelated document.",
                        [
                            _root_cause("INC-2025-003", "idempotency store unavailable", ["c3"]),
                            _root_cause(
                                "INC-2025-999-GHOST", "fabricated cause", ["chunk-never-retrieved"]
                            ),
                        ],
                    ),
                ),
                # aggregation of b0 + b1
                _mock_response(
                    "research_finding",
                    _research_finding_input("Combined analysis of three payment incidents."),
                ),
                # final answer — states exactly the verified count (2)
                _mock_response(
                    "incident_final_answer",
                    _final_answer_input(
                        "Two 2025 payment incidents were caused by connection pool "
                        "exhaustion; a third was an unrelated SWIFT duplication issue.",
                        limitations="Time range relies on retrieval relevance, not an "
                        "independent date-field check.",
                    ),
                ),
            ]
        )
        pool = [_C1, _C2, _C3, _C4]  # authorized subset (no RESTRICTED doc)
        agent = _agent(client, _mock_retriever(pool))

        result = await agent.analyze_incident_root_causes(
            question=QUESTION, roles=[Role.ANALYST], budget=ExecutionBudget()
        )

        assert len(result.recurring_root_causes) == 1
        recurring = result.recurring_root_causes[0]
        assert recurring.root_cause == "connection pool exhaustion"
        assert recurring.count == 2
        assert set(recurring.incident_ids) == {"INC-2025-001", "INC-2025-002"}

        # The hallucinated claim (bad chunk_id) never reaches supporting docs.
        assert "INC-2025-999-GHOST" not in result.supporting_document_ids
        assert set(result.supporting_document_ids) == {
            "INC-2025-001",
            "INC-2025-002",
            "INC-2025-003",
        }
        assert result.unverified_claim_count == 1

        # The LLM's narrative passed through unchanged: "2025" is a
        # legitimate number (it's in the question itself), not an invented
        # statistic, so the guard must not have replaced this summary.
        assert result.summary.startswith("Two 2025 payment incidents")
        assert client.messages.create.call_count == 4  # 2 leaves + aggregate + final

    async def test_unverified_evidence_never_inflates_the_count(self) -> None:
        """A claim whose only chunk_id was never retrieved/authorized must
        not contribute to a recurring-cause count or its incident list."""
        client = _mock_client(
            [
                _mock_response(
                    "incident_root_causes",
                    _incident_finding_input(
                        True,
                        "One verified incident, one fabricated one.",
                        [
                            _root_cause("INC-2025-001", "connection pool exhaustion", ["c1"]),
                            _root_cause(
                                "INC-2025-FAKE", "connection pool exhaustion", ["not-a-real-chunk"]
                            ),
                        ],
                    ),
                ),
                _mock_response("incident_final_answer", _final_answer_input("One incident found.")),
            ]
        )
        pool = [_C1]
        agent = _agent(client, _mock_retriever(pool))

        result = await agent.analyze_incident_root_causes(
            question=QUESTION, roles=[Role.ANALYST], budget=ExecutionBudget()
        )

        # Only one incident is verified -> below the recurring threshold (2).
        assert result.recurring_root_causes == ()
        assert result.unverified_claim_count == 1
        assert "INC-2025-FAKE" not in result.supporting_document_ids

    async def test_restricted_document_is_excluded_before_any_llm_call(self) -> None:
        client = _mock_client(
            [
                _mock_response(
                    "incident_root_causes",
                    _incident_finding_input(True, "Reviewed available incidents.", []),
                ),
                _mock_response("incident_final_answer", _final_answer_input("No recurring cause.")),
            ]
        )
        pool = [_C5_RESTRICTED]
        agent = _agent(client, _mock_retriever(pool))

        result = await agent.analyze_incident_root_causes(
            question=QUESTION, roles=[Role.ANALYST], budget=ExecutionBudget()
        )

        assert "No authorized evidence" in result.summary
        assert "does not permit access" in result.limitations
        client.messages.create.assert_not_called()

    async def test_role_without_analytics_permission_skips_pattern_detection(self) -> None:
        """RBAC respected: recurring-pattern detection never runs for a role
        that lacks analytics permission — no bypass, just an honest skip.

        Uses evidence explicitly authorized for VIEWER so the test isolates
        the analytics-permission gap from document-level authorization
        (covered separately by TestFilterByAuthorization /
        test_restricted_document_is_excluded_before_any_llm_call).
        """
        viewer_c1 = _evidence(
            "c1",
            "INC-2025-001",
            _C1.text,
            allowed_roles=(Role.VIEWER, Role.ANALYST, Role.ADMINISTRATOR),
        )
        viewer_c2 = _evidence(
            "c2",
            "INC-2025-002",
            _C2.text,
            allowed_roles=(Role.VIEWER, Role.ANALYST, Role.ADMINISTRATOR),
        )
        client = _mock_client(
            [
                _mock_response(
                    "incident_root_causes",
                    _incident_finding_input(
                        True,
                        "Two outages traced to connection pool exhaustion.",
                        [
                            _root_cause("INC-2025-001", "connection pool exhaustion", ["c1"]),
                            _root_cause("INC-2025-002", "connection pool exhaustion", ["c2"]),
                        ],
                    ),
                ),
                _mock_response(
                    "incident_final_answer", _final_answer_input("Reviewed two incidents.")
                ),
            ]
        )
        pool = [viewer_c1, viewer_c2]
        agent = _agent(client, _mock_retriever(pool))

        result = await agent.analyze_incident_root_causes(
            question=QUESTION, roles=[Role.VIEWER], budget=ExecutionBudget()
        )

        assert result.recurring_root_causes == ()
        assert "analytics permission" in result.limitations

    async def test_unsupported_statistic_in_summary_is_replaced(self) -> None:
        """The final-answer LLM call states a number that was never given to
        it — the system must reject the narrative, not just the number."""
        client = _mock_client(
            [
                _mock_response(
                    "incident_root_causes",
                    _incident_finding_input(
                        True,
                        "Two outages traced to connection pool exhaustion.",
                        [
                            _root_cause("INC-2025-001", "connection pool exhaustion", ["c1"]),
                            _root_cause("INC-2025-002", "connection pool exhaustion", ["c2"]),
                        ],
                    ),
                ),
                _mock_response(
                    "incident_final_answer",
                    _final_answer_input(
                        "This root cause affected 500 customers across 14 incidents.",
                    ),
                ),
            ]
        )
        pool = [_C1, _C2]
        agent = _agent(client, _mock_retriever(pool))

        result = await agent.analyze_incident_root_causes(
            question=QUESTION, roles=[Role.ANALYST], budget=ExecutionBudget()
        )

        assert "500" not in result.summary
        assert "14" not in result.summary
        assert "connection pool exhaustion: 2 incident(s)" in result.summary
        assert "replaced" in result.limitations

    async def test_no_documents_found_short_circuits(self) -> None:
        client = _mock_client([])
        agent = _agent(client, _mock_retriever([]))

        result = await agent.analyze_incident_root_causes(
            question=QUESTION, roles=[Role.ANALYST], budget=ExecutionBudget()
        )

        assert "No relevant documents" in result.summary
        client.messages.create.assert_not_called()

    async def test_budget_exhausted_short_circuits(self) -> None:
        client = _mock_client([])
        agent = _agent(client, _mock_retriever([_C1]))
        exhausted = ExecutionBudget(max_tokens=10, used_tokens=10)

        result = await agent.analyze_incident_root_causes(
            question=QUESTION, roles=[Role.ANALYST], budget=exhausted
        )

        assert "budget already exhausted" in result.summary
        client.messages.create.assert_not_called()

    async def test_genuine_recursion_with_more_than_two_batches(self) -> None:
        """Five authorized chunks with a batch size of 2 forces three
        batches, proving the recursive case (not just a flat map)."""
        c6 = _evidence("c6", "INC-2025-004", "Another payment gateway blip.")
        c7 = _evidence("c7", "INC-2025-005", "A fifth unrelated incident chunk.")
        client = _mock_client(
            [
                _mock_response(
                    "incident_root_causes",
                    _incident_finding_input(True, "batch 0", []),
                ),
                _mock_response(
                    "incident_root_causes",
                    _incident_finding_input(True, "batch 1", []),
                ),
                _mock_response(
                    "incident_root_causes",
                    _incident_finding_input(True, "batch 2", []),
                ),
                _mock_response("research_finding", _research_finding_input("aggregated")),
                _mock_response("incident_final_answer", _final_answer_input("No recurring cause.")),
            ]
        )
        pool = [_C1, _C2, _C3, c6, c7]  # 5 authorized chunks, batch size 2 -> 3 batches
        agent = _agent(client, _mock_retriever(pool), max_chunks_per_batch=2)

        result = await agent.analyze_incident_root_causes(
            question=QUESTION, roles=[Role.ANALYST], budget=ExecutionBudget()
        )

        assert result.recurring_root_causes == ()
        assert client.messages.create.call_count == 5  # 3 leaves + aggregate + final
