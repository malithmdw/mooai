"""Tests for ``ToolExecutionService`` — the centralized tool-call gateway.

Coverage
--------
Pipeline ordering / gating:
  - unknown tool name -> TOOL_NOT_FOUND, no handler touched
  - invalid parameters -> INVALID_PARAMETERS, no handler touched
  - unauthorized role -> PERMISSION_DENIED, no handler touched (RBAC bypass
    attempts — see CLAUDE.md "Agents must never bypass application
    authorization")
  - exhausted tool-call budget -> BUDGET_EXHAUSTED, no handler touched
  - handler timeout -> TIMEOUT, budget still consumed
  - handler raises an unexpected exception -> EXECUTION_FAILED, no raw
    traceback propagates
  - non-JSON-serializable output -> OUTPUT_VALIDATION_FAILED

Unauthorized-access matrix (explicit negative RBAC tests):
  - VIEWER denied python_analysis, find_employee, get_service, get_incident
  - ENGINEER (holds no permissions under the default policy) denied
    knowledge_search

Successful invocations for all three integrated tool categories:
  - knowledge_search (HybridRetriever)
  - python_analysis (src.tools.analysis.run_analysis)
  - find_employee / get_service / get_incident (EnterpriseDataClient / MCP)

Audit logging:
  - a structured audit event is emitted for both a denial and a success
"""

from __future__ import annotations

import asyncio
import dataclasses
import uuid
from datetime import date
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.agents.budget import ExecutionBudget
from src.mcp.client import EnterpriseDataClient, MCPClientError
from src.models.documents import DocumentMetadata
from src.models.enums import AccessLevel, Role
from src.models.tools import ToolCall
from src.retrieval.hybrid.models import RetrievalEvidence, RetrievalSource
from src.retrieval.hybrid.retriever import HybridRetriever
from src.services.tool_execution import ToolExecutionError, ToolExecutionService, _validate_output

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _call(tool_name: str, arguments: dict[str, object] | None = None) -> ToolCall:
    return ToolCall(
        tool_call_id=str(uuid.uuid4()),
        tool_name=tool_name,
        arguments=arguments or {},
    )


def _evidence(chunk_id: str = "chunk-1") -> RetrievalEvidence:
    return RetrievalEvidence(
        chunk_id=chunk_id,
        document_id="doc-1",
        title="Loan Policy",
        text="Loans over $1M require director approval.",
        metadata=DocumentMetadata(
            document_id="doc-1",
            title="Loan Policy",
            department="Risk",
            document_type="policy",
            access_level=AccessLevel.INTERNAL,
            created_date=date(2024, 1, 1),
            allowed_roles=(Role.ANALYST, Role.ADMINISTRATOR),
        ),
        source=RetrievalSource.BOTH,
        dense_score=0.9,
        sparse_score=0.8,
        final_score=0.85,
        rank=1,
    )


def _service(
    *,
    retriever: HybridRetriever | None = None,
    mcp_client: EnterpriseDataClient | None = None,
) -> ToolExecutionService:
    return ToolExecutionService(
        retriever=retriever if retriever is not None else MagicMock(spec=HybridRetriever),
        mcp_client=mcp_client if mcp_client is not None else MagicMock(spec=EnterpriseDataClient),
    )


def _budget(**overrides: int) -> ExecutionBudget:
    return ExecutionBudget(**overrides)


# ---------------------------------------------------------------------------
# 1. Tool existence validation
# ---------------------------------------------------------------------------


class TestToolExistence:
    async def test_unknown_tool_is_rejected(self) -> None:
        service = _service()
        result, budget = await service.execute(
            _call("delete_all_customers"), role=Role.ADMINISTRATOR, budget=_budget()
        )
        assert result.success is False
        assert result.error_code == "TOOL_NOT_FOUND"
        assert budget.used_tool_calls == 0

    async def test_available_tools_lists_all_three_categories(self) -> None:
        service = _service()
        assert service.available_tools == (
            "find_employee",
            "get_incident",
            "get_service",
            "knowledge_search",
            "python_analysis",
        )


# ---------------------------------------------------------------------------
# 2. Parameter validation
# ---------------------------------------------------------------------------


class TestParameterValidation:
    async def test_knowledge_search_missing_query_is_rejected(self) -> None:
        retriever = MagicMock(spec=HybridRetriever)
        retriever.search = AsyncMock()
        service = _service(retriever=retriever)

        result, budget = await service.execute(
            _call("knowledge_search", {}), role=Role.ANALYST, budget=_budget()
        )

        assert result.success is False
        assert result.error_code == "INVALID_PARAMETERS"
        assert budget.used_tool_calls == 0
        retriever.search.assert_not_called()

    async def test_knowledge_search_top_k_out_of_range_is_rejected(self) -> None:
        service = _service()
        result, _ = await service.execute(
            _call("knowledge_search", {"query": "policy", "top_k": 999}),
            role=Role.ANALYST,
            budget=_budget(),
        )
        assert result.success is False
        assert result.error_code == "INVALID_PARAMETERS"

    async def test_find_employee_blank_query_is_rejected(self) -> None:
        service = _service()
        result, _ = await service.execute(
            _call("find_employee", {"query": "   "}), role=Role.ANALYST, budget=_budget()
        )
        assert result.success is False
        assert result.error_code == "INVALID_PARAMETERS"

    async def test_python_analysis_missing_fields_is_rejected(self) -> None:
        service = _service()
        result, _ = await service.execute(
            _call("python_analysis", {"operation": "count_by"}),
            role=Role.ANALYST,
            budget=_budget(),
        )
        assert result.success is False
        assert result.error_code == "INVALID_PARAMETERS"


# ---------------------------------------------------------------------------
# 3. RBAC — unauthorized access attempts (explicit negative tests)
# ---------------------------------------------------------------------------


class TestUnauthorizedAccess:
    async def test_viewer_denied_python_analysis(self) -> None:
        service = _service()
        result, budget = await service.execute(
            _call(
                "python_analysis",
                {"operation": "count_by", "data": [{"x": "1"}], "parameters": {"field": "x"}},
            ),
            role=Role.VIEWER,
            budget=_budget(),
        )
        assert result.success is False
        assert result.error_code == "PERMISSION_DENIED"
        assert budget.used_tool_calls == 0

    async def test_viewer_denied_find_employee(self) -> None:
        mcp_client = MagicMock(spec=EnterpriseDataClient)
        mcp_client.find_employee = AsyncMock()
        service = _service(mcp_client=mcp_client)

        result, budget = await service.execute(
            _call("find_employee", {"query": "Alice"}), role=Role.VIEWER, budget=_budget()
        )

        assert result.success is False
        assert result.error_code == "PERMISSION_DENIED"
        assert budget.used_tool_calls == 0
        mcp_client.find_employee.assert_not_called()

    async def test_viewer_denied_get_service(self) -> None:
        service = _service()
        result, _ = await service.execute(
            _call("get_service", {"service_id": "SVC-001"}), role=Role.VIEWER, budget=_budget()
        )
        assert result.success is False
        assert result.error_code == "PERMISSION_DENIED"

    async def test_viewer_denied_get_incident(self) -> None:
        service = _service()
        result, _ = await service.execute(
            _call("get_incident", {"incident_id": "INC-001"}), role=Role.VIEWER, budget=_budget()
        )
        assert result.success is False
        assert result.error_code == "PERMISSION_DENIED"

    async def test_engineer_denied_knowledge_search(self) -> None:
        """ENGINEER holds no permissions under the default RBAC policy."""
        retriever = MagicMock(spec=HybridRetriever)
        retriever.search = AsyncMock(return_value=[_evidence()])
        service = _service(retriever=retriever)

        result, budget = await service.execute(
            _call("knowledge_search", {"query": "loan policy"}),
            role=Role.ENGINEER,
            budget=_budget(),
        )

        assert result.success is False
        assert result.error_code == "PERMISSION_DENIED"
        assert budget.used_tool_calls == 0
        retriever.search.assert_not_called()


# ---------------------------------------------------------------------------
# 4. Rate / budget validation
# ---------------------------------------------------------------------------


class TestBudgetEnforcement:
    async def test_exhausted_budget_denies_call(self) -> None:
        retriever = MagicMock(spec=HybridRetriever)
        retriever.search = AsyncMock(return_value=[_evidence()])
        service = _service(retriever=retriever)
        exhausted = _budget(max_tool_calls=1, used_tool_calls=1)

        result, budget = await service.execute(
            _call("knowledge_search", {"query": "loan policy"}),
            role=Role.ANALYST,
            budget=exhausted,
        )

        assert result.success is False
        assert result.error_code == "BUDGET_EXHAUSTED"
        assert budget.used_tool_calls == 1  # unchanged
        retriever.search.assert_not_called()

    async def test_successful_call_consumes_one_tool_call(self) -> None:
        retriever = MagicMock(spec=HybridRetriever)
        retriever.search = AsyncMock(return_value=[_evidence()])
        service = _service(retriever=retriever)

        result, budget = await service.execute(
            _call("knowledge_search", {"query": "loan policy"}),
            role=Role.ANALYST,
            budget=_budget(),
        )

        assert result.success is True
        assert budget.used_tool_calls == 1


# ---------------------------------------------------------------------------
# 5. Timeout
# ---------------------------------------------------------------------------


class TestTimeout:
    async def test_slow_handler_times_out(self) -> None:
        async def _slow(params: object, role: Role) -> object:
            await asyncio.sleep(1)
            return {}

        service = _service()
        slow_def = dataclasses.replace(
            service._registry["knowledge_search"], handler=_slow, timeout=0.01
        )
        service._registry["knowledge_search"] = slow_def

        result, budget = await service.execute(
            _call("knowledge_search", {"query": "loan policy"}),
            role=Role.ANALYST,
            budget=_budget(),
        )

        assert result.success is False
        assert result.error_code == "TIMEOUT"
        # Budget was consumed: the call passed authorization and budget
        # checks and reached execution before it timed out.
        assert budget.used_tool_calls == 1


# ---------------------------------------------------------------------------
# 6. Output validation
# ---------------------------------------------------------------------------


class TestOutputValidation:
    def test_non_serializable_value_is_rejected(self) -> None:
        with pytest.raises(ToolExecutionError) as excinfo:
            _validate_output("some_tool", {"bad": object()})
        assert excinfo.value.code == "OUTPUT_VALIDATION_FAILED"

    def test_oversized_output_is_rejected(self) -> None:
        with pytest.raises(ToolExecutionError) as excinfo:
            _validate_output("some_tool", {"data": "x" * 300_000})
        assert excinfo.value.code == "OUTPUT_VALIDATION_FAILED"

    def test_plain_json_safe_value_round_trips(self) -> None:
        assert _validate_output("some_tool", {"a": 1, "b": [1, 2, 3]}) == {"a": 1, "b": [1, 2, 3]}

    async def test_handler_returning_bad_output_fails_pipeline(self) -> None:
        async def _bad(params: object, role: Role) -> object:
            return {"handle": object()}

        service = _service()
        bad_def = dataclasses.replace(service._registry["knowledge_search"], handler=_bad)
        service._registry["knowledge_search"] = bad_def

        result, budget = await service.execute(
            _call("knowledge_search", {"query": "loan policy"}),
            role=Role.ANALYST,
            budget=_budget(),
        )

        assert result.success is False
        assert result.error_code == "OUTPUT_VALIDATION_FAILED"
        assert budget.used_tool_calls == 1


# ---------------------------------------------------------------------------
# 7. Unexpected handler errors
# ---------------------------------------------------------------------------


class TestUnexpectedErrors:
    async def test_handler_exception_becomes_execution_failed(self) -> None:
        retriever = MagicMock(spec=HybridRetriever)
        retriever.search = AsyncMock(side_effect=RuntimeError("pinecone is down"))
        service = _service(retriever=retriever)

        result, budget = await service.execute(
            _call("knowledge_search", {"query": "loan policy"}),
            role=Role.ANALYST,
            budget=_budget(),
        )

        assert result.success is False
        assert result.error_code == "EXECUTION_FAILED"
        assert "pinecone is down" not in (result.error_message or "")
        assert budget.used_tool_calls == 1

    async def test_mcp_not_found_is_structured_not_found(self) -> None:
        mcp_client = MagicMock(spec=EnterpriseDataClient)
        mcp_client.find_employee = AsyncMock(
            side_effect=MCPClientError("NOT_FOUND", "No employee matching 'Zzz'")
        )
        service = _service(mcp_client=mcp_client)

        result, _ = await service.execute(
            _call("find_employee", {"query": "Zzz"}), role=Role.ANALYST, budget=_budget()
        )

        assert result.success is False
        assert result.error_code == "NOT_FOUND"


# ---------------------------------------------------------------------------
# 8. Successful invocations — all three tool categories
# ---------------------------------------------------------------------------


class TestSuccessfulInvocations:
    async def test_knowledge_search_success(self) -> None:
        retriever = MagicMock(spec=HybridRetriever)
        retriever.search = AsyncMock(return_value=[_evidence()])
        service = _service(retriever=retriever)

        result, _ = await service.execute(
            _call("knowledge_search", {"query": "loan policy", "top_k": 5}),
            role=Role.ANALYST,
            budget=_budget(),
        )

        assert result.success is True
        assert isinstance(result.output, list)
        assert result.output[0]["chunk_id"] == "chunk-1"
        retriever.search.assert_awaited_once()
        _, kwargs = retriever.search.await_args
        assert kwargs["top_k"] == 5
        assert kwargs["roles"] == [Role.ANALYST]

    async def test_python_analysis_success(self) -> None:
        service = _service()
        result, budget = await service.execute(
            _call(
                "python_analysis",
                {
                    "operation": "count_by",
                    "data": [{"status": "open"}, {"status": "closed"}, {"status": "open"}],
                    "parameters": {"field": "status"},
                },
            ),
            role=Role.ANALYST,
            budget=_budget(),
        )

        assert result.success is True
        assert result.output["result"]["counts"] == {"open": 2, "closed": 1}
        assert budget.used_tool_calls == 1

    async def test_find_employee_success(self) -> None:
        mcp_client = MagicMock(spec=EnterpriseDataClient)
        mcp_client.find_employee = AsyncMock(
            return_value={"employee_id": "EMP-001", "name": "Alice Chen"}
        )
        service = _service(mcp_client=mcp_client)

        result, _ = await service.execute(
            _call("find_employee", {"query": "Alice"}), role=Role.ANALYST, budget=_budget()
        )

        assert result.success is True
        assert result.output == {"employee_id": "EMP-001", "name": "Alice Chen"}
        mcp_client.find_employee.assert_awaited_once_with("Alice", role=Role.ANALYST)

    async def test_get_service_success(self) -> None:
        mcp_client = MagicMock(spec=EnterpriseDataClient)
        mcp_client.get_service = AsyncMock(return_value={"service_id": "SVC-001"})
        service = _service(mcp_client=mcp_client)

        result, _ = await service.execute(
            _call("get_service", {"service_id": "SVC-001"}),
            role=Role.ADMINISTRATOR,
            budget=_budget(),
        )

        assert result.success is True
        assert result.output == {"service_id": "SVC-001"}

    async def test_get_incident_success(self) -> None:
        mcp_client = MagicMock(spec=EnterpriseDataClient)
        mcp_client.get_incident = AsyncMock(return_value={"incident_id": "INC-001"})
        service = _service(mcp_client=mcp_client)

        result, _ = await service.execute(
            _call("get_incident", {"incident_id": "INC-001"}),
            role=Role.ADMINISTRATOR,
            budget=_budget(),
        )

        assert result.success is True
        assert result.output == {"incident_id": "INC-001"}


# ---------------------------------------------------------------------------
# 9. Audit logging
# ---------------------------------------------------------------------------


class TestAuditLogging:
    async def test_denied_call_emits_audit_event(self, caplog: pytest.LogCaptureFixture) -> None:
        service = _service()
        with caplog.at_level("WARNING", logger="src.services.tool_execution"):
            await service.execute(
                _call("get_service", {"service_id": "SVC-001"}),
                role=Role.VIEWER,
                budget=_budget(),
            )
        assert any(getattr(r, "event_type", None) == "tool_execution.audit" for r in caplog.records)

    async def test_successful_call_emits_audit_event(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        retriever = MagicMock(spec=HybridRetriever)
        retriever.search = AsyncMock(return_value=[_evidence()])
        service = _service(retriever=retriever)

        with caplog.at_level("INFO", logger="src.services.tool_execution"):
            await service.execute(
                _call("knowledge_search", {"query": "loan policy"}),
                role=Role.ANALYST,
                budget=_budget(),
            )

        audit_records = [
            r for r in caplog.records if getattr(r, "event_type", None) == "tool_execution.audit"
        ]
        assert len(audit_records) == 1
        assert audit_records[0].fields["success"] is True
        assert audit_records[0].fields["tool_name"] == "knowledge_search"
        # Arguments and output must never be logged.
        assert "loan policy" not in caplog.text
