"""Centralized tool execution gateway — the single enforced path from an
agent's intent to invoke a tool through to that tool actually running.

See CLAUDE.md "Agents must never bypass application authorization": an
agent's (or an LLM's) decision to call a tool is never itself authorization
to do so. ``ToolExecutionService.execute`` is the one place in this codebase
that turns a ``ToolCall`` into a real side effect, and every such call passes
through the same fixed pipeline, in this order:

    1. Tool existence validation   — is ``tool_name`` a registered tool?
    2. Parameter validation        — do ``arguments`` satisfy that tool's
                                      Pydantic parameter schema?
    3. RBAC authorization          — does the caller's ``Role`` hold the
                                      ``Permission`` the tool requires
                                      (``src.security.authorization``)?
    4. Rate/budget validation      — has ``ExecutionBudget.max_tool_calls``
                                      already been exhausted for this
                                      execution? (No separate time-window
                                      rate limiter exists yet — the budget
                                      is this POC's rate/quota primitive;
                                      see ``src.agents.budget``.)
    5. Timeout                     — the handler runs under
                                      ``asyncio.wait_for`` with a per-tool
                                      timeout; a stuck tool cannot hang the
                                      agent graph.
    6. Tool execution               — the handler actually runs.
    7. Output validation           — the raw result must be JSON-safe and
                                      under a size cap before it is handed
                                      back (see CLAUDE.md "LLM output must be
                                      validated before use" — the same
                                      discipline applies to tool output,
                                      which is likewise untrusted until
                                      checked).
    8. Audit event                 — one structured log line is always
                                      emitted, success or failure, recording
                                      who, what tool, and the outcome — never
                                      the arguments or the output, both of
                                      which may carry enterprise data or PII
                                      (see CLAUDE.md "Logging requirements").

Every step before execution can reject the call; a rejection never reaches
step 6, so a rejected call is guaranteed no side effect. No other module may
call ``run_analysis``, ``EnterpriseDataClient``, or ``HybridRetriever.search``
on an agent's behalf — those integrations are wired in below as the only
three tool categories this POC exposes (knowledge search, Python analysis,
MCP enterprise data), and this service is their only caller from agent code.

Dependency boundary (see CLAUDE.md)
------------------------------------
``services`` may import ``agents``, ``retrieval``, ``memory``, ``tools``,
``security``, and ``observability``. This module needs the retrieval layer
(``HybridRetriever``) directly, which is why it lives in ``services`` rather
than ``tools`` — ``tools`` is not permitted to import ``retrieval``.
"""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from src.agents.budget import ExecutionBudget
from src.core.logging import get_logger, log_error, log_info, log_warning
from src.mcp.client import DEFAULT_TIMEOUT as DEFAULT_MCP_TIMEOUT
from src.mcp.client import EnterpriseDataClient, MCPClientError
from src.models.common import NonEmptyStr
from src.models.enums import Role
from src.models.tools import ToolCall, ToolResult
from src.observability.tracing import trace_tool_call
from src.retrieval.hybrid.retriever import HybridRetriever
from src.security.authorization import AuthorizationPolicy, Permission, PermissionDeniedError
from src.tools.analysis import (
    ANALYSIS_TIMEOUT_SECONDS,
    AnalysisError,
    AnalysisRequest,
    run_analysis,
)

_logger = get_logger(__name__)

KNOWLEDGE_SEARCH_TIMEOUT_SECONDS: float = 15.0
MAX_OUTPUT_BYTES: int = 200_000
"""Upper bound on serialized tool output — guards against a runaway or
malformed tool response flooding the agent's context (and, by extension,
what could be exfiltrated through it)."""


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class ToolExecutionError(Exception):
    """Raised internally for a structured, machine-readable rejection.

    Never escapes ``ToolExecutionService.execute`` — it is always converted
    to a failed ``ToolResult`` before returning to the caller. ``code``
    matches ``src.models.errors.ErrorCode``'s pattern (upper-snake-case).
    """

    def __init__(self, code: str, message: str) -> None:
        self.code = code
        self.message = message
        super().__init__(message)


# ---------------------------------------------------------------------------
# Per-tool parameter schemas
# ---------------------------------------------------------------------------


class KnowledgeSearchParams(BaseModel):
    """Parameters for the ``knowledge_search`` tool."""

    model_config = ConfigDict(frozen=True)

    query: NonEmptyStr
    top_k: int = Field(default=10, ge=1, le=50)


class EmployeeLookupParams(BaseModel):
    """Parameters for the ``find_employee`` tool."""

    model_config = ConfigDict(frozen=True)

    query: NonEmptyStr


class ServiceLookupParams(BaseModel):
    """Parameters for the ``get_service`` tool."""

    model_config = ConfigDict(frozen=True)

    service_id: NonEmptyStr


class IncidentLookupParams(BaseModel):
    """Parameters for the ``get_incident`` tool."""

    model_config = ConfigDict(frozen=True)

    incident_id: NonEmptyStr


# ---------------------------------------------------------------------------
# Tool registry
# ---------------------------------------------------------------------------

_ToolHandler = Callable[[BaseModel, Role], Awaitable[Any]]


@dataclass(frozen=True)
class ToolDefinition:
    """One registered tool: its permission, parameter schema, and handler."""

    name: str
    permission: Permission
    params_model: type[BaseModel]
    handler: _ToolHandler
    timeout: float
    description: str


# ---------------------------------------------------------------------------
# Output validation
# ---------------------------------------------------------------------------


def _validate_output(tool_name: str, value: Any) -> Any:
    """Return a JSON-safe copy of *value*, or raise ``ToolExecutionError``.

    A tool's raw return value is untrusted until it is proven to be plain,
    bounded, JSON-representable data — see module docstring, step 7. No
    ``default=`` fallback is used: an unrecognized type must fail this check,
    not be silently stringified.
    """
    try:
        serialized = json.dumps(value)
    except (TypeError, ValueError) as exc:
        raise ToolExecutionError(
            "OUTPUT_VALIDATION_FAILED",
            f"tool {tool_name!r} produced output that is not JSON-serializable",
        ) from exc

    if len(serialized) > MAX_OUTPUT_BYTES:
        raise ToolExecutionError(
            "OUTPUT_VALIDATION_FAILED",
            f"tool {tool_name!r} output of {len(serialized)} bytes exceeds "
            f"the {MAX_OUTPUT_BYTES}-byte limit",
        )

    return json.loads(serialized)


# ---------------------------------------------------------------------------
# ToolExecutionService
# ---------------------------------------------------------------------------


class ToolExecutionService:
    """Owns the tool registry and enforces the pipeline in the module docstring.

    Build one instance per process (it is stateless beyond its injected
    dependencies) and call ``execute()`` for every ``ToolCall`` an agent
    produces — never call ``run_analysis``, ``EnterpriseDataClient``, or
    ``HybridRetriever.search`` directly from agent code.
    """

    def __init__(
        self,
        *,
        retriever: HybridRetriever,
        mcp_client: EnterpriseDataClient | None = None,
        policy: AuthorizationPolicy | None = None,
    ) -> None:
        self._retriever = retriever
        self._mcp = mcp_client if mcp_client is not None else EnterpriseDataClient()
        self._policy = policy if policy is not None else AuthorizationPolicy()
        self._registry: dict[str, ToolDefinition] = self._build_registry()

    def _build_registry(self) -> dict[str, ToolDefinition]:
        definitions = [
            ToolDefinition(
                name="knowledge_search",
                permission=Permission.KNOWLEDGE_SEARCH,
                params_model=KnowledgeSearchParams,
                handler=self._run_knowledge_search,
                timeout=KNOWLEDGE_SEARCH_TIMEOUT_SECONDS,
                description="Search enterprise knowledge base documents relevant to a query.",
            ),
            ToolDefinition(
                name="python_analysis",
                permission=Permission.ANALYTICS,
                params_model=AnalysisRequest,
                handler=self._run_python_analysis,
                timeout=ANALYSIS_TIMEOUT_SECONDS,
                description="Run a constrained analytical operation over structured data.",
            ),
            ToolDefinition(
                name="find_employee",
                permission=Permission.MCP_TOOLS,
                params_model=EmployeeLookupParams,
                handler=self._run_find_employee,
                timeout=DEFAULT_MCP_TIMEOUT,
                description="Look up an employee by name or ID in the enterprise directory.",
            ),
            ToolDefinition(
                name="get_service",
                permission=Permission.MCP_TOOLS,
                params_model=ServiceLookupParams,
                handler=self._run_get_service,
                timeout=DEFAULT_MCP_TIMEOUT,
                description="Look up a service catalog entry by ID.",
            ),
            ToolDefinition(
                name="get_incident",
                permission=Permission.MCP_TOOLS,
                params_model=IncidentLookupParams,
                handler=self._run_get_incident,
                timeout=DEFAULT_MCP_TIMEOUT,
                description="Look up an incident record by ID.",
            ),
        ]
        return {definition.name: definition for definition in definitions}

    @property
    def available_tools(self) -> tuple[str, ...]:
        """Names of every registered tool, sorted for stable output."""
        return tuple(sorted(self._registry))

    # ------------------------------------------------------------------
    # Per-tool handlers
    # ------------------------------------------------------------------
    # Called only after existence/parameter/RBAC/budget checks pass (see
    # `execute`). Each receives its own validated params model and the
    # caller's role, and returns raw (not-yet-output-validated) data.
    # Inner permission/parameter checks performed by these integrations
    # (e.g. `EnterpriseDataClient._check`, `run_analysis`'s own
    # `_policy.authorize`) are intentionally left in place as
    # defense-in-depth, not relied upon as the primary gate.

    async def _run_knowledge_search(self, params: BaseModel, role: Role) -> Any:
        assert isinstance(params, KnowledgeSearchParams)
        results = await self._retriever.search(params.query, top_k=params.top_k, roles=[role])
        return [item.model_dump(mode="json") for item in results]

    async def _run_python_analysis(self, params: BaseModel, role: Role) -> Any:
        assert isinstance(params, AnalysisRequest)
        result = await run_analysis(params, role=role, timeout=ANALYSIS_TIMEOUT_SECONDS)
        return result.model_dump(mode="json")

    async def _run_find_employee(self, params: BaseModel, role: Role) -> Any:
        assert isinstance(params, EmployeeLookupParams)
        return await self._mcp.find_employee(params.query, role=role)

    async def _run_get_service(self, params: BaseModel, role: Role) -> Any:
        assert isinstance(params, ServiceLookupParams)
        return await self._mcp.get_service(params.service_id, role=role)

    async def _run_get_incident(self, params: BaseModel, role: Role) -> Any:
        assert isinstance(params, IncidentLookupParams)
        return await self._mcp.get_incident(params.incident_id, role=role)

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    async def execute(
        self,
        call: ToolCall,
        *,
        role: Role,
        budget: ExecutionBudget,
        conversation_id: str | None = None,
    ) -> tuple[ToolResult, ExecutionBudget]:
        """Run *call* through the full enforcement pipeline.

        Never raises for an expected failure — unknown tool, bad parameters,
        denied permission, exhausted budget, timeout, handler error, or
        invalid output all become a structured ``ToolResult(success=False,
        error_code=..., error_message=...)``. An unexpected exception from a
        handler is caught, logged with its traceback, and also converted to
        a structured result — this is the trust boundary between agent
        orchestration and real side effects; it must not propagate a raw
        traceback to the caller.

        Returns
        -------
        ``(ToolResult, ExecutionBudget)`` — the budget is unchanged unless
        the call passed rate/budget validation and reached execution, in
        which case it reflects one consumed tool call.
        """
        t0 = time.monotonic()

        # 1. Tool existence validation ----------------------------------
        tool_def = self._registry.get(call.tool_name)
        if tool_def is None:
            return self._reject(
                call,
                budget,
                role=role,
                conversation_id=conversation_id,
                t0=t0,
                code="TOOL_NOT_FOUND",
                message=f"no such tool: {call.tool_name!r}",
            )

        # 2. Parameter validation -----------------------------------------
        try:
            params = tool_def.params_model.model_validate(call.arguments)
        except ValidationError as exc:
            return self._reject(
                call,
                budget,
                role=role,
                conversation_id=conversation_id,
                t0=t0,
                code="INVALID_PARAMETERS",
                message=f"invalid parameters for {call.tool_name!r}: {exc.error_count()} error(s)",
            )

        # 3. RBAC authorization ---------------------------------------------
        try:
            self._policy.authorize(role, tool_def.permission)
        except PermissionDeniedError as exc:
            return self._reject(
                call,
                budget,
                role=role,
                conversation_id=conversation_id,
                t0=t0,
                code="PERMISSION_DENIED",
                message=str(exc),
            )

        # 4. Rate / budget validation -----------------------------------------
        if budget.is_tool_exhausted:
            return self._reject(
                call,
                budget,
                role=role,
                conversation_id=conversation_id,
                t0=t0,
                code="BUDGET_EXHAUSTED",
                message=(
                    f"tool-call budget exhausted ({budget.used_tool_calls}/{budget.max_tool_calls})"
                ),
            )
        updated_budget = budget.consume(tool_calls=1)

        # 5. + 6. Timeout + tool execution ------------------------------------
        with trace_tool_call(call.tool_name, conversation_id=conversation_id):
            try:
                raw_output = await asyncio.wait_for(
                    tool_def.handler(params, role), timeout=tool_def.timeout
                )
            except TimeoutError:
                return self._reject(
                    call,
                    updated_budget,
                    role=role,
                    conversation_id=conversation_id,
                    t0=t0,
                    code="TIMEOUT",
                    message=f"{call.tool_name!r} timed out after {tool_def.timeout}s",
                )
            except PermissionDeniedError as exc:
                return self._reject(
                    call,
                    updated_budget,
                    role=role,
                    conversation_id=conversation_id,
                    t0=t0,
                    code="PERMISSION_DENIED",
                    message=str(exc),
                )
            except (MCPClientError, AnalysisError) as exc:
                code = getattr(exc, "code", None) or "EXECUTION_FAILED"
                message = getattr(exc, "message", None) or str(exc)
                return self._reject(
                    call,
                    updated_budget,
                    role=role,
                    conversation_id=conversation_id,
                    t0=t0,
                    code=code,
                    message=message,
                )
            except Exception as exc:
                log_error(
                    _logger,
                    "tool_execution.unexpected_error",
                    "Tool handler raised an unexpected exception",
                    error=exc,
                    tool_name=call.tool_name,
                    role=str(role),
                )
                return self._reject(
                    call,
                    updated_budget,
                    role=role,
                    conversation_id=conversation_id,
                    t0=t0,
                    code="EXECUTION_FAILED",
                    message=f"{call.tool_name!r} failed unexpectedly",
                )

        # 7. Output validation --------------------------------------------
        try:
            safe_output = _validate_output(call.tool_name, raw_output)
        except ToolExecutionError as exc:
            return self._reject(
                call,
                updated_budget,
                role=role,
                conversation_id=conversation_id,
                t0=t0,
                code=exc.code,
                message=exc.message,
            )

        # 8. Audit event (success) -----------------------------------------
        result = ToolResult(tool_call_id=call.tool_call_id, success=True, output=safe_output)
        self._audit(
            call,
            role=role,
            conversation_id=conversation_id,
            success=True,
            error_code=None,
            latency_ms=round((time.monotonic() - t0) * 1000, 1),
        )
        return result, updated_budget

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _reject(
        self,
        call: ToolCall,
        budget: ExecutionBudget,
        *,
        role: Role,
        conversation_id: str | None,
        t0: float,
        code: str,
        message: str,
    ) -> tuple[ToolResult, ExecutionBudget]:
        """Build a failed ``ToolResult``, emit its audit event, and return."""
        self._audit(
            call,
            role=role,
            conversation_id=conversation_id,
            success=False,
            error_code=code,
            latency_ms=round((time.monotonic() - t0) * 1000, 1),
            message=message,
        )
        result = ToolResult(
            tool_call_id=call.tool_call_id,
            success=False,
            error_code=code,
            error_message=message,
        )
        return result, budget

    def _audit(
        self,
        call: ToolCall,
        *,
        role: Role,
        conversation_id: str | None,
        success: bool,
        error_code: str | None,
        latency_ms: float,
        message: str | None = None,
    ) -> None:
        """Emit one structured audit log line for this tool call.

        Only identifiers, the role, and the outcome are recorded — never
        ``call.arguments`` or the tool's output, either of which may carry
        enterprise data or PII; see CLAUDE.md "Logging requirements".
        """
        fields: dict[str, Any] = {
            "tool_call_id": call.tool_call_id,
            "tool_name": call.tool_name,
            "role": str(role),
            "success": success,
            "error_code": error_code,
            "latency_ms": latency_ms,
            "conversation_id": conversation_id,
        }
        if success:
            log_info(
                _logger,
                "tool_execution.audit",
                "Tool call authorized and executed",
                **fields,
            )
        else:
            log_warning(
                _logger,
                "tool_execution.audit",
                f"Tool call denied or failed: {message}",
                **fields,
            )
