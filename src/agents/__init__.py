"""Agents: LangGraph-based agent graphs (RLM-style recursive reasoning).

`agents` is the top of the dependency stack: it may depend on `core`,
`models`, `retrieval`, `memory`, `tools`, `security`, and `observability`.
No other `src` package may import from `agents`. Agents must never bypass
`security` to reach a tool or data source directly — see CLAUDE.md.

Public surface
--------------
State
- ``GraphState``        — typed TypedDict for the LangGraph StateGraph schema.
- ``initial_state``     — factory that creates a clean starting state.

Budget
- ``ExecutionBudget``   — resource limits and usage counters for one execution.

Supervisor
- ``IntentType``           — StrEnum of possible request intents.
- ``SupervisorDecision``   — Pydantic model for structured LLM output.
- ``SupervisorAgent``      — Agent class (prefer ``make_supervisor_node``).
- ``make_supervisor_node`` — Factory returning a LangGraph node callable.

Retrieval
- ``RetrievalAgent``      — Agent class (prefer ``make_retrieval_node``).
- ``make_retrieval_node`` — Factory returning a LangGraph node callable.
- ``generate_queries``    — Query-generation helper.

Response
- ``ConfidenceLevel``    — StrEnum: high / medium / low / unsupported.
- ``ResponseDecision``   — Pydantic model for structured LLM output.
- ``ResponseAgent``      — Agent class (prefer ``make_response_node``).
- ``make_response_node`` — Factory returning a LangGraph node callable.

Research (see ``docs/rlm.md``)
- ``ResearchAgent``      — Agent class (prefer ``make_research_node``).
- ``make_research_node`` — Factory returning a LangGraph node callable.

Guardrails
- ``InputGuard``              — Pattern-based validator for user message text.
- ``DocumentInjectionGuard``  — Scans retrieved documents for injected instructions.
- ``OutputGuard``             — Scans LLM answers for leakage / jailbreak success.
- ``ThreatType``              — StrEnum classifying detected injection categories.
- ``InputValidationResult``   — Result of ``InputGuard.validate()``.
- ``DocumentScanResult``      — Per-document result of ``DocumentInjectionGuard.scan()``.
- ``OutputValidationResult``  — Result of ``OutputGuard.validate()``.
- ``INJECTION_BLOCKED_RESPONSE`` — Generic refusal when user input is blocked.
- ``CitationGuardrail``       — Validates every citation against retrieved evidence.
- ``SAFE_FAILURE_RESPONSE``   — Canned message when all validation attempts fail.
"""

from src.agents.budget import ExecutionBudget
from src.agents.guardrails import (
    INJECTION_BLOCKED_RESPONSE,
    SAFE_FAILURE_RESPONSE,
    CitationGuardrail,
    DocumentInjectionGuard,
    DocumentScanResult,
    InputGuard,
    InputValidationResult,
    OutputGuard,
    OutputValidationResult,
    ThreatType,
)
from src.agents.research import ResearchAgent, make_research_node
from src.agents.response import ConfidenceLevel, ResponseAgent, ResponseDecision, make_response_node
from src.agents.retrieval import RetrievalAgent, generate_queries, make_retrieval_node
from src.agents.state import GraphState, initial_state
from src.agents.supervisor import (
    IntentType,
    SupervisorAgent,
    SupervisorDecision,
    make_supervisor_node,
)

__all__ = [
    "INJECTION_BLOCKED_RESPONSE",
    "SAFE_FAILURE_RESPONSE",
    "CitationGuardrail",
    "ConfidenceLevel",
    "DocumentInjectionGuard",
    "DocumentScanResult",
    "ExecutionBudget",
    "GraphState",
    "InputGuard",
    "InputValidationResult",
    "IntentType",
    "OutputGuard",
    "OutputValidationResult",
    "ResearchAgent",
    "ResponseAgent",
    "ResponseDecision",
    "RetrievalAgent",
    "SupervisorAgent",
    "SupervisorDecision",
    "ThreatType",
    "generate_queries",
    "initial_state",
    "make_research_node",
    "make_response_node",
    "make_retrieval_node",
    "make_supervisor_node",
]
