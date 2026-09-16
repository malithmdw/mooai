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
"""

from src.agents.budget import ExecutionBudget
from src.agents.state import GraphState, initial_state
from src.agents.supervisor import IntentType, SupervisorAgent, SupervisorDecision, make_supervisor_node

__all__ = [
    "GraphState",
    "initial_state",
    "ExecutionBudget",
    "IntentType",
    "SupervisorDecision",
    "SupervisorAgent",
    "make_supervisor_node",
]
