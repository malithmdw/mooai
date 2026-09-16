"""Supervisor agent — intent classification and execution routing.

The supervisor is the entry point of every agent graph run.  It classifies
the user's intent, decides which downstream agents are needed, builds a
task plan, and routes execution to the appropriate next stage.  It must
never generate the final answer, execute tools, or bypass retrieval.

Public surface
--------------
- ``IntentType``           — StrEnum of the five possible request intents.
- ``SupervisorDecision``   — Pydantic model wrapping the structured LLM output.
- ``SupervisorAgent``      — Agent class (prefer ``make_supervisor_node``).
- ``make_supervisor_node`` — Factory returning a LangGraph node callable.
"""

from src.agents.supervisor.models import IntentType, SupervisorDecision
from src.agents.supervisor.node import SupervisorAgent, make_supervisor_node

__all__ = [
    "IntentType",
    "SupervisorDecision",
    "SupervisorAgent",
    "make_supervisor_node",
]
