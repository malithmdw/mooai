"""Supervisor agent LangGraph node."""

from __future__ import annotations

from collections.abc import Callable, Coroutine
from typing import Any

import anthropic
from pydantic import ValidationError

from src.agents.budget import ExecutionBudget
from src.agents.state import GraphState
from src.agents.supervisor.models import IntentType, SupervisorDecision
from src.agents.supervisor.prompts import build_messages, build_system_prompt
from src.core.config import get_settings
from src.core.logging import get_logger, log_debug, log_error, log_info, log_warning
from src.models.enums import AgentState

_logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Anthropic tool definition for structured supervisor output
# ---------------------------------------------------------------------------

_SUPERVISOR_TOOL: dict[str, Any] = {
    "name": "supervisor_decision",
    "description": (
        "Record the supervisor's analysis and routing decision for the current request. "
        "Call this tool exactly once to commit the intent classification, task plan, "
        "and next-stage route."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "intent": {
                "type": "string",
                "enum": [e.value for e in IntentType],
                "description": "Classified intent of the user request.",
            },
            "requires_retrieval": {
                "type": "boolean",
                "description": "True when knowledge-base retrieval is needed.",
            },
            "requires_research": {
                "type": "boolean",
                "description": "True when multi-step recursive research is needed.",
            },
            "requires_tools": {
                "type": "boolean",
                "description": "True when external tool calls are needed.",
            },
            "task_plan": {
                "type": "array",
                "items": {"type": "string"},
                "minItems": 1,
                "description": "Ordered high-level steps to fulfil the intent.",
            },
            "route_to": {
                "type": "string",
                "enum": ["retrieval", "research", "tool_execution", "response"],
                "description": "Next agent stage for execution.",
            },
            "reasoning": {
                "type": "string",
                "description": "Brief rationale for this decision; not shown to the user.",
            },
        },
        "required": [
            "intent",
            "requires_retrieval",
            "requires_research",
            "requires_tools",
            "task_plan",
            "route_to",
            "reasoning",
        ],
    },
}


# ---------------------------------------------------------------------------
# Supervisor agent class
# ---------------------------------------------------------------------------


class SupervisorAgent:
    """Calls the Anthropic API with a forced ``supervisor_decision`` tool call
    and validates the result as a ``SupervisorDecision``.

    Construct via ``make_supervisor_node`` — that function returns a
    LangGraph-compatible async node callable.
    """

    def __init__(
        self,
        *,
        client: anthropic.AsyncAnthropic,
        model: str,
    ) -> None:
        self._client = client
        self._model = model

    async def run(self, state: GraphState) -> dict[str, Any]:
        """Execute the supervisor step and return a partial GraphState update.

        Never raises: all exceptions are captured and returned as ``errors``
        entries so the graph can continue or terminate gracefully.
        """
        budget: ExecutionBudget = state["budget"]

        if budget.is_exhausted:
            log_warning(
                _logger,
                "supervisor.budget_exhausted",
                "Supervisor skipped — execution budget already exhausted",
                conversation_id=state["conversation_id"],
            )
            return {
                "intent": IntentType.UNSUPPORTED_REQUEST,
                "task_plan": ["Budget exhausted — terminate execution"],
                "current_agent": AgentState.SUPERVISOR,
                "current_node": "supervisor",
                "errors": ["supervisor skipped: execution budget exhausted"],
                "budget": budget,
            }

        system_prompt = build_system_prompt(state["user_role"])
        messages = build_messages(list(state["messages"]))

        log_debug(
            _logger,
            "supervisor.call_start",
            "Calling Anthropic for supervisor decision",
            model=self._model,
            message_count=len(messages),
            conversation_id=state["conversation_id"],
        )

        try:
            response = await self._client.messages.create(
                model=self._model,
                max_tokens=512,
                system=system_prompt,
                tools=[_SUPERVISOR_TOOL],  # type: ignore[list-item]
                tool_choice={"type": "tool", "name": "supervisor_decision"},
                messages=messages,  # type: ignore[arg-type]
            )
        except Exception as exc:
            log_error(
                _logger,
                "supervisor.llm_error",
                "Anthropic call failed in supervisor",
                error=exc,
            )
            return {
                "intent": IntentType.UNSUPPORTED_REQUEST,
                "task_plan": [],
                "current_agent": AgentState.SUPERVISOR,
                "current_node": "supervisor",
                "errors": [f"supervisor LLM call failed: {exc}"],
                "budget": budget,
            }

        tokens_used = response.usage.input_tokens + response.usage.output_tokens
        new_budget = budget.consume(tokens=tokens_used)

        tool_block = next(
            (b for b in response.content if getattr(b, "type", None) == "tool_use"),
            None,
        )
        if tool_block is None:
            log_error(
                _logger,
                "supervisor.no_tool_block",
                "Supervisor LLM response contained no tool-use block",
                conversation_id=state["conversation_id"],
            )
            return {
                "intent": IntentType.UNSUPPORTED_REQUEST,
                "task_plan": [],
                "current_agent": AgentState.SUPERVISOR,
                "current_node": "supervisor",
                "errors": ["supervisor LLM returned no tool-use block"],
                "budget": new_budget,
            }

        try:
            decision = SupervisorDecision.model_validate(tool_block.input)
        except ValidationError as exc:
            log_error(
                _logger,
                "supervisor.validation_error",
                "Supervisor decision failed Pydantic validation",
                error=exc,
                conversation_id=state["conversation_id"],
            )
            return {
                "intent": IntentType.UNSUPPORTED_REQUEST,
                "task_plan": [],
                "current_agent": AgentState.SUPERVISOR,
                "current_node": "supervisor",
                "errors": [f"supervisor decision validation failed: {exc}"],
                "budget": new_budget,
            }

        log_info(
            _logger,
            "supervisor.decision",
            "Supervisor decision recorded",
            intent=decision.intent,
            route_to=decision.route_to,
            requires_retrieval=decision.requires_retrieval,
            requires_research=decision.requires_research,
            requires_tools=decision.requires_tools,
            task_plan_steps=len(decision.task_plan),
            conversation_id=state["conversation_id"],
        )

        return {
            "intent": decision.intent,
            "task_plan": decision.task_plan,
            "current_agent": AgentState.SUPERVISOR,
            "current_node": "supervisor",
            "budget": new_budget,
        }


# ---------------------------------------------------------------------------
# Public factory
# ---------------------------------------------------------------------------


def make_supervisor_node(
    *,
    client: anthropic.AsyncAnthropic,
    model: str | None = None,
) -> Callable[[GraphState], Coroutine[Any, Any, dict[str, Any]]]:
    """Return a LangGraph-compatible async callable for the supervisor node.

    Parameters
    ----------
    client:
        An ``anthropic.AsyncAnthropic`` instance.  Injected so callers
        control the client lifecycle; tests can pass a mock.
    model:
        Model identifier.  Defaults to ``Settings.model_name``.
    """
    resolved_model = model or get_settings().model_name
    agent = SupervisorAgent(client=client, model=resolved_model)
    return agent.run
