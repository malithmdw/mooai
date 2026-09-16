"""Domain models for the Supervisor agent decision."""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, field_validator

from src.models.enums import AgentState


class IntentType(StrEnum):
    """Classified intent of a user request routed through the supervisor."""

    KNOWLEDGE_QUESTION = "knowledge_question"
    ANALYTICAL_RESEARCH = "analytical_research"
    TOOL_REQUEST = "tool_request"
    MIXED_REQUEST = "mixed_request"
    UNSUPPORTED_REQUEST = "unsupported_request"


_VALID_ROUTES: frozenset[AgentState] = frozenset(
    {AgentState.RETRIEVAL, AgentState.RESEARCH, AgentState.TOOL_EXECUTION, AgentState.RESPONSE}
)


class SupervisorDecision(BaseModel):
    """Structured output from the Supervisor LLM call.

    Constructed by validating the ``supervisor_decision`` tool-use block
    returned by the Anthropic API.  A ``ValidationError`` on construction
    means the LLM produced an invalid decision; ``supervisor_node`` handles
    this gracefully by appending to ``GraphState.errors``.
    """

    model_config = ConfigDict(frozen=True)

    intent: IntentType
    requires_retrieval: bool = Field(
        description="Whether knowledge-base retrieval is needed to answer this request"
    )
    requires_research: bool = Field(
        description="Whether multi-step recursive research (RLM) is needed"
    )
    requires_tools: bool = Field(
        description="Whether external tool calls are needed"
    )
    task_plan: list[str] = Field(
        description="Ordered high-level steps to fulfil the intent",
        min_length=1,
    )
    route_to: AgentState = Field(description="Next agent stage for execution")
    reasoning: str = Field(
        description="Brief rationale for this decision; not exposed to the user"
    )

    @field_validator("route_to")
    @classmethod
    def _route_is_valid(cls, v: AgentState) -> AgentState:
        if v not in _VALID_ROUTES:
            raise ValueError(
                f"supervisor cannot route to {v!r}; valid routes: "
                + ", ".join(sorted(s.value for s in _VALID_ROUTES))
            )
        return v
