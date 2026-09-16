"""Execution budget — tracks resource consumption for one agent graph run.

The budget is part of ``GraphState`` and is replaced (not accumulated) when
a node returns an updated copy.  Nodes that consume LLM tokens, tool calls,
or retrieval results call ``budget.consume(...)`` to get a new budget, then
return ``{"budget": new_budget}`` as part of their state update.

All fields are validated on construction and the model is frozen so callers
cannot accidentally mutate shared budget state between nodes.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ExecutionBudget(BaseModel):
    """Limits and usage counters for one agent graph execution.

    Limits are set at graph entry and never change.  Usage counters grow as
    nodes consume resources.  ``is_exhausted`` signals the supervisor to
    terminate rather than continue planning.
    """

    model_config = ConfigDict(frozen=True)

    # --- Limits ----------------------------------------------------------------
    max_tokens: int = Field(
        default=10_000,
        gt=0,
        description="Total token budget (input + output across all LLM calls).",
    )
    max_retrieval_results: int = Field(
        default=20,
        gt=0,
        description="Maximum number of retrieved chunks across all retrieval queries.",
    )
    max_research_depth: int = Field(
        default=3,
        gt=0,
        description="Maximum number of recursive research cycles.",
    )
    max_tool_calls: int = Field(
        default=10,
        gt=0,
        description="Maximum number of external tool invocations.",
    )

    # --- Usage counters --------------------------------------------------------
    used_tokens: int = Field(default=0, ge=0)
    used_retrieval_results: int = Field(default=0, ge=0)
    used_research_steps: int = Field(default=0, ge=0)
    used_tool_calls: int = Field(default=0, ge=0)

    # --- Derived properties ---------------------------------------------------

    @property
    def remaining_tokens(self) -> int:
        """Tokens left before the token budget is exhausted."""
        return max(0, self.max_tokens - self.used_tokens)

    @property
    def remaining_tool_calls(self) -> int:
        """Tool calls left before the tool-call budget is exhausted."""
        return max(0, self.max_tool_calls - self.used_tool_calls)

    @property
    def is_token_exhausted(self) -> bool:
        """True when no token budget remains."""
        return self.used_tokens >= self.max_tokens

    @property
    def is_tool_exhausted(self) -> bool:
        """True when no tool-call budget remains."""
        return self.used_tool_calls >= self.max_tool_calls

    @property
    def is_exhausted(self) -> bool:
        """True when any primary budget (tokens or tool calls) is depleted."""
        return self.is_token_exhausted or self.is_tool_exhausted

    # --- Mutation (returns new instance) -------------------------------------

    def consume(
        self,
        *,
        tokens: int = 0,
        retrieval_results: int = 0,
        research_steps: int = 0,
        tool_calls: int = 0,
    ) -> "ExecutionBudget":
        """Return a new budget with the given amounts added to usage counters.

        Does not clamp usage to the limits — the caller is responsible for
        checking ``is_exhausted`` after consuming.
        """
        return self.model_copy(
            update={
                "used_tokens": self.used_tokens + tokens,
                "used_retrieval_results": self.used_retrieval_results + retrieval_results,
                "used_research_steps": self.used_research_steps + research_steps,
                "used_tool_calls": self.used_tool_calls + tool_calls,
            }
        )

    @model_validator(mode="after")
    def _usage_not_negative(self) -> "ExecutionBudget":
        for field, value in [
            ("used_tokens", self.used_tokens),
            ("used_retrieval_results", self.used_retrieval_results),
            ("used_research_steps", self.used_research_steps),
            ("used_tool_calls", self.used_tool_calls),
        ]:
            if value < 0:
                raise ValueError(f"{field} cannot be negative")
        return self
