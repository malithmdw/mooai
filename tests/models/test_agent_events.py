"""Tests for `src.models.agent_events.AgentEvent`."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.models.agent_events import AgentEvent
from src.models.enums import AgentState


@pytest.mark.parametrize("state", list(AgentState))
def test_supports_every_observable_state(state: AgentState) -> None:
    event = AgentEvent(
        event_id="evt-1",
        conversation_id="conv-1",
        state=state,
        message=f"entered {state.value}",
    )
    assert event.state is state


def test_expected_observable_states_are_present() -> None:
    assert {state.value for state in AgentState} == {
        "supervisor",
        "retrieval",
        "research",
        "tool_execution",
        "memory",
        "validation",
        "response",
    }


def test_event_is_immutable() -> None:
    event = AgentEvent(
        event_id="evt-1",
        conversation_id="conv-1",
        state=AgentState.SUPERVISOR,
        message="routing request",
    )
    with pytest.raises(ValidationError):
        event.state = AgentState.RESPONSE  # type: ignore[misc]


def test_rejects_empty_message() -> None:
    with pytest.raises(ValidationError, match="message"):
        AgentEvent(
            event_id="evt-1",
            conversation_id="conv-1",
            state=AgentState.SUPERVISOR,
            message="   ",
        )
