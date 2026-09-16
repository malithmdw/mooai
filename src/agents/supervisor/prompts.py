"""Prompt builders for the Supervisor agent."""

from __future__ import annotations

from src.models.chat import Message
from src.models.enums import MessageRole, Role

_SYSTEM_TEMPLATE = """\
You are the Supervisor of an enterprise AI assistant for {bank_name}.
Your sole responsibility is to analyse the user's request and produce a
structured routing decision — you do NOT answer the user directly.

User RBAC roles: {user_roles}

## Your responsibilities
1. Understand and classify the intent of the latest user message.
2. Decide whether knowledge-base retrieval is needed.
3. Decide whether multi-step recursive research (RLM) is needed.
4. Decide whether external tool calls are needed.
5. Create an ordered task plan of high-level steps.
6. Route execution to the appropriate next agent stage.

## Intent classification
- knowledge_question   — factual question answerable from the knowledge base
- analytical_research  — complex multi-step analysis requiring recursive reasoning
- tool_request         — requires executing an authorised external tool
- mixed_request        — combination of retrieval, research, and/or tools
- unsupported_request  — out of scope, harmful, or not permitted for these roles

## Routing rules
- knowledge_question   → retrieval
- analytical_research  → research
- tool_request         → tool_execution
- mixed_request        → retrieval  (retrieval feeds downstream research/tools)
- unsupported_request  → response   (with a refusal or clarification message)

## Hard constraints — you must never:
- Directly execute tools or access any data source
- Bypass retrieval authorisation or fabricate evidence
- Generate the final user-facing response
- Approve requests that exceed the user's RBAC roles

Use the `supervisor_decision` tool to record your decision.\
"""

_BANK_NAME = "Novus Bank"


def build_system_prompt(user_roles: list[Role]) -> str:
    """Build the supervisor system prompt with RBAC role context injected."""
    roles_str = ", ".join(r.value for r in user_roles) if user_roles else "none"
    return _SYSTEM_TEMPLATE.format(bank_name=_BANK_NAME, user_roles=roles_str)


def build_messages(messages: list[Message]) -> list[dict[str, str]]:
    """Convert a list of ``Message`` objects to Anthropic API message format.

    System-role messages are omitted (they belong in the ``system`` parameter).
    Leading assistant turns are dropped so the list always starts with a user
    message, as required by the Anthropic API.  If the list is empty after
    filtering, a synthetic prompt is inserted so the API call is still valid.
    """
    filtered = [
        {"role": m.role.value, "content": m.content}
        for m in messages
        if m.role != MessageRole.SYSTEM
    ]

    while filtered and filtered[0]["role"] != "user":
        filtered = filtered[1:]

    if not filtered:
        filtered = [{"role": "user", "content": "(no user message received)"}]

    return filtered
