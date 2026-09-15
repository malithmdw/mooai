"""Enumerated domain vocabularies shared across models."""

from __future__ import annotations

from enum import StrEnum


class Role(StrEnum):
    """RBAC role granted to a `User`.

    Every data/tool access is checked against these via `src.security`,
    never against agent intent — see CLAUDE.md security principles.
    """

    VIEWER = "VIEWER"
    ANALYST = "ANALYST"
    ADMINISTRATOR = "ADMINISTRATOR"


class MessageRole(StrEnum):
    """Who authored a `Message` within a `Conversation`.

    Distinct from `Role` (RBAC permissions) — this is chat-turn authorship,
    not access control.
    """

    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


class AccessLevel(StrEnum):
    """Sensitivity classification of a document, used for RBAC filtering."""

    PUBLIC = "PUBLIC"
    INTERNAL = "INTERNAL"
    CONFIDENTIAL = "CONFIDENTIAL"
    RESTRICTED = "RESTRICTED"


class AgentState(StrEnum):
    """Observable stage of agent execution, reported via `AgentEvent`.

    Supports transparent, inspectable agent execution — see CLAUDE.md
    project purpose.
    """

    SUPERVISOR = "supervisor"
    RETRIEVAL = "retrieval"
    RESEARCH = "research"
    TOOL_EXECUTION = "tool_execution"
    MEMORY = "memory"
    VALIDATION = "validation"
    RESPONSE = "response"


class ResearchStatus(StrEnum):
    """Lifecycle status of a `ResearchTask`."""

    PENDING = "PENDING"
    IN_PROGRESS = "IN_PROGRESS"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
