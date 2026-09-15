"""Shared Pydantic schemas and data models.

`models` defines data shapes only — no business logic, no I/O. It may be
imported by any other `src` package; it must not import from `api`,
`agents`, `retrieval`, `memory`, `tools`, `security`, `observability`, or
`services` (no upward or lateral dependencies).
"""

from src.models.agent_events import AgentEvent
from src.models.chat import ChatRequest, ChatResponse, Conversation, Message
from src.models.documents import DocumentMetadata, RetrievedDocument
from src.models.enums import AccessLevel, AgentState, MessageRole, ResearchStatus, Role
from src.models.errors import ErrorResponse
from src.models.evidence import Citation, Evidence
from src.models.health import HealthResponse
from src.models.research import ResearchResult, ResearchTask
from src.models.tools import ToolCall, ToolResult
from src.models.user import User
from src.models.validation import ValidationResult

__all__ = [
    "AccessLevel",
    "AgentEvent",
    "AgentState",
    "ChatRequest",
    "ChatResponse",
    "Citation",
    "Conversation",
    "DocumentMetadata",
    "ErrorResponse",
    "Evidence",
    "HealthResponse",
    "Message",
    "MessageRole",
    "ResearchResult",
    "ResearchStatus",
    "ResearchTask",
    "RetrievedDocument",
    "Role",
    "ToolCall",
    "ToolResult",
    "User",
    "ValidationResult",
]
