"""Response Agent — evidence-grounded answer synthesis with citation validation.

Produces the final user-visible answer by grounding it in retrieved documents,
validating every citation against the actually retrieved set, and clearly
identifying confidence and limitations.

Public surface
--------------
- ``ConfidenceLevel``    — StrEnum: high / medium / low / unsupported.
- ``CitedChunk``         — One LLM-generated citation to a retrieved chunk.
- ``ResponseDecision``   — Pydantic model wrapping the full structured output.
- ``ResponseAgent``      — Agent class (prefer ``make_response_node``).
- ``make_response_node`` — Factory returning a LangGraph node callable.
- ``MAX_EVIDENCE_IN_PROMPT`` — Cap on evidence items forwarded to the LLM.
"""

from src.agents.response.models import CitedChunk, ConfidenceLevel, ResponseDecision
from src.agents.response.node import MAX_EVIDENCE_IN_PROMPT, ResponseAgent, make_response_node

__all__ = [
    "ConfidenceLevel",
    "CitedChunk",
    "ResponseDecision",
    "ResponseAgent",
    "make_response_node",
    "MAX_EVIDENCE_IN_PROMPT",
]
