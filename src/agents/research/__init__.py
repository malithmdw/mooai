"""Research Agent — recursive, RLM-style multi-step research over a bounded
evidence pool.

Wraps ``HybridRetriever`` + the Anthropic API in a LangGraph node that
recursively decomposes a research question into a tree of ``ResearchTask``s,
analyzing bounded batches of evidence rather than ever placing the whole
document collection in one context window. See ``docs/rlm.md`` for the full
design and ``src.agents.research.node`` for the stage-by-stage pipeline.

Public surface
--------------
- ``ResearchAgent``      — Agent class (prefer ``make_research_node``).
- ``make_research_node`` — Factory returning a LangGraph node callable.
- ``ResearchConfidence``, ``QuestionAnalysis``, ``Claim``, ``ResearchFinding``,
  ``Contradiction``, ``FinalSynthesis`` — structured stage models.
- Tunable limits: ``MAX_SUB_QUESTIONS``, ``MAX_CHUNKS_PER_BATCH``,
  ``MAX_CHARS_PER_BATCH``, ``MAX_TOTAL_TASKS``, ``MAX_SEARCH_QUERIES``,
  ``DEFAULT_TOP_K_PER_QUERY``, ``MAX_EVIDENCE_POOL``, ``MAX_EVIDENCE_PER_RESULT``.
"""

from src.agents.research.models import (
    Claim,
    Contradiction,
    FinalSynthesis,
    QuestionAnalysis,
    ResearchConfidence,
    ResearchFinding,
)
from src.agents.research.node import (
    DEFAULT_TOP_K_PER_QUERY,
    MAX_CHARS_PER_BATCH,
    MAX_CHUNKS_PER_BATCH,
    MAX_EVIDENCE_PER_RESULT,
    MAX_EVIDENCE_POOL,
    MAX_SEARCH_QUERIES,
    MAX_SUB_QUESTIONS,
    MAX_TOTAL_TASKS,
    ResearchAgent,
    make_research_node,
)

__all__ = [
    "DEFAULT_TOP_K_PER_QUERY",
    "MAX_CHARS_PER_BATCH",
    "MAX_CHUNKS_PER_BATCH",
    "MAX_EVIDENCE_PER_RESULT",
    "MAX_EVIDENCE_POOL",
    "MAX_SEARCH_QUERIES",
    "MAX_SUB_QUESTIONS",
    "MAX_TOTAL_TASKS",
    "Claim",
    "Contradiction",
    "FinalSynthesis",
    "QuestionAnalysis",
    "ResearchAgent",
    "ResearchConfidence",
    "ResearchFinding",
    "make_research_node",
]
