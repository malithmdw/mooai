"""Tools: external tool integrations exposed to agents (via MCP).

Every tool call is a trust boundary: tools must enforce the same
authorization as the rest of the application (see
`src.security` and CLAUDE.md "Agents must never bypass application
authorization") — an agent's intent to call a tool is not itself
authorization to do so. `tools` may depend on `core`, `models`, `security`,
and `observability` only.
"""

from src.tools.analysis import (
    ANALYSIS_TIMEOUT_SECONDS,
    MAX_RECORDS,
    AnalysisError,
    AnalysisOperation,
    AnalysisRequest,
    AnalysisResult,
    run_analysis,
)

__all__ = [
    "ANALYSIS_TIMEOUT_SECONDS",
    "MAX_RECORDS",
    "AnalysisError",
    "AnalysisOperation",
    "AnalysisRequest",
    "AnalysisResult",
    "run_analysis",
]
