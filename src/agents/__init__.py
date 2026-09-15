"""Agents: LangGraph-based agent graphs (RLM-style recursive reasoning).

`agents` is the top of the dependency stack: it may depend on `core`,
`models`, `retrieval`, `memory`, `tools`, `security`, and `observability`.
No other `src` package may import from `agents`. Agents must never bypass
`security` to reach a tool or data source directly — see CLAUDE.md.

No agent graphs are implemented yet; this package currently defines the
architectural seam only.
"""
