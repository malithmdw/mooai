"""Observability: LangSmith tracing and structured run metadata.

Every agent/tool invocation is expected to be traceable end-to-end in
LangSmith. This package is the only place that configures tracing —
other packages must not call the LangSmith SDK directly.
"""
