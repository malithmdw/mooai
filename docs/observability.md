# Observability

This document explains the LangSmith tracing integration implemented in
`src/observability/tracing.py`: what gets traced, what deliberately
doesn't, how sensitive information is protected, and how to inspect a
trace as an evaluator. It complements — and does not replace — the
logging behavior documented in `src/core/logging.py`'s module docstring
and CLAUDE.md "Logging requirements".

## Design summary

- **Configurable, not hardcoded.** Tracing is entirely driven by
  `Settings.langsmith_tracing`, `Settings.langsmith_api_key`, and
  `Settings.langsmith_project` (`src/core/config.py`) — the same
  environment-variable-backed configuration as everything else in the
  application. Nothing in `src/observability` reads `os.environ` directly.
- **Decoupled from business logic.** Agents, tools, retrieval, and
  validation code call small helper functions
  (`trace_conversation`, `trace_agent_node`, `trace_tool_call`,
  `trace_retrieval`, `trace_validation`) exported from
  `src.observability`. None of that calling code imports the `langsmith`
  SDK, constructs a client, or knows how a run is represented on the wire
  — `src/observability/tracing.py` is the only module in the codebase
  that does.
- **Graceful when disabled or unavailable.** If `LANGSMITH_API_KEY` is
  absent or `LANGSMITH_TRACING=false`, every helper is a true no-op — it
  never imports the LangSmith SDK, never constructs a client, and never
  makes a network call. If LangSmith is configured but unreachable (or
  the SDK call otherwise fails), that failure is logged as a `WARNING`
  and swallowed, not raised. In both cases, **the wrapped code always
  runs exactly as if it weren't being traced at all** — its own
  exceptions still propagate normally, and its return value/behavior is
  unaffected by whether a trace was recorded.

## What is traced

Each helper below opens one LangSmith run for the duration of its `with`
block, closes it when the block exits, and records whether it succeeded
or raised:

| Helper | Run type | Covers |
|---|---|---|
| `trace_conversation(conversation_id, user_id=...)` | `chain` | One end-to-end conversation turn — the top-level run other traces nest under once agent execution exists. |
| `trace_agent_node(node_name, conversation_id=...)` | `chain` | One LangGraph node's execution (e.g. `supervisor`, `retrieval`, `research`, `tool_execution`, `memory`, `validation`, `response` — see `src.models.enums.AgentState`). |
| `trace_tool_call(tool_name, conversation_id=...)` | `tool` | One external tool invocation made via MCP. |
| `trace_retrieval(query_length=..., conversation_id=...)` | `retriever` | One hybrid-search (Pinecone + BM25) retrieval operation. |
| `trace_validation(validation_type, conversation_id=...)` | `chain` | One output-validation check (see CLAUDE.md "LLM output must be validated before use"). |

Every run records: a name, a LangSmith `run_type`, the configured
`langsmith_project`, whether it succeeded, and — on failure — the
exception message. Metadata attached to a run is limited to
**identifiers, names, and counts**: e.g. `conversation_id`, `node_name`,
`tool_name`, `query_length`. This is deliberately the same shape of
information CLAUDE.md's logging rules already consider safe to log (see
"Logging requirements": *"Log identifiers ... not the sensitive payloads
themselves"*) — tracing follows the identical policy.

## What is intentionally not logged

None of the tracing helpers accept, and therefore never transmit to
LangSmith:

- **Raw user messages or assistant responses.** `trace_conversation`
  takes a `conversation_id`, not the conversation's content.
- **Retrieved document content.** `trace_retrieval` takes `query_length`
  (an integer), never the query text itself, and never document content
  — only `src.retrieval` code (once implemented) decides what, if
  anything, gets surfaced about a retrieval result, and it must not pass
  raw content into these helpers.
- **Tool arguments or tool output.** `trace_tool_call` takes a
  `tool_name`, not the arguments passed to the tool or what it returned
  — those may contain confidential document content or PII pulled in via
  a tool.
- **Secrets of any kind.** API keys, passwords, and authorization tokens
  are never accepted as trace metadata by design (there is no parameter
  for them), the same way `src.core.logging` never accepts them.

This is enforced at the API level, not just by convention: the public
helpers' signatures simply have no parameter through which message text,
document content, or tool payloads could flow. A future caller would have
to deliberately misuse the low-level `_traced_run(name, run_type,
**metadata)` primitive to attach something like `document_text=...` — see
the next section for what happens if they do.

## How sensitive information is protected

Defense in depth, matching CLAUDE.md "Security principles":

1. **By construction (primary defense).** As above, the five public
   helpers' parameters don't accept message/document/tool-payload content
   in the first place.
2. **By redaction (safety net).** Whatever metadata *does* reach a run is
   still passed through `src.core.redaction.sanitize_fields` — the exact
   same function `src.core.logging` uses for structured log fields, so
   the two systems can never drift apart on what counts as sensitive:
   - Any field whose key contains `api_key`, `password`, `token`,
     `authorization`, `secret`, or `credential` is replaced with
     `***REDACTED***`.
   - Any string field whose key suggests document content (`content`,
     `excerpt`, `body`, `document_text`, `raw_text`) is truncated to a
     200-character preview if longer, never sent in full.
3. **By failure isolation.** A LangSmith outage, misconfiguration, or SDK
   bug cannot expose secrets by crashing into a stack trace visible
   elsewhere, and cannot take the application down either — see
   "Graceful when disabled or unavailable" above. Errors talking to
   LangSmith are caught, logged (via `src.core.logging`, which applies
   the same redaction), and swallowed.

## How an evaluator can inspect traces

1. **Enable tracing.** Set `LANGSMITH_TRACING=true` and a real
   `LANGSMITH_API_KEY` in `.env` (copy `.env.example` first — see
   README.md "Getting started"). `LANGSMITH_PROJECT` controls which
   LangSmith project runs are grouped under; it defaults to
   `enterprise-ai-assistant`.
2. **Run the application.** `configure_tracing()` runs once at startup
   (wired into `src.api.main.create_app`) and logs whether tracing ended
   up enabled — check the structured startup logs
   (`event_type: "observability.tracing_configured"`) to confirm before
   assuming traces are being recorded.
3. **Open the LangSmith project** in the LangSmith UI (or via its API)
   under the configured project name. Each conversation turn appears as a
   top-level `conversation` run; once agent execution exists, agent node,
   tool, retrieval, and validation runs will nest underneath it in
   execution order, giving a full timeline of one request.
4. **Correlate with structured logs.** Every run's `conversation_id` (and,
   where applicable, `user_id`) matches the `conversation_id`/`user_id`
   fields on the corresponding JSON log lines emitted by
   `src.core.logging` (see that module's docstring) — join on those
   fields to go from "what LangSmith shows happened" to "what the
   application logged about it," or vice versa.
5. **Expect gaps, not errors, when unconfigured.** If `LANGSMITH_API_KEY`
   is absent, no runs will appear in LangSmith at all, but the
   application logs and behaves identically — this is the intended
   "graceful" behavior, not a bug to chase.

## Current scope

This is the observability *foundation*: the tracing primitives, their
safety guarantees, and their tests exist now, ahead of the agent graph
they will eventually wrap (per CLAUDE.md "Foundation before features").
No agent, tool, retrieval, or validation code exists yet to call these
helpers in production — they are exercised today only by
`tests/observability/test_tracing.py`, against a mocked LangSmith client.
