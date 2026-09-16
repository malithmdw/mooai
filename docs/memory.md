# Conversational Memory

The memory module manages how the assistant stores, retrieves, and compresses
conversation history so that agents receive the right amount of context on
every turn — never the full unbounded history.

---

## Overview

```
User message
     │
     ▼
 MemoryService.get_window()
     │
     ├── ConversationStore.load_conversation()   ← PostgreSQL
     │         returns all messages
     │
     ├── ConversationStore.load_latest_summary() ← PostgreSQL
     │         returns compressed older context
     │
     └── build_window()                          ← pure function
               │
               ▼
         MemoryWindow
           ├── recent_messages (last 10, verbatim)
           └── summary (compressed older turns, or None)
```

The `MemoryWindow` is injected into the LLM prompt instead of the raw
conversation history.

---

## Short-term memory

Short-term memory is the **in-process conversation state** that exists for the
duration of a single request:

| Component | Contents |
|-----------|----------|
| `MemoryWindow.recent_messages` | The last `RECENT_WINDOW_SIZE` (default: 10) messages, verbatim |
| `MemoryWindow.summary` | A `ConversationSummary` covering older turns, or `None` |
| `MemoryWindow.user_context` | User identity (`user_id`, `roles`, `display_name`) |
| `MemoryWindow.total_message_count` | Total historical messages for this conversation |

Short-term memory is rebuilt from the database at the start of each request
via `MemoryService.get_window()`.  It is not cached between requests.

---

## Persistent memory

All conversation state is persisted in **PostgreSQL** through three tables:

### `conversations`
One row per conversation thread.

| Column | Type | Notes |
|--------|------|-------|
| `conversation_id` | `VARCHAR(128)` PK | Unique conversation identifier |
| `user_id` | `VARCHAR(128)` | Owner |
| `title` | `TEXT` | Optional human-readable title |
| `created_at` | `TIMESTAMPTZ` | Creation timestamp |

### `messages`
One row per message turn.

| Column | Type | Notes |
|--------|------|-------|
| `message_id` | `VARCHAR(128)` PK | Unique message identifier |
| `conversation_id` | `VARCHAR(128)` FK | Parent conversation |
| `role` | `VARCHAR(16)` | `user` / `assistant` / `system` |
| `content` | `TEXT` | Message body |
| `created_at` | `TIMESTAMPTZ` | Insertion timestamp (used for ordering) |

### `conversation_summaries`
One row per generated summary.

| Column | Type | Notes |
|--------|------|-------|
| `summary_id` | `VARCHAR(128)` PK | Unique summary identifier |
| `conversation_id` | `VARCHAR(128)` FK | Parent conversation |
| `content` | `TEXT` | LLM-generated (or plain-text) summary |
| `message_count` | `INTEGER` | Number of messages compressed by this summary |
| `created_at` | `TIMESTAMPTZ` | Generation timestamp |

`ConversationStore.load_latest_summary()` returns the most recently created
summary for a conversation so the newest compression is always used.

### LangGraph checkpoint tables

Two additional tables store LangGraph agent graph state:

| Table | Purpose |
|-------|---------|
| `langgraph_checkpoints` | One row per graph checkpoint (serialised state) |
| `langgraph_checkpoint_writes` | Pending channel writes for interrupted graphs |

These are managed by `PostgresCheckpointSaver` and passed to the agent graph
via `graph.compile(checkpointer=saver)`.

---

## Context-window control

Sending the full conversation history to the LLM on every turn is expensive
and can exceed the model's context limit.  Two constants control the budget:

| Constant | Default | Meaning |
|----------|---------|---------|
| `RECENT_WINDOW_SIZE` | 10 | Maximum verbatim messages per turn |
| `SUMMARISE_THRESHOLD` | 20 | Total messages before summarisation is triggered |

### How it works

1. If `len(conversation.messages) <= SUMMARISE_THRESHOLD`: all messages fit
   in `recent_messages` (up to 10) with no summary needed.
2. If `len(conversation.messages) > SUMMARISE_THRESHOLD`:
   - `should_summarise()` returns `True`.
   - `MemoryService.maybe_summarise()` invokes the `Summariser` on the
     messages **outside** the recent window.
   - The resulting `ConversationSummary` is persisted and returned with
     subsequent `MemoryWindow` objects.
3. On each request, `build_window()` always provides at most
   `RECENT_WINDOW_SIZE` verbatim messages regardless of total history length.

### Summariser

The `Summariser` protocol has a single method:

```python
async def summarise(
    self,
    conversation_id: str,
    messages: Sequence[Message],
) -> ConversationSummary: ...
```

The production implementation should call the LLM to generate a coherent
prose summary.  The default `NullSummariser` concatenates message content
without LLM involvement and is intended for testing and local development.

---

## LangGraph checkpointing

`PostgresCheckpointSaver` implements `BaseCheckpointSaver` from LangGraph.
Wire it into a compiled graph:

```python
saver = make_memory_saver(settings.postgres_url)
await saver.create_tables()

graph = my_graph.compile(checkpointer=saver)

# Each invocation uses thread_id to isolate conversation state
result = await graph.ainvoke(
    {"messages": [...]},
    config={"configurable": {"thread_id": conversation_id}},
)
```

> **Only async methods are implemented.**  The agent graph must use
> `ainvoke` / `astream`; calling `invoke` or `stream` will raise
> `NotImplementedError`.

---

## Privacy considerations

- **No secrets in summaries.** The `Summariser` receives full message
  content.  When using an LLM-backed summariser, the content is sent to an
  external provider.  Ensure this is acceptable under your data classification
  policy before enabling it for CONFIDENTIAL or RESTRICTED conversations.
- **No API keys in messages.** The `ConversationStore` stores `content` as
  plain text.  Callers must not allow API keys, passwords, or session tokens
  to appear in message content.
- **User separation.** Each conversation row stores the `user_id`.
  Application-layer RBAC must verify ownership before calling
  `load_conversation`.  The store itself does not enforce access control.
- **Checkpoint serialisation.** LangGraph checkpoints are stored as JSON in
  the `langgraph_checkpoints` table.  Agent state that contains sensitive
  values (e.g., API responses with PII) will appear in the database.

---

## Failure behaviour

| Failure | Behaviour |
|---------|-----------|
| Database unavailable at request time | SQLAlchemy raises `OperationalError`; the caller (API handler) should catch and return HTTP 503 |
| `load_conversation` returns `None` | `MemoryService.get_window()` returns an empty `MemoryWindow` — the agent starts a fresh context |
| `Summariser.summarise` raises | `maybe_summarise` does not catch; the caller should handle to avoid blocking message saves |
| `aput` fails mid-write | The transaction is rolled back; the previous checkpoint is still valid |
| No checkpoint found | `aget_tuple` returns `None`; LangGraph starts a fresh graph execution |

Tables are created with `CREATE TABLE IF NOT EXISTS` via `create_tables()`.
This must be called once at application startup before any store or
checkpoint operations.
