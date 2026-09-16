# Recursive Language Model (RLM) research capability

This document covers the Research Agent (`src/agents/research/`): what
problem it solves, how the nine-stage recursive pipeline works, why it
qualifies as a Recursive Language Model (RLM) style architecture rather than
repeated flat LLM calls, and its known limitations.

## The objective

A single LLM call has a fixed context window. An enterprise document
collection — incident reports, runbooks, policies, meeting notes — does not.
The naive way to "research" a question across many documents is to retrieve
everything relevant and concatenate it into one giant prompt. That approach
degrades badly (context dilution, lost-in-the-middle effects) and simply
stops working once the relevant evidence no longer fits, regardless of the
model's stated context limit.

The Research Agent instead demonstrates that the system can reason over a
document collection **without ever placing the whole thing in one context
window**: it recursively decomposes a research question into a tree of
bounded sub-tasks, each of which only ever sees a small, targeted slice of
evidence (or, one level up, a handful of already-condensed findings from its
own children) — never the full pool, and never an entire document.

## The nine pipeline stages

| # | Stage | Implementation | LLM call? |
|---|-------|-----------------|-----------|
| 1 | Research question analysis | `ResearchAgent._analyze_question` | Yes — `question_analysis` tool |
| 2 | Search-plan generation | `_build_search_plan` | No — deterministic |
| 3 | Document discovery | `ResearchAgent._discover_documents` | No — `HybridRetriever.search` |
| 4 | Task decomposition | `ResearchAgent.run` (semantic, by sub-question) | No — deterministic |
| 5 | Partitioning of evidence into batches | `_partition_into_batches` | No — deterministic |
| 6 | Recursive / sub-agent analysis | `ResearchAgent._process_task` → `_analyze_batch` | Yes — `research_finding` tool |
| 7 | Intermediate result aggregation | `ResearchAgent._aggregate` | Yes — `research_finding` tool |
| 8 | Contradiction detection | `_detect_contradictions` | No — deterministic |
| 9 | Final evidence synthesis | `ResearchAgent._synthesize_final` | Yes — `final_synthesis` tool |

Stages 1, 6, 7, and 9 call the Anthropic API with a forced tool call
(`tool_choice={"type": "tool", "name": ...}`), mirroring the same pattern
used by the Supervisor and Response agents (`src/agents/supervisor/node.py`,
`src/agents/response/node.py`) — the model can only respond by populating a
Pydantic-validated schema, never free text.

## Two decomposition mechanisms, cleanly separated

**Stage 4 (semantic decomposition)** happens once, at the root: Stage 1
breaks the question into 2–6 focused sub-questions
(`QuestionAnalysis.sub_questions`), and the root task creates one child
`ResearchTask` per sub-question — each initially scoped to the full
discovered evidence pool. This is *what* to investigate.

**Stage 5 (mechanical decomposition)**, inside `_partition_into_batches`,
happens independently for every task, however deep: it splits a task's
assigned evidence into batches bounded by both chunk count
(`MAX_CHUNKS_PER_BATCH`, default 3) and total characters
(`MAX_CHARS_PER_BATCH`, default 3 000). This is *how much* evidence any one
LLM call is allowed to see. It has nothing to do with the question's
semantics — it exists purely to keep every prompt small regardless of how
large the evidence pool is.

## The recursion (stages 5–7)

`ResearchAgent._process_task` is a literal recursive function, not a loop:

```
_process_task(task):
    chunks = evidence assigned to task
    batches = partition(chunks)                 # Stage 5

    if len(batches) <= 1 or depth/task budget reached:
        return analyze(task, batches[0])         # Stage 6 — base case

    for batch in batches:                        # Stage 5, continued
        child = ResearchTask(depth = task.depth + 1, parent = task)
        child.result = _process_task(child)       # <-- recursive call
    return aggregate(task, children)              # Stage 7
```

A task whose evidence fits in one batch is a **leaf**: one sub-agent call
(Stage 6) analyzes it directly and returns a condensed `ResearchFinding`
(a summary plus a small list of structured `Claim`s — never raw text). A
task whose evidence spans multiple batches is an **internal node**: it
spawns one child task per batch, recurses into each — which may itself
recurse further — and then combines its children's *already-condensed*
findings (Stage 7). A parent's aggregation call never sees the evidence
its children saw, only their summaries.

```
root (depth 0, full pool, never given to one LLM call)
 ├─ root-sq1 (depth 1, "What was the root cause?")
 │   ├─ root-sq1-b0 (depth 2, batch of 3 chunks)  ─┐
 │   ├─ root-sq1-b1 (depth 2, batch of 3 chunks)   ├─ aggregated (Stage 7)
 │   └─ root-sq1-b2 (depth 2, batch of 3 chunks)  ─┘
 └─ root-sq2 (depth 1, "What other incidents occurred?")
     └─ single batch → leaf (Stage 6 only, no recursion needed)
```

## Explicit state: `ResearchTask`

Every node in the tree is a `ResearchTask` (`src/models/research.py`):

```python
class ResearchTask(BaseModel):
    task_id: EntityId
    question: NonEmptyStr
    document_ids: tuple[str, ...]   # chunk ids only — never a full document
    status: ResearchStatus          # PENDING -> IN_PROGRESS -> COMPLETED/FAILED
    parent_task_id: EntityId | None
    depth: int
    result: ResearchResult | None
```

`GraphState.research_tasks` holds the *entire* tree for one run (replaced,
not accumulated, each run — see `src/agents/state.py`), so the full
decomposition is inspectable after the fact: which sub-questions were asked,
how each was batched, what depth each task reached, and what each one
concluded. `GraphState.research_results` accumulates every `ResearchResult`
produced along the way (leaves, aggregations, and the final synthesis), not
just the last one.

## Depth/recursion budget — preventing infinite recursion

Two independent, hard bounds stop the recursion, checked in
`_process_task` before a task is allowed to spawn children:

1. **Depth budget** — `task.depth >= ExecutionBudget.max_research_depth`
   (default 3, part of `GraphState.budget`, the same budget object every
   other agent node consumes from). At the limit, a task with more evidence
   than fits in one batch is forced to analyze only its highest-scoring
   batch directly, and the result notes the truncation explicitly rather
   than silently dropping evidence.
2. **Total task cap** — `MAX_TOTAL_TASKS` (default 50), independent of
   depth. Defense in depth: even a shallow tree with a very large evidence
   pool and a small batch size cannot fan out unboundedly. Both the Stage 4
   semantic decomposition loop and the Stage 5/6/7 recursive loop check
   this before creating another task.

Both are unconditional stops — recursion cannot run away regardless of
corpus size, LLM behaviour, or adversarial input, because the check happens
in Python before the next LLM call is even issued, not by hoping the model
decides to stop.

## Contradiction detection (stage 8)

Every leaf and aggregation call also extracts structured `Claim`s —
`(subject, attribute, value, chunk_ids)` — e.g.
`(subject="INC-2024-001", attribute="root_cause", value="certificate sync
misconfiguration")`. These accumulate across the *entire* run, across every
branch of the tree. `_detect_contradictions` groups claims by normalized
`(subject, attribute)`; a group with more than one distinct `value` is a
contradiction, regardless of which sub-question or which batch produced
each claim. This is what lets the agent catch, for example, an early
on-call note and a finalized root-cause analysis disagreeing about the same
incident even though they were examined by two different sub-agent calls
that never saw each other's input.

Detection is deterministic (a grouping pass over already-extracted
structured data, not another LLM call), which keeps it cheap regardless of
how many claims were collected, and testable without mocking an LLM.

## Worked example: payment incident reports

This mirrors `tests/agents/test_research.py::TestPaymentIncidentExample` and
uses the flavour of the real payment-incident data in
`data/knowledge/incidents/` (`INC-2024-001` "Payment Gateway Timeout
Cascade", `INC-2024-002` "SWIFT MT103 Message Duplication").

**Scenario**: an analyst asks *"What caused our recent payment
incidents?"*. The evidence pool includes an early on-call note for
INC-2024-001 ("looks like the connection pool was just too small") and the
finalized incident report's root-cause section ("the ExternalSecret
operator failed to sync the rotated certificate, exhausting the pool") —
two documents that describe the same incident differently because one was
written mid-incident and the other after the postmortem.

1. **Question analysis** splits this into sub-questions, e.g. "What was the
   root cause of INC-2024-001?" and "What other payment incidents occurred
   recently?".
2. **Search plan**: the root question plus each sub-question.
3. **Document discovery** retrieves the early note, the final RCA, and the
   INC-2024-002 summary — a 3-chunk pool.
4. **Task decomposition** creates `root-sq1` and `root-sq2`, each initially
   scoped to all 3 chunks.
5. **Batch partitioning**: 3 chunks fit under the default batch size (3
   chunks / 3 000 characters), so each sub-question task is a single batch
   — no further recursion needed for this small example (a larger pool
   would trigger the recursive case shown in the tree diagram above).
6. **Sub-agent analysis**: the `root-sq1` call extracts a claim
   `(INC-2024-001, root_cause, "connection pool sized too small")` from the
   early note; the `root-sq2` call — which also saw the final RCA chunk —
   extracts `(INC-2024-001, root_cause, "certificate sync
   misconfiguration")` alongside a claim about INC-2024-002.
7. **Aggregation** is skipped for both (single batch = leaf, no children to
   combine).
8. **Contradiction detection** finds two claims sharing
   `(INC-2024-001, root_cause)` with different values and reports:
   *"Conflicting root_cause reported for INC-2024-001: connection pool
   sized too small vs. certificate sync misconfiguration."*
9. **Final synthesis** sees only: the two sub-question summaries and the
   one contradiction description — never the raw incident text — and
   produces an answer that explicitly flags the disagreement rather than
   silently picking one root cause.

The point of the example is what the final synthesis call's prompt
*doesn't* contain: the full text of either incident report. It contains two
short summaries and one contradiction sentence, regardless of whether the
underlying pool had 3 chunks or 300.

## Why this qualifies as RLM-style, not repeated LLM calls

A flat "repeated LLM calls" approach would retrieve documents and then loop
over them, calling the LLM once per document (or once per chunk) with the
*same* accumulating prompt or the *same* fixed instructions, and finally
concatenate all the outputs — there is no tree, no depth, no aggregation
step distinct from the per-item step, and every call is interchangeable.

This implementation differs in the ways that matter:

- **Recursive structure, not a loop.** `_process_task` calls itself. A
  task's own resolution depends on first resolving its children, which may
  themselves recurse — an actual call tree with parent/child relationships
  recorded in `ResearchTask.parent_task_id` and `ResearchTask.depth`, not a
  flat `for` loop over a fixed list.
- **Bounded work at every level, independent of corpus size.** A leaf call
  sees at most `MAX_CHUNKS_PER_BATCH` chunks; an aggregation or synthesis
  call sees only its children's condensed summaries. Doubling the size of
  the evidence pool increases the *width* of the tree (more sibling tasks),
  not the size of any individual prompt — this is the property a "repeated
  calls over a growing context" approach cannot offer.
- **Two distinct reduction operations.** Leaf analysis (Stage 6) and
  aggregation (Stage 7) are different operations over different inputs
  (raw evidence vs. condensed findings), reused recursively at every
  internal level of the tree — this is a genuine reduce-over-a-tree, not a
  single per-item transform repeated N times.
- **Emergent cross-branch analysis.** Contradiction detection (Stage 8)
  operates over facts extracted by *different, independent* branches of the
  recursion and finds relationships between them (two sub-agents that never
  communicated, disagreeing about the same subject) — something a flat
  per-document loop has no mechanism to produce, because it never combines
  outputs from separate calls into a joint analysis.
- **An explicit, inspectable state tree**, not a transient loop variable —
  `GraphState.research_tasks` is the actual decomposition, not a summary of
  one; every task's depth, parent, evidence, and status can be audited
  after the run completes.

## Limitations and known gaps

| Limitation | Mitigation / rationale |
|------------|------------|
| Sibling tasks/batches are processed sequentially (`await`ed one after another), not concurrently | Keeps budget accounting and LLM call ordering simple and deterministic for testing; parallelizing siblings with `asyncio.gather` is a natural follow-up once budget accounting is made concurrency-safe |
| Stage 4's semantic decomposition assigns the *entire* discovered pool to every sub-question task, rather than filtering evidence by relevance per sub-question | Simpler and avoids a second NLP-driven relevance step; Stage 5's batching still bounds any individual call regardless. A future iteration could re-rank the pool per sub-question before batching |
| Contradiction detection is a normalized-string equality check on `(subject, attribute)`, not semantic similarity | Two claims about the same fact phrased very differently (no shared subject string) will not be linked. This is a deliberate, cheap, deterministic first pass — CLAUDE.md's "prefer the more conservative, more auditable, more explicit implementation" — not a general-purpose NLP contradiction detector |
| The Research Agent is not yet wired into `ChatService`'s live LangGraph (`src/services/chat.py`) | Matches this project's established pattern: the Supervisor, Retrieval, and Response agents were each built and tested standalone before a later commit wired them together (see CLAUDE.md "Foundation before features"); wiring the Research Agent in — including the supervisor's existing `route_to: "research"` decision — is the natural next step |
| No mid-run persistence: if the process crashes mid-recursion, the partial task tree is lost | `GraphState` (and therefore `research_tasks`) is only returned once `run()` completes; LangGraph checkpointing (`src/memory/checkpoint.py`) persists *between* graph nodes, not within one node's internal recursion |
| A task truncated by the depth/task budget silently drops the remaining batches' evidence (beyond noting the truncation in its summary) | The alternative — refusing to answer — is worse for a research assistant; the truncation is explicit in both the result text and a `research.truncated` log line, never silent |

## File inventory

| File | Role |
|------|------|
| `src/agents/research/node.py` | `ResearchAgent`, `make_research_node`, all nine pipeline stages |
| `src/agents/research/models.py` | `QuestionAnalysis`, `Claim`, `ResearchFinding`, `Contradiction`, `FinalSynthesis`, `ResearchConfidence` |
| `src/agents/research/prompts.py` | System prompts for stages 1, 6, 7, 9 — all evidence blocks are bounded and XML-delimited as untrusted data |
| `src/models/research.py` | `ResearchTask`, `ResearchResult` — the explicit state |
| `src/agents/budget.py` | `ExecutionBudget.max_research_depth` / `used_research_steps` — the depth budget |
| `src/agents/state.py` | `GraphState.research_tasks` / `research_results` fields |
| `tests/agents/test_research.py` | Full pipeline tests, including the payment-incident worked example |
| `tests/models/test_research.py` | `ResearchTask` / `ResearchResult` model tests |
