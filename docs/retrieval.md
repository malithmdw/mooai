# Retrieval — Pinecone Vector Index

This document covers the Pinecone vector index: what is stored, how to set it
up, how the access-control filter works, and how to run the indexing pipeline.

## Architecture overview

```
data/knowledge/
    ├── incidents/          JSON incident reports
    ├── architecture/       Markdown architecture docs
    ├── runbooks/           Markdown operational runbooks
    ├── products/           Markdown product specs
    ├── policies/           Markdown policy documents
    └── meetings/           Markdown meeting notes
         │
         ▼  scripts/index.py
    [parse + chunk]         src/retrieval/ingestion/
         │
         ▼  OpenAI text-embedding-3-large
    [embed]                 src/retrieval/embedding/
         │
         ▼  Pinecone upsert
    [index]                 src/retrieval/indexing/
         │
         ▼
    Pinecone serverless index  (cosine, 3072-dim)
```

## Pinecone account setup

1. Sign up at [pinecone.io](https://www.pinecone.io) and create a project.
2. Copy your **API key** from the Pinecone console → API Keys.
3. Note the **cloud** and **region** for the project (e.g. `aws / us-east-1`).
4. The index will be created automatically by `scripts/index.py --init`.

## Environment variables

Add these to `.env` (copy `.env.example` first):

```bash
OPENAI_API_KEY=sk-...           # For generating embeddings
PINECONE_API_KEY=pcsk_...       # Pinecone project API key
PINECONE_INDEX_NAME=enterprise-knowledge   # Name of the Pinecone index
PINECONE_NAMESPACE=             # Optional namespace within the index
```

| Variable               | Default                  | Purpose                              |
|------------------------|--------------------------|--------------------------------------|
| `PINECONE_API_KEY`     | _(required)_             | Pinecone authentication              |
| `PINECONE_INDEX_NAME`  | `enterprise-knowledge`   | Index to read/write                  |
| `PINECONE_NAMESPACE`   | `""` (empty)             | Namespace for multi-tenant isolation |

## Index specification

| Parameter   | Value                            |
|-------------|----------------------------------|
| Metric      | cosine                           |
| Dimension   | 3 072 (text-embedding-3-large)   |
| Type        | Serverless (AWS us-east-1)       |

If you use `text-embedding-3-small`, set `EMBEDDING_MODEL=text-embedding-3-small`;
`ensure_index` will automatically use dimension 1 536.

## Creating the index

Run once — before the first full index build:

```bash
python -m scripts.index --init
```

This calls `pc.create_index(...)` with a `ServerlessSpec`.  If the index
already exists, it is a no-op.  You can change the cloud/region by
editing `ensure_index`'s keyword defaults in `src/retrieval/indexing/service.py`.

## Running a full index build

```bash
# Index everything in data/knowledge/
python -m scripts.index

# Custom data directory
python -m scripts.index --data-dir /path/to/docs

# Write to a specific namespace
python -m scripts.index --namespace staging

# Parse and embed but do not write to Pinecone
python -m scripts.index --dry-run

# Suppress all progress output (errors still printed)
python -m scripts.index --quiet
```

The pipeline runs in order:

1. **Discover** — scan `--data-dir` for `.md` and `.json` files.
2. **Parse** — extract frontmatter metadata and body text.
3. **Chunk** — split each document at H2 headings, sub-split on paragraphs.
4. **Embed** — call OpenAI `embeddings.create` in batches of 512.
5. **Upsert** — write vectors to Pinecone in batches of `--batch-size` (default 100).

## Deleting a document

Remove all chunks for a single document without re-indexing:

```bash
python -m scripts.index --delete ARCH-001
```

This calls `index.delete(filter={"document_id": {"$eq": "ARCH-001"}}, namespace=...)`.

## Metadata schema

Every vector stored in Pinecone carries the following metadata fields.  All
are filterable; `allowed_roles` supports the `$in` operator required for RBAC.

| Field           | Type           | Example                               | Purpose                          |
|-----------------|----------------|---------------------------------------|----------------------------------|
| `chunk_id`      | `str`          | `"ARCH-001-chunk-0002"`               | Unique vector ID (also the Pinecone vector ID) |
| `document_id`   | `str`          | `"ARCH-001"`                          | Source document; used for filter-based delete |
| `title`         | `str`          | `"Core Banking Ledger"`               | Display title for search results |
| `section`       | `str`          | `"## 3. Data Flow"`                   | H2 section heading containing this chunk |
| `chunk_index`   | `int`          | `2`                                   | 0-based position within document |
| `chunk_total`   | `int`          | `5`                                   | Total chunks for this document   |
| `text`          | `str`          | `"The ledger records…"`               | Full chunk text (self-contained retrieval) |
| `department`    | `str`          | `"Core Banking"`                      | Owning department                |
| `document_type` | `str`          | `"architecture_document"`             | Document category                |
| `access_level`  | `str`          | `"INTERNAL"`                          | Sensitivity classification       |
| `created_date`  | `str` (ISO)    | `"2024-01-15"`                        | Document creation date           |
| `allowed_roles` | `list[str]`    | `["ENGINEER", "ANALYST"]`             | RBAC roles permitted to see this chunk |

The text is stored in metadata (not in a separate store) so retrieved vectors
are fully self-contained — no secondary database lookup is needed.

## Access-control filtering

Every query against the index **must** include an access filter to enforce
RBAC.  Use `build_access_filter` from `src.retrieval.indexing`:

```python
from src.retrieval.indexing import build_access_filter
from src.models.enums import AccessLevel, Role

# Show only documents accessible to an ENGINEER
filt = build_access_filter([Role.ENGINEER])
# → {"allowed_roles": {"$in": ["ENGINEER"]}}

# Restrict to INTERNAL documents only
filt = build_access_filter(
    [Role.ANALYST],
    access_levels=[AccessLevel.INTERNAL],
)
# → {"allowed_roles": {"$in": ["ANALYST"]}, "access_level": {"$in": ["INTERNAL"]}}
```

Pass the filter to Pinecone's `query` call:

```python
index.query(
    vector=query_vector,
    top_k=10,
    namespace="",
    filter=filt,
    include_metadata=True,
)
```

`allowed_roles` uses `$in` — the user's role must appear **somewhere** in the
document's role list.  A document with `allowed_roles: ["ENGINEER", "ANALYST"]`
is visible to both an `ENGINEER` and an `ANALYST` user.

## Namespaces

Pinecone namespaces provide logical isolation within a single index.  Common
uses:

| Namespace    | Purpose                                          |
|--------------|--------------------------------------------------|
| `""`         | Default (production corpus)                      |
| `"staging"`  | Pre-production documents, for QA                 |
| `"dev"`      | Developer sandbox with a small subset of docs    |

Set `PINECONE_NAMESPACE` in `.env` or pass `--namespace` to `scripts/index.py`.
Queries must specify the same namespace to reach the right partition.

## Source files

| File                                         | Responsibility                                   |
|----------------------------------------------|--------------------------------------------------|
| `src/retrieval/indexing/metadata.py`         | `chunk_to_metadata`, `build_access_filter`       |
| `src/retrieval/indexing/service.py`          | `PineconeIndexService`, `ensure_index`, `make_pinecone_service` |
| `scripts/index.py`                           | CLI: ingest → embed → upsert pipeline            |
| `tests/retrieval/indexing/test_metadata.py`  | Metadata serialisation unit tests                |
| `tests/retrieval/indexing/test_service.py`   | Service unit tests (mocked Pinecone)             |

## Operational notes

- **Re-indexing**: run `scripts/index.py` again at any time.  Pinecone
  upsert is idempotent by vector ID — existing vectors are overwritten, not
  duplicated.
- **Partial updates**: to refresh a single document, delete it first with
  `--delete DOC_ID`, then re-run the ingestion pointing at just that file.
- **Index stats**: call `service.describe_stats()` to get total vector
  count, dimension, and per-namespace counts without querying.
- **Rate limits**: the embedding step uses the `OpenAIEmbeddingProvider`
  which retries on 429s automatically.  For very large corpora, consider
  running the embedding step with `--dry-run` to generate vectors first,
  then upserting from a cached file.

---

# BM25 Sparse Retrieval

BM25 (Okapi BM25) is a classical keyword-frequency ranking function that
complements Pinecone's dense vector search.  The two retrievers have
different strengths; combining them (hybrid retrieval) produces better
recall than either alone.

## Architecture

```
DocumentChunk list
      │
      ▼  src/retrieval/bm25/tokenizer.py
 [tokenize]     lowercase + alphanumeric split, stop-word removal
      │
      ▼  rank_bm25.BM25Okapi (in-process, no network)
 [BM25Corpus]   in-memory index, RBAC-aware search()
      │
      ▼  asyncio.to_thread
 [BM25Service]  async facade — never blocks the event loop
```

The corpus is built once from a ``list[DocumentChunk]`` and is immutable.
Rebuild it whenever documents are re-indexed.

## Usage

```python
from src.retrieval.bm25 import BM25Service
from src.models.enums import Role

# Build once (synchronous BM25Okapi construction)
service = BM25Service.from_chunks(chunks)

# Search asynchronously — runs in a thread pool
results = await service.search(
    "FPS certificate expiry ERR-503",
    top_k=10,
    roles=[Role.ENGINEER],
)

for r in results:
    print(r.rank, r.score, r.chunk.chunk_id, r.chunk.title)
```

## Tokenizer

`src/retrieval/bm25/tokenizer.py` — `tokenize(text) -> list[str]`

- Splits on any non-alphanumeric character.
- Lowercases all tokens.
- Drops tokens shorter than two characters (single letters, lone digits).
- Removes common English stop words.
- **Preserves technical identifiers**: `FPS-ERR-429` → `["fps", "err",
  "429"]`; `HMAC-SHA256` → `["hmac", "sha256"]`.

The same function is used for both corpus indexing and query tokenization
so the vocabulary is always consistent.

## Metadata and RBAC filtering

`BM25Corpus.search` accepts the same filter parameters as Pinecone's
`build_access_filter`:

| Parameter       | Type                      | Effect                                   |
|-----------------|---------------------------|------------------------------------------|
| `roles`         | `Sequence[Role] \| None`  | Keep chunks whose `allowed_roles` overlap |
| `access_levels` | `Sequence[AccessLevel] \| None` | Keep chunks at these sensitivity levels |
| `department`    | `str \| None`             | Exact match on `department`              |
| `document_type` | `str \| None`             | Exact match on `document_type`           |

Chunks that fail any predicate are excluded before ranking — they never
appear in results regardless of their BM25 score.

## Source files

| File                                             | Responsibility                                  |
|--------------------------------------------------|-------------------------------------------------|
| `src/retrieval/bm25/tokenizer.py`                | `tokenize` — shared tokenizer                   |
| `src/retrieval/bm25/corpus.py`                   | `BM25Corpus`, `BM25Result`                      |
| `src/retrieval/bm25/service.py`                  | `BM25Service` — async facade                    |
| `tests/retrieval/bm25/test_tokenizer.py`         | Tokenizer unit tests                            |
| `tests/retrieval/bm25/test_corpus.py`            | Corpus unit tests (incl. BM25 advantage cases)  |
| `tests/retrieval/bm25/test_service.py`           | Service unit tests                              |

## BM25 vs dense retrieval — trade-offs

Hybrid retrieval combines both signals because each retriever has distinct
failure modes.

| Scenario                        | BM25 (sparse)                          | Dense (Pinecone)                          |
|---------------------------------|----------------------------------------|-------------------------------------------|
| Exact error codes (`ERR-429`)   | ✅ Token match; high precision          | ❌ May embed to semantically adjacent doc  |
| Rare terms (`idempotency`)      | ✅ High IDF → high rank for rare tokens | ⚠ Under-weighted if rare in training data |
| Numeric identifiers (`20240115`)| ✅ Exact token match                   | ❌ Embedding may not distinguish numbers   |
| Algorithm names (`HMAC-SHA256`) | ✅ Exact token match on both sub-tokens | ⚠ Depends on pre-training vocabulary      |
| Full query-term coverage        | ✅ All terms must appear for high score | ⚠ Score is cosine of aggregate embedding  |
| Semantic paraphrasing           | ❌ Vocabulary mismatch → zero score    | ✅ Synonyms and related concepts matched   |
| Spelling variation              | ❌ Must match exact tokens              | ✅ Embedding is robust to minor variants   |
| Cross-lingual queries           | ❌ Language-specific stop words         | ✅ Multilingual embeddings available       |
| Zero-shot concepts              | ❌ OOV terms never match               | ✅ Contextual embedding generalises        |
| No network / no API cost        | ✅ Fully local, instant                | ❌ Requires embedding API call             |
| Deterministic results           | ✅ Same query → same ranking always    | ⚠ Model updates can shift rankings        |
| Explainability                  | ✅ Score = weighted token frequencies  | ❌ Cosine similarity is opaque             |

### Recommended query routing

- Use **BM25 first** when the query contains: error codes, document IDs,
  algorithm names, version strings, or any verbatim identifier.
- Use **dense first** when the query is a natural-language question or
  contains likely synonyms/paraphrases.
- For production: run both in parallel and merge results (Reciprocal Rank
  Fusion or a learned ranker).

### BM25 parameters

| Parameter | Default | Effect                                              |
|-----------|---------|-----------------------------------------------------|
| `k1`      | 1.5     | TF saturation — higher = TF keeps contributing longer |
| `b`       | 0.75    | Length normalisation — 1.0 = full normalisation      |

For short banking documents (chunks ≤ 1 500 chars) the defaults are
generally appropriate.  Decrease `b` if document lengths vary widely.
