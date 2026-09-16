"""BM25 sparse retrieval.

Public surface
--------------
- ``BM25Corpus`` — in-memory BM25 index with RBAC-aware ``search()``.
- ``BM25Result`` — typed result containing the matched chunk, BM25 score,
  and 1-based rank.
- ``BM25Service`` — async facade over ``BM25Corpus`` (uses
  ``asyncio.to_thread``).
- ``tokenize`` — shared tokenizer used for both corpus indexing and query
  tokenization.

Design notes
------------
The corpus is built once from a list of ``DocumentChunk`` objects and is
immutable.  To reflect document changes, rebuild a new corpus from the
updated chunk list.  This is intentional — it keeps the implementation
stateless and trivially testable without any mock infrastructure.
"""

from src.retrieval.bm25.corpus import BM25Corpus, BM25Result
from src.retrieval.bm25.service import BM25Service
from src.retrieval.bm25.tokenizer import tokenize

__all__ = [
    "BM25Corpus",
    "BM25Result",
    "BM25Service",
    "tokenize",
]
