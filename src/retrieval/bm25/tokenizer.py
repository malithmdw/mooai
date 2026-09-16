"""Text tokenizer for BM25 keyword search.

The same ``tokenize`` function is used for both corpus indexing and query
tokenization so the vocabulary is always consistent.  The tokenizer is
intentionally simple: lowercase-alphanumeric splitting with stop word
removal.  Technical identifiers — error codes, algorithm names, version
numbers with three or more characters — survive unchanged because they
are never in the stop list and always meet the minimum-length threshold.
"""

from __future__ import annotations

import re

_TOKEN_RE = re.compile(r"[a-z0-9]+")

_STOP_WORDS: frozenset[str] = frozenset(
    {
        "a",
        "an",
        "the",
        "and",
        "or",
        "but",
        "if",
        "in",
        "on",
        "at",
        "to",
        "for",
        "of",
        "with",
        "by",
        "from",
        "as",
        "is",
        "was",
        "are",
        "were",
        "be",
        "been",
        "being",
        "have",
        "has",
        "had",
        "do",
        "does",
        "did",
        "will",
        "would",
        "could",
        "should",
        "may",
        "might",
        "shall",
        "can",
        "not",
        "no",
        "nor",
        "so",
        "yet",
        "both",
        "either",
        "neither",
        "this",
        "that",
        "these",
        "those",
        "it",
        "its",
        "he",
        "she",
        "they",
        "we",
        "you",
        "me",
        "him",
        "her",
        "us",
        "them",
        "who",
        "which",
        "what",
        "when",
        "where",
        "how",
        "all",
        "each",
        "every",
        "any",
        "some",
        "few",
        "more",
        "most",
        "other",
        "into",
        "through",
        "during",
        "than",
        "then",
        "there",
        "here",
        "about",
        "above",
        "after",
        "before",
        "between",
        "up",
        "down",
        "out",
        "off",
        "over",
        "under",
        "again",
        "while",
        "also",
        "just",
        "only",
        "own",
        "same",
        "too",
        "very",
        "such",
        "per",
        "re",
        "eg",
        "ie",
    }
)


def tokenize(text: str) -> list[str]:
    """Tokenize *text* into lowercase alphanumeric tokens, removing stop words.

    - Splits on any non-alphanumeric character (hyphens, dots, underscores,
      spaces, punctuation).
    - Lowercases everything so queries are case-insensitive.
    - Drops tokens shorter than two characters (single letters, lone digits).
    - Removes common English stop words.
    - Preserves technical identifiers: ``FPS-ERR-429`` → ``["fps", "err",
      "429"]``; ``HMAC-SHA256`` → ``["hmac", "sha256"]``.

    The function is deterministic: the same input always produces the same
    output list (including repeated tokens, which BM25 needs for TF scoring).
    """
    tokens = _TOKEN_RE.findall(text.lower())
    return [t for t in tokens if len(t) >= 2 and t not in _STOP_WORDS]
