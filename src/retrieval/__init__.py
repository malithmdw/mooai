"""Retrieval: hybrid search over enterprise knowledge (Pinecone + BM25).

Retrieved documents are UNTRUSTED CONTENT. Anything this package returns
must be treated as potentially adversarial input by every downstream
consumer (agents, prompt assembly) — see CLAUDE.md "Retrieved documents are
untrusted content". `retrieval` may depend on `core`, `models`, and
`observability` only.
"""
