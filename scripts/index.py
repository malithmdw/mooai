"""Pinecone indexing CLI.

Runs the full pipeline: discover → parse → chunk → embed → upsert.

Usage
-----
    python -m scripts.index [OPTIONS]

Options
-------
    --data-dir PATH     Root directory to scan (default: data/knowledge)
    --max-chars INT     Maximum characters per chunk (default: 1500)
    --namespace TEXT    Pinecone namespace (default: $PINECONE_NAMESPACE or "")
    --batch-size INT    Vectors per upsert batch (default: 100)
    --init              Create the Pinecone index if it does not exist, then exit
    --dry-run           Parse, chunk, embed — but do not write to Pinecone
    --delete DOC_ID     Delete all chunks for DOC_ID from the index, then exit
    --quiet             Suppress progress output (errors still go to stderr)

Exit codes: 0 = success, 1 = error.

Required environment variables
-------------------------------
    OPENAI_API_KEY      For embedding
    PINECONE_API_KEY    For indexing
    PINECONE_INDEX_NAME Index to write to (default: enterprise-knowledge)
    PINECONE_NAMESPACE  Namespace within the index (default: "")
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from pathlib import Path


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m scripts.index",
        description="Discover, parse, chunk, embed, and index enterprise knowledge documents.",
    )
    p.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/knowledge"),
        metavar="PATH",
        help="Root directory to scan (default: data/knowledge)",
    )
    p.add_argument(
        "--max-chars",
        type=int,
        default=1500,
        metavar="INT",
        help="Maximum characters per chunk (default: 1500)",
    )
    p.add_argument(
        "--namespace",
        type=str,
        default=None,
        metavar="TEXT",
        help="Pinecone namespace (default: PINECONE_NAMESPACE env var or empty string)",
    )
    p.add_argument(
        "--batch-size",
        type=int,
        default=100,
        metavar="INT",
        help="Vectors per Pinecone upsert batch (default: 100)",
    )
    p.add_argument(
        "--init",
        action="store_true",
        help="Create the Pinecone index if it does not exist, then exit",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Parse, chunk, and embed — but do not write to Pinecone",
    )
    p.add_argument(
        "--delete",
        metavar="DOC_ID",
        default=None,
        help="Delete all chunks for DOC_ID from the index, then exit",
    )
    p.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress progress output (errors still go to stderr)",
    )
    return p


def _log(msg: str, *, quiet: bool) -> None:
    if not quiet:
        print(msg, file=sys.stderr)


async def _async_main(args: argparse.Namespace) -> int:
    quiet: bool = args.quiet

    # Late imports keep Settings import-time side-effects after env defaults.
    from src.core.config import Settings
    from src.retrieval.embedding.openai_provider import make_openai_provider
    from src.retrieval.indexing.service import ensure_index, make_pinecone_service
    from src.retrieval.ingestion.pipeline import ingest

    # Namespace CLI override: re-read Settings after updating the env var.
    if args.namespace is not None:
        os.environ["PINECONE_NAMESPACE"] = args.namespace

    settings = Settings(_env_file=None)

    # --init: create the index and exit.
    if args.init:
        _log("Initialising Pinecone index…", quiet=quiet)
        await ensure_index(settings)
        _log("Done.", quiet=quiet)
        return 0

    # --delete DOC_ID: remove a document and exit.
    if args.delete:
        doc_id: str = args.delete
        _log(f"Deleting document {doc_id!r} from index…", quiet=quiet)
        svc = make_pinecone_service(settings)
        await svc.delete_document(doc_id)
        _log("Done.", quiet=quiet)
        return 0

    data_dir: Path = args.data_dir
    max_chars: int = args.max_chars
    batch_size: int = args.batch_size

    if not data_dir.exists():
        print(f"error: data directory not found: {data_dir}", file=sys.stderr)
        return 1

    # --- Ingest ---
    _log(f"Ingesting from {data_dir} (max_chars={max_chars})", quiet=quiet)
    chunks = ingest(data_dir, max_chars=max_chars)
    if not chunks:
        print("warning: no chunks produced — check parse errors above", file=sys.stderr)
        return 1
    doc_ids = {c.document_id for c in chunks}
    _log(f"Ingested: {len(doc_ids)} documents, {len(chunks)} chunks", quiet=quiet)

    # --- Embed ---
    embedder = make_openai_provider(settings)
    _log(f"Embedding {len(chunks)} chunks with {embedder.model}…", quiet=quiet)
    vectors = await embedder.embed([c.text for c in chunks])
    _log(f"Embedded {len(vectors)} vectors", quiet=quiet)

    if args.dry_run:
        _log("Dry run — skipping Pinecone upsert.", quiet=quiet)
        return 0

    # --- Upsert ---
    service = make_pinecone_service(settings)
    _log(
        f"Upserting to namespace={service.namespace!r} (batch_size={batch_size})…",
        quiet=quiet,
    )
    result = await service.upsert_chunks(chunks, vectors, batch_size=batch_size)
    _log(
        f"Indexed: {result.upserted_count} vectors in {result.batch_count} batch(es)",
        quiet=quiet,
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    """Run the indexing pipeline and return an exit code."""
    # Provide dummy DSN values so Settings validation passes.  The indexing
    # CLI never connects to Postgres or Redis.
    os.environ.setdefault("POSTGRES_URL", "postgresql+asyncpg://unused:unused@localhost/unused")
    os.environ.setdefault("REDIS_URL", "redis://localhost:6379/0")

    args = _build_parser().parse_args(argv)
    return asyncio.run(_async_main(args))


if __name__ == "__main__":
    sys.exit(main())
