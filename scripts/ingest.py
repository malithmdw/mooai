"""Document ingestion CLI.

Usage
-----
    python -m scripts.ingest [OPTIONS]

Options
-------
    --data-dir PATH     Root directory to scan (default: data/knowledge)
    --max-chars INT     Maximum characters per chunk (default: 1500)
    --output PATH       Write JSONL output to PATH instead of stdout
    --dry-run           Discover and parse only; do not chunk or write output
    --quiet             Suppress progress output (errors still go to stderr)

Exit codes: 0 = success, 1 = no output produced or data dir not found.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m scripts.ingest",
        description="Discover, parse, and chunk enterprise knowledge documents.",
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
        "--output",
        type=Path,
        default=None,
        metavar="PATH",
        help="Write JSONL output to PATH (default: stdout)",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Discover and parse only; do not chunk or write output",
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


def main(argv: list[str] | None = None) -> int:
    """Run the ingestion pipeline and return an exit code."""
    # Provide dummy DSN values so `Settings` validation passes before any
    # real infrastructure is required.  The ingestion CLI never connects to
    # Postgres or Redis.
    os.environ.setdefault("POSTGRES_URL", "postgresql+asyncpg://unused:unused@localhost/unused")
    os.environ.setdefault("REDIS_URL", "redis://localhost:6379/0")

    args = _build_parser().parse_args(argv)
    data_dir: Path = args.data_dir
    max_chars: int = args.max_chars
    output: Path | None = args.output
    dry_run: bool = args.dry_run
    quiet: bool = args.quiet

    # Late imports: keeps import-time side-effects after env defaults above.
    from src.retrieval.ingestion.pipeline import discover, ingest

    if not data_dir.exists():
        print(f"error: data directory not found: {data_dir}", file=sys.stderr)
        return 1

    if dry_run:
        paths = discover(data_dir)
        _log(f"Discovered {len(paths)} file(s) in {data_dir}", quiet=quiet)
        for p in paths:
            _log(f"  {p.relative_to(data_dir)}", quiet=quiet)
        return 0

    _log(f"Ingesting from {data_dir} (max_chars={max_chars})", quiet=quiet)

    chunks = ingest(data_dir, max_chars=max_chars)

    if not chunks:
        print("warning: no chunks produced — check parse errors above", file=sys.stderr)
        return 1

    doc_ids = {c.document_id for c in chunks}
    _log(f"Documents: {len(doc_ids)}  Chunks: {len(chunks)}", quiet=quiet)

    lines = [
        json.dumps(
            {
                "chunk_id": c.chunk_id,
                "document_id": c.document_id,
                "title": c.title,
                "section": c.section,
                "chunk_index": c.chunk_index,
                "chunk_total": c.chunk_total,
                "text": c.text,
                "metadata": c.metadata.model_dump(mode="json"),
            },
            ensure_ascii=False,
        )
        for c in chunks
    ]

    if output:
        output.write_text("\n".join(lines) + "\n", encoding="utf-8")
        _log(f"Written to {output}", quiet=quiet)
    else:
        sys.stdout.write("\n".join(lines) + "\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
