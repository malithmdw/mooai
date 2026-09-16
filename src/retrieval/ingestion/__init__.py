"""Document ingestion pipeline: discover → parse → chunk.

Entry point: `src.retrieval.ingestion.pipeline.ingest`.
CLI: `python -m scripts.ingest`.

No embedding or indexing is performed here — chunked `DocumentChunk`
objects are the output boundary. Downstream indexing picks them up.
"""
