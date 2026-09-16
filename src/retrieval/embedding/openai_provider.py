"""OpenAI implementation of the `EmbeddingProvider` protocol.

Features
--------
- **Batching**: splits large input lists into batches (default 512 texts)
  to stay within OpenAI's per-request limit.
- **Retry with exponential backoff**: automatically retries transient
  failures (`TimeoutError`, rate-limit 429s, 5xx server errors, connection
  errors) up to `max_retries` times with capped binary-exponential delay.
  Non-retryable 4xx errors surface immediately.
- **Timeout**: each individual API call is wrapped in `asyncio.wait_for`
  so a hung request cannot block the caller indefinitely.
- **Structured logging**: every significant event (batch start/complete,
  retry, exhausted retries) is emitted through the project's logging
  helpers — API keys never appear in log output.
- **Dependency injection**: the `AsyncOpenAI` client is passed in via the
  constructor, so tests can inject an `AsyncMock` without patching globals.

Usage
-----
    # Production
    from src.retrieval.embedding import make_openai_provider
    provider = make_openai_provider()          # reads Settings

    # Tests
    from src.retrieval.embedding import OpenAIEmbeddingProvider
    provider = OpenAIEmbeddingProvider(client=mock_client)
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Sequence
from typing import Final

from openai import APIConnectionError, APIStatusError, AsyncOpenAI, RateLimitError

from src.core.config import Settings, get_settings
from src.core.logging import get_logger, log_debug, log_error, log_info, log_warning

_logger = get_logger(__name__)

_BASE_DELAY_SECONDS: Final[float] = 1.0
_MAX_DELAY_SECONDS: Final[float] = 60.0

DEFAULT_BATCH_SIZE: Final[int] = 512
DEFAULT_MAX_RETRIES: Final[int] = 3
DEFAULT_TIMEOUT_SECONDS: Final[float] = 30.0


class OpenAIEmbeddingProvider:
    """Batched, retry-capable embedding provider backed by the OpenAI API.

    Instantiate via `make_openai_provider()` in production, or directly
    with an injected `AsyncOpenAI` client in tests.

    This class satisfies `EmbeddingProvider` structurally — no inheritance
    from the protocol is required or used.
    """

    def __init__(
        self,
        client: AsyncOpenAI,
        *,
        model: str = "text-embedding-3-large",
        batch_size: int = DEFAULT_BATCH_SIZE,
        max_retries: int = DEFAULT_MAX_RETRIES,
        timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
    ) -> None:
        self._client = client
        self._model = model
        self._batch_size = batch_size
        self._max_retries = max_retries
        self._timeout_seconds = timeout_seconds

    @property
    def model(self) -> str:
        return self._model

    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed *texts* in batches, returning vectors in input order.

        An empty sequence returns an empty list without making any API call.
        Large inputs are split into batches of at most `batch_size` texts and
        sent as separate requests; results are re-assembled in the original
        input order.
        """
        if not texts:
            return []

        text_list = list(texts)
        batches = [
            text_list[i : i + self._batch_size]
            for i in range(0, len(text_list), self._batch_size)
        ]

        log_debug(
            _logger,
            "embedding.batch_start",
            "Starting batch embedding",
            text_count=len(text_list),
            batch_count=len(batches),
            model=self._model,
        )

        t0 = time.monotonic()
        vectors: list[list[float]] = []
        for batch in batches:
            vectors.extend(await self._embed_batch_with_retry(batch))

        elapsed_ms = round((time.monotonic() - t0) * 1000, 1)
        log_info(
            _logger,
            "embedding.batch_complete",
            "Batch embedding completed",
            text_count=len(text_list),
            vector_count=len(vectors),
            elapsed_ms=elapsed_ms,
            model=self._model,
        )
        return vectors

    async def embed_one(self, text: str) -> list[float]:
        """Embed a single text, returning its float vector."""
        results = await self.embed([text])
        return results[0]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _embed_batch_with_retry(self, texts: list[str]) -> list[list[float]]:
        """Call the API with retry and exponential backoff.

        Retries on: `TimeoutError` (asyncio-level), `RateLimitError` (429),
        `APIConnectionError` (network / SDK-level timeout), and 5xx
        `APIStatusError`.  Non-retryable 4xx errors re-raise immediately.
        After exhausting all retries, the last exception is re-raised.
        """
        last_exc: BaseException | None = None

        for attempt in range(self._max_retries + 1):
            try:
                return await self._call_api(texts)

            except RateLimitError as exc:
                # Must appear before APIStatusError: RateLimitError IS an APIStatusError
                # (status 429), but it is always retryable, unlike other 4xx codes.
                last_exc = exc
                log_warning(
                    _logger,
                    "embedding.rate_limited",
                    "Rate limited by OpenAI",
                    attempt=attempt,
                    batch_size=len(texts),
                )

            except APIStatusError as exc:
                if exc.status_code < 500:
                    log_error(
                        _logger,
                        "embedding.client_error",
                        "Non-retryable client error from OpenAI",
                        error=exc,
                        status_code=exc.status_code,
                        batch_size=len(texts),
                    )
                    raise
                last_exc = exc
                log_warning(
                    _logger,
                    "embedding.server_error",
                    "Server error from OpenAI",
                    attempt=attempt,
                    status_code=exc.status_code,
                    batch_size=len(texts),
                )

            except APIConnectionError as exc:
                # Also catches openai.APITimeoutError (a subclass).
                last_exc = exc
                log_warning(
                    _logger,
                    "embedding.connection_error",
                    "Connection error during embedding",
                    attempt=attempt,
                    error_type=type(exc).__name__,
                    batch_size=len(texts),
                )

            except TimeoutError as exc:
                # Raised by asyncio.wait_for when our per-call deadline expires.
                last_exc = exc
                log_warning(
                    _logger,
                    "embedding.timeout",
                    "Embedding API call timed out",
                    attempt=attempt,
                    timeout_seconds=self._timeout_seconds,
                    batch_size=len(texts),
                )

            if attempt < self._max_retries:
                delay = min(_BASE_DELAY_SECONDS * (2.0**attempt), _MAX_DELAY_SECONDS)
                log_debug(
                    _logger,
                    "embedding.retry_wait",
                    "Waiting before retry",
                    attempt=attempt,
                    next_attempt=attempt + 1,
                    delay_seconds=delay,
                )
                await asyncio.sleep(delay)

        assert last_exc is not None  # guaranteed: loop always sets it before reaching here
        log_error(
            _logger,
            "embedding.exhausted_retries",
            "Embedding failed after all retries",
            error=last_exc,
            max_retries=self._max_retries,
            batch_size=len(texts),
        )
        raise last_exc

    async def _call_api(self, texts: list[str]) -> list[list[float]]:
        """Make one API call, enforcing the per-call timeout.

        Results are sorted by the `index` field returned by the API before
        building the output list, so the order is stable even if OpenAI
        returns embeddings out of order (an undocumented edge case in batch
        processing).
        """
        try:
            response = await asyncio.wait_for(
                self._client.embeddings.create(
                    model=self._model,
                    input=texts,
                ),
                timeout=self._timeout_seconds,
            )
        except asyncio.TimeoutError as exc:
            raise TimeoutError(
                f"OpenAI embeddings API timed out after {self._timeout_seconds}s"
            ) from exc

        return [item.embedding for item in sorted(response.data, key=lambda e: e.index)]


def make_openai_provider(settings: Settings | None = None) -> OpenAIEmbeddingProvider:
    """Build an `OpenAIEmbeddingProvider` configured from application settings.

    This is the production entry point.  Tests should construct
    `OpenAIEmbeddingProvider` directly with an injected mock client rather
    than calling this factory — doing so avoids reading the environment and
    never creates a real network client.

    The API key is read from `Settings.openai_api_key` (a `SecretStr`) and
    passed directly to the `AsyncOpenAI` constructor; it is never stored as
    an attribute of `OpenAIEmbeddingProvider`.
    """
    s = settings or get_settings()
    client = AsyncOpenAI(
        api_key=s.openai_api_key.get_secret_value(),
        max_retries=0,  # retries are handled by OpenAIEmbeddingProvider
    )
    return OpenAIEmbeddingProvider(
        client=client,
        model=s.embedding_model,
    )
