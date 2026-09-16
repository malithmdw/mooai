"""Unit tests for OpenAIEmbeddingProvider.

All tests use injected `AsyncMock` / `MagicMock` clients — no real network
calls are made.  `asyncio.sleep` is patched to a no-op `AsyncMock` in every
retry test so the suite runs instantly.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from openai import APIConnectionError, APIStatusError, RateLimitError

from src.retrieval.embedding.base import EmbeddingProvider
from src.retrieval.embedding.openai_provider import (
    DEFAULT_BATCH_SIZE,
    OpenAIEmbeddingProvider,
    make_openai_provider,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_DUMMY_REQUEST = httpx.Request("POST", "https://api.openai.com/v1/embeddings")


def _mock_response(status_code: int) -> httpx.Response:
    return httpx.Response(status_code, request=_DUMMY_REQUEST)


def _make_api_response(vectors: list[list[float]]) -> MagicMock:
    """Build a minimal mock of `openai.types.CreateEmbeddingResponse`."""
    response = MagicMock()
    items = []
    for i, vec in enumerate(vectors):
        item = MagicMock()
        item.embedding = vec
        item.index = i
        items.append(item)
    response.data = items
    return response


def _make_provider(
    *,
    vectors: list[list[float]] | None = None,
    batch_size: int = DEFAULT_BATCH_SIZE,
    max_retries: int = 3,
    timeout_seconds: float = 30.0,
) -> tuple[OpenAIEmbeddingProvider, AsyncMock]:
    """Return ``(provider, mock_create)`` with `embeddings.create` as an AsyncMock."""
    mock_client = MagicMock()
    mock_create = AsyncMock(
        return_value=_make_api_response(vectors or [[0.1, 0.2, 0.3]])
    )
    mock_client.embeddings.create = mock_create
    provider = OpenAIEmbeddingProvider(
        client=mock_client,
        model="text-embedding-3-large",
        batch_size=batch_size,
        max_retries=max_retries,
        timeout_seconds=timeout_seconds,
    )
    return provider, mock_create


# ---------------------------------------------------------------------------
# Protocol conformance
# ---------------------------------------------------------------------------


class TestEmbeddingProviderProtocol:
    def test_openai_provider_satisfies_protocol(self) -> None:
        provider, _ = _make_provider()
        assert isinstance(provider, EmbeddingProvider)

    def test_model_property_returns_string(self) -> None:
        provider, _ = _make_provider()
        assert isinstance(provider.model, str)
        assert provider.model == "text-embedding-3-large"


# ---------------------------------------------------------------------------
# embed() — basic behaviour
# ---------------------------------------------------------------------------


class TestEmbed:
    async def test_empty_input_returns_empty_list(self) -> None:
        provider, mock_create = _make_provider()
        result = await provider.embed([])
        assert result == []
        mock_create.assert_not_called()

    async def test_single_text_returns_single_vector(self) -> None:
        vec = [0.1, 0.2, 0.3]
        provider, _ = _make_provider(vectors=[vec])
        result = await provider.embed(["hello"])
        assert result == [vec]

    async def test_multiple_texts_return_multiple_vectors(self) -> None:
        vecs = [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]
        provider, _ = _make_provider(vectors=vecs)
        result = await provider.embed(["a", "b", "c"])
        assert result == vecs

    async def test_output_order_matches_input_order(self) -> None:
        # The API returns embeddings with an `index` field; we sort by it.
        # Simulate out-of-order API response by reversing the items' index.
        mock_client = MagicMock()
        response = MagicMock()
        items = []
        for i, vec in enumerate([[0.1], [0.2], [0.3]]):
            item = MagicMock()
            item.embedding = vec
            item.index = 2 - i  # reversed: index 2, 1, 0
            items.append(item)
        response.data = items
        mock_client.embeddings.create = AsyncMock(return_value=response)

        provider = OpenAIEmbeddingProvider(client=mock_client)
        result = await provider.embed(["a", "b", "c"])

        assert result == [[0.3], [0.2], [0.1]]

    async def test_large_input_batched_into_multiple_api_calls(self) -> None:
        batch_size = 3
        text_count = 7
        texts = [f"text-{i}" for i in range(text_count)]

        def _side_effect(**kwargs: object) -> MagicMock:
            n = len(kwargs.get("input", []))  # type: ignore[arg-type]
            return _make_api_response([[float(i)] for i in range(n)])

        mock_client = MagicMock()
        mock_client.embeddings.create = AsyncMock(side_effect=_side_effect)
        provider = OpenAIEmbeddingProvider(
            client=mock_client,
            batch_size=batch_size,
        )
        result = await provider.embed(texts)

        expected_calls = (text_count + batch_size - 1) // batch_size  # ceil division
        assert mock_client.embeddings.create.call_count == expected_calls
        assert len(result) == text_count

    async def test_returns_list_of_float_lists(self) -> None:
        provider, _ = _make_provider(vectors=[[1.0, 2.0]])
        result = await provider.embed(["x"])
        assert isinstance(result, list)
        assert isinstance(result[0], list)
        assert all(isinstance(v, float) for v in result[0])


# ---------------------------------------------------------------------------
# embed_one()
# ---------------------------------------------------------------------------


class TestEmbedOne:
    async def test_returns_single_vector(self) -> None:
        vec = [0.9, 0.8, 0.7]
        provider, _ = _make_provider(vectors=[vec])
        result = await provider.embed_one("hello world")
        assert result == vec

    async def test_calls_embed_with_single_text(self) -> None:
        provider, mock_create = _make_provider()
        await provider.embed_one("single")
        mock_create.assert_called_once()
        _, kwargs = mock_create.call_args
        assert kwargs["input"] == ["single"]


# ---------------------------------------------------------------------------
# Retry behaviour
# ---------------------------------------------------------------------------


class TestRetryBehavior:
    @patch("src.retrieval.embedding.openai_provider.asyncio.sleep", new_callable=AsyncMock)
    async def test_retries_on_timeout_error(self, mock_sleep: AsyncMock) -> None:
        provider, mock_create = _make_provider(max_retries=2)
        mock_create.side_effect = TimeoutError("timed out")
        with pytest.raises(TimeoutError):
            await provider.embed(["text"])
        assert mock_create.call_count == 3  # initial + 2 retries
        assert mock_sleep.call_count == 2

    @patch("src.retrieval.embedding.openai_provider.asyncio.sleep", new_callable=AsyncMock)
    async def test_retries_on_rate_limit(self, mock_sleep: AsyncMock) -> None:
        provider, mock_create = _make_provider(max_retries=2)
        mock_create.side_effect = RateLimitError(
            "rate limited",
            response=_mock_response(429),
            body=None,
        )
        with pytest.raises(RateLimitError):
            await provider.embed(["text"])
        assert mock_create.call_count == 3
        assert mock_sleep.call_count == 2

    @patch("src.retrieval.embedding.openai_provider.asyncio.sleep", new_callable=AsyncMock)
    async def test_retries_on_connection_error(self, mock_sleep: AsyncMock) -> None:
        provider, mock_create = _make_provider(max_retries=1)
        mock_create.side_effect = APIConnectionError(request=_DUMMY_REQUEST)
        with pytest.raises(APIConnectionError):
            await provider.embed(["text"])
        assert mock_create.call_count == 2
        assert mock_sleep.call_count == 1

    @patch("src.retrieval.embedding.openai_provider.asyncio.sleep", new_callable=AsyncMock)
    async def test_retries_on_5xx_server_error(self, mock_sleep: AsyncMock) -> None:
        provider, mock_create = _make_provider(max_retries=2)
        mock_create.side_effect = APIStatusError(
            "server error",
            response=_mock_response(503),
            body=None,
        )
        with pytest.raises(APIStatusError):
            await provider.embed(["text"])
        assert mock_create.call_count == 3

    async def test_no_retry_on_4xx_client_error(self) -> None:
        provider, mock_create = _make_provider(max_retries=3)
        mock_create.side_effect = APIStatusError(
            "bad request",
            response=_mock_response(400),
            body=None,
        )
        with pytest.raises(APIStatusError):
            await provider.embed(["text"])
        assert mock_create.call_count == 1  # called once, no retry

    @patch("src.retrieval.embedding.openai_provider.asyncio.sleep", new_callable=AsyncMock)
    async def test_succeeds_on_second_attempt(self, mock_sleep: AsyncMock) -> None:
        provider, mock_create = _make_provider(max_retries=3)
        vec = [0.5, 0.6]
        mock_create.side_effect = [
            RateLimitError("rate limited", response=_mock_response(429), body=None),
            _make_api_response([vec]),
        ]
        result = await provider.embed(["text"])
        assert result == [vec]
        assert mock_create.call_count == 2
        assert mock_sleep.call_count == 1

    @patch("src.retrieval.embedding.openai_provider.asyncio.sleep", new_callable=AsyncMock)
    async def test_exponential_backoff_delay_increases(self, mock_sleep: AsyncMock) -> None:
        provider, mock_create = _make_provider(max_retries=3)
        mock_create.side_effect = TimeoutError("always timeout")
        with pytest.raises(TimeoutError):
            await provider.embed(["text"])
        delays = [call.args[0] for call in mock_sleep.call_args_list]
        assert delays == sorted(delays), "Delays must be non-decreasing"
        assert delays[0] < delays[-1], "Delay must grow across retries"

    @patch("src.retrieval.embedding.openai_provider.asyncio.sleep", new_callable=AsyncMock)
    async def test_no_sleep_after_final_retry(self, mock_sleep: AsyncMock) -> None:
        provider, mock_create = _make_provider(max_retries=2)
        mock_create.side_effect = TimeoutError("timeout")
        with pytest.raises(TimeoutError):
            await provider.embed(["text"])
        # 2 retries → 2 sleeps (not 3)
        assert mock_sleep.call_count == 2


# ---------------------------------------------------------------------------
# Timeout handling
# ---------------------------------------------------------------------------


class TestTimeoutHandling:
    async def test_asyncio_timeout_converted_to_timeout_error(self) -> None:
        """asyncio.wait_for cancels a slow request and we surface it as TimeoutError."""

        async def _slow(**_: object) -> MagicMock:
            await asyncio.sleep(10)
            return _make_api_response([[0.1]])

        mock_client = MagicMock()
        mock_client.embeddings.create = AsyncMock(side_effect=_slow)
        provider = OpenAIEmbeddingProvider(
            client=mock_client,
            max_retries=0,
            timeout_seconds=0.001,
        )
        with pytest.raises(TimeoutError):
            await provider._call_api(["text"])  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# API key safety
# ---------------------------------------------------------------------------


class TestApiKeySafety:
    def test_api_key_not_stored_on_provider(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The OpenAI API key must not appear in provider instance attributes."""
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test-secret-key-should-not-leak")
        monkeypatch.setenv("EMBEDDING_MODEL", "text-embedding-3-large")

        from src.core.config import Settings

        provider = make_openai_provider(Settings(_env_file=None))

        provider_state = str(vars(provider))
        assert "sk-test-secret-key-should-not-leak" not in provider_state

    def test_model_is_accessible_without_key_exposure(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test-key")
        monkeypatch.setenv("EMBEDDING_MODEL", "text-embedding-3-small")

        from src.core.config import Settings

        provider = make_openai_provider(Settings(_env_file=None))
        assert provider.model == "text-embedding-3-small"


# ---------------------------------------------------------------------------
# make_openai_provider factory
# ---------------------------------------------------------------------------


class TestMakeOpenAIProvider:
    def test_returns_openai_embedding_provider(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        monkeypatch.setenv("EMBEDDING_MODEL", "text-embedding-3-large")

        from src.core.config import Settings

        provider = make_openai_provider(Settings(_env_file=None))
        assert isinstance(provider, OpenAIEmbeddingProvider)

    def test_uses_embedding_model_from_settings(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        monkeypatch.setenv("EMBEDDING_MODEL", "text-embedding-3-small")

        from src.core.config import Settings

        provider = make_openai_provider(Settings(_env_file=None))
        assert provider.model == "text-embedding-3-small"

    def test_satisfies_embedding_provider_protocol(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")

        from src.core.config import Settings

        provider = make_openai_provider(Settings(_env_file=None))
        assert isinstance(provider, EmbeddingProvider)
