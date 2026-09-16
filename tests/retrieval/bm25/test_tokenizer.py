"""Unit tests for the BM25 tokenizer.

The tokenizer must be consistent between corpus indexing and query time.
Tests verify: lowercasing, punctuation splitting, stop word removal,
minimum length filtering, and preservation of technical identifiers.
"""

from __future__ import annotations

import pytest

from src.retrieval.bm25.tokenizer import tokenize


class TestTokenize:
    def test_empty_string_returns_empty_list(self) -> None:
        assert tokenize("") == []

    def test_whitespace_only_returns_empty_list(self) -> None:
        assert tokenize("   \t\n  ") == []

    def test_lowercases_all_tokens(self) -> None:
        tokens = tokenize("FastAPI UVICORN Pinecone")
        assert all(t == t.lower() for t in tokens)

    def test_splits_on_punctuation(self) -> None:
        tokens = tokenize("error-code:429")
        assert "error" in tokens
        assert "code" in tokens
        assert "429" in tokens

    def test_splits_on_dots(self) -> None:
        tokens = tokenize("service.payment.timeout")
        assert "service" in tokens
        assert "payment" in tokens
        assert "timeout" in tokens

    def test_splits_on_underscores(self) -> None:
        tokens = tokenize("fps_err_503")
        assert "fps" in tokens
        assert "err" in tokens
        assert "503" in tokens

    def test_drops_single_character_tokens(self) -> None:
        tokens = tokenize("a b c x y z")
        assert tokens == []

    def test_drops_single_digit_number(self) -> None:
        tokens = tokenize("version 1 patch 2")
        assert "1" not in tokens
        assert "2" not in tokens

    def test_preserves_two_digit_numbers(self) -> None:
        tokens = tokenize("retry after 30 seconds")
        assert "30" in tokens

    def test_preserves_three_digit_error_code(self) -> None:
        tokens = tokenize("HTTP status 429")
        assert "429" in tokens
        assert "503" not in tokens  # not in this string

    def test_removes_stop_words(self) -> None:
        tokens = tokenize("the payment was processed successfully")
        assert "the" not in tokens
        assert "was" not in tokens
        assert "payment" in tokens
        assert "processed" in tokens
        assert "successfully" in tokens

    def test_removes_common_prepositions(self) -> None:
        tokens = tokenize("failure in the core banking system")
        assert "in" not in tokens
        assert "the" not in tokens
        assert "failure" in tokens
        assert "core" in tokens
        assert "banking" in tokens
        assert "system" in tokens

    def test_preserves_error_code_with_hyphens(self) -> None:
        tokens = tokenize("FPS-ERR-429")
        assert "fps" in tokens
        assert "err" in tokens
        assert "429" in tokens

    def test_preserves_algorithm_identifier(self) -> None:
        tokens = tokenize("HMAC-SHA256")
        assert "hmac" in tokens
        assert "sha256" in tokens

    def test_preserves_rare_technical_term(self) -> None:
        tokens = tokenize("idempotency key required")
        assert "idempotency" in tokens
        assert "key" in tokens

    def test_preserves_repeated_tokens_for_tf_scoring(self) -> None:
        tokens = tokenize("payment payment payment timeout")
        assert tokens.count("payment") == 3
        assert "timeout" in tokens

    def test_mixed_technical_content(self) -> None:
        text = "The FPS service returned ERR-429 after a timeout on HMAC-SHA256 validation"
        tokens = tokenize(text)
        assert "fps" in tokens
        assert "service" in tokens
        assert "returned" in tokens
        assert "err" in tokens
        assert "429" in tokens
        assert "timeout" in tokens
        assert "hmac" in tokens
        assert "sha256" in tokens
        assert "validation" in tokens
        # Stop words removed
        assert "the" not in tokens
        assert "after" not in tokens
        assert "on" not in tokens

    def test_numeric_id_preserved(self) -> None:
        tokens = tokenize("incident INC-20240115 resolved")
        assert "inc" in tokens
        assert "20240115" in tokens
        assert "resolved" in tokens

    def test_consecutive_delimiters_produce_no_empty_tokens(self) -> None:
        tokens = tokenize("...---   ...payment---timeout")
        assert "" not in tokens
        assert "payment" in tokens
        assert "timeout" in tokens

    def test_result_is_a_list(self) -> None:
        result = tokenize("payment processing")
        assert isinstance(result, list)

    def test_deterministic_same_input_same_output(self) -> None:
        text = "core banking ledger FPS-ERR-429 timeout"
        assert tokenize(text) == tokenize(text)
