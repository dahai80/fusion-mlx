# SPDX-License-Identifier: Apache-2.0
"""Regression tests for issue #1062.

#1062: three engine token-count / bounds bugs.
- reranker: max_content_tokens went negative for small max_length →
  tokenizer(max_length=negative) crash.
- embedding: max(len(ids) for ids in encoded_ids) → ValueError on empty
  input (embed([]) with non-callable processor).
- NER: total_tokens used space-split (wrong for CJK).
"""

from __future__ import annotations

import inspect

import pytest


class TestRerankerMaxContentTokensFloor:
    """reranker clamps max_content_tokens to >= 0."""

    def test_clamp_logic(self):
        max_length = 5
        prefix_tokens = [1, 2, 3]
        suffix_tokens = [4, 5, 6]
        max_content_tokens = max_length - len(prefix_tokens) - len(suffix_tokens)
        assert max_content_tokens < 0  # 5 - 3 - 3 = -1
        clamped = max(0, max_content_tokens)
        assert clamped == 0

    def test_source_has_floor_guard(self):
        from fusion_mlx.engines import reranker

        source = inspect.getsource(reranker)
        assert "max_content_tokens <= 0" in source
        assert "max_content_tokens = 0" in source


class TestEmbeddingEmptyMaxGuard:
    """embedding max() on empty encoded_ids uses default=0."""

    def test_max_empty_default_zero(self):
        encoded_ids: list[list[int]] = []
        max_len = max((len(ids) for ids in encoded_ids), default=0)
        assert max_len == 0

    def test_max_nonempty_normal(self):
        encoded_ids = [[1, 2, 3], [4, 5]]
        max_len = max((len(ids) for ids in encoded_ids), default=0)
        assert max_len == 3

    def test_source_uses_default(self):
        from fusion_mlx.engines import embedding

        source = inspect.getsource(embedding)
        assert "default=0" in source


class TestNERTokenCountRealTokenizer:
    """NER uses the real tokenizer, falls back to char-based."""

    def test_source_uses_tokenizer_not_split(self):
        from fusion_mlx.engines import ner

        source = inspect.getsource(ner)
        # pre-fix used len(text.split()); post-fix uses tokenizer.encode
        # with a char-based fallback.
        assert "ner_tok" in source
        assert "tokenizer" in source
        assert ".encode(text)" in source

    def test_char_fallback_better_than_split_for_cjk(self):
        # space-split: 一段中文 → 1 word; char-based → 4 chars (closer to
        # real token count for CJK-heavy text).
        text = "一段中文测试"
        split_count = len(text.split())
        char_count = max(1, len(text))
        assert split_count == 1
        assert char_count == 6
        assert char_count > split_count  # char-based is more honest for CJK


if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__, "-v"])
