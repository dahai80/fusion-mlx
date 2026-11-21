# SPDX-License-Identifier: Apache-2.0
"""Tests for hybrid-model prompt cache construction.

Regression: mlx-lm's generic make_prompt_cache returns per-layer KVCache
when the top-level model lacks make_cache. VLM wrappers (qwen3_5) keep
make_cache on the inner language_model — linear-attention layers need
ArraysCache, and a plain KVCache raises "TypeError: 'KVCache' object is
not subscriptable" at the first prefill forward.
"""

import mlx.nn as nn

from fusion_mlx.scheduler.helpers import make_model_prompt_cache


class _FakeLanguageModel(nn.Module):
    def __init__(self, n_layers):
        super().__init__()
        self.n_layers = n_layers

    def make_cache(self):
        class _Linear:
            state = "linear"

        class _KV:
            state = "kv"

        return [_Linear() if i % 2 == 0 else _KV() for i in range(self.n_layers)]


class _FakeVLMWrapper(nn.Module):
    """Mirrors mlx_vlm qwen3_5.Model: no make_cache, inner language_model has it."""

    def __init__(self, n_layers=4):
        super().__init__()
        self.language_model = _FakeLanguageModel(n_layers)


class _FakeTextModel(nn.Module):
    """No make_cache anywhere — must fall back to mlx-lm generic path."""

    def __init__(self, n_layers=2):
        super().__init__()
        self.layers = [nn.Linear(4, 4) for _ in range(n_layers)]


def test_vlm_wrapper_uses_language_model_make_cache():
    model = _FakeVLMWrapper(n_layers=4)
    cache = make_model_prompt_cache(model)
    assert len(cache) == 4
    # Hybrid mix preserved: linear layers NOT plain KVCache stand-ins
    kinds = [type(c).__name__ for c in cache]
    assert len(set(kinds)) == 2


def test_model_with_own_make_cache_preferred():
    class _Own(nn.Module):
        def make_cache(self):
            return ["top"]

    assert make_model_prompt_cache(_Own()) == ["top"]


def test_text_model_falls_back_to_generic():
    model = _FakeTextModel(n_layers=2)
    cache = make_model_prompt_cache(model)
    assert len(cache) == 2
    assert all(type(c).__name__ == "KVCache" for c in cache)
