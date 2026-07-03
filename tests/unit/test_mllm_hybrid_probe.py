# SPDX-License-Identifier: Apache-2.0
"""Tests for the startup probe that blocks hybrid models from --mllm mode (#352).

Adapted from Rapid-MLX. The ``_probe_mllm_cache_type`` function lives in
``fusion_mlx.engine.batched``.
"""

from __future__ import annotations

from unittest.mock import MagicMock

from mlx_lm.models.cache import KVCache, RotatingKVCache

from fusion_mlx.engine.batched import _probe_mllm_cache_type


class _FakeArraysCache:
    """Stand-in for mlx-lm's ArraysCache that doesn't import the real one
    (which requires building a layer). Type name is what the probe uses
    for the error message, so a class with the right __name__ is enough."""

    pass


_FakeArraysCache.__name__ = "ArraysCache"


def _model_with_cache(cache_obj):
    """Build a fake language_model whose make_prompt_cache hook returns
    a single-layer cache containing ``cache_obj``."""
    m = MagicMock()
    m.make_cache = MagicMock(return_value=[cache_obj])
    return m


def test_probe_returns_none_for_kvcache():
    model = _model_with_cache(KVCache())
    assert _probe_mllm_cache_type(model) is None


def test_probe_returns_none_for_rotating_kvcache():
    model = _model_with_cache(RotatingKVCache(max_size=256))
    assert _probe_mllm_cache_type(model) is None


def test_probe_returns_class_name_for_arrayscache():
    model = _model_with_cache(_FakeArraysCache())
    assert _probe_mllm_cache_type(model) == "ArraysCache"


def test_probe_returns_none_when_make_prompt_cache_raises():
    broken = MagicMock()
    broken.make_cache = MagicMock(side_effect=RuntimeError("not loaded"))
    result = _probe_mllm_cache_type(broken)
    assert result is None


def test_probe_returns_none_for_empty_cache():
    model = MagicMock()
    model.make_cache = MagicMock(return_value=[])
    assert _probe_mllm_cache_type(model) is None
