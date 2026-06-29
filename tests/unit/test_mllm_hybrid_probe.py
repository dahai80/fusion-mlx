# SPDX-License-Identifier: Apache-2.0
"""Tests for the startup probe that blocks hybrid models from --mllm mode (#352).

Adapted from Rapid-MLX. The ``_probe_mllm_cache_type`` function lives in
``vllm_mlx.engine.batched`` which does not exist in fusion-mlx — all tests
are skipped.
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.skip(reason="rapid-mlx-only: fusion_mlx.engine.batched does not exist")


def test_probe_returns_none_for_kvcache():
    pass


def test_probe_returns_none_for_rotating_kvcache():
    pass


def test_probe_returns_class_name_for_arrayscache():
    pass


def test_probe_returns_none_when_make_prompt_cache_raises():
    pass


def test_probe_returns_none_for_empty_cache():
    pass
