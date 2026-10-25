# SPDX-License-Identifier: Apache-2.0
"""#1403: slim install — text training paths must not require mlx_vlm."""

from __future__ import annotations

import sys

import mlx.core as mx
import pytest
from mlx_lm.models import base as mlx_base

from fusion_mlx.patches import turboquant_attention
from fusion_mlx.training.service import format_job_error


@pytest.fixture
def restore_patch(monkeypatch):
    original = mlx_base.scaled_dot_product_attention
    monkeypatch.setattr(turboquant_attention, "_PATCHED", False)
    monkeypatch.setattr(turboquant_attention, "_TQ_TYPES", None)
    monkeypatch.setattr(turboquant_attention, "_TQ_TYPES_TRIED", False)
    monkeypatch.setattr(mlx_base, "scaled_dot_product_attention", original)
    yield original
    mlx_base.scaled_dot_product_attention = original


def _attn_inputs():
    q = mx.zeros((1, 2, 4, 8), dtype=mx.float32)
    return q, q, q


class TestPatchWithoutMlxVlm:
    def test_sdpa_passes_through_when_mlx_vlm_missing(self, restore_patch, monkeypatch):
        # sys.modules[name] = None makes `import mlx_vlm` raise ImportError,
        # which is what a slim install looks like from inside the patch.
        monkeypatch.setitem(sys.modules, "mlx_vlm", None)
        monkeypatch.setitem(sys.modules, "mlx_vlm.turboquant", None)
        assert turboquant_attention.apply_turboquant_attention_patch() is True
        assert turboquant_attention._tq_cache_types() == ()

        q, k, v = _attn_inputs()
        patched = mlx_base.scaled_dot_product_attention
        expected = restore_patch(q, k, v, None, 0.5, None)
        got = patched(q, k, v, None, 0.5, None)
        assert mx.allclose(got, expected).item()

    def test_types_resolved_once(self, restore_patch, monkeypatch):
        import builtins

        monkeypatch.setitem(sys.modules, "mlx_vlm", None)
        monkeypatch.setitem(sys.modules, "mlx_vlm.turboquant", None)
        assert turboquant_attention._tq_cache_types() == ()
        calls = []
        real_import = builtins.__import__

        def counting(name, *args, **kwargs):
            calls.append(name)
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", counting)
        turboquant_attention._tq_cache_types()
        assert not [c for c in calls if c.startswith("mlx_vlm")]


class TestPatchWithMlxVlm:
    def test_tq_types_include_state_length_helper(self, restore_patch):
        pytest.importorskip("mlx_vlm")
        tq = turboquant_attention._tq_cache_types()
        assert len(tq) == 3
        assert callable(tq[2])

    def test_plain_cache_still_uses_tq_path_when_available(self, restore_patch):
        pytest.importorskip("mlx_vlm")
        assert turboquant_attention.apply_turboquant_attention_patch() is True
        q, k, v = _attn_inputs()
        expected = restore_patch(q, k, v, None, 0.5, None)
        got = mlx_base.scaled_dot_product_attention(q, k, v, None, 0.5, None)
        assert mx.allclose(got, expected).item()


class TestFormatJobError:
    def test_message_first_then_frames(self):
        def inner():
            raise ValueError("No module named 'mlx_vlm'")

        try:
            inner()
        except ValueError as exc:
            text = format_job_error(exc)
        lines = text.splitlines()
        assert lines[0].endswith("No module named 'mlx_vlm'")
        assert "Traceback (most recent call last)" in text
        assert "in inner" in text

    def test_truncated_to_budget(self):
        def deep(n):
            if n:
                return deep(n - 1)
            raise RuntimeError("deep")

        try:
            deep(40)
        except RuntimeError as exc:
            text = format_job_error(exc, limit=40)
        assert len(text) <= 4000
        assert text.startswith("RuntimeError: deep")

    def test_exception_without_traceback_still_reports(self):
        text = format_job_error(RuntimeError("no frames"))
        assert "no frames" in text
