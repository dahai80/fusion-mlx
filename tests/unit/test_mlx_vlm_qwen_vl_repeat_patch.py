# SPDX-License-Identifier: Apache-2.0
"""Tests for the qwen2_5_vl mx.repeat array-repeats compat patch and the
grid_thw ``or``-truthiness regression in VLMBatchedEngine._compute_vision_features."""

import importlib
import types

import mlx.core as mx


def test_repeat_proxy_converts_array_repeats():
    from fusion_mlx.patches.mlx_vlm_qwen_vl_repeat import _MxRepeatProxy

    proxy = _MxRepeatProxy(mx)
    expected = mx.repeat(mx.array([1, 2, 3]), 2)
    got = proxy.repeat(mx.array([1, 2, 3]), mx.array(2))
    assert (got == expected).all()


def test_apply_patches_vision_module_repeat():
    vision_mod = importlib.import_module("mlx_vlm.models.qwen2_5_vl.vision")

    from fusion_mlx.patches import mlx_vlm_qwen_vl_repeat

    assert mlx_vlm_qwen_vl_repeat.apply_mlx_vlm_qwen_vl_repeat_patch() is True
    # The module-level mx is now the proxy; a length-1 array repeats value
    # (what vision.py __call__ passes as grid_thw[i, 0]) must not raise
    # TypeError against MLX 0.32's int-only signature.
    vision_mod.mx.repeat(mx.array([1, 2]), mx.array(3))


def test_pre_load_dispatch_applies_qwen_vl_repeat_patch(tmp_path, monkeypatch):
    (tmp_path / "config.json").write_text('{"model_type": "qwen2_5_vl"}')

    from fusion_mlx.patches import mlx_vlm_qwen_vl_repeat
    from fusion_mlx.utils.model_loading import maybe_apply_pre_load_patches

    calls = []
    monkeypatch.setattr(mlx_vlm_qwen_vl_repeat, "_APPLIED", False)
    monkeypatch.setattr(
        mlx_vlm_qwen_vl_repeat,
        "apply_mlx_vlm_qwen_vl_repeat_patch",
        lambda: calls.append(True) or True,
    )

    maybe_apply_pre_load_patches(str(tmp_path), for_vlm=True)

    assert calls == [True]


def test_compute_vision_features_grid_thw_array_no_bool():
    # Regression: ``extra.get("image_grid_thw") or extra.get("video_grid_thw")``
    # evaluated bool() on the shape-(n,3) mx.array grid, raising
    # "[convert] Only length-1 arrays can be converted to Python scalars"
    # before Strategy 2 ever reached the vision tower.
    from fusion_mlx.engines.vlm import VLMBatchedEngine

    class _BoolRaises:
        def __bool__(self):
            raise ValueError(
                "[convert] Only length-1 arrays can be converted to " "Python scalars."
            )

    tower_calls = {}

    class _Tower:
        def __call__(self, pv, grid_thw):
            tower_calls["grid"] = grid_thw
            return ("features",)

    tower = _Tower()
    tower.patch_embed = types.SimpleNamespace(
        proj=types.SimpleNamespace(weight=types.SimpleNamespace(dtype=mx.float16))
    )
    model = types.SimpleNamespace(
        vision_tower=tower,
        config=types.SimpleNamespace(model_type="qwen2_5_vl"),
    )

    engine = object.__new__(VLMBatchedEngine)
    engine._vlm_model = model

    grid = _BoolRaises()
    features = engine._compute_vision_features(
        [0.0, 1.0, 2.0], {"image_grid_thw": grid}
    )

    assert features == "features"
    assert tower_calls["grid"] is grid
