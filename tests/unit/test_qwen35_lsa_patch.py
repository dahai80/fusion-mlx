# SPDX-License-Identifier: Apache-2.0
"""Tests for the qwen3_5 LSA (latent sparse attention) pre-load patch."""

from __future__ import annotations

import logging

import pytest

from fusion_mlx.model_settings import ModelSettings
from fusion_mlx.utils.model_loading import maybe_apply_pre_load_patches


def _write_config(tmp_path) -> str:
    (tmp_path / "config.json").write_text(
        '{"model_type": "qwen3_5", "hidden_size": 256, "num_hidden_layers": 8, '
        '"num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": 64, '
        '"full_attention_interval": 4}'
    )
    return str(tmp_path)


def _write_map(tmp_path) -> str:
    # a real (tiny) safetensors map, so apply validates an actual load
    import numpy as np
    from safetensors.numpy import save_file

    path = str(tmp_path / "lsa-map.safetensors")
    save_file(
        {
            "layer3.query_projection": np.zeros((8, 4), dtype=np.float32),
            "layer3.key_projection": np.zeros((8, 4), dtype=np.float32),
        },
        path,
        metadata={
            "model_arch": "qwen38_27b",
            "indexer_layers": "3",
            "latent_dim": "8",
            "index_dim": "4",
        },
    )
    return path


def reset_patch_state():
    from fusion_mlx.patches import qwen35_lsa

    qwen35_lsa.reset_state()
    yield
    qwen35_lsa.reset_state()


@pytest.fixture(autouse=True)
def _reset():
    yield from reset_patch_state()


def test_pre_load_dispatch_applies_lsa_patch(tmp_path, caplog):
    config_dir = _write_config(tmp_path)
    map_path = _write_map(tmp_path)
    settings = ModelSettings(lsa_enabled=True, lsa_map_path=map_path)
    with caplog.at_level(logging.INFO):
        maybe_apply_pre_load_patches(config_dir, model_settings=settings)
    from fusion_mlx.patches import qwen35_lsa

    assert qwen35_lsa.is_active()
    assert qwen35_lsa.map_path() == map_path
    assert any("armed" in r.message.lower() for r in caplog.records)


def test_dispatch_noop_without_flag(tmp_path):
    config_dir = _write_config(tmp_path)
    maybe_apply_pre_load_patches(config_dir, model_settings=ModelSettings())
    from fusion_mlx.patches import qwen35_lsa

    assert not qwen35_lsa.is_active()


def test_dispatch_fails_loud_on_missing_map(tmp_path):
    config_dir = _write_config(tmp_path)
    settings = ModelSettings(
        lsa_enabled=True, lsa_map_path=str(tmp_path / "absent.safetensors")
    )
    with pytest.raises(Exception):
        maybe_apply_pre_load_patches(config_dir, model_settings=settings)


def test_dispatch_noop_for_other_model_types(tmp_path):
    (tmp_path / "config.json").write_text('{"model_type": "glm_moe_dsa"}')
    map_path = _write_map(tmp_path)
    settings = ModelSettings(lsa_enabled=True, lsa_map_path=map_path)
    maybe_apply_pre_load_patches(str(tmp_path), model_settings=settings)
    from fusion_mlx.patches import qwen35_lsa

    assert not qwen35_lsa.is_active()


def test_ensure_all_registered_includes_qwen35_lsa():
    from fusion_mlx.patches import ensure_all_registered, list_patches

    ensure_all_registered()
    ids = {entry["patch_id"] for entry in list_patches()}
    assert any("qwen35_lsa" in pid for pid in ids)


def test_settings_require_map_when_enabled():
    with pytest.raises(ValueError):
        ModelSettings(lsa_enabled=True)


def test_settings_allow_lsa_with_turboquant():
    # coexistence is allowed at the settings level; the serve command gates
    # it to k8v4 mode (8-bit keys) and the engine gates auto-enable off
    settings = ModelSettings(
        lsa_enabled=True, lsa_map_path="m.safetensors", turboquant_kv_enabled=True
    )
    assert settings.lsa_enabled and settings.turboquant_kv_enabled
