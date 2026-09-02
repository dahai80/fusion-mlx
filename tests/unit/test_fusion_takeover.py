from __future__ import annotations

import mlx.nn as nn

from fusion_mlx.fusion_takeover import (
    FusionConfig,
    apply_fusion_takeover,
)
from fusion_mlx.fusion_takeover.patcher import _is_linear_like, _iter_linear
from fusion_mlx.model_settings import ModelSettings
from fusion_mlx.utils.model_loading import apply_post_load_transforms


class _FakeQwen(nn.Module):
    def __init__(self):
        super().__init__()
        self.model_type = "qwen2"
        self.model = _Inner()

    def __call__(self, *a, **k):
        return None


class _Inner(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = [_Block() for _ in range(2)]


class _Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn = _Attn()
        self.mlp = _MLP()


class _Attn(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj = nn.Linear(4, 4)
        self.k_proj = nn.Linear(4, 4)
        self.v_proj = nn.Linear(4, 4)
        self.o_proj = nn.Linear(4, 4)


class _MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate_proj = nn.Linear(4, 8)
        self.up_proj = nn.Linear(4, 8)
        self.down_proj = nn.Linear(8, 4)


def _make_model():
    return _FakeQwen()


def test_config_defaults_off():
    cfg = FusionConfig()
    assert cfg.enabled is False
    assert cfg.paged_kv_enabled is False
    assert cfg.is_supported_model_type("qwen2") is True


def test_config_from_model_settings_off():
    ms = ModelSettings()
    cfg = FusionConfig.from_model_settings(ms)
    assert cfg.enabled is False


def test_config_from_model_settings_on():
    ms = ModelSettings(
        fusion_takeover_enabled=True,
        fusion_quant="nvfp4",
        fusion_paged_kv_enabled=True,
        fusion_target_model_types=("qwen2",),
    )
    cfg = FusionConfig.from_model_settings(ms)
    assert cfg.enabled is True
    assert cfg.quant == "nvfp4"
    assert cfg.paged_kv_enabled is True
    assert cfg.is_supported_model_type("qwen2") is True
    assert cfg.is_supported_model_type("llama") is False


def test_config_invalid_block_size():
    import pytest

    with pytest.raises(ValueError):
        FusionConfig(enabled=True, paged_kv_block_size=0)


def test_apply_fusion_takeover_off_passthrough():
    model = _make_model()
    ms = ModelSettings()
    out = apply_fusion_takeover(model, ms)
    assert out is model
    assert getattr(out, "_fusion_takeover_applied", False) is False


def test_apply_fusion_takeover_none_settings():
    model = _make_model()
    out = apply_fusion_takeover(model, None)
    assert out is model


def test_apply_fusion_takeover_tags_linears():
    model = _make_model()
    ms = ModelSettings(
        fusion_takeover_enabled=True,
        fusion_quant="nvfp4",
        fusion_target_model_types=("qwen2",),
    )
    out = apply_fusion_takeover(model, ms)
    assert getattr(out, "_fusion_takeover_applied", False) is True
    assert out.model.layers[0].self_attn.q_proj._fusion_quant == "nvfp4"
    assert out.model.layers[0].mlp.down_proj._fusion_quant == "nvfp4"
    tagged = sum(
        1 for _, _, _, mo, _ in _iter_linear(out) if getattr(mo, "_fusion_quant", None)
    )
    assert tagged == 2 * (4 + 3)


def test_apply_fusion_takeover_target_filter():
    model = _make_model()
    ms = ModelSettings(
        fusion_takeover_enabled=True,
        fusion_quant="nvfp4",
        fusion_target_model_types=("llama",),
    )
    out = apply_fusion_takeover(model, ms)
    assert getattr(out, "_fusion_takeover_applied", False) is False


def test_apply_fusion_takeover_idempotent():
    model = _make_model()
    ms = ModelSettings(
        fusion_takeover_enabled=True,
        fusion_quant="nvfp4",
        fusion_target_model_types=("qwen2",),
    )
    out = apply_fusion_takeover(model, ms)
    out2 = apply_fusion_takeover(out, ms)
    assert out2 is out


def test_apply_fusion_takeover_non_module_passthrough():
    ms = ModelSettings(fusion_takeover_enabled=True, fusion_quant="nvfp4")
    out = apply_fusion_takeover("not_a_model", ms)
    assert out == "not_a_model"


def test_post_load_transforms_off_chain():
    model = _make_model()
    ms = ModelSettings()
    out = apply_post_load_transforms(model, ms)
    assert getattr(out, "_fusion_takeover_applied", "none") == "none"


def test_post_load_transforms_on_chain():
    model = _make_model()
    ms = ModelSettings(
        fusion_takeover_enabled=True,
        fusion_quant="mxfp8",
        fusion_target_model_types=("qwen2",),
    )
    out = apply_post_load_transforms(model, ms)
    assert getattr(out, "_fusion_takeover_applied", False) is True
    assert out.model.layers[0].self_attn.q_proj._fusion_quant == "mxfp8"


def test_iter_linear_handles_list_blocks():
    model = _make_model()
    names = [n for _, _, n, _, _ in _iter_linear(model)]
    assert "model.layers.0.self_attn.q_proj" in names
    assert "model.layers.1.mlp.down_proj" in names
    assert len(names) == 2 * (4 + 3)


def test_is_linear_like_quantized():
    try:
        from mlx.nn.layers.quantized import QuantizedLinear
    except Exception:
        return
    ql = QuantizedLinear.from_linear(nn.Linear(32, 32), group_size=32, bits=4)
    assert _is_linear_like(ql) is True
