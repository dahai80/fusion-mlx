"""C DiT 权重量化单元测试.

验证:
  - _dit_quantization_config: env 解析 w8a16/w4/off/非法值/大小写
  - tiny 真实 DiT nn.quantize(bits=8) 消费 config dict: Linear -> QuantizedLinear
"""

import mlx.core as mx
import mlx.nn as nn

from fusion_mlx.video.skyreels_v3.pipelines import SkyReelsBasePipeline


def _pipe():
    # _dit_quantization_config 只读 env, 不需 self 状态, 绕过 __init__.
    return SkyReelsBasePipeline.__new__(SkyReelsBasePipeline)


def test_quant_config_default_off(monkeypatch):
    monkeypatch.delenv("FUSION_SKYREELS_QUANT", raising=False)
    assert _pipe()._dit_quantization_config() is None


def test_quant_config_off_aliases(monkeypatch):
    for v in ("0", "off", "none", "bf16", ""):
        monkeypatch.setenv("FUSION_SKYREELS_QUANT", v)
        assert _pipe()._dit_quantization_config() is None, f"off alias {v!r}"


def test_quant_config_w8a16(monkeypatch):
    for v in ("w8a16", "w8", "int8", "8"):
        monkeypatch.setenv("FUSION_SKYREELS_QUANT", v)
        cfg = _pipe()._dit_quantization_config()
        assert cfg == {"bits": 8, "group_size": 64}, f"w8a16 alias {v!r}: {cfg}"


def test_quant_config_w4(monkeypatch):
    for v in ("w4", "nf4", "int4", "4"):
        monkeypatch.setenv("FUSION_SKYREELS_QUANT", v)
        cfg = _pipe()._dit_quantization_config()
        assert cfg == {"bits": 4, "group_size": 64}, f"w4 alias {v!r}: {cfg}"


def test_quant_config_case_insensitive(monkeypatch):
    monkeypatch.setenv("FUSION_SKYREELS_QUANT", "W8A16")
    assert _pipe()._dit_quantization_config() == {"bits": 8, "group_size": 64}
    monkeypatch.setenv("FUSION_SKYREELS_QUANT", "  W4  ")
    assert _pipe()._dit_quantization_config() == {"bits": 4, "group_size": 64}


def test_quant_config_invalid_warns(monkeypatch):
    monkeypatch.setenv("FUSION_SKYREELS_QUANT", "foo")
    assert _pipe()._dit_quantization_config() is None


# ---------------------------------------------------------------------------
# tiny 真实 DiT 量化 smoke (dims 整除 group_size=64)
# ---------------------------------------------------------------------------

_TINY_CFG = dict(
    num_layers=4,
    dim=128,
    ffn_dim=256,
    num_heads=4,
    num_kv_heads=4,
    patch_size=(1, 2, 2),
    in_dim=16,
    out_dim=16,
    text_dim=128,
    text_len=8,
    freq_dim=128,
)


def test_quant_smoke_w8a16_consumes_config():
    # config dict 可被 nn.quantize 消费: block Linear -> QuantizedLinear, 不崩.
    from fusion_mlx.video.skyreels_v3.transformer_r2v import SkyReelsR2VDiT

    dit = SkyReelsR2VDiT(_TINY_CFG)
    assert type(dit.blocks[0].self_attn.q).__name__ == "Linear"
    cfg = {"bits": 8, "group_size": 64}
    nn.quantize(dit, group_size=cfg["group_size"], bits=cfg["bits"])
    assert type(dit.blocks[0].self_attn.q).__name__ == "QuantizedLinear"
    # 量化后仍可前向 (shape 不变)
    x = mx.random.normal((1, 16, 2, 4, 4))
    t = mx.array([0.5])
    ctx = mx.random.normal((1, 8, 128))
    out = dit(x, t, ctx, [8], [(2, 2, 2)])
    mx.eval(out)
    assert out.shape == (1, 16, 2, 4, 4)
