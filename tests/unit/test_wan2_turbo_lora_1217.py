# SPDX-License-Identifier: Apache-2.0
# Tests for issue #1217: Wan2.2 few-step turbo LoRA on the TI2V-5B LoRA
# path. The community turbo artifact (hum-ma Wan22_TI2V_5B_Turbo_lora_rank_64)
# ships the ComfyUI extraction format: lora_down/lora_up pairs PLUS direct
# ``diff_b`` (bias) and ``.diff`` (norm weight) deltas under the
# ``diffusion_model.`` prefix with ``ffn.0/ffn.2`` naming — the loader and
# both apply paths must consume all of them, and the backend must resolve
# the request knob + apply the few-step preset (steps=4, cfg=1).

import mlx.core as mx
import mlx.nn as nn
import pytest

pytest.importorskip("mlx")


def _write_turbo_style_lora(path, dim=64, rank=4, ffn_dim=128):
    # Minimal ComfyUI-format turbo LoRA: prefix + ffn.0/ffn.2 naming,
    # lora pairs, diff_b bias deltas, norm_q/norm_k/norm3 weight diffs.
    from safetensors.numpy import save_file

    tensors = {
        # Linear module with lora pair + bias delta
        "diffusion_model.blocks.0.self_attn.q.lora_down.weight": mx.zeros((rank, dim))
        + 0.5,
        "diffusion_model.blocks.0.self_attn.q.lora_up.weight": mx.zeros((dim, rank))
        + 0.25,
        "diffusion_model.blocks.0.self_attn.q.diff_b": mx.zeros((dim,)) + 0.125,
        # ffn module under ComfyUI numbering
        "diffusion_model.blocks.0.ffn.0.lora_down.weight": mx.zeros((rank, dim)) + 0.5,
        "diffusion_model.blocks.0.ffn.0.lora_up.weight": mx.zeros((ffn_dim, rank))
        + 0.25,
        "diffusion_model.blocks.0.ffn.0.diff_b": mx.zeros((ffn_dim,)) + 0.125,
        # Norm-only modules (no low-rank pair)
        "diffusion_model.blocks.0.self_attn.norm_q.diff": mx.zeros((dim,)) + 0.0625,
        "diffusion_model.blocks.0.norm3.diff": mx.zeros((dim,)) + 0.0625,
        "diffusion_model.blocks.0.norm3.diff_b": mx.zeros((dim,)) + 0.03125,
    }
    import numpy as np

    save_file(
        {k: np.asarray(v, dtype=np.float32) for k, v in tensors.items()}, str(path)
    )
    return str(path)


def test_loader_parses_comfyui_turbo_format(tmp_path):
    from fusion_mlx.video.wan2.lora import load_lora_weights

    lora_file = tmp_path / "turbo.safetensors"
    _write_turbo_style_lora(lora_file)
    lw = load_lora_weights(lora_file)

    # 2 lora-pair modules + 2 diff-only modules
    pairs = {k for k, v in lw.items() if v.lora_A is not None}
    diffs = {k for k, v in lw.items() if v.lora_A is None}
    assert pairs == {
        "diffusion_model.blocks.0.self_attn.q",
        "diffusion_model.blocks.0.ffn.0",
    }
    assert "diffusion_model.blocks.0.self_attn.norm_q" in diffs
    assert "diffusion_model.blocks.0.norm3" in diffs
    # bias delta attached to the Linear module
    q = lw["diffusion_model.blocks.0.self_attn.q"]
    assert q.bias_diff is not None
    assert float(q.bias_diff.max()) == pytest.approx(0.125)
    # norm module carries weight + bias diffs, no pair
    n3 = lw["diffusion_model.blocks.0.norm3"]
    assert n3.lora_A is None and n3.lora_B is None
    assert n3.weight_diff is not None and n3.bias_diff is not None
    assert n3.scale == 0.0  # rank-0 module must not divide by zero


def test_apply_loras_to_weights_merges_pairs_bias_and_norm(tmp_path):
    from fusion_mlx.video.wan2.lora import apply_loras_to_weights, load_lora_weights

    lora_file = tmp_path / "turbo.safetensors"
    _write_turbo_style_lora(lora_file)
    lw = load_lora_weights(lora_file)
    module_to_loras = {k: [(v, 1.0)] for k, v in lw.items()}

    dim, ffn_dim, group = 64, 128, 64
    lin_w = mx.random.normal((dim, dim))
    q, s, b = mx.quantize(lin_w, group_size=group, bits=8)
    base = {
        "blocks.0.self_attn.q.weight": q,
        "blocks.0.self_attn.q.scales": s,
        "blocks.0.self_attn.q.biases": b,
        "blocks.0.self_attn.q.bias": mx.zeros((dim,)),
        "blocks.0.ffn.fc1.weight": mx.random.normal((ffn_dim, dim)),
        "blocks.0.ffn.fc1.bias": mx.zeros((ffn_dim,)),
        "blocks.0.self_attn.norm_q.weight": mx.ones((dim,)),
        "blocks.0.norm3.weight": mx.ones((dim,)),
        "blocks.0.norm3.bias": mx.zeros((dim,)),
    }

    merged = apply_loras_to_weights(base, module_to_loras, quantization_bits=8)

    # Linear weight moved (dequant + lora delta + requant)
    assert not mx.array_equal(
        base["blocks.0.self_attn.q.weight"], merged["blocks.0.self_attn.q.weight"]
    )
    # ffn.0 remapped onto ffn.fc1
    assert not mx.array_equal(
        base["blocks.0.ffn.fc1.weight"], merged["blocks.0.ffn.fc1.weight"]
    )
    # bias deltas applied
    assert float(merged["blocks.0.self_attn.q.bias"].max()) == pytest.approx(0.125)
    assert float(merged["blocks.0.ffn.fc1.bias"].max()) == pytest.approx(0.125)
    # norm weight delta applied (fp32 target: 1.0 + 0.0625)
    assert float(merged["blocks.0.self_attn.norm_q.weight"].max()) == pytest.approx(
        1.0625
    )
    assert float(merged["blocks.0.norm3.weight"].max()) == pytest.approx(1.0625)
    assert float(merged["blocks.0.norm3.bias"].max()) == pytest.approx(0.03125)
    # no NaN anywhere
    for v in merged.values():
        assert not mx.isnan(v.astype(mx.float32)).any()


def test_apply_loras_to_model_handles_norm_and_quantized_bias(tmp_path):
    from fusion_mlx.video.wan2.lora import apply_loras_to_model, load_lora_weights

    lora_file = tmp_path / "turbo.safetensors"
    _write_turbo_style_lora(lora_file)
    lw = load_lora_weights(lora_file)
    module_to_loras = {k: [(v, 1.0)] for k, v in lw.items()}

    dim = 64

    class Block(nn.Module):
        def __init__(self):
            class Attn(nn.Module):
                def __init__(self):
                    self.q = nn.Linear(dim, dim, bias=True)
                    self.norm_q = nn.RMSNorm(dim)

                def __call__(self, x):
                    return self.norm_q(self.q(x))

            self.self_attn = Attn()
            self.norm3 = nn.LayerNorm(dim)

        def __call__(self, x):
            return self.norm3(self.self_attn(x))

    class FakeModel(nn.Module):
        def __init__(self):
            self.blocks = [Block()]

        def __call__(self, x):
            return self.blocks[0](x)

    model = FakeModel()
    mx.eval(model.parameters())
    q_weight = model.blocks[0].self_attn.q.weight
    q_bias = model.blocks[0].self_attn.q.bias
    nq_weight = model.blocks[0].self_attn.norm_q.weight
    n3_weight = model.blocks[0].norm3.weight
    n3_bias = model.blocks[0].norm3.bias

    applied = apply_loras_to_model(model, module_to_loras, verbose=False)
    mx.eval(model.parameters())
    # ffn.0 target absent in the fake model -> skipped; q + norms applied
    assert applied >= 3
    assert not mx.array_equal(q_weight, model.blocks[0].self_attn.q.weight)
    assert not mx.array_equal(q_bias, model.blocks[0].self_attn.q.bias)
    assert not mx.array_equal(nq_weight, model.blocks[0].self_attn.norm_q.weight)
    assert not mx.array_equal(n3_weight, model.blocks[0].norm3.weight)
    assert not mx.array_equal(n3_bias, model.blocks[0].norm3.bias)
    assert float(model.blocks[0].norm3.bias.max()) == pytest.approx(0.03125)


def test_resolve_turbo_lora(tmp_path, monkeypatch):
    from fusion_mlx.engines.video_backends import wan2 as wan2_backend

    # None / False -> off
    assert wan2_backend._resolve_turbo_lora(None, None) is None
    assert wan2_backend._resolve_turbo_lora(False, None) is None
    # True with no candidates anywhere -> fail visibly
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()
    monkeypatch.setattr(
        wan2_backend, "_TURBO_LORA_FALLBACK_DIR", str(tmp_path / "nofallback")
    )
    with pytest.raises(ValueError, match="lora=True"):
        wan2_backend._resolve_turbo_lora(True, str(empty_dir))
    # True auto-detects next to the model
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    turbo = model_dir / "turbo_lora.safetensors"
    turbo.write_bytes(b"x")
    assert wan2_backend._resolve_turbo_lora(True, str(model_dir)) == [(str(turbo), 1.0)]
    # str path wins; missing file fails visibly
    assert wan2_backend._resolve_turbo_lora(str(turbo), None) == [(str(turbo), 1.0)]
    with pytest.raises(FileNotFoundError):
        wan2_backend._resolve_turbo_lora(str(tmp_path / "missing.safetensors"), None)


def test_wan2_generate_turbo_preset_and_clamp(tmp_path, monkeypatch):
    import asyncio

    from fusion_mlx.engines.video_backends import wan2 as wan2_backend
    from fusion_mlx.engines.video_backends.base import VideoGenParams

    captured: dict = {}

    def fake_generate_one(model_dir, **kwargs):
        captured.update(kwargs)
        return b"fake"

    monkeypatch.setattr(wan2_backend, "_generate_one", fake_generate_one)

    async def run(lora=None, steps=None, guide_scale=None):
        captured.clear()
        backend = wan2_backend.Wan2Backend("fake-model")
        backend._loaded = True
        backend._model_dir = str(tmp_path)
        params = VideoGenParams(
            prompt="p",
            num_frames=81,
            width=704,
            height=1280,
            num_inference_steps=steps,
            guide_scale=guide_scale,
            lora=lora,
        )
        out = await backend.generate(params)
        assert out == [b"fake"]
        return captured

    resolved = [("/fake/turbo.safetensors", 1.0)]
    monkeypatch.setattr(
        wan2_backend, "_resolve_turbo_lora", lambda lora, md: resolved if lora else None
    )

    # Preset: unset steps -> 4, unset guide_scale -> 1.0
    cap = asyncio.run(run(lora=True))
    assert cap["steps"] == 4
    assert cap["guide_scale"] == 1.0
    assert cap["loras"] == resolved
    # Explicit values are honored (6 is inside the useful range)
    cap = asyncio.run(run(lora=True, steps=6, guide_scale=1.5))
    assert cap["steps"] == 6
    assert cap["guide_scale"] == 1.5
    # Out-of-range steps clamped to 8
    cap = asyncio.run(run(lora=True, steps=20))
    assert cap["steps"] == 8
    # No lora -> loras off, no preset
    cap = asyncio.run(run())
    assert cap["loras"] is None
    assert cap["steps"] is None
    assert cap["guide_scale"] is None
