# SPDX-License-Identifier: Apache-2.0
# #1212 Turbo LoRA：apply_lora 附加、qkv 行置换、完备性校验、旁路数学。
import mlx.core as mx
import pytest

from fusion_mlx.video.minimax_h3.config import H3Config
from fusion_mlx.video.minimax_h3.transformer import (
    LoRALinear,
    MiniMaxH3DiTModel,
    apply_lora,
    reorder_interleaved_qkv,
)


def _tiny_config():
    cfg = H3Config()
    cfg.dim = 64
    cfg.num_layers = 2
    cfg.token_refiner_layers = 1
    cfg.num_heads = 4
    cfg.head_dim = 16
    cfg.ffn_dim = 128
    cfg.latents_dim = 4
    cfg.audio_latents_dim = 6
    cfg.patch_size = (1, 2, 2)
    cfg.text_dim = 32
    cfg.timestep_input_dim = 16
    cfg.time_embed_hidden = 64
    cfg.time_embed_dim = 32
    cfg.adaln_out = 6 * 64 * 3
    cfg.final_adaln_out = 2 * 64
    cfg.rope_inv_freq_len = 2
    return cfg


def _module_paths(cfg):
    # 与 larryvrh Turbo LoRA 同构的完整目标集（tiny 深度）。
    per_block = [
        "attn.qkv_proj",
        "attn.out_proj",
        "mlp.fc1",
        "mlp.fc2",
        "adaln_proj.linear",
    ]
    paths = []
    for i in range(cfg.num_layers):
        for suffix in per_block:
            paths.append(f"blocks.{i}.{suffix}")
    for i in range(cfg.token_refiner_layers):
        for suffix in per_block[:4]:
            paths.append(f"token_refiner.blocks.{i}.{suffix}")
    paths.append("final_layer.adaln_proj.linear")
    return paths


def _module_shapes(model, cfg):
    shapes = {}
    for path in _module_paths(cfg):
        parts = path.split(".")
        obj = model
        for part in parts:
            obj = obj[int(part)] if part.isdigit() else getattr(obj, part)
        shapes[path] = (obj.weight.shape[1], obj.weight.shape[0])
    return shapes


def _write_lora(tmp_path, model, cfg, rank=4, poison=None, drop=None):
    shapes = _module_shapes(model, cfg)
    tensors = {}
    for path, (in_dim, out_dim) in shapes.items():
        if drop and path in drop:
            continue
        a = mx.random.normal((rank, in_dim)) * 0.1
        b = mx.random.normal((out_dim, rank)) * 0.1
        if poison and path in poison:
            # 行序按原始 per-head 交错源（qkv 附加时会做行置换）。
            pass
        tensors[f"{path}.lora_A.weight"] = a.astype(mx.bfloat16)
        tensors[f"{path}.lora_B.weight"] = b.astype(mx.bfloat16)
    out = tmp_path / "lora.safetensors"
    mx.save_safetensors(str(out), tensors)
    return str(out), tensors


class TestApplyLora:
    def test_attaches_every_module(self, tmp_path):
        cfg = _tiny_config()
        model = MiniMaxH3DiTModel(cfg)
        lora_path, _ = _write_lora(tmp_path, model, cfg)
        attached = apply_lora(model, lora_path)
        expected = len(_module_paths(cfg))
        assert attached == expected
        # 目标模块都被 LoRALinear 包裹。
        for path in _module_paths(cfg):
            parts = path.split(".")
            obj = model
            for part in parts:
                obj = obj[int(part)] if part.isdigit() else getattr(obj, part)
            assert isinstance(obj, LoRALinear), path

    def test_sidepath_math(self, tmp_path):
        # y = base(x) + scale * (x @ A^T) @ B^T（W_eff = W + B @ A 的前向等价）。
        cfg = _tiny_config()
        model = MiniMaxH3DiTModel(cfg)
        lora_path, tensors = _write_lora(tmp_path, model, cfg)
        base = model.blocks[0].mlp.fc1
        w = base.weight
        x = mx.random.normal((2, 5, w.shape[1]))
        y_base = base(x)
        apply_lora(model, lora_path)
        wrapped = model.blocks[0].mlp.fc1
        y = wrapped(x)
        a = tensors["blocks.0.mlp.fc1.lora_A.weight"].astype(mx.float32)
        b = tensors["blocks.0.mlp.fc1.lora_B.weight"].astype(mx.float32)
        expected = y_base + (x @ a.T) @ b.T
        mx.eval(y, expected)
        assert mx.allclose(y, expected, atol=1e-4)

    def test_scale_scales_delta(self, tmp_path):
        cfg = _tiny_config()
        model = MiniMaxH3DiTModel(cfg)
        lora_path, _ = _write_lora(tmp_path, model, cfg)
        apply_lora(model, lora_path, scale=0.0)
        wrapped = model.blocks[0].mlp.fc1
        assert wrapped.scale == 0.0
        # scale=0 -> 退化为基座输出。
        w = wrapped.base.weight
        x = mx.random.normal((2, 5, w.shape[1]))
        y = wrapped(x)
        expected = wrapped.base(x)
        mx.eval(y, expected)
        assert mx.allclose(y, expected, atol=1e-5)

    def test_incomplete_pair_raises(self, tmp_path):
        # 只有 lora_A 没有 lora_B 的目标 = 半挂载 Turbo，必须报错。
        cfg = _tiny_config()
        model = MiniMaxH3DiTModel(cfg)
        shapes = _module_shapes(model, cfg)
        tensors = {}
        for path, (in_dim, out_dim) in shapes.items():
            tensors[f"{path}.lora_A.weight"] = (
                mx.random.normal((4, in_dim)) * 0.1
            ).astype(mx.bfloat16)
        # 单独丢一个 B（其余 B 行数 = out_dim）。
        drop = "blocks.1.attn.out_proj"
        for path in shapes:
            if path != drop:
                tensors[f"{path}.lora_B.weight"] = (
                    mx.random.normal((shapes[path][1], 4)) * 0.1
                ).astype(mx.bfloat16)
        out = tmp_path / "half.safetensors"
        mx.save_safetensors(str(out), tensors)
        with pytest.raises(ValueError, match="failed to attach"):
            apply_lora(model, str(out))

    def test_unknown_module_raises(self, tmp_path):
        # 目标模块树里不存在的键（错版 LoRA）→ 完备性校验报错。
        cfg = _tiny_config()
        model = MiniMaxH3DiTModel(cfg)
        tensors = {
            "blocks.99.attn.qkv_proj.lora_A.weight": (
                mx.random.normal((4, 64)) * 0.1
            ).astype(mx.bfloat16),
            "blocks.99.attn.qkv_proj.lora_B.weight": (
                mx.random.normal((192, 4)) * 0.1
            ).astype(mx.bfloat16),
        }
        out = tmp_path / "bad.safetensors"
        mx.save_safetensors(str(out), tensors)
        with pytest.raises(ValueError, match="failed to attach"):
            apply_lora(model, str(out))

    def test_qkv_lora_b_row_permutation(self):
        # qkv lora_B 行序为原始 per-head 交错（与原始 checkpoint 同源），
        # apply_lora 内部对齐 _remap_transformer_weights 的行置换。
        # 用行标识值验证置换：B[r,:] = r，置换后读出原行号。
        num_heads, head_dim = 2, 2
        rows = num_heads * 3 * head_dim
        b = mx.arange(rows, dtype=mx.float32).reshape(rows, 1) @ mx.ones((1, 3))
        out = reorder_interleaved_qkv(b, num_heads, head_dim)
        got = out[:, 0].tolist()
        # 交错 [h0:q,k,v | h1:q,k,v] -> [q_all, k_all, v_all]
        # h0 q=[0,1] k=[2,3] v=[4,5]; h1 q=[6,7] k=[8,9] v=[10,11]
        assert got == [0, 1, 6, 7, 2, 3, 8, 9, 4, 5, 10, 11]

    def test_forward_runs_with_lora(self, tmp_path):
        # 附加后整个 DiT 前向可跑（packed 形状不变，旁路不破坏图）。
        cfg = _tiny_config()
        model = MiniMaxH3DiTModel(cfg)
        lora_path, _ = _write_lora(tmp_path, model, cfg)
        apply_lora(model, lora_path)
        num_text, num_video, num_audio = 6, 8, 4
        seq_len = num_text + num_audio + num_video
        video_indices = mx.arange(num_text + num_audio, seq_len)
        audio_indices = mx.arange(num_text, num_text + num_audio)
        text_indices = mx.arange(0, num_text)
        position_ids = mx.zeros((seq_len, 3))
        token_tags = mx.zeros((seq_len,), dtype=mx.int32)
        token_tags[text_indices] = 1
        token_tags[audio_indices] = 2
        timestep_indices = mx.zeros((seq_len,), dtype=mx.int32)
        video = mx.random.normal((1, num_video, cfg.latents_dim * 2 * 2))
        audio = mx.random.normal((1, num_audio, cfg.audio_latents_dim))
        text = mx.random.normal((1, num_text, cfg.text_dim))
        timestep = mx.array([0.8], dtype=mx.float32)
        video_out, audio_out = model(
            video,
            audio,
            text,
            timestep,
            timestep_indices,
            token_tags,
            position_ids,
            video_indices,
            audio_indices,
            text_indices,
        )
        mx.eval(video_out, audio_out)
        assert video_out.shape == (1, num_video, cfg.latents_dim * 2 * 2)
        assert mx.all(mx.isfinite(video_out))
