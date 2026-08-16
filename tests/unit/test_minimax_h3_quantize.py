# SPDX-License-Identifier: Apache-2.0
# 运行时量化单元测试：验证 quantize_dit / quantize_text_encoder 的 predicate 跳过逻辑。
# 不依赖真实权重，用合成 mlx.nn.Module 验证哪些层被量化、哪些保持原精度。

import mlx.core as mx
import mlx.nn as nn

from fusion_mlx.video.minimax_h3.quantize import (
    _DIT_KEEP_PREFIXES,
    _dit_predicate,
    _te_predicate,
    quantize_dit,
    quantize_text_encoder,
)


class _TinyDiT(nn.Module):
    # 模拟 MiniMaxH3DiTModel 关键子模块的命名结构。
    def __init__(self):
        super().__init__()
        self.time_embedder_proj = nn.Linear(64, 64)
        self.video_patch_proj = nn.Linear(64, 64)
        self.rope_inv_freq = nn.Linear(64, 64)
        self.final_layer_video_out = nn.Linear(64, 64)
        self.condition_proj = nn.Linear(64, 64)
        self.blocks_0_adaln_proj_linear = nn.Linear(64, 64)
        self.blocks_0_attn_qkv_proj = nn.Linear(64, 64)
        self.blocks_0_mlp_fc1 = nn.Linear(64, 64)
        self.embed = nn.Embedding(128, 64)


class _TinyTE(nn.Module):
    # 模拟 Qwen3-VL language_model 命名结构。
    def __init__(self):
        super().__init__()
        self.model = nn.Module()
        self.model.embed_tokens = nn.Embedding(128, 64)
        self.model.layers_0_self_attn_q_proj = nn.Linear(64, 64)
        self.model.layers_0_mlp_gate_proj = nn.Linear(64, 64)


class TestDitPredicate:
    def test_keep_prefixes_skipped(self):
        # keep 前缀内的 Linear 不应被量化。
        lin = nn.Linear(8, 8)
        for prefix in _DIT_KEEP_PREFIXES:
            assert _dit_predicate(prefix, lin) is False, f"{prefix} should be kept"

    def test_quantize_targets_accepted(self):
        # adaln / attn / mlp 的 Linear 应被量化。
        lin = nn.Linear(8, 8)
        for path in (
            "blocks.0.adaln_proj.linear",
            "blocks.0.attn.qkv_proj",
            "blocks.0.attn.out_proj",
            "blocks.0.mlp.fc1",
            "blocks.0.mlp.fc2",
        ):
            assert _dit_predicate(path, lin) is True, f"{path} should be quantized"

    def test_embedding_skipped(self):
        emb = nn.Embedding(16, 8)
        assert _dit_predicate("blocks.0.embed", emb) is False

    def test_norm_skipped(self):
        # RMSNorm/LayerNorm 无 to_quantized，天然跳过。
        norm = nn.LayerNorm(8)
        assert _dit_predicate("blocks.0.norm1", norm) is False


class TestTePredicate:
    def test_embed_tokens_skipped(self):
        emb = nn.Embedding(16, 8)
        for path in (
            "model.embed_tokens",
            "language_model.embed_tokens",
            "embed_tokens",
        ):
            assert _te_predicate(path, emb) is False

    def test_attn_mlp_quantized(self):
        lin = nn.Linear(8, 8)
        assert _te_predicate("model.layers.0.self_attn.q_proj", lin) is True
        assert _te_predicate("model.layers.0.mlp.gate_proj", lin) is True


class TestQuantizeInPlace:
    def test_quantize_dit_reduces_bytes_and_keeps_output_finite(self):
        from mlx.utils import tree_flatten

        dit = _TinyDiT()
        before = sum(v.size * v.itemsize for _, v in tree_flatten(dit.parameters()))
        quantize_dit(dit, bits=8, group_size=32)
        after = sum(
            v.size * v.itemsize
            for _, v in tree_flatten(dit.parameters())
            if hasattr(v, "itemsize")
        )
        # 量化后总字节数应下降（大 Linear 被压缩，keep 层保持 bf16）。
        assert after < before
        # time_embedder 等保持原 Linear（未被替换为 QuantizedLinear）。
        assert type(dit.time_embedder_proj).__name__ == "Linear"
        # 量化后的 adaln 输出仍 finite。
        x = mx.zeros((1, 64))
        out = dit.blocks_0_adaln_proj_linear(x)
        mx.eval(out)
        assert mx.all(mx.isfinite(out)).item() is True

    def test_quantize_text_encoder_keeps_embed(self):
        te = _TinyTE()
        quantize_text_encoder(te, bits=4, group_size=32)
        # embed_tokens 保持 Embedding（未被量化）。
        assert type(te.model.embed_tokens).__name__ == "Embedding"
        # attn Linear 应被量化。
        assert type(te.model.layers_0_self_attn_q_proj).__name__ != "Linear"
