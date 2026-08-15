# SPDX-License-Identifier: Apache-2.0
# P2 Transformer checkpoint：MiniMaxH3DiTModel 前向形状 + 权重键树匹配。
# 无真实权重时跳过键匹配；前向用 tiny config 验证 packed-sequence 形状正确。
import os

import mlx.core as mx
import pytest

from fusion_mlx.video.minimax_h3.config import H3Config
from fusion_mlx.video.minimax_h3.transformer import (
    AdaLayerNormModulation,
    AdaLayerNormOut,
    Attention,
    FeedForward,
    MiniMaxH3DiTModel,
    RotaryPosEmbed,
    TokenRefiner,
    TransformerBlock,
    apply_rotary_emb,
    get_timestep_embedding,
    reorder_interleaved_qkv,
    scatter_rows,
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


def _packed_layout(num_text, num_video, num_audio, num_cond=0):
    num_text_tokens = num_text
    num_cond_rows = num_cond
    num_audio_rows = num_audio
    num_video_rows = num_video
    seq_len = num_text_tokens + num_cond_rows + num_audio_rows + num_video_rows

    text_indices = mx.arange(0, num_text_tokens)
    cond_start = num_text_tokens
    audio_start = cond_start + num_cond_rows
    video_start = audio_start + num_audio_rows

    if num_cond_rows > 0:
        video_indices = mx.concatenate(
            [mx.arange(cond_start, audio_start), mx.arange(video_start, seq_len)]
        )
    else:
        video_indices = mx.arange(video_start, seq_len)
    audio_indices = mx.arange(audio_start, video_start)

    position_ids = mx.zeros((seq_len, 3), dtype=mx.float32)
    position_ids[:num_text_tokens, 0] = mx.arange(num_text_tokens, dtype=mx.float32)
    pos_t = mx.arange(num_video_rows, dtype=mx.float32) + num_text_tokens
    if num_cond_rows == 0:
        position_ids[video_start:, 0] = pos_t
    token_tags = mx.zeros((seq_len,), dtype=mx.int32)
    token_tags[text_indices] = 1
    token_tags[video_indices] = 0
    token_tags[audio_indices] = 2
    return position_ids, token_tags, video_indices, audio_indices, text_indices, seq_len


class TestRotary:
    def test_rope_output_shape(self):
        rope = RotaryPosEmbed(rope_freq_dim=2)
        position_ids = mx.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
        cos, sin = rope(position_ids)
        assert cos.shape == (1, 2, 2 * 3 * 2)
        assert sin.shape == (1, 2, 2 * 3 * 2)

    def test_rope_zero_position(self):
        rope = RotaryPosEmbed(rope_freq_dim=2)
        position_ids = mx.zeros((1, 3, 3), dtype=mx.float32)
        cos, sin = rope(position_ids)
        cos0 = cos[0, 0]
        assert abs(float(cos0[0]) - 1.0) < 1e-5
        assert abs(float(sin[0, 0][0])) < 1e-5


class TestApplyRotary:
    def test_rotate_preserves_shape(self):
        hidden = mx.ones((1, 5, 4, 16))
        cos = mx.ones((1, 5, 8))
        sin = mx.zeros((1, 5, 8))
        out = apply_rotary_emb(hidden, cos, sin)
        assert out.shape == (1, 5, 4, 16)

    def test_pass_through_channels(self):
        hidden = mx.ones((1, 2, 2, 16))
        cos = mx.ones((1, 2, 4))
        sin = mx.zeros((1, 2, 4))
        out = apply_rotary_emb(hidden, cos, sin)
        assert abs(float(out[0, 0, 0, 12]) - 1.0) < 1e-5


class TestTimestepEmbedding:
    def test_shape(self):
        t = mx.array([0.5, 0.9])
        emb = get_timestep_embedding(
            t, 16, flip_sin_to_cos=True, downscale_freq_shift=0.0
        )
        assert emb.shape == (2, 16)


class TestAttention:
    def test_nonsquare_shapes(self):
        attn = Attention(hidden_size=64, heads=4, head_dim=16)
        assert attn.qkv_proj.weight.shape == (3 * 4 * 16, 64)
        assert attn.out_proj.weight.shape == (64, 4 * 16)
        x = mx.ones((1, 8, 64))
        out = attn(x)
        assert out.shape == (1, 8, 64)

    def test_with_rope(self):
        attn = Attention(hidden_size=64, heads=4, head_dim=16)
        x = mx.ones((1, 6, 64))
        cos = mx.ones((1, 6, 8))
        sin = mx.zeros((1, 6, 8))
        out = attn(x, rotary_emb=(cos, sin))
        assert out.shape == (1, 6, 64)


class TestFeedForward:
    def test_gate_value(self):
        ff = FeedForward(hidden_size=64, ffn_dim=128)
        assert ff.fc1.weight.shape == (2 * 128, 64)
        assert ff.fc2.weight.shape == (64, 128)
        x = mx.ones((1, 5, 64))
        out = ff(x)
        assert out.shape == (1, 5, 64)


class TestAdaLayerNorm:
    def test_modulation_shape(self):
        adaln = AdaLayerNormModulation(time_embed_dim=32, hidden_size=64)
        assert adaln.linear.weight.shape == (6 * 64 * 3, 32)
        temb = mx.ones((2, 32))
        chunks = adaln(temb)
        assert len(chunks) == 6
        for c in chunks:
            assert c.shape == (2 * 3, 64)

    def test_out_shape(self):
        out_norm = AdaLayerNormOut(hidden_size=64, time_embed_dim=32, eps=1e-5)
        assert out_norm.linear.weight.shape == (2 * 64, 32)
        hidden = mx.ones((1, 10, 64))
        temb = mx.ones((2, 32))
        ts_idx = mx.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1], dtype=mx.int32)
        out = out_norm(hidden, temb, ts_idx)
        assert out.shape == (1, 10, 64)


class TestTokenRefiner:
    def test_forward_shape(self):
        refiner = TokenRefiner(
            hidden_size=64,
            num_heads=4,
            head_dim=16,
            ffn_dim=128,
            num_layers=1,
            norm_eps=1e-5,
            qk_norm_eps=1e-5,
            final_norm_eps=1e-5,
        )
        x = mx.ones((1, 8, 64))
        out = refiner(x)
        assert out.shape == (1, 8, 64)


class TestTransformerBlock:
    def test_forward_shape(self):
        block = TransformerBlock(
            hidden_size=64,
            num_heads=4,
            head_dim=16,
            ffn_dim=128,
            time_embed_dim=32,
            norm_eps=1e-5,
            qk_norm_eps=1e-5,
        )
        hidden = mx.ones((1, 12, 64))
        temb = mx.ones((2, 32))
        adaln_idx = mx.array([0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5], dtype=mx.int32)
        cos = mx.ones((1, 12, 8))
        sin = mx.zeros((1, 12, 8))
        out = block(hidden, temb, adaln_idx, (cos, sin))
        assert out.shape == (1, 12, 64)


class TestScatterRows:
    def test_scatter(self):
        buf = mx.zeros((1, 6, 4))
        indices = mx.array([2, 4], dtype=mx.int32)
        values = mx.array([[[1.0, 1.0, 1.0, 1.0], [2.0, 2.0, 2.0, 2.0]]])
        out = scatter_rows(buf, indices, values, axis=1)
        assert abs(float(out[0, 2, 0]) - 1.0) < 1e-5
        assert abs(float(out[0, 4, 0]) - 2.0) < 1e-5
        assert abs(float(out[0, 0, 0])) < 1e-5


class TestDiTForward:
    def test_param_tree_keys(self):
        cfg = _tiny_config()
        model = MiniMaxH3DiTModel(cfg)
        flat = _flatten_for_test(model.parameters())
        assert "video_patch_proj.weight" in flat
        assert "audio_patch_proj.weight" in flat
        assert "condition_proj.weight" in flat
        assert "time_embedder.proj_in.weight" in flat
        assert "time_embedder.proj_out.weight" in flat
        assert "rope.inv_freq" in flat
        assert "token_refiner.blocks.0.norm1.weight" in flat
        assert "token_refiner.final_norm.weight" in flat
        assert "blocks.0.norm1.weight" in flat
        assert "blocks.0.attn.qkv_proj.weight" in flat
        assert "blocks.0.attn.q_norm.weight" in flat
        assert "blocks.0.attn.k_norm.weight" in flat
        assert "blocks.0.attn.out_proj.weight" in flat
        assert "blocks.0.mlp.fc1.weight" in flat
        assert "blocks.0.mlp.fc2.weight" in flat
        assert "blocks.0.adaln_proj.linear.weight" in flat
        assert "final_layer.norm.weight" in flat
        assert "final_layer.adaln_proj.linear.weight" in flat
        assert "final_layer.video_out.weight" in flat
        assert "final_layer.audio_out.weight" in flat

    def test_param_shapes(self):
        cfg = _tiny_config()
        model = MiniMaxH3DiTModel(cfg)
        flat = _flatten_for_test(model.parameters())
        video_patch_dim = cfg.latents_dim * 1 * 2 * 2
        assert flat["video_patch_proj.weight"].shape == (cfg.dim, video_patch_dim)
        assert flat["audio_patch_proj.weight"].shape == (cfg.dim, cfg.audio_latents_dim)
        assert flat["condition_proj.weight"].shape == (cfg.dim, cfg.text_dim)
        assert flat["time_embedder.proj_in.weight"].shape == (
            cfg.time_embed_hidden,
            cfg.timestep_input_dim,
        )
        assert flat["time_embedder.proj_out.weight"].shape == (
            cfg.time_embed_dim,
            cfg.time_embed_hidden,
        )
        inner = cfg.num_heads * cfg.head_dim
        assert flat["blocks.0.attn.qkv_proj.weight"].shape == (3 * inner, cfg.dim)
        assert flat["blocks.0.attn.out_proj.weight"].shape == (cfg.dim, inner)
        assert flat["blocks.0.mlp.fc1.weight"].shape == (2 * cfg.ffn_dim, cfg.dim)
        assert flat["blocks.0.mlp.fc2.weight"].shape == (cfg.dim, cfg.ffn_dim)
        assert flat["blocks.0.adaln_proj.linear.weight"].shape == (
            6 * cfg.dim * 3,
            cfg.time_embed_dim,
        )
        assert flat["final_layer.adaln_proj.linear.weight"].shape == (
            2 * cfg.dim,
            cfg.time_embed_dim,
        )
        assert flat["final_layer.video_out.weight"].shape == (video_patch_dim, cfg.dim)
        assert flat["final_layer.audio_out.weight"].shape == (
            cfg.audio_latents_dim,
            cfg.dim,
        )

    def test_forward_shape(self):
        cfg = _tiny_config()
        model = MiniMaxH3DiTModel(cfg)
        num_text, num_video, num_audio = 6, 8, 4
        (
            position_ids,
            token_tags,
            video_indices,
            audio_indices,
            text_indices,
            seq_len,
        ) = _packed_layout(num_text, num_video, num_audio)

        video_patch_dim = cfg.latents_dim * 1 * 2 * 2
        hidden_states = mx.ones((1, num_video, video_patch_dim))
        audio_hidden_states = mx.ones((1, num_audio, cfg.audio_latents_dim))
        encoder_hidden_states = mx.ones((1, num_text, cfg.text_dim))
        timestep = mx.array([0.9, 0.5], dtype=mx.float32)
        timestep_indices = mx.array(
            [0] * num_text + [1] * num_audio + [0] * num_video,
            dtype=mx.int32,
        )

        video_out, audio_out = model(
            hidden_states,
            audio_hidden_states,
            encoder_hidden_states,
            timestep,
            timestep_indices,
            token_tags,
            position_ids,
            video_indices,
            audio_indices,
            text_indices,
        )
        assert video_out.shape == (1, num_video, video_patch_dim)
        assert audio_out.shape == (1, num_audio, cfg.audio_latents_dim)

    def test_forward_finite(self):
        cfg = _tiny_config()
        model = MiniMaxH3DiTModel(cfg)
        num_text, num_video, num_audio = 4, 6, 3
        (
            position_ids,
            token_tags,
            video_indices,
            audio_indices,
            text_indices,
            seq_len,
        ) = _packed_layout(num_text, num_video, num_audio)
        video_patch_dim = cfg.latents_dim * 1 * 2 * 2
        hidden_states = mx.random.normal((1, num_video, video_patch_dim))
        audio_hidden_states = mx.random.normal((1, num_audio, cfg.audio_latents_dim))
        encoder_hidden_states = mx.random.normal((1, num_text, cfg.text_dim))
        timestep = mx.array([0.8], dtype=mx.float32)
        timestep_indices = mx.zeros((seq_len,), dtype=mx.int32)
        video_out, audio_out = model(
            hidden_states,
            audio_hidden_states,
            encoder_hidden_states,
            timestep,
            timestep_indices,
            token_tags,
            position_ids,
            video_indices,
            audio_indices,
            text_indices,
        )
        assert mx.all(mx.isfinite(video_out)), "video output has nan/inf"
        assert mx.all(mx.isfinite(audio_out)), "audio output has nan/inf"


class TestReorderQKV:
    def test_deinterleave(self):
        num_heads, head_dim = 3, 4
        total = num_heads * 3 * head_dim
        weight = mx.arange(total).reshape(total, 1).astype(mx.float32)
        reordered = reorder_interleaved_qkv(weight, num_heads, head_dim)
        assert reordered.shape == (total, 1)
        q_rows = reordered[: num_heads * head_dim]
        expected_q = mx.array(
            [0, 1, 2, 3, 12, 13, 14, 15, 24, 25, 26, 27], dtype=mx.float32
        )
        assert mx.all(q_rows.reshape(-1) == expected_q)


class TestKeySetMatch:
    def test_key_match_skips_without_weights(self):
        model_dir = os.path.expanduser(
            "~/.fusion-mlx/models/minimax-h3/FL2VA/transformer"
        )
        if not os.path.isdir(model_dir) or not any(
            f.endswith(".safetensors") for f in os.listdir(model_dir)
        ):
            pytest.skip("no H3 transformer weights downloaded")
        from fusion_mlx.video.minimax_h3.transformer import load_dit_from_pretrained

        model = load_dit_from_pretrained(model_dir, config=H3Config.fl2va())
        flat = _flatten_for_test(model.parameters())
        assert "blocks.0.attn.qkv_proj.weight" in flat


def _flatten_for_test(params, prefix=""):
    out = {}
    if isinstance(params, dict):
        for k, v in params.items():
            key = f"{prefix}.{k}" if prefix else k
            out.update(_flatten_for_test(v, key))
    elif isinstance(params, mx.array):
        out[prefix] = params
    elif isinstance(params, list):
        for i, v in enumerate(params):
            out.update(_flatten_for_test(v, f"{prefix}.{i}"))
    return out
