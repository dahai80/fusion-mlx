# SPDX-License-Identifier: Apache-2.0
# MiniMax H3 Omni-Transformer 纯 MLX 移植：packed-sequence 联合视频+音频 DiT。
# 源码（权威）：/tmp/h3src/transformer_minimax_h3.py（diffusers GitHub main）
# 权重键树对齐开源 checkpoint（/FL2VA/transformer/*.safetensors），from_pretrained
# 仅做两处变换：
#   - attn.qkv_proj：原始 checkpoint 行为 per-head 交错 [h0:q,k,v,h1:q,k,v,...]，
#     加载时 deinterleave 成 [q_all;k_all;v_all]（reorder_interleaved_qkv，无转置）。
#   - mlp.fc1：原始 checkpoint 为 [gate;value]，forward 用 silu(gate)*value，保持顺序不交换。
#
# 关键架构细节（逐条对齐 diffusers 源码 + convert.py）：
#   - 非方阵 attention：inner_dim=7168(56*128) > hidden=5376。
#     qkv: 5376→7168*3；out_proj: 7168→5376。
#   - 无 cross-attention：文本/视频/音频 scatter 进单一 packed 序列，全自注意力 is_causal=False 无 mask。
#   - AdaLN 3-modality 表：MINIMAX_H3_MODALITY_NUM=3(0=video,1=text,2=audio)，
#     adaln_indices = timestep_indices*3 + token_tags，adaln_proj.linear [96768,2688]=6*5376*3。
#   - RoPE：inv_freq[16]=1/theta^(arange(0,32,2)/32)，3 轴共享；
#     cos/sin [seq,2*3*16=96]，旋转 head_dim=128 的前 96 通道，后 32 通道直通。
#   - final_layer.norm 用 timestep_indices（非 adaln_indices），每 timestep 一行，不拆 modality。
#   - time_embedder：sinusoidal Timesteps(flip_sin_to_cos=True,downscale_freq_shift=0)
#     → proj_out(silu(proj_in(x))) → [num_timesteps,2688]。
#   - token_refiner：pre-norm 普通 transformer block，无 AdaLN 无 RoPE。
import glob
import logging
import math
import os

import mlx.core as mx
import mlx.nn as nn

from .config import H3Config

logger = logging.getLogger(__name__)

MINIMAX_H3_MODALITY_NUM = 3


def _silu(x):
    return x * mx.sigmoid(x)


def _rms_norm(x, weight, eps):
    orig_dtype = x.dtype
    x = x.astype(mx.float32)
    var = mx.mean(x * x, axis=-1, keepdims=True)
    x = x * mx.rsqrt(var + eps)
    return (weight.astype(mx.float32) * x).astype(orig_dtype)


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.weight = mx.ones((dim,))
        self.eps = eps

    def __call__(self, x):
        return _rms_norm(x, self.weight, self.eps)


def get_timestep_embedding(
    timesteps,
    embedding_dim,
    flip_sin_to_cos=True,
    downscale_freq_shift=0.0,
    scale=1.0,
    max_period=10000,
):
    half_dim = embedding_dim // 2
    exponent = -math.log(max_period) * mx.arange(0, half_dim, dtype=mx.float32)
    exponent = exponent / (half_dim - downscale_freq_shift)
    emb = mx.exp(exponent)
    emb = timesteps.reshape(-1, 1).astype(mx.float32) * emb.reshape(1, -1)
    emb = scale * emb
    emb = mx.concatenate([mx.sin(emb), mx.cos(emb)], axis=-1)
    if flip_sin_to_cos:
        emb = mx.concatenate([emb[:, half_dim:], emb[:, :half_dim]], axis=-1)
    return emb


class TimestepEmbedder(nn.Module):
    def __init__(self, freq_dim, hidden_dim, out_dim):
        super().__init__()
        self.proj_in = nn.Linear(freq_dim, hidden_dim, bias=True)
        self.proj_out = nn.Linear(hidden_dim, out_dim, bias=True)

    def __call__(self, temb):
        return self.proj_out(_silu(self.proj_in(temb)))


class RotaryPosEmbed(nn.Module):
    def __init__(self, rope_freq_dim=16, rope_theta=10000.0):
        super().__init__()
        self.rope_freq_dim = rope_freq_dim
        inv_freq = 1.0 / (
            rope_theta
            ** (
                mx.arange(0, 2 * rope_freq_dim, 2, dtype=mx.float32)
                / (2 * rope_freq_dim)
            )
        )
        self.inv_freq = inv_freq

    def __call__(self, position_ids):
        position_ids = position_ids.astype(mx.float32)
        freqs = position_ids.reshape(1, -1, 3, 1) * self.inv_freq.reshape(1, 1, 1, -1)
        freqs_t = freqs[:, :, 0, :]
        freqs_h = freqs[:, :, 1, :]
        freqs_w = freqs[:, :, 2, :]
        freqs = mx.concatenate([freqs_t, freqs_h, freqs_w], axis=-1)
        freqs = mx.concatenate([freqs, freqs], axis=-1)
        return freqs.cos(), freqs.sin()


def apply_rotary_emb(hidden_states, cos, sin):
    rotary_dim = cos.shape[-1]
    x_rot = hidden_states[..., :rotary_dim]
    x_pass = hidden_states[..., rotary_dim:]
    cos = cos.reshape(1, cos.shape[1], 1, cos.shape[2])
    sin = sin.reshape(1, sin.shape[1], 1, sin.shape[2])
    x1, x2 = mx.split(x_rot, 2, axis=-1)
    x_rotated = mx.concatenate([-x2, x1], axis=-1)
    x_rot = x_rot * cos + x_rotated * sin
    return mx.concatenate([x_rot, x_pass], axis=-1)


class Attention(nn.Module):
    def __init__(self, hidden_size, heads, head_dim, qk_norm_eps=1e-5):
        super().__init__()
        self.heads = heads
        self.head_dim = head_dim
        self.inner_dim = heads * head_dim
        self.scale = head_dim**-0.5
        self.qkv_proj = nn.Linear(hidden_size, 3 * self.inner_dim, bias=False)
        self.q_norm = RMSNorm(head_dim, eps=qk_norm_eps)
        self.k_norm = RMSNorm(head_dim, eps=qk_norm_eps)
        self.out_proj = nn.Linear(self.inner_dim, hidden_size, bias=False)

    def __call__(self, hidden_states, rotary_emb=None):
        B, L, _ = hidden_states.shape
        qkv = self.qkv_proj(hidden_states)
        q, k, v = mx.split(qkv, 3, axis=-1)
        q = q.reshape(B, L, self.heads, self.head_dim)
        k = k.reshape(B, L, self.heads, self.head_dim)
        v = v.reshape(B, L, self.heads, self.head_dim)
        q = self.q_norm(q)
        k = self.k_norm(k)
        if rotary_emb is not None:
            cos, sin = rotary_emb
            q = apply_rotary_emb(q, cos, sin)
            k = apply_rotary_emb(k, cos, sin)
        q = q.transpose(0, 2, 1, 3)
        k = k.transpose(0, 2, 1, 3)
        v = v.transpose(0, 2, 1, 3)
        out = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.scale)
        out = out.transpose(0, 2, 1, 3).reshape(B, L, self.inner_dim)
        return self.out_proj(out)


class FeedForward(nn.Module):
    def __init__(self, hidden_size, ffn_dim):
        super().__init__()
        self.fc1 = nn.Linear(hidden_size, 2 * ffn_dim, bias=False)
        self.fc2 = nn.Linear(ffn_dim, hidden_size, bias=False)

    def __call__(self, x):
        gate, value = mx.split(self.fc1(x), 2, axis=-1)
        return self.fc2(_silu(gate) * value)


class AdaLayerNormModulation(nn.Module):
    def __init__(self, time_embed_dim, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.linear = nn.Linear(
            time_embed_dim, 6 * hidden_size * MINIMAX_H3_MODALITY_NUM, bias=True
        )

    def __call__(self, temb):
        temb = self.linear(_silu(temb))
        temb = temb.reshape(-1, 6 * self.hidden_size)
        return mx.split(temb, 6, axis=-1)


class AdaLayerNormOut(nn.Module):
    def __init__(self, hidden_size, time_embed_dim, eps):
        super().__init__()
        self.norm = RMSNorm(hidden_size, eps=eps)
        self.linear = nn.Linear(time_embed_dim, 2 * hidden_size, bias=True)

    def __call__(self, hidden_states, temb, timestep_indices):
        shift, scale = mx.split(self.linear(_silu(temb)), 2, axis=-1)
        hidden_states = self.norm(hidden_states)
        scale_row = mx.take(scale, timestep_indices, axis=0)
        shift_row = mx.take(shift, timestep_indices, axis=0)
        return hidden_states * (1.0 + scale_row) + shift_row


class TokenRefinerBlock(nn.Module):
    def __init__(
        self, hidden_size, num_heads, head_dim, ffn_dim, norm_eps, qk_norm_eps
    ):
        super().__init__()
        self.norm1 = RMSNorm(hidden_size, eps=norm_eps)
        self.attn = Attention(hidden_size, num_heads, head_dim, qk_norm_eps)
        self.norm2 = RMSNorm(hidden_size, eps=norm_eps)
        self.mlp = FeedForward(hidden_size, ffn_dim)

    def __call__(self, hidden_states):
        hidden_states = hidden_states + self.attn(self.norm1(hidden_states))
        hidden_states = hidden_states + self.mlp(self.norm2(hidden_states))
        return hidden_states


class TokenRefiner(nn.Module):
    def __init__(
        self,
        hidden_size,
        num_heads,
        head_dim,
        ffn_dim,
        num_layers,
        norm_eps,
        qk_norm_eps,
        final_norm_eps,
    ):
        super().__init__()
        self.blocks = [
            TokenRefinerBlock(
                hidden_size, num_heads, head_dim, ffn_dim, norm_eps, qk_norm_eps
            )
            for _ in range(num_layers)
        ]
        self.final_norm = RMSNorm(hidden_size, eps=final_norm_eps)

    def __call__(self, hidden_states):
        for block in self.blocks:
            hidden_states = block(hidden_states)
        return self.final_norm(hidden_states)


class TransformerBlock(nn.Module):
    def __init__(
        self,
        hidden_size,
        num_heads,
        head_dim,
        ffn_dim,
        time_embed_dim,
        norm_eps,
        qk_norm_eps,
    ):
        super().__init__()
        self.norm1 = RMSNorm(hidden_size, eps=norm_eps)
        self.attn = Attention(hidden_size, num_heads, head_dim, qk_norm_eps)
        self.norm2 = RMSNorm(hidden_size, eps=norm_eps)
        self.mlp = FeedForward(hidden_size, ffn_dim)
        self.adaln_proj = AdaLayerNormModulation(time_embed_dim, hidden_size)

    def __call__(self, hidden_states, temb, adaln_indices, rotary_emb):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.adaln_proj(temb)
        )

        residual = hidden_states
        norm_h = self.norm1(hidden_states)
        norm_h = norm_h * (1.0 + mx.take(scale_msa, adaln_indices, axis=0)) + mx.take(
            shift_msa, adaln_indices, axis=0
        )
        attn_out = self.attn(norm_h, rotary_emb)
        hidden_states = residual + mx.take(gate_msa, adaln_indices, axis=0) * attn_out

        residual = hidden_states
        norm_h = self.norm2(hidden_states)
        norm_h = norm_h * (1.0 + mx.take(scale_mlp, adaln_indices, axis=0)) + mx.take(
            shift_mlp, adaln_indices, axis=0
        )
        ff_out = self.mlp(norm_h)
        hidden_states = residual + mx.take(gate_mlp, adaln_indices, axis=0) * ff_out
        return hidden_states


class FinalLayer(nn.Module):
    def __init__(self, hidden_size, time_embed_dim, video_patch_dim, audio_in_dim, eps):
        super().__init__()
        self.norm = RMSNorm(hidden_size, eps=eps)
        self.adaln_proj = AdaLayerNormOut(hidden_size, time_embed_dim, eps)
        self.video_out = nn.Linear(hidden_size, video_patch_dim, bias=True)
        self.audio_out = nn.Linear(hidden_size, audio_in_dim, bias=True)

    def __call__(self, hidden_states, temb, timestep_indices):
        hidden_states = self.adaln_proj(hidden_states, temb, timestep_indices)
        return self.video_out(hidden_states), self.audio_out(hidden_states)


def scatter_rows(buffer, indices, values, axis=1):
    idx = mx.asarray(indices).astype(mx.int32)
    idx = idx.reshape(1, idx.shape[0], 1)
    return mx.put_along_axis(buffer, idx, values, axis=axis)


class MiniMaxH3DiTModel(nn.Module):
    def __init__(self, config: H3Config):
        super().__init__()
        self.config = config
        cfg = config
        self.hidden_size = cfg.dim
        self.heads = cfg.num_heads
        self.head_dim = cfg.head_dim
        self.inner_dim = cfg.num_heads * cfg.head_dim
        self.patch_size = cfg.patch_size
        video_patch_dim = (
            cfg.latents_dim * cfg.patch_size[0] * cfg.patch_size[1] * cfg.patch_size[2]
        )

        self.video_patch_proj = nn.Linear(video_patch_dim, cfg.dim, bias=True)
        self.audio_patch_proj = nn.Linear(cfg.audio_latents_dim, cfg.dim, bias=True)
        self.condition_proj = nn.Linear(cfg.text_dim, cfg.dim, bias=True)

        self.time_embedder = TimestepEmbedder(
            cfg.timestep_input_dim, cfg.time_embed_hidden, cfg.time_embed_dim
        )
        self.rope = RotaryPosEmbed(cfg.rope_inv_freq_len, rope_theta=10000.0)

        self.token_refiner = TokenRefiner(
            hidden_size=cfg.dim,
            num_heads=cfg.num_heads,
            head_dim=cfg.head_dim,
            ffn_dim=cfg.ffn_dim,
            num_layers=cfg.token_refiner_layers,
            norm_eps=cfg.norm_eps,
            qk_norm_eps=cfg.qk_norm_eps,
            final_norm_eps=cfg.final_norm_eps,
        )

        self.blocks = [
            TransformerBlock(
                hidden_size=cfg.dim,
                num_heads=cfg.num_heads,
                head_dim=cfg.head_dim,
                ffn_dim=cfg.ffn_dim,
                time_embed_dim=cfg.time_embed_dim,
                norm_eps=cfg.norm_eps,
                qk_norm_eps=cfg.qk_norm_eps,
            )
            for _ in range(cfg.num_layers)
        ]

        self.final_layer = FinalLayer(
            hidden_size=cfg.dim,
            time_embed_dim=cfg.time_embed_dim,
            video_patch_dim=video_patch_dim,
            audio_in_dim=cfg.audio_latents_dim,
            eps=cfg.final_norm_eps,
        )

    def __call__(
        self,
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
    ):
        if position_ids.ndim != 2 or position_ids.shape[-1] != 3:
            raise ValueError(
                f"position_ids must be (seq_len, 3), got {list(position_ids.shape)}"
            )
        seq_len = position_ids.shape[0]
        if token_tags.shape != (seq_len,) or timestep_indices.shape != (seq_len,):
            raise ValueError(
                f"token_tags/timestep_indices must be ({seq_len},), got "
                f"{list(token_tags.shape)} and {list(timestep_indices.shape)}"
            )

        rotary_emb = self.rope(position_ids)

        video_embeds = self.video_patch_proj(hidden_states)
        audio_embeds = self.audio_patch_proj(audio_hidden_states)
        text_embeds = self.condition_proj(encoder_hidden_states)
        text_embeds = self.token_refiner(text_embeds)

        B = text_embeds.shape[0]
        packed = mx.zeros((B, seq_len, self.hidden_size), dtype=text_embeds.dtype)
        packed = scatter_rows(packed, text_indices, text_embeds, axis=1)
        packed = scatter_rows(packed, video_indices, video_embeds, axis=1)
        packed = scatter_rows(packed, audio_indices, audio_embeds, axis=1)

        temb = get_timestep_embedding(
            timestep,
            self.config.timestep_input_dim,
            flip_sin_to_cos=True,
            downscale_freq_shift=0.0,
        )
        temb = self.time_embedder(temb)

        adaln_indices = timestep_indices * MINIMAX_H3_MODALITY_NUM + token_tags

        for block in self.blocks:
            packed = block(packed, temb, adaln_indices, rotary_emb)

        video_out, audio_out = self.final_layer(packed, temb, timestep_indices)
        video_output = mx.take(video_out, video_indices, axis=1)
        audio_output = mx.take(audio_out, audio_indices, axis=1)
        return video_output, audio_output


def reorder_interleaved_qkv(weight, num_heads, head_dim):
    expected_rows = num_heads * 3 * head_dim
    if weight.shape[0] != expected_rows:
        raise ValueError(
            f"fused qkv weight has {weight.shape[0]} rows, expected {expected_rows}"
        )
    grouped = weight.reshape(num_heads, 3 * head_dim, *weight.shape[1:])
    q, k, v = mx.split(grouped, 3, axis=1)
    return mx.concatenate(
        [t.reshape(num_heads * head_dim, *weight.shape[1:]) for t in (q, k, v)],
        axis=0,
    )


def _flatten_params(params, prefix=""):
    out = {}
    if isinstance(params, dict):
        for k, v in params.items():
            key = f"{prefix}.{k}" if prefix else k
            out.update(_flatten_params(v, key))
    elif isinstance(params, mx.array):
        out[prefix] = params
    return out


def _update_module(module, flat_params):
    children = {}
    for k, v in flat_params.items():
        parts = k.split(".")
        if len(parts) == 1:
            if isinstance(v, mx.array):
                setattr(module, k, v)
            continue
        child_name = parts[0]
        rest = ".".join(parts[1:])
        if child_name not in children:
            children[child_name] = {}
        children[child_name][rest] = v
    for child_name, child_params in children.items():
        child = getattr(module, child_name, None)
        if child is not None and isinstance(child, nn.Module):
            _update_module(child, child_params)
        elif child is not None and isinstance(child, mx.array):
            if "." not in child_name and len(child_params) == 1 and "" in child_params:
                setattr(module, child_name, child_params[""])


def _remap_transformer_weights(params, config):
    out = {}
    dropped = {"rope.inv_freq"}
    for k, v in params.items():
        if k in dropped:
            continue
        if k.endswith(".attn.qkv_proj.weight"):
            reordered = reorder_interleaved_qkv(v, config.num_heads, config.head_dim)
            out[k] = reordered
        else:
            out[k] = v
    return out


def load_dit_from_pretrained(model_path, config=None):
    if config is None:
        config = H3Config.fl2va()
    model = MiniMaxH3DiTModel(config)
    if os.path.isfile(model_path) and model_path.endswith(".safetensors"):
        safetensor_files = [model_path]
    else:
        safetensor_files = sorted(glob.glob(os.path.join(model_path, "*.safetensors")))
    if not safetensor_files:
        logger.warning("minimax_h3 dit: no safetensors at %s, random init", model_path)
        return model
    all_params = {}
    for sf in safetensor_files:
        all_params.update(mx.load(sf))
    mapped = _remap_transformer_weights(all_params, config)
    flat_model = _flatten_params(model.parameters())
    loaded = {}
    matched = 0
    shape_mismatches = []
    unmatched_model = []
    for k, v in flat_model.items():
        if k in mapped:
            w = mapped[k]
            if list(v.shape) == list(w.shape):
                loaded[k] = w.astype(v.dtype)
                matched += 1
            else:
                shape_mismatches.append((k, list(v.shape), list(w.shape)))
                loaded[k] = v
        else:
            loaded[k] = v
            unmatched_model.append(k)
    logger.info(
        "minimax_h3 dit: matched %d/%d model params from %d weight keys",
        matched,
        len(flat_model),
        len(mapped),
    )
    if shape_mismatches:
        logger.warning(
            "minimax_h3 dit: %d shape mismatches: %s",
            len(shape_mismatches),
            shape_mismatches[:10],
        )
    if unmatched_model:
        logger.debug("minimax_h3 dit: unmatched model keys: %s", unmatched_model[:20])
    _update_module(model, loaded)
    return model
