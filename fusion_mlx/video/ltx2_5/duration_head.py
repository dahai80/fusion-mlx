# SPDX-License-Identifier: Apache-2.0
# Pure-MLX port of LTX-2.5 duration-head.
# Upstream: packages/ltx-core/src/ltx_core/duration_head/duration_head.py
# (Lightricks/LTX-2 repo, Apache-2.0).
#
# Predicts shot duration (seconds) from frozen Connector outputs:
#   - video_tokens: (B, T_v, 4096) from video_embeddings_connector
#   - audio_tokens: (B, T_a, 2048) from audio_embeddings_connector
# Each modality is projected to a shared pooler_hidden_dim (256), tagged with
# a learnable modality embedding, concatenated, then an attention pooler
# (learnable query tokens cross-attending to the token stream) produces a
# fixed-shape vector. A 2-layer MLP maps it to a log-duration; exp() yields
# seconds. Training target is log-seconds; inference returns seconds.
#
# Checkpoint: model_patches/ltx-2.5-duration-head-bf16.safetensors
# Key prefix: "duration_head." (stripped on load).
from __future__ import annotations

import logging
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn

logger = logging.getLogger(__name__)

# LTX 帧约束：num_frames % 8 == 1。
_LTX_FRAME_BLOCK = 8
# duration-head 输出上限/下限（秒），防止极端值。
_MIN_DURATION_S = 0.5
_MAX_DURATION_S = 60.0

# 默认维度对齐 upstream DurationHeadConfigurator.from_metadata：
# transformer.cross_attention_dim=4096 (video), audio_cross_attention_dim=2048。
_DEFAULT_VIDEO_DIM = 4096
_DEFAULT_AUDIO_DIM = 2048
_DEFAULT_POOLER_DIM = 256
_DEFAULT_NUM_QUERIES = 1
_DEFAULT_NUM_HEADS = 4
_DEFAULT_MLP_HIDDEN = 256


def _split_heads(x: mx.array, num_heads: int) -> mx.array:
    # (B, T, D) -> (B, num_heads, T, head_dim)
    b, t, _ = x.shape
    head_dim = x.shape[-1] // num_heads
    return x.reshape(b, t, num_heads, head_dim).transpose(0, 2, 1, 3)


def _merge_heads(x: mx.array) -> mx.array:
    # (B, num_heads, T, head_dim) -> (B, T, D)
    b, h, t, hd = x.shape
    return x.transpose(0, 2, 1, 3).reshape(b, t, h * hd)


class _MultiHeadAttn(nn.Module):
    # nn.MultiheadAttention(batch_first=True) 的最小 MLX 实现，仅 cross-attn。
    # 权键对齐 PyTorch 命名：in_proj_weight/in_proj_bias/out_proj.weight/out_proj.bias。

    def __init__(self, hidden_dim: int, num_heads: int) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.in_proj_weight = mx.zeros((3 * hidden_dim, hidden_dim))
        self.in_proj_bias = mx.zeros((3 * hidden_dim,))
        self.out_proj = nn.Linear(hidden_dim, hidden_dim, bias=True)

    def __call__(self, query: mx.array, key_value: mx.array) -> mx.array:
        # query: (B, Nq, D), key_value: (B, T, D) -> (B, Nq, D)
        w = self.in_proj_weight
        b = self.in_proj_bias
        hd = self.hidden_dim
        q = mx.matmul(query, w[:hd].T) + b[:hd]
        k = mx.matmul(key_value, w[hd : 2 * hd].T) + b[hd : 2 * hd]
        v = mx.matmul(key_value, w[2 * hd :].T) + b[2 * hd :]
        nh = self.num_heads
        q = _split_heads(q, nh)
        k = _split_heads(k, nh)
        v = _split_heads(v, nh)
        head_dim = hd // nh
        scale = 1.0 / mx.sqrt(mx.array(head_dim, dtype=mx.float32))
        attn = mx.matmul(q, k.transpose(0, 1, 3, 2)) * scale
        attn = mx.softmax(attn.astype(mx.float32), axis=-1).astype(query.dtype)
        out = mx.matmul(attn, v)
        out = _merge_heads(out)
        return self.out_proj(out)


class AttentionPooler(nn.Module):
    # Cross-attend num_queries learnable tokens against the input token stream.
    # 输出固定 (B, num_queries, hidden_dim)，与输入序列长度无关。
    # 所有位置均可 attend（connector 已用 learnable registers 替换 padding），
    # 故无需 attention mask。

    def __init__(
        self,
        hidden_dim: int = _DEFAULT_POOLER_DIM,
        num_queries: int = _DEFAULT_NUM_QUERIES,
        num_heads: int = _DEFAULT_NUM_HEADS,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_queries = num_queries
        self.num_heads = num_heads
        # learnable query tokens: (num_queries, hidden_dim)
        self.query_tokens = mx.zeros((num_queries, hidden_dim))
        self.cross_attn = _MultiHeadAttn(hidden_dim, num_heads)

    def __call__(self, tokens: mx.array) -> mx.array:
        batch_size = tokens.shape[0]
        queries = mx.broadcast_to(
            self.query_tokens, (batch_size, self.num_queries, self.hidden_dim)
        )
        return self.cross_attn(queries, tokens)


class DurationHead(nn.Module):
    # 从 video/audio Connector 输出预测 log-duration（秒）。
    # 至少传入 video_tokens 或 audio_tokens 之一。

    def __init__(
        self,
        video_cross_attention_dim: int = _DEFAULT_VIDEO_DIM,
        audio_cross_attention_dim: int = _DEFAULT_AUDIO_DIM,
        pooler_hidden_dim: int = _DEFAULT_POOLER_DIM,
        num_queries: int = _DEFAULT_NUM_QUERIES,
        num_pooler_heads: int = _DEFAULT_NUM_HEADS,
        mlp_hidden: int = _DEFAULT_MLP_HIDDEN,
    ) -> None:
        super().__init__()
        self.pooler_hidden_dim = pooler_hidden_dim
        self.num_queries = num_queries

        self.video_input_proj = nn.Linear(video_cross_attention_dim, pooler_hidden_dim, bias=True)
        self.video_modality_emb = mx.zeros((pooler_hidden_dim,))

        self.audio_input_proj = nn.Linear(audio_cross_attention_dim, pooler_hidden_dim, bias=True)
        self.audio_modality_emb = mx.zeros((pooler_hidden_dim,))

        self.attention_pooler = AttentionPooler(
            hidden_dim=pooler_hidden_dim,
            num_queries=num_queries,
            num_heads=num_pooler_heads,
        )
        self.mlp_hidden = nn.Linear(pooler_hidden_dim * num_queries, mlp_hidden, bias=True)
        self.mlp_out = nn.Linear(mlp_hidden, 1, bias=True)

    def __call__(
        self,
        video_tokens: mx.array | None = None,
        audio_tokens: mx.array | None = None,
    ) -> mx.array:
        # 返回秒数 (B,)。内部为 log-duration，此处 exp()。
        if video_tokens is None and audio_tokens is None:
            raise ValueError(
                "DurationHead requires at least one of video_tokens / audio_tokens"
            )
        groups = []
        if video_tokens is not None:
            groups.append(self.video_input_proj(video_tokens) + self.video_modality_emb)
        if audio_tokens is not None:
            groups.append(self.audio_input_proj(audio_tokens) + self.audio_modality_emb)
        tokens = mx.concatenate(groups, axis=1)
        pooled = self.attention_pooler(tokens)
        pooled_flat = pooled.reshape(pooled.shape[0], -1)
        hidden = nn.gelu_approx(self.mlp_hidden(pooled_flat))
        log_duration = self.mlp_out(hidden).squeeze(-1)
        return mx.exp(log_duration)

    def predict_duration(
        self,
        video_tokens: mx.array | None = None,
        audio_tokens: mx.array | None = None,
        *,
        clamp: bool = True,
    ) -> mx.array:
        duration = self.__call__(video_tokens, audio_tokens)
        if clamp:
            duration = mx.clip(duration, _MIN_DURATION_S, _MAX_DURATION_S)
        return duration


def duration_to_num_frames(duration_s: float, fps: float) -> int:
    # LTX-2.5 帧换算：round(duration*fps/8)*8 + 1（满足 num_frames % 8 == 1）。
    raw = round(duration_s * fps / _LTX_FRAME_BLOCK) * _LTX_FRAME_BLOCK
    frames = int(raw) + 1
    logger.debug(
        "duration_to_num_frames: duration=%.3fs fps=%.1f -> frames=%d",
        duration_s,
        fps,
        frames,
    )
    return frames


def infer_num_frames(
    head: DurationHead,
    video_tokens: mx.array | None,
    audio_tokens: mx.array | None,
    fps: float,
) -> int:
    # 端到端：connector tokens -> duration -> num_frames。
    duration = float(head.predict_duration(video_tokens, audio_tokens).item())
    return duration_to_num_frames(duration, fps)


def _sanitize_duration_head(weights: dict[str, mx.array]) -> dict[str, mx.array]:
    # 剥去 "duration_head." 前缀（与 upstream DURATION_HEAD_KEY_OPS 一致）。
    prefix = "duration_head."
    out = {}
    dropped = 0
    for k, v in weights.items():
        if k.startswith(prefix):
            out[k[len(prefix):]] = v
        else:
            dropped += 1
    if dropped:
        logger.warning(
            "duration_head sanitize: dropped %d keys without '%s' prefix (first 10: %s)",
            dropped,
            prefix,
            sorted(k for k in weights if not k.startswith(prefix))[:10],
        )
    return out


def load_duration_head(
    weights_path: str | Path,
    video_cross_attention_dim: int = _DEFAULT_VIDEO_DIM,
    audio_cross_attention_dim: int = _DEFAULT_AUDIO_DIM,
    pooler_hidden_dim: int = _DEFAULT_POOLER_DIM,
    num_queries: int = _DEFAULT_NUM_QUERIES,
    num_pooler_heads: int = _DEFAULT_NUM_HEADS,
    mlp_hidden: int = _DEFAULT_MLP_HIDDEN,
    strict: bool = True,
) -> DurationHead:
    # 从 model_patches/ltx-2.5-duration-head-bf16.safetensors 加载。
    weights_path = Path(weights_path)
    if not weights_path.exists():
        raise FileNotFoundError(f"duration-head weights not found: {weights_path}")

    head = DurationHead(
        video_cross_attention_dim=video_cross_attention_dim,
        audio_cross_attention_dim=audio_cross_attention_dim,
        pooler_hidden_dim=pooler_hidden_dim,
        num_queries=num_queries,
        num_pooler_heads=num_pooler_heads,
        mlp_hidden=mlp_hidden,
    )
    weights = mx.load(str(weights_path))
    weights = _sanitize_duration_head(weights)

    try:
        from mlx.utils import tree_flatten

        model_params = dict(tree_flatten(head.parameters()))
        model_keys = set(model_params.keys())
        weight_keys = set(weights.keys())
        unmatched = sorted(weight_keys - model_keys)
        missing = sorted(model_keys - weight_keys)
        logger.info(
            "duration_head audit: model_params=%d weights=%d unmatched=%d missing=%d",
            len(model_keys),
            len(weight_keys),
            len(unmatched),
            len(missing),
        )
        if unmatched:
            logger.warning("duration_head unmatched (first 20): %s", unmatched[:20])
        if missing:
            logger.warning("duration_head missing (first 20): %s", missing[:20])
        if strict and (unmatched or missing):
            raise RuntimeError(
                f"duration_head strict load failed: unmatched={len(unmatched)} missing={len(missing)}"
            )
    except RuntimeError:
        raise
    except Exception as audit_err:
        logger.warning("duration_head audit skipped: %s", audit_err)

    head.load_weights(list(weights.items()), strict=False)
    loaded = len(weights)
    logger.info("load_duration_head: loaded %d keys from %s", loaded, weights_path)
    return head
