# SPDX-License-Identifier: Apache-2.0
# P6 Generate checkpoint：t2va video-only 去噪循环结构（mock dit/vae，无真实权重）。
# 验证 generate_t2va_video 调用 dit 的参数契约 + 输出帧形状，不依赖权重。

import mlx.core as mx
import numpy as np

from fusion_mlx.video.minimax_h3 import generate_t2va_video


class _FakeDiT:
    # 记录每次调用的 packed 参数，返回与 video token 形状匹配的输出。
    def __init__(self):
        self.calls = []

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
        self.calls.append(
            {
                "timestep": [float(x) for x in timestep],
                "token_tags": [int(t) for t in token_tags],
                "video_indices_len": int(video_indices.shape[0]),
                "text_indices_len": int(text_indices.shape[0]),
            }
        )
        # video_output 与 hidden_states 同形（patch tokens）。
        return mx.zeros_like(hidden_states), mx.zeros((1, 0, 32))


class _FakeVAE:
    def decode(self, z):
        # z (1, 24, t, h, w) → (1, 3, t, h*16, w*16) 近似，返回 [0,1] 范围。
        b, c, t, h, w = z.shape
        return mx.zeros((b, 3, t, h * 16, w * 16))


class TestGenerateT2va:
    def test_denoise_loop_calls_dit(self):
        # 小 latent：(1,24,2,4,4) → n_video=8；2 步去噪。
        text = mx.zeros((1, 4, 5120))
        dit = _FakeDiT()
        vae = _FakeVAE()
        frames = generate_t2va_video(
            dit=dit,
            vae=vae,
            text_embeds=text,
            num_frames=2,
            height=64,
            width=64,
            seed=0,
            num_inference_steps=4,
            z_channels=24,
            vae_ratio=16,
            vae_ratio_t=4,
            compute_dtype=mx.float32,
        )
        # 至少调用 dit 2 次（每步一次）。
        assert len(dit.calls) >= 2
        # 第一次调用：t2va 单一 video_timestep（text 继承 video timestep，#602），
        # 不再把 text 钉 1.0。4 步 shift=12 → 首步 t=1-sigma[0]，sigma[0]=1.0 → t=0.0。
        first = dit.calls[0]
        assert len(first["timestep"]) == 1
        # token_tags 含 text(1) + video(0)。
        assert 1 in first["token_tags"]
        assert 0 in first["token_tags"]
        # 输出帧。
        assert len(frames) >= 1
        assert frames[0].dtype == np.uint8
        assert frames[0].ndim == 3
