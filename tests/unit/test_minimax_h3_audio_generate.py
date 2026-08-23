# SPDX-License-Identifier: Apache-2.0
# #588 native audio：t2va joint audio+video 去噪循环结构（mock dit/vae/audio_vae）。
# 验证 generate_t2va_av 双 scheduler + 三模态 packed + (frames,waveform) 输出契约。
# 不依赖真实权重。

import mlx.core as mx
import numpy as np

from fusion_mlx.video.minimax_h3 import generate_t2va_av


class _FakeDiT:
    # 记录每次调用的 packed 参数，返回 video token + audio token 形状匹配的输出。
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
                "audio_indices_len": int(audio_indices.shape[0]),
                "text_indices_len": int(text_indices.shape[0]),
                "n_audio_hidden": int(audio_hidden_states.shape[1]),
            }
        )
        # video_output 与 hidden_states 同形；audio_output 与 audio_hidden_states 步数同。
        return (
            mx.zeros_like(hidden_states),
            mx.zeros_like(audio_hidden_states),
        )


class _FakeVAE:
    def decode(self, z):
        b, c, t, h, w = z.shape
        return mx.zeros((b, 3, t, h * 16, w * 16))


class _FakeAudioVAE:
    # z (1, T_audio, 32) → (1, T_audio*800, 1)（hop_length=800）。
    def decode(self, z):
        b, t, _c = z.shape
        return mx.zeros((b, t * 800, 1))


class TestGenerateT2vaAv:
    def test_joint_loop_returns_frames_and_waveform(self):
        # 小 latent：(1,24,2,4,4) → n_video=8；audio_latent_steps(2,24)=ceil(2/24*32000/800)=ceil(3.33)=4。
        text = mx.zeros((1, 4, 5120))
        dit = _FakeDiT()
        vae = _FakeVAE()
        audio_vae = _FakeAudioVAE()
        frames, waveform = generate_t2va_av(
            dit=dit,
            vae=vae,
            audio_vae=audio_vae,
            text_embeds=text,
            num_frames=2,
            height=64,
            width=64,
            fps=24,
            seed=0,
            num_inference_steps=4,
            z_channels=24,
            vae_ratio=16,
            vae_ratio_t=4,
            compute_dtype=mx.float32,
        )
        # 双 scheduler：每步 dit 收到 2 个 timestep（video idx0 + audio idx1）。
        first = dit.calls[0]
        assert len(first["timestep"]) == 2, first["timestep"]
        # 三模态 token_tags：text(1) + video(0) + audio(2)。
        assert 1 in first["token_tags"]
        assert 0 in first["token_tags"]
        assert 2 in first["token_tags"]
        # audio_indices 非空，audio_hidden_states 步数 = audio_latent_steps。
        assert first["audio_indices_len"] > 0
        assert first["n_audio_hidden"] == 4  # ceil(2/24*32000/800)
        # 至少 2 步。
        assert len(dit.calls) >= 2
        # 输出帧。
        assert len(frames) >= 1
        assert frames[0].dtype == np.uint8
        assert frames[0].ndim == 3
        # 输出波形：mono，长度 = T_audio*800。
        assert waveform.ndim == 1
        assert waveform.shape[0] == 4 * 800

    def test_dual_scheduler_distinct_shifts(self):
        # video shift=12 / audio shift=3 → 第 2 步起 timestep 不同（不同 sigma 网格）。
        # 首步 sigma[0]=1.0 → timestep=0.0 两 scheduler 相同（linspace 起点 1.0，
        # shift 只影响 1.0 之后的 sigma），故从第 2 步（calls[1]）断言差异。
        text = mx.zeros((1, 4, 5120))
        dit = _FakeDiT()
        frames, waveform = generate_t2va_av(
            dit=dit,
            vae=_FakeVAE(),
            audio_vae=_FakeAudioVAE(),
            text_embeds=text,
            num_frames=2,
            height=64,
            width=64,
            fps=24,
            seed=1,
            num_inference_steps=6,
            z_channels=24,
            vae_ratio=16,
            vae_ratio_t=4,
            compute_dtype=mx.float32,
        )
        assert len(dit.calls) >= 2
        second = dit.calls[1]
        t_video, t_audio = second["timestep"]
        # 不同 shift → 第 2 步 timestep 必不同（shift=12 sigma 衰减更快）。
        assert t_video != t_audio, (t_video, t_audio)
