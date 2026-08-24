# SPDX-License-Identifier: Apache-2.0
# P6 Condition checkpoint：packed-sequence 组装 + patchify/unpatchify + latent 归一化。
# 纯结构测试（无真实权重），验证 transformer __call__ 契约的输入形状。

import mlx.core as mx
import pytest

from fusion_mlx.video.minimax_h3.condition import (
    TAG_TEXT,
    TAG_VIDEO,
    build_fl2va_packed,
    build_t2va_packed,
    denormalize_latents,
    normalize_latents,
    patchify_video_latents,
    unpatchify_video_tokens,
    video_latents_mean_std,
    video_position_grid,
)


class TestPatchify:
    def test_patchify_shape(self):
        # (1, 24, 4, 8, 8) → n = 4*(8/2)*(8/2) = 4*4*4 = 64 tokens, dim = 24*1*2*2 = 96。
        z = mx.zeros((1, 24, 4, 8, 8))
        tokens = patchify_video_latents(z, (1, 2, 2))
        assert tokens.shape == (1, 64, 96)

    def test_patchify_unpatchify_roundtrip(self):
        z = mx.random.normal((1, 24, 4, 8, 8))
        tokens = patchify_video_latents(z, (1, 2, 2))
        z2 = unpatchify_video_tokens(tokens, z.shape, (1, 2, 2))
        assert mx.allclose(z, z2, atol=1e-6)

    def test_patchify_not_divisible(self):
        # h=3 不可被 ph=2 整除。
        z = mx.zeros((1, 24, 2, 3, 8))
        with pytest.raises(ValueError):
            patchify_video_latents(z, (1, 2, 2))


class TestPositionGrid:
    def test_grid_shape_and_order(self):
        # (1, 24, 4, 8, 8) → nt=4 nh=4 nw=4 → 64 rows (t,h,w)。
        # 空间轴 aspect-normalized [0,32)：sqrt_area=8, ratio=1, linspace(0,1,4)*32=[0,8,16,24]。
        # 时间轴非均匀 _temporal_position_grid(4, origin=0)=[0, 5/3, 25/3, 15]。
        grid = video_position_grid((1, 24, 4, 8, 8), (1, 2, 2), origin=0.0)
        assert grid.shape == (64, 3)
        # 第一行 (t0=0, h0=0, w0=0)，第二行 (t0=0, h0=0, w1=8)（w 内层）。
        assert [float(grid[0, i]) for i in range(3)] == [0.0, 0.0, 0.0]
        assert [float(grid[1, i]) for i in range(3)] == [0.0, 0.0, 8.0]
        # 第 4 行跳到 (0, h1=8, w0=0)（h 中层，nw=4）。
        assert [float(grid[4, i]) for i in range(3)] == [0.0, 8.0, 0.0]


class TestNormalize:
    def test_mean_std_shape(self):
        mean, std = video_latents_mean_std()
        assert mean.shape == (1, 24, 1, 1, 1)
        assert std.shape == (1, 24, 1, 1, 1)

    def test_roundtrip(self):
        z = mx.random.normal((2, 24, 4, 8, 8))
        z2 = denormalize_latents(normalize_latents(z))
        assert mx.allclose(z, z2, atol=1e-5)


class TestBuildPacked:
    def test_t2va_packed_structure(self):
        # video (1,24,2,4,4) → n_video = 2*2*2 = 8；text n_text=5。
        video = mx.random.normal((1, 24, 2, 4, 4))
        text = mx.random.normal((1, 5, 5120))
        packed = build_t2va_packed(video, text, timestep_video=0.5)

        assert packed["hidden_states"].shape == (1, 8, 96)
        assert packed["encoder_hidden_states"].shape == (1, 5, 5120)
        # audio 空。
        assert packed["audio_hidden_states"].shape == (1, 0, 32)

        seq = 5 + 8
        assert packed["token_tags"].shape == (seq,)
        assert packed["timestep_indices"].shape == (seq,)
        assert packed["position_ids"].shape == (seq, 3)
        assert packed["video_indices"].shape == (8,)
        assert packed["text_indices"].shape == (5,)
        assert packed["audio_indices"].shape == (0,)

        # tag：text=1 在前，video=0 在后。
        tags = [int(t) for t in packed["token_tags"]]
        assert tags == [TAG_TEXT] * 5 + [TAG_VIDEO] * 8
        # timestep_indices：t2va 无 condition 行，官方 build_row_timesteps 给全序列
        # 赋 video_timestep（text 继承 video timestep）→ 单一去重水平，全 0（#602）。
        tis = [int(t) for t in packed["timestep_indices"]]
        assert tis == [0] * seq
        # timestep = [0.5]（仅 video_timestep，text 不再钉 1.0 clean）。
        assert [float(x) for x in packed["timestep"]] == [0.5]
        # video_indices = [5..12]（text 占 0..4）。
        assert [int(i) for i in packed["video_indices"]] == list(range(5, 13))
        # text position：time=arange(n_text)，h/w=0（对照 before_denoise.py）。
        for i in range(5):
            assert [float(p) for p in packed["position_ids"][i]] == [float(i), 0.0, 0.0]
        # latent_shape 透传。
        assert tuple(packed["latent_shape"]) == (1, 24, 2, 4, 4)

    def test_t2va_single_timestep(self):
        video = mx.zeros((1, 24, 2, 4, 4))
        text = mx.zeros((1, 3, 5120))
        packed = build_t2va_packed(video, text, timestep_video=0.123)
        # t2va 无 condition 行 → 全序列单一 video_timestep（#602）。
        assert packed["timestep"].shape == (1,)
        assert float(packed["timestep"][0]) == pytest.approx(0.123)
        # 全行 timestep_indices=0（指向唯一去重水平）。
        assert [int(t) for t in packed["timestep_indices"]] == [0] * 11


class TestBuildFl2vaPacked:
    # 对照 diffusers before_denoise.py:268 build_packed_sequence。
    # fl2va = t2va + keyframe 条件帧。i2va = anchors=('first',)，l2va = ('last',)。
    # 行序 [text | keyframe conditions | target audio | target video]（无音频时 audio 空）。
    # 条件行 tag=TAG_VIDEO（0），timestep 钉在 max(video_t, keyframe_noise_aug)=0.999，
    # 每步不变（conditioning rides through，FL2VAPrepareLatentsStep L970-1009）。
    # 条件行 position：'first'→anchor_time=n_text；'last'→n_text+spans.sum()-rescale；
    # 空间 = 同 frame_grid。

    def test_fl2va_first_anchor_structure(self):
        # video (1,24,2,4,4) → 8 target video tokens；condition 1 帧 → 4 tokens；text n=5。
        video = mx.random.normal((1, 24, 2, 4, 4))
        cond = mx.random.normal((1, 24, 1, 4, 4))  # 1 条件帧
        text = mx.random.normal((1, 5, 5120))
        packed = build_fl2va_packed(
            video,
            cond,
            text,
            timestep_video=0.5,
            keyframe_anchors=("first",),
        )
        n_text, n_cond, n_gen = 5, 4, 8
        seq = n_text + n_cond + n_gen  # 17（无音频）
        assert packed["hidden_states"].shape == (1, n_cond + n_gen, 96)  # cond+gen
        assert packed["encoder_hidden_states"].shape == (1, 5, 5120)
        assert packed["audio_hidden_states"].shape == (1, 0, 32)
        assert packed["token_tags"].shape == (seq,)
        assert packed["video_indices"].shape == (n_cond + n_gen,)
        assert packed["text_indices"].shape == (5,)
        assert packed["audio_indices"].shape == (0,)
        # tag：text=1 (5)，cond+gen video=0 (12)。
        tags = [int(t) for t in packed["token_tags"]]
        assert tags == [TAG_TEXT] * 5 + [TAG_VIDEO] * 12
        # video_indices = [5..16]（text 占 0..4）。
        assert [int(i) for i in packed["video_indices"]] == list(range(5, 17))
        # timestep：video_t=0.5 < keyframe_noise_aug=0.999 → [0.5, 0.999] sorted。
        ts = [float(x) for x in packed["timestep"]]
        assert len(ts) == 2
        assert ts[0] == pytest.approx(0.5, abs=1e-6)
        assert ts[1] == pytest.approx(0.999, abs=1e-6)
        # timestep_indices：cond 行 → idx1(0.999)，gen video 行 → idx0(0.5)，text→idx0。
        tis = [int(t) for t in packed["timestep_indices"]]
        assert tis == [0] * 5 + [1] * 4 + [0] * 8
        # 条件行 position：'first' anchor_time = n_text = 5，空间 = target 同 frame_grid。
        for i in range(n_cond):
            row = n_text + i
            assert float(packed["position_ids"][row, 0]) == pytest.approx(5.0)

    def test_fl2va_last_anchor_position(self):
        # 'last' anchor：anchor_time = n_text + spans.sum() - _ROPE_FRAME_RESCALE。
        # nt=2 → spans = 5/3*(1,4) → sum = 5/3*5 = 25/3；rescale=5/3。
        # anchor_time = 5 + 25/3 - 5/3 = 5 + 20/3 ≈ 11.6667。
        video = mx.random.normal((1, 24, 2, 4, 4))
        cond = mx.random.normal((1, 24, 1, 4, 4))
        text = mx.random.normal((1, 5, 5120))
        packed = build_fl2va_packed(
            video,
            cond,
            text,
            timestep_video=0.5,
            keyframe_anchors=("last",),
        )
        n_cond = 4
        expected = (
            5.0 + (5.0 / 3.0) * (1 + 4) - (5.0 / 3.0)
        )  # n_text + sum(1,4)*resc - resc
        for i in range(n_cond):
            row = 5 + i
            assert float(packed["position_ids"][row, 0]) == pytest.approx(
                expected, abs=1e-4
            )

    def test_fl2va_condition_timestep_pinned_when_video_cleaner(self):
        # video_timestep < keyframe_noise_aug → 条件行钉 0.999，gen 行用 video_t。
        video = mx.random.normal((1, 24, 2, 4, 4))
        cond = mx.random.normal((1, 24, 1, 4, 4))
        text = mx.random.normal((1, 5, 5120))
        packed = build_fl2va_packed(
            video,
            cond,
            text,
            timestep_video=0.1,
            keyframe_anchors=("first",),
        )
        ts = [float(x) for x in packed["timestep"]]
        assert len(ts) == 2
        assert ts[0] == pytest.approx(0.1, abs=1e-6)
        assert ts[1] == pytest.approx(0.999, abs=1e-6)

    def test_fl2va_condition_timestep_capped_when_video_noisier(self):
        # video_timestep > keyframe_noise_aug → max() → 条件行同 video_t（单一水平）。
        video = mx.random.normal((1, 24, 2, 4, 4))
        cond = mx.random.normal((1, 24, 1, 4, 4))
        text = mx.random.normal((1, 5, 5120))
        packed = build_fl2va_packed(
            video,
            cond,
            text,
            timestep_video=0.9999,
            keyframe_anchors=("first",),
        )
        # max(0.9999, 0.999) = 0.9999 → 与 gen 相同 → 单一去重水平。
        assert packed["timestep"].shape == (1,)
        assert float(packed["timestep"][0]) == pytest.approx(0.9999)

    def test_fl2va_two_keyframes(self):
        # i2va+l2va 联合：anchors=('first','last') → 2 条件帧。
        video = mx.random.normal((1, 24, 2, 4, 4))
        cond = mx.random.normal((1, 24, 2, 4, 4))  # 2 条件帧
        text = mx.random.normal((1, 5, 5120))
        packed = build_fl2va_packed(
            video,
            cond,
            text,
            timestep_video=0.5,
            keyframe_anchors=("first", "last"),
        )
        n_cond = 8  # 2 帧 * 4 tokens
        # hidden_states = cond(8) + gen(8) = 16。
        assert packed["hidden_states"].shape == (1, 16, 96)
        # 第一条件块 'first'→t=5，第二 'last'→t≈11.6667。
        assert float(packed["position_ids"][5, 0]) == pytest.approx(5.0)
        last_expected = 5.0 + (5.0 / 3.0) * 5 - (5.0 / 3.0)
        assert float(packed["position_ids"][9, 0]) == pytest.approx(
            last_expected, abs=1e-4
        )

    def test_fl2va_condition_row_count_mismatch_raises(self):
        # condition 帧数 != len(keyframe_anchors) → shape-mismatch guard（L959）。
        video = mx.random.normal((1, 24, 2, 4, 4))
        cond = mx.random.normal((1, 24, 1, 4, 4))  # 1 帧
        text = mx.random.normal((1, 5, 5120))
        with pytest.raises(ValueError, match="(?i)condition"):
            build_fl2va_packed(
                video,
                cond,
                text,
                timestep_video=0.5,
                keyframe_anchors=("first", "last"),  # 2 anchors vs 1 帧
            )

    def test_fl2va_invalid_anchor_raises(self):
        video = mx.random.normal((1, 24, 2, 4, 4))
        cond = mx.random.normal((1, 24, 1, 4, 4))
        text = mx.random.normal((1, 5, 5120))
        with pytest.raises(ValueError, match="(?i)anchor"):
            build_fl2va_packed(
                video,
                cond,
                text,
                timestep_video=0.5,
                keyframe_anchors=("middle",),
            )
