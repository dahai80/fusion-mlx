# SPDX-License-Identifier: Apache-2.0
# P6 Condition checkpoint：packed-sequence 组装 + patchify/unpatchify + latent 归一化。
# 纯结构测试（无真实权重），验证 transformer __call__ 契约的输入形状。

import mlx.core as mx
import pytest

from fusion_mlx.video.minimax_h3.condition import (
    TAG_TEXT,
    TAG_VIDEO,
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
        grid = video_position_grid((1, 24, 4, 8, 8), (1, 2, 2))
        assert grid.shape == (64, 3)
        # 第一行 (0,0,0)，第二行 (0,0,1)（w 内层）。
        assert [float(grid[0, i]) for i in range(3)] == [0.0, 0.0, 0.0]
        assert [float(grid[1, i]) for i in range(3)] == [0.0, 0.0, 1.0]
        # 第 4 行应跳到 (0,1,0)（h 中层，nw=4）。
        assert [float(grid[4, i]) for i in range(3)] == [0.0, 1.0, 0.0]


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
        # text position 全 0。
        for i in range(5):
            assert [float(p) for p in packed["position_ids"][i]] == [0.0, 0.0, 0.0]
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
