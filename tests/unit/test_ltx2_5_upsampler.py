# SPDX-License-Identifier: Apache-2.0
# P5b checkpoint: LTX-2.5 temporal upsampler (latent frame-count x2).
from __future__ import annotations

import mlx.core as mx
import pytest

from fusion_mlx.video.ltx2_5.upsampler import (
    LatentTemporalUpsampler,
    TemporalPixelShuffle,
    TemporalUpsampler2x,
    load_spatial_upsampler_2_5,
    load_temporal_upsampler,
)


class TestTemporalPixelShuffle:
    def test_doubles_time_dim(self):
        shuf = TemporalPixelShuffle(upscale_factor=2)
        # (n, d, h, w, c) with c divisible by r
        x = mx.zeros((1, 4, 8, 8, 16))
        out = shuf(x)
        assert out.shape == (1, 8, 8, 8, 8)

    def test_channels_divided(self):
        shuf = TemporalPixelShuffle(upscale_factor=2)
        x = mx.zeros((1, 3, 4, 4, 8))
        out = shuf(x)
        assert out.shape[4] == 4

    def test_preserves_spatial(self):
        shuf = TemporalPixelShuffle(upscale_factor=2)
        x = mx.zeros((1, 2, 5, 7, 12))
        out = shuf(x)
        assert out.shape[2] == 5
        assert out.shape[3] == 7


class TestTemporalUpsampler2x:
    def test_time_doubles(self):
        up = TemporalUpsampler2x(mid_channels=16, upscale_factor=2)
        # (n, d, h, w, c=mid)
        x = mx.zeros((1, 4, 8, 8, 16))
        out = up(x)
        assert out.shape[1] == 8
        assert out.shape[4] == 16

    def test_preserves_spatial_dims(self):
        up = TemporalUpsampler2x(mid_channels=16, upscale_factor=2)
        x = mx.zeros((1, 3, 5, 7, 16))
        out = up(x)
        assert out.shape[2] == 5
        assert out.shape[3] == 7


class TestLatentTemporalUpsampler:
    def test_forward_doubles_time_channels_first(self):
        up = LatentTemporalUpsampler(
            in_channels=8, mid_channels=32, num_blocks_per_stage=2, temporal_scale=2.0
        )
        # latent channels-first: (n, c, d, h, w)
        latent = mx.random.normal((1, 8, 4, 8, 8))
        out = up(latent)
        assert out.shape[0] == 1
        assert out.shape[1] == 8  # channels preserved
        assert out.shape[2] == 8  # time doubled 4->8
        assert out.shape[3] == 8  # spatial preserved
        assert out.shape[4] == 8

    def test_forward_preserves_spatial(self):
        up = LatentTemporalUpsampler(
            in_channels=8, mid_channels=32, num_blocks_per_stage=1, temporal_scale=2.0
        )
        latent = mx.random.normal((1, 8, 2, 6, 10))
        out = up(latent)
        assert out.shape[3] == 6
        assert out.shape[4] == 10
        assert out.shape[2] == 4

    def test_batch_gt_one(self):
        up = LatentTemporalUpsampler(
            in_channels=8, mid_channels=32, num_blocks_per_stage=1, temporal_scale=2.0
        )
        latent = mx.random.normal((2, 8, 3, 8, 8))
        out = up(latent)
        assert out.shape[0] == 2
        assert out.shape[2] == 6

    def test_temporal_scale_3(self):
        up = LatentTemporalUpsampler(
            in_channels=8, mid_channels=32, num_blocks_per_stage=1, temporal_scale=3.0
        )
        latent = mx.random.normal((1, 8, 2, 4, 4))
        out = up(latent)
        assert out.shape[2] == 6


class TestLoadTemporalUpsampler:
    def test_missing_weights_raises(self, tmp_path):
        with pytest.raises(
            FileNotFoundError, match="temporal upsampler weights not found"
        ):
            load_temporal_upsampler(tmp_path / "nope.safetensors")


class TestLoadSpatialUpsampler25:
    def test_missing_weights_raises(self, tmp_path):
        # The ltx2 spatial loader will raise on a missing file path.
        with pytest.raises(Exception):
            load_spatial_upsampler_2_5(tmp_path / "nope.safetensors")
