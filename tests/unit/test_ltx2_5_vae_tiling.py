# SPDX-License-Identifier: Apache-2.0
"""#945: LTX-2.5 VAE temporal-only tiled decode.

Validates the tiling param dispatch (previously dead code) routes to
temporal-only TilingConfig for ltx2_5 (spatial tiling disabled — conv
decoder REFLECT seam artifacts, #937/#939).

Parity (temporal-only vs full decode) requires real MLX + Metal; gated
skipif. On headless CI only the dispatch logic tests run.
"""

import pytest

from fusion_mlx.video.ltx2_5.generate import (
    _LTX2_5_AUTO_TILING_FRAME_THRESHOLD,
    _resolve_ltx2_5_tiling_config,
)


def test_none_returns_full_decode():
    cfg = _resolve_ltx2_5_tiling_config("none", 113)
    assert cfg is None


def test_auto_below_threshold_no_tiling():
    cfg = _resolve_ltx2_5_tiling_config("auto", 33)
    assert cfg is None


def test_auto_above_threshold_temporal_only():
    cfg = _resolve_ltx2_5_tiling_config("auto", 113)
    assert cfg is not None
    assert cfg.spatial_config is None
    assert cfg.temporal_config is not None
    assert cfg.temporal_config.tile_size_in_frames == 128
    assert cfg.temporal_config.tile_overlap_in_frames == 64


def test_temporal_explicit():
    cfg = _resolve_ltx2_5_tiling_config("temporal", 33)
    assert cfg is not None
    assert cfg.spatial_config is None
    assert cfg.temporal_config is not None


@pytest.mark.parametrize("mode", ["default", "aggressive", "conservative", "spatial"])
def test_spatial_modes_fall_back_to_temporal_only(mode):
    cfg = _resolve_ltx2_5_tiling_config(mode, 113)
    assert cfg is not None
    assert (
        cfg.spatial_config is None
    ), f"{mode} must not enable spatial tiling for ltx2_5"
    assert cfg.temporal_config is not None


def test_threshold_boundary():
    n = _LTX2_5_AUTO_TILING_FRAME_THRESHOLD
    assert _resolve_ltx2_5_tiling_config("auto", n) is None
    assert _resolve_ltx2_5_tiling_config("auto", n + 1) is not None


def test_unknown_mode_falls_back_auto():
    cfg = _resolve_ltx2_5_tiling_config("bogus", 113)
    assert cfg is not None
    assert cfg.spatial_config is None
