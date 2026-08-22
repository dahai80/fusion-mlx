# SPDX-License-Identifier: Apache-2.0
# H3 DiT position-ids 校正测试（#605）。
# 对照官方 diffusers before_denoise.py 的 _spatial_position_grid /
# _temporal_position_grid / _frame_position_grid。
# 根因：MLX video_position_grid 用裸 arange(n)（空间范围随分辨率线性增长），
# 官方空间网格固定 [0,32) aspect-normalized → 分辨率无关。
# 旧实现致 16x16 latent 解码偏暗（YAVG 9.9）vs 32x32 偏亮（146）。

import math

import mlx.core as mx
import pytest

from fusion_mlx.video.minimax_h3.condition import (
    build_t2va_packed,
)

ROPE_SPATIAL_SCALE = 32.0
ROPE_FRAME_RESCALE = 5.0 / 3.0
ROPE_FRAMES_PER_LATENT = (1, 4, 4, 4, 4)


def _ref_spatial_grid(dim, patch, sqrt_area):
    import numpy as np

    ratio = dim / sqrt_area
    left = (1.0 - ratio) / 2.0
    n = dim // patch
    return np.linspace(left, left + ratio, n, endpoint=False) * ROPE_SPATIAL_SCALE


def _ref_temporal_grid(num_latent_frames, origin):
    import numpy as np

    spans = np.array(
        [
            ROPE_FRAME_RESCALE * ROPE_FRAMES_PER_LATENT[i % len(ROPE_FRAMES_PER_LATENT)]
            for i in range(num_latent_frames)
        ],
        dtype=np.float64,
    )
    out = origin + np.concatenate([np.zeros(1), spans[:-1].cumsum()])
    return out


class TestSpatialGrid:
    def test_square_constant_range(self):
        # 正方形：ratio=1，范围 [0,32) 与分辨率无关。
        g16 = build_t2va_packed(
            mx.zeros((1, 24, 13, 16, 16)), mx.zeros((1, 4, 5120)), 0.5
        )
        g32 = build_t2va_packed(
            mx.zeros((1, 24, 13, 32, 32)), mx.zeros((1, 4, 5120)), 0.5
        )
        import numpy as np

        # video 行 = 后 n_video 行；取 h 轴（列1）。
        v16 = np.array(g16["position_ids"])[4:]
        v32 = np.array(g32["position_ids"])[4:]
        h16 = v16[:, 1]
        w16 = v16[:, 2]
        h32 = v32[:, 1]
        w32 = v32[:, 2]
        # 范围上限 ≈ 32（endpoint=False → 最大 < 32）。
        assert h16.max() < 32.0 and h16.min() >= 0.0
        assert h32.max() < 32.0 and h32.min() >= 0.0
        # 同分辨率 h==w（正方形）。
        assert pytest.approx(float(h16.max())) == float(w16.max())

    def test_aspect_normalized(self):
        # 矩形 16x32：h ratio=16/sqrt(512)=0.707，w ratio=1.414。
        import numpy as np

        packed = build_t2va_packed(
            mx.zeros((1, 24, 13, 16, 32)), mx.zeros((1, 4, 5120)), 0.5
        )
        v = np.array(packed["position_ids"])[4:]
        h = v[:, 1]
        w = v[:, 2]
        sqrt_area = math.sqrt(16 * 32)
        ref_h = _ref_spatial_grid(16, 2, sqrt_area)
        ref_w = _ref_spatial_grid(32, 2, sqrt_area)
        assert pytest.approx(float(h.min()), abs=1e-4) == float(ref_h.min())
        assert pytest.approx(float(h.max()), abs=1e-4) == float(ref_h.max())
        assert pytest.approx(float(w.min()), abs=1e-4) == float(ref_w.min())
        assert pytest.approx(float(w.max()), abs=1e-4) == float(ref_w.max())

    def test_not_raw_arange(self):
        # 旧 bug：裸 arange → 16x16 max≈15，32x32 max≈31。修复后两者 max≈30(<32)。
        import numpy as np

        g16 = build_t2va_packed(
            mx.zeros((1, 24, 13, 16, 16)), mx.zeros((1, 4, 5120)), 0.5
        )
        v16 = np.array(g16["position_ids"])[4:]
        assert float(v16[:, 1].max()) < 32.0
        assert float(v16[:, 1].max()) > 20.0  # 接近 32，非 15


class TestTemporalGrid:
    def test_nonuniform_spacing(self):
        # 官方：spans = 5/3*(1,4,4,4,4) 重复，origin=n_text。
        # video 行按 t 外 h 中 w 内排列：每 (nh*nw) 行一帧。
        import numpy as np

        n_text = 4
        packed = build_t2va_packed(
            mx.zeros((1, 24, 13, 16, 16)), mx.zeros((1, n_text, 5120)), 0.5
        )
        v = np.array(packed["position_ids"])[n_text:]
        nh, nw = 8, 8  # 16//2
        t = v[:, 0][:: nh * nw]  # 每帧取一个 t
        ref = _ref_temporal_grid(13, float(n_text))
        for i in range(13):
            assert pytest.approx(float(t[i]), abs=1e-4) == float(ref[i])

    def test_origin_is_ntext(self):
        # 首帧 t = n_text（非 0）。
        import numpy as np

        for n_text in (4, 8, 16):
            packed = build_t2va_packed(
                mx.zeros((1, 24, 13, 16, 16)), mx.zeros((1, n_text, 5120)), 0.5
            )
            v = np.array(packed["position_ids"])[n_text:]
            nh, nw = 8, 8
            assert pytest.approx(float(v[0, 0]), abs=1e-4) == float(n_text)


class TestTextPosition:
    def test_text_time_is_arange(self):
        # 官方 text 行 time = arange(n_text)，非全 0。
        import numpy as np

        n_text = 5
        packed = build_t2va_packed(
            mx.zeros((1, 24, 13, 16, 16)), mx.zeros((1, n_text, 5120)), 0.5
        )
        txt = np.array(packed["position_ids"])[:n_text]
        assert [float(x) for x in txt[:, 0]] == [0.0, 1.0, 2.0, 3.0, 4.0]
        # text h/w 仍 0。
        assert float(txt[:, 1].max()) == 0.0
        assert float(txt[:, 2].max()) == 0.0
