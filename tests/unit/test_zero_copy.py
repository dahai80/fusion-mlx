# SPDX-License-Identifier: Apache-2.0
"""#913: IOSurface↔MTLBuffer↔CVPixelBuffer zero-copy bridge tests."""

import mlx.core as mx
import numpy as np
import pytest

from fusion_mlx.metal.zero_copy import (
    MetalZeroCopyBridge,
    _native_zero_copy_available,
    make_bridge,
)


def test_make_bridge_factory():
    b = make_bridge(64, 48)
    assert isinstance(b, MetalZeroCopyBridge)
    assert b.width == 64
    assert b.height == 48


def test_native_available_returns_bool():
    assert isinstance(_native_zero_copy_available(), bool)


def test_array_to_cvbuffer_wrong_ndim():
    b = make_bridge(8, 8)
    with pytest.raises(ValueError):
        b.array_to_cvbuffer(mx.zeros((8, 8)))


def test_array_to_cvbuffer_wrong_channels():
    b = make_bridge(8, 8)
    with pytest.raises(ValueError):
        b.array_to_cvbuffer(mx.zeros((8, 8, 2)))


def test_array_to_cvbuffer_size_mismatch():
    b = make_bridge(16, 16)
    with pytest.raises(ValueError):
        b.array_to_cvbuffer(mx.zeros((8, 8, 3)))


def test_array_to_cvbuffer_rgb_returns_pointer():
    b = make_bridge(4, 4)
    arr = mx.zeros((4, 4, 3))
    ptr = b.array_to_cvbuffer(arr)
    assert ptr is not None


def test_array_to_cvbuffer_single_channel():
    b = make_bridge(4, 4)
    arr = mx.ones((4, 4, 1)) * 0.5
    ptr = b.array_to_cvbuffer(arr)
    assert ptr is not None


def test_array_to_cvbuffer_four_channel():
    b = make_bridge(4, 4)
    arr = mx.zeros((4, 4, 4))
    ptr = b.array_to_cvbuffer(arr)
    assert ptr is not None


def test_denormalize_default_mapping():
    # [-1,1] → [0,255] with scale=127.5, offset=1.0.
    b = make_bridge(2, 2)
    arr = mx.array([[[[-1.0, 0.0, 1.0]]]])  # (1,1,1,3)→ reshape to (2,2,3)? no
    arr = mx.array(np.full((2, 2, 3), -1.0))
    ptr = b.array_to_cvbuffer(arr)
    assert ptr is not None


def test_custom_scale_offset():
    b = make_bridge(2, 2)
    arr = mx.array(np.full((2, 2, 3), 0.0, dtype=np.float32))
    ptr = b.array_to_cvbuffer(arr, scale=255.0, offset=0.0)
    assert ptr is not None
