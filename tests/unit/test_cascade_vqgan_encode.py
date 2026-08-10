# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import mlx.core as mx
import pytest

from fusion_mlx.image.cascade.vqgan import PaellaVQModel


def _tiny_model(**overrides):
    defaults = dict(
        in_channels=3,
        out_channels=3,
        up_down_scale_factor=2,
        levels=2,
        bottleneck_blocks=2,
        embed_dim=16,
        latent_channels=4,
        scale_factor=0.3764,
    )
    defaults.update(overrides)
    return PaellaVQModel(**defaults)


def test_encode_returns_downsampled_latent():
    m = _tiny_model()
    img = mx.random.normal((1, 3, 16, 16))
    lat = m.encode(img)
    # pixel-unshuffle x2 -> 8, then one stride-2 conv (levels=2, i>0) -> 4
    assert lat.shape == (1, 4, 4, 4), f"expected (1,4,4,4) got {lat.shape}"


def test_encode_applies_scale_factor():
    m = _tiny_model(scale_factor=2.0)
    img = mx.ones((1, 3, 16, 16))
    m.scale_factor = 1.0
    lat_unscaled = m.encode(img)
    m.scale_factor = 2.0
    lat_scaled = m.encode(img)
    assert mx.allclose(lat_scaled, lat_unscaled * 2.0, atol=1e-5).item()


def test_encode_decode_roundtrip_shapes():
    m = _tiny_model()
    img = mx.random.normal((1, 3, 16, 16))
    lat = m.encode(img)
    out = m.decode(lat)
    assert (
        out.shape == img.shape
    ), f"roundtrip must restore image shape: {out.shape} vs {img.shape}"


def test_encode_accepts_batch():
    m = _tiny_model()
    img = mx.random.normal((3, 3, 16, 16))
    lat = m.encode(img)
    assert lat.shape[0] == 3, "batch dim preserved"


def test_encode_nhwc_internal_no_channels_first_leak():
    m = _tiny_model()
    img = mx.random.normal((1, 3, 16, 16))
    lat = m.encode(img)
    assert (
        lat.shape[1] == 4
    ), f"channels-first: dim 1 must be latent_channels(4), got {lat.shape}"


@pytest.mark.parametrize("size", [8, 16, 32])
def test_encode_various_spatial_sizes(size):
    m = _tiny_model()
    img = mx.random.normal((1, 3, size, size))
    lat = m.encode(img)
    assert lat.shape[2] == size // 4
    assert lat.shape[3] == size // 4
