# SPDX-License-Identifier: Apache-2.0
"""#gap5: LTX2_5VideoVAE adopts VideoVAEBase (PRD v1 §4.2).

Thin wrapper over the existing function-loaded VideoEncoder + LTX2VideoDecoder
so LTX2_5 is polymorphic with MiniMaxH3VideoVAE under the unified base.
"""

import mlx.core as mx
import mlx.nn as nn
import pytest

from fusion_mlx.video.common import VideoVAEBase
from fusion_mlx.video.ltx2_5.video_vae import LTX2_5VideoVAE


class _FakeEncoder(nn.Module):
    def __call__(self, sample):
        return sample * 2


class _FakeDecoder(nn.Module):
    class _PCS:
        mean = mx.array([0.1])
        std = mx.array([0.2])

    per_channel_statistics = _PCS()

    def __call__(self, sample):
        return sample + 1

    def decode_tiled(self, sample, tiling_config=None, **kw):
        return sample + 10


def test_is_video_vae_base():
    vae = LTX2_5VideoVAE(_FakeEncoder(), _FakeDecoder())
    assert isinstance(vae, VideoVAEBase)
    assert vae.name == "ltx2_5"


def test_encode_delegates_to_encoder():
    vae = LTX2_5VideoVAE(_FakeEncoder(), _FakeDecoder())
    x = mx.array([1.0])
    out = vae.encode(x)
    assert float(out[0]) == 2.0


def test_decode_delegates_to_decoder():
    vae = LTX2_5VideoVAE(_FakeEncoder(), _FakeDecoder())
    x = mx.array([1.0])
    out = vae.decode(x)
    assert float(out[0]) == 2.0


def test_decode_tiled_delegates():
    vae = LTX2_5VideoVAE(_FakeEncoder(), _FakeDecoder())
    x = mx.array([1.0])
    out = vae.decode_tiled(x, tile_size=256)
    assert float(out[0]) == 11.0


def test_stats_reports_mean_std():
    vae = LTX2_5VideoVAE(_FakeEncoder(), _FakeDecoder())
    s = vae.stats()
    assert s["vae"] == "ltx2_5"
    assert "latent_mean" in s
    assert "latent_std" in s


def test_encode_without_encoder_raises():
    vae = LTX2_5VideoVAE(None, _FakeDecoder())
    with pytest.raises(RuntimeError):
        vae.encode(mx.array([1.0]))


def test_decode_without_decoder_raises():
    vae = LTX2_5VideoVAE(_FakeEncoder(), None)
    with pytest.raises(RuntimeError):
        vae.decode(mx.array([1.0]))


def test_release_is_noop_base():
    vae = LTX2_5VideoVAE(_FakeEncoder(), _FakeDecoder())
    vae.release()  # base no-op, must not raise
