# SPDX-License-Identifier: Apache-2.0
"""Regression tests for issue #1018.

#1018: skyreels_v3 had multiple zero-tensor stub outputs (UMT5/CLIP text
encoder, VAE decode, A2V audio/text embeds) that silently produced distorted/
all-black video with no request-level error or warning.

Fix (#1018): added is_stub() to UMT5Encoder + SkyReelsVAE (CLIPTextEncoder
already had one), log WARNING on every stub return so degradation is visible.
"""

from __future__ import annotations

import logging

import pytest

from fusion_mlx.video.skyreels_v3.text_encoder import UMT5Encoder
from fusion_mlx.video.skyreels_v3.vae import SkyReelsVAE


class TestUMT5EncoderIsStub:
    def test_real_encoder_not_stub(self):
        enc = UMT5Encoder.__new__(UMT5Encoder)
        enc.encoder = object()  # non-None = real
        enc._tokenizer = None
        assert enc.is_stub() is False

    def test_none_encoder_is_stub(self):
        enc = UMT5Encoder.__new__(UMT5Encoder)
        enc.encoder = None
        enc._tokenizer = None
        assert enc.is_stub() is True

    def test_stub_tokenizer_is_stub(self):
        enc = UMT5Encoder.__new__(UMT5Encoder)
        enc.encoder = object()
        enc._tokenizer = "stub"
        assert enc.is_stub() is True


class TestSkyReelsVAEIsStub:
    def test_real_vae_not_stub(self):
        vae = SkyReelsVAE.__new__(SkyReelsVAE)
        vae.vae = object()  # non-None = real
        from fusion_mlx.video.skyreels_v3.vae import SkyReelsVAEConfig

        vae.config = SkyReelsVAEConfig()
        assert vae.is_stub() is False

    def test_none_vae_is_stub(self):
        vae = SkyReelsVAE.__new__(SkyReelsVAE)
        vae.vae = None
        from fusion_mlx.video.skyreels_v3.vae import SkyReelsVAEConfig

        vae.config = SkyReelsVAEConfig()
        assert vae.is_stub() is True


class TestStubWarnings:
    def test_umt5_encode_text_stub_logs_warning(self, caplog):
        enc = UMT5Encoder.__new__(UMT5Encoder)
        from fusion_mlx.video.skyreels_v3.text_encoder import UMT5Config

        enc.config = UMT5Config()
        enc.encoder = None
        enc._tokenizer = "stub"
        enc._text_cache = None
        with caplog.at_level(
            logging.WARNING, logger="fusion_mlx.video.skyreels_v3.text_encoder"
        ):
            result = enc.encode_text("test prompt")
        assert result.shape == (1, 512, enc.config.d_model)
        assert any("stub" in r.getMessage().lower() for r in caplog.records)

    def test_umt5_call_stub_logs_warning(self, caplog):
        import mlx.core as mx

        enc = UMT5Encoder.__new__(UMT5Encoder)
        from fusion_mlx.video.skyreels_v3.text_encoder import UMT5Config

        enc.config = UMT5Config()
        enc.encoder = None
        with caplog.at_level(
            logging.WARNING, logger="fusion_mlx.video.skyreels_v3.text_encoder"
        ):
            result = enc(mx.zeros((1, 10), dtype=mx.int32))
        assert result.shape == (1, 10, enc.config.d_model)
        assert any("stub" in r.getMessage().lower() for r in caplog.records)

    def test_vae_decode_stub_logs_warning(self, caplog):
        import mlx.core as mx

        vae = SkyReelsVAE.__new__(SkyReelsVAE)
        from fusion_mlx.video.skyreels_v3.vae import SkyReelsVAEConfig

        vae.config = SkyReelsVAEConfig()
        vae.vae = None
        vae.vae_mean = mx.zeros((1, 16, 1, 1, 1))
        vae.vae_std = mx.ones((1, 16, 1, 1, 1))
        latent = mx.zeros((1, 16, 2, 8, 8))
        with caplog.at_level(
            logging.WARNING, logger="fusion_mlx.video.skyreels_v3.vae"
        ):
            result = vae.decode(latent)
        assert result.shape == (1, 3, 2, 64, 64)
        assert any("stub" in r.getMessage().lower() for r in caplog.records)


if __name__ == "__main__":  # pragma: no cover — convenience only
    pytest.main([__file__, "-v"])
