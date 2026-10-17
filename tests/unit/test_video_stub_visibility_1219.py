# SPDX-License-Identifier: Apache-2.0
# #1219: video pipeline weight loading must fail loud, never random-init.
# Stub backends (svd / hunyuanvideo / opensora) previously produced
# "successful" garbage generations when weights were missing or unconverted.

from pathlib import Path

import pytest


class TestSVDStubVisibility:
    def test_unet_refuses_missing_weights(self, tmp_path: Path):
        from fusion_mlx.video.svd.unet import SVDTemporalUNet

        with pytest.raises(FileNotFoundError, match="SVD UNet"):
            SVDTemporalUNet.from_pretrained(str(tmp_path))

    def test_vae_refuses_missing_weights(self, tmp_path: Path):
        from fusion_mlx.video.svd.vae import SVDVideoVAE

        with pytest.raises(FileNotFoundError, match="SVD VAE"):
            SVDVideoVAE.from_pretrained(str(tmp_path))

    def test_clip_vision_refuses_missing_weights(self, tmp_path: Path):
        from fusion_mlx.video.svd.clip_vision import SVDCLIPVisionEncoder

        enc = SVDCLIPVisionEncoder(model_path=str(tmp_path))
        with pytest.raises(FileNotFoundError, match="CLIP vision"):
            enc.encode_image(str(tmp_path / "unused.jpg"))


class TestHunyuanVideoStubVisibility:
    def test_dit_refuses_missing_weights(self, tmp_path: Path):
        from fusion_mlx.video.hunyuanvideo.dit import HunyuanVideoDiT

        with pytest.raises(FileNotFoundError, match="DiT"):
            HunyuanVideoDiT.from_pretrained(str(tmp_path))

    def test_vae_refuses_missing_weights(self, tmp_path: Path):
        from fusion_mlx.video.hunyuanvideo.vae import HunyuanVideoVAE

        with pytest.raises(FileNotFoundError, match="VAE"):
            HunyuanVideoVAE.from_pretrained(str(tmp_path))

    def test_text_encoder_refuses_missing_weights(self, tmp_path: Path):
        from fusion_mlx.video.hunyuanvideo.text_encoder import (
            HunyuanDualTextEncoder,
        )

        with pytest.raises(FileNotFoundError, match="text encoder"):
            HunyuanDualTextEncoder.from_pretrained(str(tmp_path))

    def test_vae_refuses_partial_weight_load(self, tmp_path: Path):
        # #1219: VAE remap must cover every param — partially-random model
        # produces garbage decode indistinguishable from success.
        import mlx.core as mx
        import numpy as np

        from fusion_mlx.video.hunyuanvideo.vae import (
            HunyuanVideoVAE,
            _remap_vae_weights,
        )

        vae = HunyuanVideoVAE()
        # Feed a diffusers-named subset that remaps cleanly but is incomplete.
        partial = {
            "encoder.conv_in.conv.weight": mx.array(
                np.zeros((128, 3, 3, 3, 3), dtype=np.float32)
            ),
            "encoder.conv_in.conv.bias": mx.array(np.zeros((128,), dtype=np.float32)),
        }
        assert set(_remap_vae_weights(partial)) == {
            "encoder.conv_in.conv.weight",
            "encoder.conv_in.conv.bias",
        }
        mx.save_safetensors(
            str(tmp_path / "vae.safetensors"),
            partial,
        )
        with pytest.raises(ValueError, match="unmatched"):
            HunyuanVideoVAE.from_pretrained(str(tmp_path))


class TestOpenSoraStubVisibility:
    def test_generate_refuses_missing_text_encoder(self, tmp_path: Path):
        from fusion_mlx.video.opensora.generate import generate_video

        with pytest.raises(RuntimeError, match="text encoder"):
            generate_video(model_dir=str(tmp_path), prompt="test")

    def test_generate_refuses_missing_vae(self, tmp_path: Path):
        from fusion_mlx.video.opensora.generate import generate_video

        class FakeEnc:
            def encode(self, texts):
                import mlx.core as mx

                return mx.zeros((len(texts), 8, 16)), mx.zeros((len(texts), 4))

        with pytest.raises(RuntimeError, match="VAE"):
            generate_video(
                model_dir=str(tmp_path), prompt="test", text_encoder=FakeEnc()
            )

    def test_generate_refuses_missing_weights(self, tmp_path: Path):
        from fusion_mlx.video.opensora.generate import generate_video

        class FakeEnc:
            def encode(self, texts):
                import mlx.core as mx

                return mx.zeros((len(texts), 8, 16)), mx.zeros((len(texts), 4))

        class FakeVAE:
            def decode(self, x):
                return x

        with pytest.raises(FileNotFoundError, match="weights.npz"):
            generate_video(
                model_dir=str(tmp_path),
                prompt="test",
                text_encoder=FakeEnc(),
                vae=FakeVAE(),
            )

    def test_backend_start_refuses_missing_artifacts(self, tmp_path: Path):
        import asyncio

        from fusion_mlx.engines.video_backends.opensora import OpenSoraBackend

        backend = OpenSoraBackend("opensora-test")
        with pytest.raises(FileNotFoundError, match="weights.npz"):
            asyncio.run(backend.start(str(tmp_path)))

    def test_backend_start_refuses_missing_t5(self, tmp_path: Path):
        import asyncio

        from fusion_mlx.engines.video_backends.opensora import OpenSoraBackend

        (tmp_path / "weights.npz").write_bytes(b"")
        backend = OpenSoraBackend("opensora-test")
        with pytest.raises(FileNotFoundError, match="t5_encoder"):
            asyncio.run(backend.start(str(tmp_path)))

    def test_backend_start_refuses_missing_vae(self, tmp_path: Path):
        import asyncio

        from fusion_mlx.engines.video_backends.opensora import OpenSoraBackend

        (tmp_path / "weights.npz").write_bytes(b"")
        (tmp_path / "t5_encoder").mkdir()
        backend = OpenSoraBackend("opensora-test")
        with pytest.raises(FileNotFoundError, match="vae"):
            asyncio.run(backend.start(str(tmp_path)))
