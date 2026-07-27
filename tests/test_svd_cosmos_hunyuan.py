# SPDX-License-Identifier: Apache-2.0
# Tests for SVD (#212), Cosmos (#213), HunyuanVideo (#214) backends.

import asyncio
from pathlib import Path

import pytest

from fusion_mlx.engines.video_backends.base import VideoGenParams, validate_params
from fusion_mlx.engines.video_backends.cosmos import CosmosBackend
from fusion_mlx.engines.video_backends.hunyuanvideo import HunyuanVideoBackend
from fusion_mlx.engines.video_backends.svd import SVDBackend

# ---------------------------------------------------------------------------
# SVD (#212)
# ---------------------------------------------------------------------------


class TestSVDBackend:
    def test_detect_svd(self):
        assert SVDBackend.detect("svd-xt")

    def test_detect_stable_video_diffusion(self):
        assert SVDBackend.detect("stable-video-diffusion-xt")

    def test_detect_img2vid(self):
        assert SVDBackend.detect("img2vid-xt")

    def test_no_detect_other(self):
        assert not SVDBackend.detect("wan2.1-1.3b")

    def test_name_and_i2v(self):
        assert SVDBackend.name == "svd"
        assert SVDBackend.supports_i2v is True

    def test_constraints(self):
        b = SVDBackend("svd-test")
        c = b.constraints()
        assert c.supports_i2v is True
        assert c.max_n == 4
        assert c.dim_divisibility == 64
        assert c.num_frames_validator(14)
        assert c.num_frames_validator(25)

    def test_start_stop(self):
        b = SVDBackend("svd-test")
        asyncio.run(b.start("svd-test"))
        assert b._loaded
        asyncio.run(b.stop())
        assert not b._loaded

    def test_generate_mp4(self, monkeypatch, tmp_path):
        b = SVDBackend("svd-test")
        b._loaded = True

        def fake_generate_video(*args, **kwargs):
            output_path = kwargs.get("output_path")
            Path(output_path).write_bytes(b"FAKE_SVD_MP4")

        monkeypatch.setattr(
            "fusion_mlx.video.svd.generate.generate_video",
            fake_generate_video,
        )

        params = VideoGenParams(
            prompt="test", image="test.jpg", num_frames=14, output_format="mp4"
        )
        result = asyncio.run(b.generate(params))
        assert len(result) == 1
        assert isinstance(result[0], bytes)

    def test_generate_no_image_passes_validation(self):
        b = SVDBackend("svd-test")
        c = b.constraints()
        validate_params(
            c,
            num_frames=14,
            width=576,
            height=1024,
            n=1,
            image=None,
        )


# ---------------------------------------------------------------------------
# Cosmos (#213)
# ---------------------------------------------------------------------------


class TestCosmosBackend:
    def test_detect_cosmos(self):
        assert CosmosBackend.detect("cosmos-7b")

    def test_detect_predict2(self):
        assert CosmosBackend.detect("cosmos-predict2-2b")

    def test_detect_video2world(self):
        assert CosmosBackend.detect("video2world-2b")

    def test_no_detect_other(self):
        assert not CosmosBackend.detect("wan2.1")

    def test_name(self):
        assert CosmosBackend.name == "cosmos"

    def test_predict2_i2v(self):
        b = CosmosBackend("cosmos-predict2-2b")
        assert b._is_predict2 is True
        c = b.constraints()
        assert c.supports_i2v is True

    def test_7b_no_i2v(self):
        b = CosmosBackend("cosmos-7b")
        assert b._is_predict2 is False
        c = b.constraints()
        assert c.supports_i2v is False

    def test_7b_num_frames_validator(self):
        b = CosmosBackend("cosmos-7b")
        c = b.constraints()
        assert c.num_frames_validator(1)
        assert c.num_frames_validator(5)
        assert not c.num_frames_validator(2)

    def test_start_stop(self):
        b = CosmosBackend("cosmos-test")
        asyncio.run(b.start("cosmos-test"))
        assert b._loaded
        asyncio.run(b.stop())
        assert not b._loaded

    def test_generate_mp4_predict2(self, monkeypatch, tmp_path):
        b = CosmosBackend("cosmos-predict2-2b")
        b._loaded = True

        def fake_generate_video(*args, **kwargs):
            output_path = kwargs.get("output_path")
            Path(output_path).write_bytes(b"FAKE_COSMOS_MP4")

        monkeypatch.setattr(
            "fusion_mlx.video.cosmos.generate.generate_video",
            fake_generate_video,
        )

        params = VideoGenParams(
            prompt="test",
            image="test.jpg",
            num_frames=41,
            output_format="mp4",
        )
        result = asyncio.run(b.generate(params))
        assert len(result) == 1
        assert isinstance(result[0], bytes)

    def test_7b_rejects_i2v(self):
        b = CosmosBackend("cosmos-7b")
        c = b.constraints()
        with pytest.raises(ValueError, match="does not support"):
            validate_params(
                c,
                num_frames=121,
                width=848,
                height=480,
                n=1,
                image="test.jpg",
            )


# ---------------------------------------------------------------------------
# HunyuanVideo (#214)
# ---------------------------------------------------------------------------


class TestHunyuanVideoBackend:
    def test_detect_hunyuanvideo(self):
        assert HunyuanVideoBackend.detect("hunyuanvideo")

    def test_detect_hunyuan_video(self):
        assert HunyuanVideoBackend.detect("hunyuan-video")

    def test_detect_hunyuan_video_underscore(self):
        assert HunyuanVideoBackend.detect("hunyuan_video")

    def test_no_detect_other(self):
        assert not HunyuanVideoBackend.detect("wan2.1")

    def test_name_and_i2v(self):
        assert HunyuanVideoBackend.name == "hunyuanvideo"
        assert HunyuanVideoBackend.supports_i2v is True

    def test_constraints(self):
        b = HunyuanVideoBackend("hunyuanvideo-test")
        c = b.constraints()
        assert c.supports_i2v is True
        assert c.max_n == 1
        assert c.dim_divisibility == 16
        assert c.num_frames_validator(33)
        assert c.num_frames_validator(65)
        assert c.num_frames_validator(129)
        assert not c.num_frames_validator(10)

    def test_start_stop(self):
        b = HunyuanVideoBackend("hunyuanvideo-test")
        asyncio.run(b.start("hunyuanvideo-test"))
        assert b._loaded
        asyncio.run(b.stop())
        assert not b._loaded

    def test_generate_mp4(self, monkeypatch, tmp_path):
        b = HunyuanVideoBackend("hunyuanvideo-test")
        b._loaded = True

        def fake_generate_video(*args, **kwargs):
            output_path = kwargs.get("output_path")
            Path(output_path).write_bytes(b"FAKE_HV_MP4")

        monkeypatch.setattr(
            "fusion_mlx.video.hunyuanvideo.generate.generate_video",
            fake_generate_video,
        )

        params = VideoGenParams(
            prompt="test",
            num_frames=33,
            width=720,
            height=480,
            output_format="mp4",
        )
        result = asyncio.run(b.generate(params))
        assert len(result) == 1
        assert isinstance(result[0], bytes)

    def test_generate_with_image(self, monkeypatch, tmp_path):
        b = HunyuanVideoBackend("hunyuanvideo-test")
        b._loaded = True

        captured = {}

        def fake_generate_video(*args, **kwargs):
            captured.update(kwargs)
            output_path = kwargs.get("output_path")
            Path(output_path).write_bytes(b"FAKE_HV_I2V_MP4")

        monkeypatch.setattr(
            "fusion_mlx.video.hunyuanvideo.generate.generate_video",
            fake_generate_video,
        )

        params = VideoGenParams(
            prompt="test i2v",
            image="test.jpg",
            num_frames=33,
            width=720,
            height=480,
            output_format="mp4",
        )
        result = asyncio.run(b.generate(params))
        assert len(result) == 1
        assert captured.get("image") == "test.jpg"


# ---------------------------------------------------------------------------
# Registry integration
# ---------------------------------------------------------------------------


class TestRegistryIntegration:
    def test_backends_registered(self):
        from fusion_mlx.engines.video_backends import BACKENDS

        assert "svd" in BACKENDS
        assert "cosmos" in BACKENDS
        assert "hunyuanvideo" in BACKENDS

    def test_aliases(self):
        from fusion_mlx.engines.video_backends import _ALIASES

        assert _ALIASES.get("svd") == "svd"
        assert _ALIASES.get("stable-video-diffusion") == "svd"
        assert _ALIASES.get("cosmos") == "cosmos"
        assert _ALIASES.get("cosmos-predict2") == "cosmos"
        assert _ALIASES.get("hunyuan-video") == "hunyuanvideo"

    def test_resolve_svd(self):
        from fusion_mlx.engines.video_backends import resolve_backend

        b = resolve_backend("svd-xt", explicit="svd")
        assert isinstance(b, SVDBackend)

    def test_resolve_cosmos(self):
        from fusion_mlx.engines.video_backends import resolve_backend

        b = resolve_backend("cosmos-7b", explicit="cosmos")
        assert isinstance(b, CosmosBackend)

    def test_resolve_hunyuanvideo(self):
        from fusion_mlx.engines.video_backends import resolve_backend

        b = resolve_backend("hunyuanvideo", explicit="hunyuanvideo")
        assert isinstance(b, HunyuanVideoBackend)
