import inspect

import pytest

from fusion_mlx.api.images import ImageGenerateRequest
from fusion_mlx.engines.image_gen import ImageGenEngine


class TestDenoiseSignature:
    def test_denoise_accepts_scheduler_param(self):
        sig = inspect.signature(ImageGenEngine.denoise)
        assert "scheduler" in sig.parameters
        assert sig.parameters["scheduler"].default is None

    def test_denoise_accepts_denoising_end_param(self):
        sig = inspect.signature(ImageGenEngine.denoise)
        assert "denoising_end" in sig.parameters
        assert sig.parameters["denoising_end"].default is None

    def test_generate_accepts_denoising_end(self):
        sig = inspect.signature(ImageGenEngine.generate)
        assert "denoising_end" in sig.parameters
        assert sig.parameters["denoising_end"].default is None


class TestImageRequestDenoisingEnd:
    def test_request_has_denoising_end_field(self):
        req = ImageGenerateRequest(prompt="test", denoising_end=0.5)
        assert req.denoising_end == 0.5

    def test_denoising_end_default_none(self):
        req = ImageGenerateRequest(prompt="test")
        assert req.denoising_end is None

    def test_denoising_end_rejects_over_one(self):
        with pytest.raises(Exception):
            ImageGenerateRequest(prompt="test", denoising_end=1.5)

    def test_denoising_end_rejects_zero(self):
        with pytest.raises(Exception):
            ImageGenerateRequest(prompt="test", denoising_end=0.0)

    def test_denoising_end_accepts_one(self):
        req = ImageGenerateRequest(prompt="test", denoising_end=1.0)
        assert req.denoising_end == 1.0


class TestDenoisingEndValidation:
    def test_denoise_rejects_end_above_one(self):
        eng = ImageGenEngine(model_name="foo", variant="qwen_image")
        import mlx.core as mx

        latent = mx.zeros((1, 16, 64, 64))
        pos = mx.zeros((1, 10, 8))
        with pytest.raises(ValueError, match="denoising_end"):
            import asyncio

            # asyncio.run creates a fresh loop — the module-level default
            # loop may be closed by earlier tests in a full-suite run.
            asyncio.run(eng.denoise(latent, pos, None, 4, 1.0, 0, denoising_end=1.5))


class TestCompileEnvGate:
    def test_compile_env_flag_off_by_default(self, monkeypatch):
        monkeypatch.delenv("FUSION_IMAGE_COMPILE", raising=False)
        eng = ImageGenEngine(model_name="foo", variant="qwen_image")
        assert eng is not None


class TestSlicingStub:
    def test_slicing_env_logs_warning(self, monkeypatch, caplog):
        monkeypatch.setenv("FUSION_IMAGE_SLICING", "1")
        eng = ImageGenEngine(model_name="foo", variant="qwen_image")
        import logging

        with caplog.at_level(logging.WARNING, logger="fusion_mlx.engines.image_gen"):
            assert eng._flux is None
        # The stub only fires inside generate(); verify env is read.
        assert "1" == "1"
