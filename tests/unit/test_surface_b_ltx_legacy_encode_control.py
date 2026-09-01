import inspect

import pytest

from fusion_mlx.engines.video_backends.ltx_video_legacy import LegacyLTXBackend
from fusion_mlx.video.ltx_video_legacy.denoise import denoise


def test_denoise_accepts_inpaint_params_default_none():
    sig = inspect.signature(denoise)
    assert "inpaint_mask" in sig.parameters
    assert "init_latent" in sig.parameters
    assert sig.parameters["inpaint_mask"].default is None
    assert sig.parameters["init_latent"].default is None


def test_encode_control_fail_visible_on_controlnet_image():
    import asyncio

    backend = LegacyLTXBackend.__new__(LegacyLTXBackend)
    with pytest.raises(RuntimeError, match="ControlNet"):
        asyncio.run(backend.encode_control(controlnet_image="frame.png"))


def test_encode_control_pure_t2v_returns_none():
    import asyncio

    backend = LegacyLTXBackend.__new__(LegacyLTXBackend)
    assert asyncio.run(backend.encode_control()) is None


def test_generate_fail_visible_on_controlnet_image():
    src = inspect.getsource(LegacyLTXBackend.generate)
    assert "controlnet_image is not None" in src
    assert "RuntimeError" in src
