import inspect

import pytest

from fusion_mlx.engines.video_backends.cogvideox import CogVideoBackend
from fusion_mlx.video.cogvideox.generate import generate_video, run_denoise


def test_generate_video_accepts_surface_kwargs():
    sig = inspect.signature(generate_video)
    for name in (
        "controlnet_image",
        "controlnet_adapter",
        "controlnet_latent",
        "inpaint_mask",
        "init_latent",
    ):
        assert name in sig.parameters, name
        assert sig.parameters[name].default is None, name


def test_run_denoise_threads_controlnet_params():
    sig = inspect.signature(run_denoise)
    assert "controlnet_adapter" in sig.parameters
    assert "controlnet_latent" in sig.parameters
    assert sig.parameters["controlnet_adapter"].default is None
    assert sig.parameters["controlnet_latent"].default is None


def test_encode_control_fail_visible_on_controlnet_image():
    # #731 Surface B: cogvideox has no per-backend ControlNet model. A caller
    # asking for ControlNet must fail visibly (Rule 12), not silently degrade
    # to T2V. encode_control is async -> drive it via the coroutine.
    import asyncio

    backend = CogVideoBackend.__new__(CogVideoBackend)
    with pytest.raises(RuntimeError, match="ControlNet"):
        asyncio.run(backend.encode_control(controlnet_image="frame.png"))


def test_encode_control_pure_t2v_returns_none():
    import asyncio

    backend = CogVideoBackend.__new__(CogVideoBackend)
    assert asyncio.run(backend.encode_control()) is None


def test_generate_video_fail_visible_on_controlnet_image():
    # generate_video itself must raise before any model load when a caller
    # threads controlnet_image directly (engine-layer guard mirrors backend).
    src = inspect.getsource(generate_video)
    assert "controlnet_image is not None" in src
    assert "RuntimeError" in src
