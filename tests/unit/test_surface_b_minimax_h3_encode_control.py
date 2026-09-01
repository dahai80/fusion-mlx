import asyncio
import inspect

import pytest

from fusion_mlx.engines.video_backends.minimax_h3 import MiniMaxH3Backend
from fusion_mlx.video.minimax_h3.generate import generate_video


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


def test_encode_control_fail_visible_on_controlnet_image():
    # #736 Surface B: minimax_h3 has no per-backend ControlNet model. A caller
    # asking for ControlNet must fail visibly (Rule 12), not silently degrade
    # to T2V. encode_control is async -> drive it via the coroutine.
    backend = MiniMaxH3Backend.__new__(MiniMaxH3Backend)
    with pytest.raises(RuntimeError, match="ControlNet"):
        asyncio.run(backend.encode_control(controlnet_image="frame.png"))


def test_encode_control_pure_t2v_returns_none():
    backend = MiniMaxH3Backend.__new__(MiniMaxH3Backend)
    assert asyncio.run(backend.encode_control()) is None


def test_generate_video_fail_visible_on_controlnet_image():
    # generate_video itself must raise before any model load when a caller
    # threads controlnet_image directly (engine-layer guard mirrors backend).
    src = inspect.getsource(generate_video)
    assert "controlnet_image is not None" in src
    assert "RuntimeError" in src
