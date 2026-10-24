# SPDX-License-Identifier: Apache-2.0
# H5 (#1354 audit, 1019 report): every registered video backend must honour
# the async VideoBackend contract (base.py). cogvideox used sync overrides of
# start/stop/generate, so selecting it through _ALIASES raised
# "TypeError: object NoneType can't be used in 'await' expression".

import concurrent.futures
import inspect
import threading

import pytest

from fusion_mlx.engines.video_backends import (
    VideoBackend,
)
from fusion_mlx.engines.video_backends import (
    base as video_base_mod,
)
from fusion_mlx.engines.video_backends import cogvideox as cog_mod
from fusion_mlx.engines.video_backends.base import VideoGenParams
from fusion_mlx.engines.video_backends.cogvideox import CogVideoBackend

_ASYNC_LIFECYCLE = ("start", "stop", "generate")
_BASE_CALL_ARGS = {
    "start": ("a-model",),
    "stop": (),
    "generate": (video_base_mod.VideoGenParams(prompt="p"),),
}


def _registry():
    import fusion_mlx.engines.video_backends as vb

    return vb._get_backends()


def test_every_registered_backend_overrides_async_contract():
    registry = _registry()
    assert registry, "video backend registry is empty"
    problems = []
    for key, cls in registry.items():
        for name in _ASYNC_LIFECYCLE:
            base_impl = getattr(VideoBackend, name)
            impl = getattr(cls, name, None)
            if impl is None or impl is base_impl:
                problems.append(f"{key}.{name}: not overridden")
                continue
            if not inspect.iscoroutinefunction(impl):
                problems.append(f"{key}.{name}: sync override of async base")
                continue
            try:
                inspect.signature(impl).bind(None, *_BASE_CALL_ARGS[name])
            except TypeError as exc:
                problems.append(f"{key}.{name}: incompatible signature ({exc})")
    assert not problems, "video backend contract violations: " + "; ".join(problems)


def test_alias_targets_are_registered_backends():
    import fusion_mlx.engines.video_backends as vb

    registry = _registry()
    missing = sorted(set(vb._ALIASES.values()) - set(registry))
    assert not missing, f"aliases point at unregistered backends: {missing}"


@pytest.mark.asyncio
async def test_cogvideox_generate_is_awaitable_and_offloads(monkeypatch):
    loop_ident = threading.get_ident()
    calls = []

    def fake_generate_video(**kwargs):
        calls.append(
            {
                "thread": threading.get_ident(),
                "prompt": kwargs["prompt"],
                "negative_prompt": kwargs["negative_prompt"],
                "num_frames": kwargs["num_frames"],
                "steps": kwargs["steps"],
                "guide_scale": kwargs["guide_scale"],
                "seed": kwargs["seed"],
                "session_id": kwargs["session_id"],
                "on_step_sync": kwargs["on_step_sync"],
                "output_path": kwargs["output_path"],
            }
        )
        with open(kwargs["output_path"], "wb") as f:
            f.write(b"COGVIDEO-MP4")

    import fusion_mlx.video.cogvideox.generate as cog_generate_mod

    monkeypatch.setattr(cog_generate_mod, "generate_video", fake_generate_video)
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    monkeypatch.setattr(cog_mod, "get_executor", lambda _name: executor)

    backend = CogVideoBackend("fake-cogvideox-model")
    backend._model_dir = "/tmp/fake-cogvideox-model"
    backend._loaded = True

    coro = backend.generate(
        VideoGenParams(
            prompt="a cat surfing",
            negative_prompt="blurry",
            num_frames=49,
            width=720,
            height=480,
            num_inference_steps=30,
            guide_scale=6.0,
            seed=7,
            session_id="session-1",
        )
    )
    assert inspect.isawaitable(coro)
    try:
        result = await coro
    finally:
        executor.shutdown(wait=True)

    assert result == [b"COGVIDEO-MP4"]
    assert len(calls) == 1
    assert calls[0]["thread"] != loop_ident
    assert calls[0]["prompt"] == "a cat surfing"
    assert calls[0]["negative_prompt"] == "blurry"
    assert calls[0]["num_frames"] == 49
    assert calls[0]["steps"] == 30
    assert calls[0]["guide_scale"] == 6.0
    assert calls[0]["seed"] == 7
    assert calls[0]["session_id"] == "session-1"
    assert calls[0]["on_step_sync"] is None


@pytest.mark.asyncio
async def test_cogvideox_stop_offloads_clear_and_drops_staged_encoder(monkeypatch):
    loop_ident = threading.get_ident()
    seen = {}

    def fake_synchronize():
        seen["synchronize"] = threading.get_ident()

    def fake_clear_cache():
        seen["clear_cache"] = threading.get_ident()

    monkeypatch.setattr(cog_mod.mx, "synchronize", fake_synchronize)
    monkeypatch.setattr(cog_mod.mx, "clear_cache", fake_clear_cache)
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    monkeypatch.setattr(cog_mod, "get_executor", lambda _name: executor)

    backend = CogVideoBackend("fake-cogvideox-model")
    backend._model_dir = "/tmp/fake-cogvideox-model"
    backend._loaded = True
    backend._stage_vae_encoder = object()
    backend._stage_flags["vae_encoder"] = True

    coro = backend.stop()
    assert inspect.isawaitable(coro)
    try:
        await coro
    finally:
        executor.shutdown(wait=True)

    assert backend._stage_vae_encoder is None
    assert "vae_encoder" not in backend._stage_flags
    assert backend._model_dir is None
    assert backend._loaded is False
    assert seen.get("synchronize") is not None
    assert seen.get("clear_cache") is not None
    assert seen["synchronize"] != loop_ident
    assert seen["clear_cache"] != loop_ident
