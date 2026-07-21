# SPDX-License-Identifier: Apache-2.0
# Tests for issue #171 streaming on_step progress callback.
# Monkeypatched / fake-model only - no real mflux/MLX model load.

import asyncio
from unittest.mock import AsyncMock

import mlx.core as mx

from fusion_mlx.engines._progress import make_sync_step_callback
from fusion_mlx.engines.image_gen import _StepProgressInLoop


def test_make_sync_step_callback_returns_none_when_no_callback():
    loop = asyncio.new_event_loop()
    try:
        assert make_sync_step_callback(None, loop) is None
        assert make_sync_step_callback(None, None) is None
        cb = lambda s, t: None  # noqa: E731
        assert make_sync_step_callback(cb, None) is None
    finally:
        loop.close()


def test_make_sync_step_callback_schedules_and_drains():
    loop = asyncio.new_event_loop()
    received: list[tuple[int, int]] = []

    async def on_step(step, total):
        received.append((step, total))

    sync_cb = make_sync_step_callback(on_step, loop)
    assert sync_cb is not None
    sync_cb(1, 3)
    sync_cb(2, 3)
    sync_cb(3, 3)
    loop.run_until_complete(asyncio.sleep(0.05))
    assert received == [(1, 3), (2, 3), (3, 3)]
    loop.close()


def test_step_progress_in_loop_via_generation_context():
    from mflux.callbacks.callback_registry import CallbackRegistry
    from mflux.callbacks.generation_context import GenerationContext

    received: list[tuple[int, int]] = []
    sync_cb = lambda step, total: received.append((step, total))  # noqa: E731
    sub = _StepProgressInLoop(sync_cb, total=3)
    registry = CallbackRegistry()
    registry.register(sub)
    assert sub in registry.in_loop
    ctx = GenerationContext(registry, seed=1, prompt="p", config=None)
    for t in [0.9, 0.5, 0.1]:
        ctx.in_loop(t, None, time_steps=[0.9, 0.5, 0.1])
    assert received == [(1, 3), (2, 3), (3, 3)]


def test_step_progress_in_loop_count_resets_per_instance():
    received: list[tuple[int, int]] = []
    sync_cb = lambda step, total: received.append((step, total))  # noqa: E731
    sub = _StepProgressInLoop(sync_cb, total=4)
    for _ in range(4):
        sub.call_in_loop(0.5, 1, "p", None, None, None)
    assert received == [(1, 4), (2, 4), (3, 4), (4, 4)]
    sub2 = _StepProgressInLoop(sync_cb, total=4)
    for _ in range(2):
        sub2.call_in_loop(0.5, 1, "p", None, None, None)
    assert received[-1] == (2, 4)


def test_video_engine_generate_sets_on_step_param(monkeypatch):
    from fusion_mlx.engines.video import VideoGenEngine

    captured: dict = {}

    class FakeBackend:
        _loaded = True

        async def generate(self, params):
            captured["params"] = params
            return [b"fake"]

    monkeypatch.setattr(
        "fusion_mlx.engines.video.resolve_backend", lambda *a, **k: FakeBackend()
    )
    eng = VideoGenEngine("fake-model")
    eng._begin_activity = lambda *a, **k: "id"
    eng._update_activity = lambda *a, **k: None
    eng._finish_activity = AsyncMock()

    async def cb(step, total):
        pass

    result = asyncio.run(eng.generate("prompt", on_step=cb))
    assert result == [b"fake"]
    assert captured["params"].on_step is cb


def test_video_engine_generate_on_step_defaults_none(monkeypatch):
    from fusion_mlx.engines.video import VideoGenEngine

    captured: dict = {}

    class FakeBackend:
        _loaded = True

        async def generate(self, params):
            captured["params"] = params
            return [b"fake"]

    monkeypatch.setattr(
        "fusion_mlx.engines.video.resolve_backend", lambda *a, **k: FakeBackend()
    )
    eng = VideoGenEngine("fake-model")
    eng._begin_activity = lambda *a, **k: "id"
    eng._update_activity = lambda *a, **k: None
    eng._finish_activity = AsyncMock()

    asyncio.run(eng.generate("prompt"))
    assert captured["params"].on_step is None


def test_legacy_denoise_on_step_sync_fires_per_step():
    from fusion_mlx.video.ltx_video_legacy.denoise import denoise

    n_tokens = 4
    c = 8
    latents = mx.zeros((1, n_tokens, c), dtype=mx.float32)
    pixel_coords = mx.zeros((1, 3, n_tokens), dtype=mx.float32)
    prompt_embeds = mx.zeros((1, 2, c), dtype=mx.float32)
    prompt_mask = mx.ones((1, 2), dtype=mx.int32)
    steps = 3

    class FakeScheduler:
        init_noise_sigma = 1.0

        def set_timesteps(self, n, samples_shape=None):
            self.timesteps = mx.array([0.9, 0.5, 0.1])

        def scale_model_input(self, x, t):
            return x

        def step(self, noise_pred, t, latents):
            return latents

    def fake_transformer(
        latent_in,
        *,
        indices_grid,
        encoder_hidden_states,
        encoder_attention_mask,
        timestep,
    ):
        return mx.zeros_like(latent_in)

    received: list[tuple[int, int]] = []

    def on_step_sync(step, total):
        received.append((step, total))

    denoise(
        fake_transformer,
        FakeScheduler(),
        latents,
        pixel_coords,
        prompt_embeds,
        prompt_mask,
        negative_embeds=mx.zeros_like(prompt_embeds),
        negative_attn_mask=mx.zeros_like(prompt_mask),
        guidance_scale=1.0,
        num_inference_steps=steps,
        frame_rate=24.0,
        latent_shape=(1, c, 2, 2, 2),
        dtype=mx.float32,
        on_step_sync=on_step_sync,
    )
    assert received == [(1, steps), (2, steps), (3, steps)]


def test_wan2_generate_video_accepts_on_step_sync_param():
    import inspect

    from fusion_mlx.video.wan2.generate import generate_video

    sig = inspect.signature(generate_video)
    assert "on_step_sync" in sig.parameters
    assert sig.parameters["on_step_sync"].default is None
