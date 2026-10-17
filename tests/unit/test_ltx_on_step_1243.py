# SPDX-License-Identifier: Apache-2.0
"""#1243: ltx2/ltx2_5 denoise loops fire the per-step on_step hook.

The backends previously accepted on_step but never emitted it, so the #1189
video watchdog heartbeat never advanced and >600s renders were falsely
poisoned as hung."""

import pytest

pytest.importorskip("mlx")  # suite needs mlx runtime; skip if absent

import mlx.core as mx


def _fake_transformer(video, audio):
    return mx.zeros_like(video.latent), None


def test_ltx2_denoise_distilled_fires_on_step_per_step():
    from fusion_mlx.video.ltx2.denoise import denoise_distilled

    latents = mx.zeros((1, 8, 3, 4, 4), dtype=mx.float32)
    positions = mx.zeros((1, 3, 3), dtype=mx.float32)
    text_embeddings = mx.zeros((1, 2, 8), dtype=mx.float32)
    sigmas = [1.0, 0.5, 0.1, 0.0]

    received: list[tuple[int, int]] = []

    def on_step(step, total):
        received.append((step, total))

    out, audio_out = denoise_distilled(
        latents,
        positions,
        text_embeddings,
        _fake_transformer,
        sigmas,
        verbose=False,
        on_step=on_step,
    )
    mx.eval(out)
    assert audio_out is None
    assert received == [(1, 3), (2, 3), (3, 3)]


def test_ltx2_5_denoise_distilled_t2v_fires_on_step_per_step():
    from fusion_mlx.video.ltx2_5.denoise import denoise_distilled_t2v

    latents = mx.zeros((1, 8, 3, 4, 4), dtype=mx.float32)
    positions = mx.zeros((1, 3, 3), dtype=mx.float32)
    text_embeddings = mx.zeros((1, 2, 8), dtype=mx.float32)
    sigmas = [1.0, 0.5, 0.1, 0.0]

    received: list[tuple[int, int]] = []

    def on_step(step, total):
        received.append((step, total))

    out = denoise_distilled_t2v(
        latents,
        positions,
        text_embeddings,
        _fake_transformer,
        sigmas,
        verbose=False,
        on_step=on_step,
    )
    mx.eval(out)
    assert received == [(1, 3), (2, 3), (3, 3)]


def test_ltx2_denoise_on_step_none_is_noop():
    from fusion_mlx.video.ltx2.denoise import denoise_distilled

    latents = mx.zeros((1, 8, 3, 4, 4), dtype=mx.float32)
    positions = mx.zeros((1, 3, 3), dtype=mx.float32)
    text_embeddings = mx.zeros((1, 2, 8), dtype=mx.float32)
    sigmas = [1.0, 0.0]

    out, _ = denoise_distilled(
        latents,
        positions,
        text_embeddings,
        _fake_transformer,
        sigmas,
        verbose=False,
    )
    mx.eval(out)
