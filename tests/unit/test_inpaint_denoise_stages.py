# SPDX-License-Identifier: Apache-2.0
# O5.5 inpaint denoise loop — unit test for stage methods.
# Tests prepare_latents/noise_latent/mask_latent/composite/inpaint_latent
# with mock VAE/scheduler/flux. Real-model MSE=0 parity is gated on
# FUSION_MLX_REAL_MODEL_TESTS=1 + a flux model (separate integration test).

import pytest


def test_mask_latent_downsample():
    import mlx.core as mx

    from fusion_mlx.engines._inpaint_denoise import mask_latent

    # 64x64 mask -> 8x8 latent (factor 8)
    mask_full = mx.zeros((64, 64))
    mask_full[0:32, :] = 1.0  # top half = regen
    latent_shape = (4, 8, 8)  # (C, H_lat, W_lat)
    mask = mask_latent(mask_full, latent_shape)
    assert mask.shape == (4, 8, 8)
    # top half of latent should be 1.0, bottom 0.0
    assert float(mx.mean(mask[:, 0:4, :]).item()) == pytest.approx(1.0, abs=0.1)
    assert float(mx.mean(mask[:, 4:8, :]).item()) == pytest.approx(0.0, abs=0.1)


def test_composite_white_keeps_denoised():
    import mlx.core as mx

    from fusion_mlx.engines._inpaint_denoise import composite

    denoised = mx.ones((4, 8, 8)) * 2.0
    noisy_init = mx.ones((4, 8, 8)) * 5.0
    mask = mx.ones((4, 8, 8))  # all white = regen
    result = composite(denoised, noisy_init, mask)
    assert float(mx.mean(result).item()) == pytest.approx(2.0)


def test_composite_black_keeps_init():
    import mlx.core as mx

    from fusion_mlx.engines._inpaint_denoise import composite

    denoised = mx.ones((4, 8, 8)) * 2.0
    noisy_init = mx.ones((4, 8, 8)) * 5.0
    mask = mx.zeros((4, 8, 8))  # all black = freeze
    result = composite(denoised, noisy_init, mask)
    assert float(mx.mean(result).item()) == pytest.approx(5.0)


def test_composite_mixed():
    import mlx.core as mx

    from fusion_mlx.engines._inpaint_denoise import composite

    denoised = mx.ones((4, 8, 8)) * 2.0
    noisy_init = mx.ones((4, 8, 8)) * 5.0
    mask = mx.zeros((4, 8, 8))
    mask[:, 0:4, :] = 1.0  # top half regen, bottom freeze
    result = composite(denoised, noisy_init, mask)
    top_mean = float(mx.mean(result[:, 0:4, :]).item())
    bot_mean = float(mx.mean(result[:, 4:8, :]).item())
    assert top_mean == pytest.approx(2.0)
    assert bot_mean == pytest.approx(5.0)


def test_noise_latent_linear_fallback():
    import mlx.core as mx

    from fusion_mlx.engines._inpaint_denoise import noise_latent

    latent = mx.ones((4, 8, 8))
    # scheduler=None -> linear mix fallback
    noisy = noise_latent(latent, None, t_start=0.5, seed=42)
    assert noisy.shape == latent.shape
    # t_start=0.5 -> alpha_bar=0.5 -> noisy = sqrt(0.5)*1 + sqrt(0.5)*noise
    # mean should be near sqrt(0.5) ~ 0.707 (noise has mean ~0)
    assert abs(float(mx.mean(noisy).item()) - 0.7071) < 0.3


def test_prepare_latents_calls_encode():
    import mlx.core as mx

    from fusion_mlx.engines._inpaint_denoise import prepare_latents

    class MockVAE:
        def __init__(self):
            self.called = False

        def encode(self, x):
            self.called = True
            return mx.zeros((4, 8, 8))

    vae = MockVAE()
    img = mx.zeros((64, 64, 3))
    result = prepare_latents(vae, img)
    assert vae.called
    assert result.shape == (4, 8, 8)


def test_inpaint_latent_per_step_composite():
    import mlx.core as mx

    from fusion_mlx.engines._inpaint_denoise import inpaint_latent

    class MockScheduler:
        timesteps = [1.0, 0.5, 0.0]

        def step(self, eps, latents, t):
            return latents - eps * 0.01

    class MockFlux:
        scheduler = MockScheduler()

        def transformer(self, latents, t, guidance):
            return mx.ones_like(latents) * 0.1

    noisy_init = mx.ones((4, 8, 8))
    mask = mx.zeros((4, 8, 8))
    mask[:, :4, :] = 1.0  # top half regen

    steps_run = []

    def on_step(img_idx, step, total):
        steps_run.append((step, total))

    result = inpaint_latent(
        MockFlux(),
        noisy_init,
        mask,
        num_inference_steps=3,
        guidance=3.5,
        on_step=on_step,
    )
    assert len(steps_run) == 3
    assert steps_run[-1] == (3, 3)
    assert result.shape == (4, 8, 8)
