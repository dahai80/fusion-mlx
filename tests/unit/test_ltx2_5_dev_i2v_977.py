# SPDX-License-Identifier: Apache-2.0
# #977: ltx2_5 dev I2V single-stage conditioning wiring. Verifies the dev path
# no longer raises NotImplementedError, and the shared _build_i2v_state helper
# builds a LatentState with the condition frame frozen clean (mask=0) and free
# frames renoised to sig[0]. No real 22B model load (OOM gate).
from __future__ import annotations

import mlx.core as mx

from fusion_mlx.video.ltx2.conditioning import LatentState
from fusion_mlx.video.ltx2_5 import generate as gen_mod


def test_build_i2v_state_condition_frame_frozen_clean():
    b, c, f, h, w = 1, 128, 4, 2, 2
    latent_shape = (b, c, f, h, w)
    image_latent = mx.ones((1, c, 1, h, w)) * 7.0
    mx.random.seed(0)
    state = gen_mod._build_i2v_state(
        image_latent,
        latent_shape,
        noise_scale=0.909,
        image_frame_idx=0,
        image_strength=1.0,
        model_dtype=mx.float32,
    )
    mx.eval(state.latent, state.clean_latent, state.denoise_mask)
    assert isinstance(state, LatentState)
    # strength=1.0 -> condition frame mask = 0 (frozen clean)
    assert mx.all(state.denoise_mask[:, :, 0:1] == 0.0).item()
    # free frames mask = 1 (renoised)
    assert mx.all(state.denoise_mask[:, :, 1:] == 1.0).item()
    # clean_latent at frame 0 == image latent
    assert mx.all(state.clean_latent[:, :, 0:1] == 7.0).item()
    # condition frame latent = image (apply_conditioning splices image into
    # latent; mask=0 -> noise*0 + base*1 = base = image_latent)
    assert mx.all(state.latent[:, :, 0:1] == 7.0).item()


def test_build_i2v_state_free_frames_renoised():
    b, c, f, h, w = 1, 2, 3, 1, 1
    latent_shape = (b, c, f, h, w)
    image_latent = mx.ones((1, c, 1, h, w)) * 5.0
    mx.random.seed(1)
    state = gen_mod._build_i2v_state(
        image_latent,
        latent_shape,
        noise_scale=1.0,
        image_frame_idx=0,
        image_strength=1.0,
        model_dtype=mx.float32,
    )
    mx.eval(state.latent)
    # noise_scale=1.0, free frame mask=1 -> latent = noise*1 + base*0 = noise (!=0)
    free = state.latent[:, :, 1:]
    assert not mx.all(free == 0.0).item()
    # condition frame latent = image (5.0)
    assert mx.all(state.latent[:, :, 0:1] == 5.0).item()


def test_build_i2v_state_partial_strength_keeps_some_noise_on_condition():
    b, c, f, h, w = 1, 2, 2, 1, 1
    latent_shape = (b, c, f, h, w)
    image_latent = mx.ones((1, c, 1, h, w)) * 3.0
    mx.random.seed(2)
    state = gen_mod._build_i2v_state(
        image_latent,
        latent_shape,
        noise_scale=1.0,
        image_frame_idx=0,
        image_strength=0.5,
        model_dtype=mx.float32,
    )
    mx.eval(state.denoise_mask)
    # strength=0.5 -> condition mask = 1-0.5 = 0.5 (partially denoised)
    assert mx.allclose(
        state.denoise_mask[:, :, 0:1], mx.array([[[0.5]]]), atol=1e-5
    ).item()
