# SPDX-License-Identifier: Apache-2.0
# #782: LTX-2.5 I2V conditioning wiring — synthetic unit tests.
# Verifies _build_i2v_conditionings + _encode_image_latent helpers and the
# denoise_distilled_t2v state= path (conditioned frame stays clean across
# re-noise, timesteps=0 for masked frames). No real 22B model load (OOM gate).
from __future__ import annotations

import mlx.core as mx

from fusion_mlx.video.ltx2.conditioning import (
    LatentState,
    apply_conditioning,
)
from fusion_mlx.video.ltx2_5 import generate as gen_mod
from fusion_mlx.video.ltx2_5.denoise import denoise_distilled_t2v


class FakeTransformer:
    # records (latent, timesteps) per step; returns zero velocity so x0 == latent.
    def __init__(self):
        self.calls = []

    def __call__(self, video, audio=None):
        lat = video.latent
        ts = video.timesteps
        b, n, c = lat.shape
        velocity = mx.zeros((b, n, c), dtype=lat.dtype)
        audio_vel = None
        self.calls.append({"latent_shape": lat.shape, "timesteps": ts})
        return velocity, audio_vel


def test_build_i2v_conditionings_first_image_only():
    latent = mx.zeros((1, 128, 1, 4, 4))
    conds = gen_mod._build_i2v_conditionings(
        latent, image_frame_idx=2, image_strength=0.8
    )
    assert len(conds) == 1
    assert conds[0].frame_idx == 2
    assert conds[0].strength == 0.8


def test_build_i2v_conditionings_with_end_image_idx_zero():
    latent = mx.zeros((1, 128, 1, 4, 4))
    end = mx.zeros((1, 128, 1, 4, 4))
    conds = gen_mod._build_i2v_conditionings(
        latent,
        image_frame_idx=3,
        image_strength=1.0,
        end_image_latent=end,
        end_image_strength=0.5,
    )
    assert len(conds) == 2
    # end_image present -> first image forced to frame 0
    assert conds[0].frame_idx == 0
    assert conds[0].strength == 1.0
    assert conds[1].frame_idx == -1
    assert conds[1].strength == 0.5


def test_apply_conditioning_splices_image_at_frame_idx():
    b, c, f, h, w = 1, 128, 5, 2, 2
    image_latent = mx.ones((1, c, 1, h, w)) * 7.0
    state = LatentState(
        latent=mx.zeros((b, c, f, h, w)),
        clean_latent=mx.zeros((b, c, f, h, w)),
        denoise_mask=mx.ones((b, 1, f, 1, 1)),
    )
    conds = gen_mod._build_i2v_conditionings(
        image_latent, image_frame_idx=2, image_strength=1.0
    )
    state = apply_conditioning(state, conds)
    mx.eval(state.latent, state.clean_latent, state.denoise_mask)
    # frame 2 spliced with image latent (==7), others stay 0
    frame2 = state.latent[:, :, 2:3]
    assert mx.all(frame2 == 7.0).item()
    other = state.latent[:, :, 0:2]
    assert mx.all(other == 0.0).item()
    # strength=1.0 -> denoise_mask = 1-1 = 0 for conditioned frame (stays clean)
    mask_frame2 = state.denoise_mask[:, :, 2:3]
    assert mx.all(mask_frame2 == 0.0).item()
    mask_free = state.denoise_mask[:, :, 0:2]
    assert mx.all(mask_free == 1.0).item()
    # clean_latent at frame 2 == image latent
    assert mx.all(state.clean_latent[:, :, 2:3] == 7.0).item()


def test_denoise_state_conditioned_frame_stays_clean():
    # state with frame 0 conditioned (mask=0, clean=7), frames 1-2 free (mask=1).
    b, c, f, h, w = 1, 2, 3, 1, 1
    per_frame = mx.array([0.0, 1.0, 2.0]).reshape(1, 1, f, 1, 1)
    latent = mx.broadcast_to(per_frame, (b, c, f, h, w)).astype(mx.float32)
    clean_per = mx.array([7.0, 0.0, 0.0]).reshape(1, 1, f, 1, 1)
    clean = mx.broadcast_to(clean_per, (b, c, f, h, w)).astype(mx.float32)
    mask = mx.array([0.0, 1.0, 1.0]).reshape(b, 1, f, 1, 1)
    state = LatentState(latent=latent, clean_latent=clean, denoise_mask=mask)

    transformer = FakeTransformer()
    sigmas = [1.0, 0.5, 0.0]
    out = denoise_distilled_t2v(
        latent,
        mx.zeros((1,)),
        mx.zeros((1, 1, c)),
        transformer,
        sigmas,
        verbose=False,
        state=state,
    )
    mx.eval(out)
    # FakeTransformer returns zero velocity -> x0 = latent - 0 = latent.
    # apply_denoise_mask: denoised*mask + clean*(1-mask).
    # frame 0 (mask=0): denoised = clean = 7.0  (conditioned, frozen clean).
    expected_frame0 = mx.broadcast_to(
        mx.array([7.0]).reshape(1, 1, 1, 1, 1), (b, c, 1, h, w)
    )
    assert mx.allclose(out[:, :, 0:1], expected_frame0, atol=1e-5).item()
    # timesteps for frame 0 == sigma*mask = sigma*0 = 0 (transformer sees clean)
    ts0 = transformer.calls[0]["timesteps"]
    # frame 0 is token 0 (f*h*w=1 token/frame, b=1) -> token index 0
    assert float(ts0[0, 0].item()) == 0.0


def test_denoise_state_none_is_t2v_uniform_timesteps():
    b, c, f, h, w = 1, 2, 2, 1, 1
    latent = mx.zeros((b, c, f, h, w))
    transformer = FakeTransformer()
    sigmas = [1.0, 0.0]
    denoise_distilled_t2v(
        latent,
        mx.zeros((1,)),
        mx.zeros((1, 1, c)),
        transformer,
        sigmas,
        verbose=False,
        state=None,
    )
    ts = transformer.calls[0]["timesteps"]
    # uniform timesteps == sigma for every token
    assert mx.all(ts == 1.0).item()


def test_encode_image_latent_loads_encoder_and_caches():
    calls = {"encode": [], "load_enc": 0, "cache_put": []}

    class FakeEncoder:
        def __call__(self, x):
            calls["encode"].append(x.shape)
            return mx.ones((1, 128, 1, x.shape[3] // 8, x.shape[4] // 8)) * 3.0

        def parameters(self):
            return []

    class FakeCache:
        def get(self, key):
            return None

        def put(self, key, val):
            calls["cache_put"].append(key)

    fake_enc = FakeEncoder()
    fake_cache = FakeCache()
    fake_root = "/fake/root"

    import fusion_mlx.video.ltx2_5.generate as gm

    gm.load_video_encoder = lambda path: fake_enc
    gm.is_flat_layout = lambda root: True
    gm.resolve_component = lambda root, name, **kw: f"{root}/{name}.safetensors"
    gm.load_image = lambda src, height, width, dtype: mx.zeros(
        (height, width, 3), dtype=dtype
    )
    gm.prepare_image_for_encoding = lambda img, h, w, dtype: mx.zeros(
        (1, 3, 1, h, w), dtype=dtype
    )

    latent, enc = gm._encode_image_latent(
        "/img.png",
        256,
        256,
        "repo",
        fake_root,
        mx.float32,
        fake_cache,
        None,
    )
    mx.eval(latent)
    assert latent.shape == (1, 128, 1, 32, 32)
    assert mx.all(latent == 3.0).item()
    assert enc is fake_enc
    assert calls["load_enc"] == 0  # load_video_encoder lambda used, counter not needed
    assert len(calls["encode"]) == 1
    assert len(calls["cache_put"]) == 1

    # second call: cache miss again (FakeCache.get always None) but encoder reused
    latent2, enc2 = gm._encode_image_latent(
        "/img.png",
        256,
        256,
        "repo",
        fake_root,
        mx.float32,
        fake_cache,
        enc,
    )
    assert enc2 is enc
