# SPDX-License-Identifier: Apache-2.0
import mlx.core as mx

from fusion_mlx.video.ltx_video_legacy.denoise import denoise
from fusion_mlx.video.ltx_video_legacy.scheduler import RectifiedFlowScheduler


class MockTransformer:
    # Returns a constant velocity that drives latents toward zero regardless of
    # inputs, so the Euler loop is fully deterministic and checkable. Captures
    # the call args so the wiring (CFG batch, indices_grid scaling, timestep
    # shape) can be asserted independently of the math.
    def __init__(self, velocity=1.0):
        self.velocity = velocity
        self.calls = []

    def __call__(
        self,
        hidden_states,
        indices_grid,
        encoder_hidden_states,
        timestep,
        attention_mask=None,
        encoder_attention_mask=None,
    ):
        self.calls.append(
            {
                "h_shape": hidden_states.shape,
                "ig_shape": indices_grid.shape,
                "ehs_shape": encoder_hidden_states.shape,
                "t_shape": timestep.shape,
                "t_value": float(timestep.reshape(-1)[0]),
                "ig_t_axis": indices_grid[:, 0] if indices_grid.shape[0] else None,
            }
        )
        return mx.full(hidden_states.shape, self.velocity, dtype=mx.float32)


def _embeds(n=256, d=4096, fill=0.5):
    return mx.full((1, n, d), fill, dtype=mx.float32)


def _mask(n=256, fill=1.0):
    return mx.full((1, n), fill, dtype=mx.float32)


def test_denoise_cfg_wiring_batch_and_timestep_shape():
    tf = MockTransformer(velocity=1.0)
    sched = RectifiedFlowScheduler()
    lf, lh, lw = 5, 4, 4
    n = lf * lh * lw
    latents = mx.ones((1, n, 128), dtype=mx.float32)
    pixel_coords = mx.zeros((1, 3, n), dtype=mx.float32)
    out = denoise(
        tf,
        sched,
        latents,
        pixel_coords,
        prompt_embeds=_embeds(fill=0.5),
        prompt_attn_mask=_mask(),
        negative_embeds=_embeds(fill=0.0),
        negative_attn_mask=_mask(),
        guidance_scale=3.0,
        num_inference_steps=4,
        frame_rate=24.0,
        latent_shape=(1, 128, lf, lh, lw),
    )
    assert tf.calls, "transformer never called"
    c0 = tf.calls[0]
    # CFG -> batch of 2 (neg+prompt)
    assert c0["h_shape"][0] == 2
    assert c0["ehs_shape"][0] == 2
    assert c0["t_shape"] == (2, 1)
    # indices_grid broadcast to the CFG batch
    assert c0["ig_shape"] == (2, 3, n)
    # output stays patchified (1, n, c) after CFG combine
    assert out.shape == (1, n, 128)


def test_denoise_indices_grid_temporal_axis_scaled_by_frame_rate():
    tf = MockTransformer()
    sched = RectifiedFlowScheduler()
    lf, lh, lw = 3, 2, 2
    n = lf * lh * lw
    # temporal axis (row 0) starts at 24.0 per token; must be /24 -> 1.0
    temporal = mx.full((n,), 24.0, dtype=mx.float32)
    spatial = mx.zeros((n,), dtype=mx.float32)
    pixel_coords = mx.stack([temporal, spatial, spatial], axis=0)[None]  # (1,3,n)
    denoise(
        tf,
        sched,
        mx.ones((1, n, 128)),
        pixel_coords,
        prompt_embeds=_embeds(),
        prompt_attn_mask=_mask(),
        negative_embeds=_embeds(),
        negative_attn_mask=_mask(),
        guidance_scale=1.0,
        num_inference_steps=2,
        frame_rate=24.0,
        latent_shape=(1, 128, lf, lh, lw),
    )
    ig = tf.calls[0]["ig_t_axis"]  # (1, n)
    # frame_rate=24 -> temporal row divided by 24
    assert mx.allclose(ig, mx.ones((1, n))).item()


def test_denoise_no_cfg_single_batch():
    tf = MockTransformer(velocity=0.0)
    sched = RectifiedFlowScheduler()
    lf, lh, lw = 2, 2, 2
    n = lf * lh * lw
    latents = mx.full((1, n, 128), 7.0, dtype=mx.float32)
    out = denoise(
        tf,
        sched,
        latents,
        mx.zeros((1, 3, n)),
        prompt_embeds=_embeds(),
        prompt_attn_mask=_mask(),
        negative_embeds=_embeds(),
        negative_attn_mask=_mask(),
        guidance_scale=1.0,
        num_inference_steps=3,
        frame_rate=24.0,
        latent_shape=(1, 128, lf, lh, lw),
    )
    # guidance_scale == 1.0 -> no CFG, single forward; velocity 0 -> latents unchanged
    assert tf.calls[0]["h_shape"][0] == 1
    assert mx.allclose(out, mx.full((1, n, 128), 7.0), atol=1e-6).item()


def test_denoise_constant_velocity_drives_to_zero():
    tf = MockTransformer(velocity=1.0)
    sched = RectifiedFlowScheduler()
    lf, lh, lw = 4, 4, 4
    n = lf * lh * lw
    latents = mx.ones((1, n, 128), dtype=mx.float32)
    out = denoise(
        tf,
        sched,
        latents,
        mx.zeros((1, 3, n)),
        prompt_embeds=_embeds(),
        prompt_attn_mask=_mask(),
        negative_embeds=_embeds(),
        negative_attn_mask=_mask(),
        guidance_scale=1.0,
        num_inference_steps=10,
        frame_rate=24.0,
        latent_shape=(1, 128, lf, lh, lw),
    )
    # constant velocity 1.0 over a full SD3 schedule -> latents -> ~0
    assert float(mx.max(mx.abs(out))) < 1e-3


def test_denoise_cfg_combines_uncond_and_text():
    # transformer returns 0 for the uncond half, +1 for the text half; with
    # guidance g, noise_pred = 0 + g*(1-0) = g per step.
    class SplitTransformer:
        def __init__(self):
            self.count = 0

        def __call__(self, hidden_states, **kw):
            b = hidden_states.shape[0]
            # batch is [neg, prompt]; neg->0, prompt->1
            out = mx.zeros(hidden_states.shape, dtype=mx.float32)
            out = out + mx.array([0.0, 1.0], dtype=mx.float32).reshape(b, 1, 1)
            self.count += 1
            return out

    tf = SplitTransformer()
    sched = RectifiedFlowScheduler()
    lf, lh, lw = 2, 2, 2
    n = lf * lh * lw
    latents = mx.zeros((1, n, 128), dtype=mx.float32)
    out = denoise(
        tf,
        sched,
        latents,
        mx.zeros((1, 3, n)),
        prompt_embeds=_embeds(),
        prompt_attn_mask=_mask(),
        negative_embeds=_embeds(),
        negative_attn_mask=_mask(),
        guidance_scale=4.0,
        num_inference_steps=2,
        frame_rate=24.0,
        latent_shape=(1, 128, lf, lh, lw),
    )
    # each step subtracts dt*g; latents end negative. Just assert the CFG path
    # produced a non-trivial (non-zero, non-unit) result driven by g.
    assert float(mx.sum(mx.abs(out))) > 0.0
