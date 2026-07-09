# SPDX-License-Identifier: Apache-2.0
import math

import mlx.core as mx
import numpy as np

from fusion_mlx.video.ltx_video_legacy.scheduler import (
    RectifiedFlowScheduler,
    _normal_shift,
    _stretch_to_terminal,
    _time_shift,
)

MIN_TOKENS = 1024
MAX_TOKENS = 4096
MIN_SHIFT = 0.95
MAX_SHIFT = 2.05


def np_time_shift(mu, sigma, t):
    return np.exp(mu) / (np.exp(mu) + (1.0 / t - 1.0) ** sigma)


def np_normal_shift(n_tokens):
    m = (MAX_SHIFT - MIN_SHIFT) / (MAX_TOKENS - MIN_TOKENS)
    b = MIN_SHIFT - m * MIN_TOKENS
    return m * n_tokens + b


def test_normal_shift_linear_bounds():
    assert math.isclose(_normal_shift(MIN_TOKENS), MIN_SHIFT, abs_tol=1e-6)
    assert math.isclose(_normal_shift(MAX_TOKENS), MAX_SHIFT, abs_tol=1e-6)
    mid = _normal_shift((MIN_TOKENS + MAX_TOKENS) / 2)
    assert MIN_SHIFT < mid < MAX_SHIFT


def test_time_shift_monotone_in_t():
    mu = 2.0
    ts = [0.1, 0.3, 0.5, 0.7, 0.9]
    vals = [float(_time_shift(mu, 1.0, t)) for t in ts]
    for a, b in zip(vals, vals[1:]):
        assert a < b
    # matches numpy oracle
    for t in ts:
        assert math.isclose(
            float(_time_shift(mu, 1.0, t)),
            float(np_time_shift(mu, 1.0, t)),
            rel_tol=1e-6,
        )


def test_stretch_fixes_terminal():
    raw = mx.array([0.95, 0.7, 0.4, 0.2], dtype=mx.float32)
    out = _stretch_to_terminal(raw, terminal=0.1)
    # last sample must land on the terminal after stretching
    assert math.isclose(float(out[-1]), 0.1, abs_tol=1e-6)
    # stays in (0,1), monotonic decreasing
    o = [float(x) for x in out.tolist()]
    assert all(0.0 < v <= 1.0 for v in o)
    assert all(a > b for a, b in zip(o, o[1:]))


def test_set_timesteps_shape_and_monotonic():
    sched = RectifiedFlowScheduler()
    shape = (1, 128, 9, 32, 32)  # B, C, F, H, W -> 9216 latent tokens
    t = sched.set_timesteps(20, samples_shape=shape)
    assert t.shape == (20,)
    vals = [float(x) for x in t.tolist()]
    assert all(0.0 < v <= 1.0 for v in vals)
    assert all(a > b for a, b in zip(vals, vals[1:]))  # strictly decreasing
    # last step lands at the configured terminal (0.1)
    assert math.isclose(vals[-1], 0.1, abs_tol=1e-5)
    assert sched.sigmas is t or sched.sigmas.shape == t.shape
    assert sched.num_inference_steps == 20


def test_set_timesteps_without_shift_uses_uniform_linspace():
    sched = RectifiedFlowScheduler(shifting=None)
    t = sched.set_timesteps(5, samples_shape=(1, 128, 9, 32, 32))
    vals = [float(x) for x in t.tolist()]
    assert math.isclose(vals[0], 1.0, abs_tol=1e-6)
    assert math.isclose(vals[-1], 1.0 / 5, abs_tol=1e-6)
    assert math.isclose(vals[1], 0.8, abs_tol=1e-6)


def test_scale_model_input_identity():
    sched = RectifiedFlowScheduler()
    x = mx.array([[1.0, 2.0, 3.0]])
    out = sched.scale_model_input(x, 0.5)
    assert mx.allclose(out, x).item()


def test_add_noise_endpoints():
    sched = RectifiedFlowScheduler()
    orig = mx.full((2, 3), 5.0)
    noise = mx.full((2, 3), 10.0)
    # t=0 -> pure original, t=1 -> pure noise
    assert mx.allclose(sched.add_noise(orig, noise, 0.0), orig).item()
    assert mx.allclose(sched.add_noise(orig, noise, 1.0), noise).item()
    # linear blend midpoint
    mid = sched.add_noise(orig, noise, 0.5)
    assert mx.allclose(mid, mx.full((2, 3), 7.5), atol=1e-6).item()


def test_step_reduces_signal_and_terminal_reaches_clean():
    sched = RectifiedFlowScheduler()
    shape = (1, 128, 9, 32, 32)
    sched.set_timesteps(10, samples_shape=shape)
    sample = mx.ones((1, 128))
    # constant velocity toward zero -> each step subtracts dt*v
    vel = mx.full((1, 128), 1.0)
    ts = [float(x) for x in sched.timesteps.tolist()]
    cur = sample
    for t in ts:
        cur = sched.step(vel, t, cur)
    # after a full schedule the latent should be driven to ~0 (sum dt == t0)
    assert float(mx.max(mx.abs(cur))) < 1e-4


def test_step_dt_is_gap_to_next_lower_sigma():
    sched = RectifiedFlowScheduler(shifting=None)
    sched.set_timesteps(4, samples_shape=(1, 128, 9, 8, 8))
    ts = [float(x) for x in sched.timesteps.tolist()]  # [1.0, 0.75, 0.5, 0.25]
    sample = mx.zeros((1, 4))
    vel = mx.ones((1, 4))
    out0 = sched.step(vel, ts[0], sample)
    # dt should be t0 - t1 = 0.25
    assert math.isclose(float(out0[0, 0]), -(ts[0] - ts[1]), abs_tol=1e-6)
    # last step: dt == tN-1 - 0 = tN-1
    out_last = sched.step(vel, ts[-1], sample)
    assert math.isclose(float(out_last[0, 0]), -ts[-1], abs_tol=1e-6)


def test_init_noise_sigma_is_one():
    assert RectifiedFlowScheduler().init_noise_sigma == 1.0
