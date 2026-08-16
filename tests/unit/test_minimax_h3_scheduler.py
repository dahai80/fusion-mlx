# SPDX-License-Identifier: Apache-2.0
# P3 Scheduler checkpoint：MiniMaxH3Scheduler rectified-flow Euler + shift。

import mlx.core as mx
import pytest

from fusion_mlx.video.minimax_h3.scheduler import MiniMaxH3Scheduler


class TestSetTimesteps:
    def test_default_shift(self):
        sched = MiniMaxH3Scheduler(shift=12.0)
        sched.set_timesteps(num_inference_steps=40)
        assert sched.timesteps.shape[0] == sched.num_inference_steps
        assert float(sched.sigmas[-1]) == pytest.approx(0.0, abs=1e-6)
        assert float(sched.sigmas[0]) == pytest.approx(1.0, abs=1e-6)
        assert float(sched.timesteps[0]) == pytest.approx(0.0, abs=1e-5)
        assert float(sched.timesteps[-1]) > 0.0

    def test_audio_shift(self):
        sched = MiniMaxH3Scheduler(shift=3.0)
        sched.set_timesteps(num_inference_steps=40)
        assert float(sched.sigmas[0]) == pytest.approx(1.0, abs=1e-6)
        assert float(sched.sigmas[-1]) == pytest.approx(0.0, abs=1e-6)

    def test_shift_compresses_grid(self):
        s12 = MiniMaxH3Scheduler(shift=12.0)
        s12.set_timesteps(num_inference_steps=40)
        s3 = MiniMaxH3Scheduler(shift=3.0)
        s3.set_timesteps(num_inference_steps=40)
        assert float(s12.sigmas[20]) > float(s3.sigmas[20])

    def test_timesteps_strictly_increasing(self):
        sched = MiniMaxH3Scheduler(shift=12.0)
        sched.set_timesteps(num_inference_steps=20)
        ts = sched.timesteps
        assert bool(mx.all(ts[1:] > ts[:-1]))

    def test_invalid_shift(self):
        with pytest.raises(ValueError):
            MiniMaxH3Scheduler(shift=0.0)
        with pytest.raises(ValueError):
            MiniMaxH3Scheduler(shift=-1.0)

    def test_too_few_steps(self):
        sched = MiniMaxH3Scheduler(shift=12.0)
        with pytest.raises(ValueError):
            sched.set_timesteps(num_inference_steps=1)

    def test_explicit_sigmas(self):
        sched = MiniMaxH3Scheduler(shift=12.0)
        sigmas = mx.array([1.0, 0.5, 0.0], dtype=mx.float32)
        sched.set_timesteps(sigmas=sigmas)
        assert sched.num_inference_steps == 2
        assert float(sched.timesteps[0]) == pytest.approx(0.0, abs=1e-6)
        assert float(sched.timesteps[1]) == pytest.approx(0.5, abs=1e-6)

    def test_explicit_sigmas_invalid(self):
        sched = MiniMaxH3Scheduler(shift=12.0)
        with pytest.raises(ValueError):
            sched.set_timesteps(sigmas=mx.array([1.0, 0.5], dtype=mx.float32))
        with pytest.raises(ValueError):
            sched.set_timesteps(sigmas=mx.array([0.5, 1.0, 0.0], dtype=mx.float32))


class TestScaleNoise:
    def test_clean_timestep(self):
        sched = MiniMaxH3Scheduler(shift=12.0)
        sample = mx.array([1.0, 2.0, 3.0])
        noise = mx.array([10.0, 20.0, 30.0])
        out = sched.scale_noise(sample, 1.0, noise)
        assert mx.allclose(out, sample)

    def test_noisy_timestep(self):
        sched = MiniMaxH3Scheduler(shift=12.0)
        sample = mx.array([1.0, 2.0, 3.0])
        noise = mx.array([10.0, 10.0, 30.0])
        out = sched.scale_noise(sample, 0.0, noise)
        assert mx.allclose(out, noise)

    def test_mid_timestep(self):
        sched = MiniMaxH3Scheduler(shift=12.0)
        sample = mx.array([2.0])
        noise = mx.array([4.0])
        out = sched.scale_noise(sample, 0.5, noise)
        assert mx.allclose(out, mx.array([3.0]))


class TestStep:
    def test_step_shape(self):
        sched = MiniMaxH3Scheduler(shift=12.0)
        sched.set_timesteps(num_inference_steps=10)
        sample = mx.zeros((1, 8, 16), dtype=mx.float32)
        model_output = mx.ones((1, 8, 16), dtype=mx.float32)
        t = float(sched.timesteps[0])
        prev = sched.step(model_output, t, sample)
        assert prev.shape == sample.shape
        assert sched.step_index == 1

    def test_velocity_sign_standard(self):
        # 官方 diffusers：data-ward velocity，x0 = x_t + sigma*v（PLUS）。
        sched = MiniMaxH3Scheduler(shift=12.0)
        sched.set_timesteps(num_inference_steps=10)
        sample = mx.zeros((1, 4), dtype=mx.float32)
        v = mx.ones((1, 4), dtype=mx.float32)
        t = float(sched.timesteps[0])
        prev = sched.step(v, t, sample)
        sigma = float(sched.sigmas[0])
        sigma_next = float(sched.sigmas[1])
        ratio = sigma_next / sigma
        expected = ratio * 0.0 + (1.0 - ratio) * (sigma * 1.0)
        assert mx.allclose(prev, mx.full((1, 4), float(expected)), atol=1e-5)

    def test_integer_timestep_rejected(self):
        sched = MiniMaxH3Scheduler(shift=12.0)
        sched.set_timesteps(num_inference_steps=10)
        sample = mx.zeros((1, 4), dtype=mx.float32)
        v = mx.ones((1, 4), dtype=mx.float32)
        with pytest.raises(ValueError):
            sched.step(v, 0, sample)

    def test_full_loop(self):
        sched = MiniMaxH3Scheduler(shift=12.0)
        sched.set_timesteps(num_inference_steps=5)
        sample = mx.zeros((1, 4), dtype=mx.float32)
        for t in sched.timesteps:
            v = mx.zeros((1, 4), dtype=mx.float32)
            sample = sched.step(v, float(t), sample)
        assert mx.allclose(sample, mx.zeros((1, 4), dtype=mx.float32), atol=1e-5)


class TestUniqueConsecutive:
    def test_collapses_duplicates(self):
        from fusion_mlx.video.minimax_h3.scheduler import _unique_consecutive

        x = mx.array([1.0, 1.0, 0.5, 0.5, 0.0], dtype=mx.float32)
        out = _unique_consecutive(x)
        assert list(out) == [1.0, 0.5, 0.0]
