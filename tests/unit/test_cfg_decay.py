# SPDX-License-Identifier: Apache-2.0
# Unit tests for T2-3: SkyReels progressive CFG decay.


import math

import pytest


def _cfg_decay_scale(step_idx, cfg_keep_steps, base_scale, decay_mode, decay_ratio):
    # Standalone copy of the pipeline method for unit testing without MLX import.
    if base_scale <= 0:
        return 1.0
    if decay_mode in ("", "0", "off", "none"):
        return base_scale
    if step_idx >= cfg_keep_steps:
        return 1.0
    decay_ratio = max(0.0, min(1.0, decay_ratio))
    decay_steps = max(1, int(cfg_keep_steps * decay_ratio))
    decay_start = cfg_keep_steps - decay_steps
    if step_idx < decay_start:
        return base_scale
    progress = (step_idx - decay_start) / decay_steps
    if decay_mode in ("linear", "1"):
        scale = base_scale + (1.0 - base_scale) * progress
    elif decay_mode in ("cosine", "2"):
        cosine_progress = 0.5 * (1.0 + math.cos(math.pi * (1.0 - progress)))
        scale = base_scale + (1.0 - base_scale) * cosine_progress
    elif decay_mode in ("step", "3"):
        scale = 1.0 if progress >= 0.5 else base_scale
    else:
        return base_scale
    return scale


class TestCfgDecayScale:
    def test_off_returns_base_scale(self):
        assert _cfg_decay_scale(0, 10, 5.0, "off", 0.3) == 5.0
        assert _cfg_decay_scale(9, 10, 5.0, "off", 0.3) == 5.0

    def test_empty_returns_base_scale(self):
        assert _cfg_decay_scale(5, 10, 5.0, "", 0.3) == 5.0

    def test_past_keep_steps_returns_1(self):
        assert _cfg_decay_scale(10, 10, 5.0, "linear", 0.3) == 1.0

    def test_linear_decay_full_range(self):
        # decay_ratio=1.0 => all 10 steps decay, progress = step/10
        assert _cfg_decay_scale(0, 10, 5.0, "linear", 1.0) == pytest.approx(5.0)
        assert _cfg_decay_scale(5, 10, 5.0, "linear", 1.0) == pytest.approx(3.0)
        assert _cfg_decay_scale(9, 10, 5.0, "linear", 1.0) == pytest.approx(1.4)

    def test_cosine_decay_midpoint(self):
        # cosine at progress=0.5: cos(pi*0.5)=0 => 0.5*(1+0)=0.5
        # scale = 5.0 + (1.0-5.0)*0.5 = 3.0
        mid = _cfg_decay_scale(5, 10, 5.0, "cosine", 1.0)
        assert mid == pytest.approx(3.0)

    def test_step_decay_half(self):
        # progress < 0.5 => base_scale; progress >= 0.5 => 1.0
        assert _cfg_decay_scale(4, 10, 5.0, "step", 1.0) == 5.0
        assert _cfg_decay_scale(5, 10, 5.0, "step", 1.0) == 1.0

    def test_decay_ratio_partial(self):
        # keep=10, decay_ratio=0.3 => decay_steps=3, decay_start=7
        assert _cfg_decay_scale(6, 10, 5.0, "linear", 0.3) == 5.0
        assert _cfg_decay_scale(7, 10, 5.0, "linear", 0.3) == pytest.approx(5.0)
        assert _cfg_decay_scale(9, 10, 5.0, "linear", 0.3) == pytest.approx(5.0 - 4.0 * 2 / 3)

    def test_unknown_mode_returns_base(self):
        assert _cfg_decay_scale(5, 10, 5.0, "bogus", 0.3) == 5.0

    def test_zero_base_scale_returns_1(self):
        # base_scale=0 → 1.0 (guard against nonsensical CFG scale)
        assert _cfg_decay_scale(0, 10, 0.0, "linear", 1.0) == 1.0

    def test_negative_base_scale_returns_1(self):
        # base_scale<0 → 1.0 (guard)
        assert _cfg_decay_scale(0, 10, -2.0, "linear", 1.0) == 1.0

    def test_before_decay_start_returns_base(self):
        # decay_ratio=0.2, keep=10 => decay_steps=2, decay_start=8
        assert _cfg_decay_scale(7, 10, 5.0, "linear", 0.2) == 5.0
