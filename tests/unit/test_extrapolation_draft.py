# SPDX-License-Identifier: Apache-2.0
# Tests for issue #177 Phase-4: ExtrapolationDraft velocity-prediction strategy.
# Covers: history tracking, linear/quadratic extrapolation, integration with
# speculative_denoise loop, acceptance rates, backward-compat with layer_pruned.

import logging
import os

os.environ.setdefault("FUSION_DISABLE_COMPILE", "1")

import pytest

import mlx.core as mx

from fusion_mlx.video.skyreels_v3.speculative_denoise import (
    ExtrapolationDraft,
    SpecStats,
    SpeculativeConfig,
    baseline_euler,
    create_extrap_draft,
    speculative_denoise,
)
from fusion_mlx.video.skyreels_v3.scheduler.fm_solvers_unipc import perform_guidance

logger = logging.getLogger(__name__)


class TestExtrapolationDraftBasic:
    def test_no_history_returns_zeros(self):
        draft = ExtrapolationDraft(mode="linear")
        x = mx.random.normal((2, 16, 1, 8, 8))
        t = mx.array([0.5, 0.4])
        out = draft(x, t)
        assert out.shape == x.shape
        assert float(mx.sum(mx.abs(out)).item()) == 0.0

    def test_linear_constant_extrapolation(self):
        draft = ExtrapolationDraft(mode="linear")
        v1 = mx.ones((16, 1, 8, 8)) * 2.0
        draft.record_verified(v1, 0.5)
        x = mx.random.normal((3, 16, 1, 8, 8))
        t = mx.array([0.4, 0.3, 0.2])
        out = draft(x, t)
        for j in range(3):
            diff = float(mx.max(mx.abs(out[j] - v1)).item())
            assert diff < 1e-6, f"linear mode: step {j} should repeat last v"

    def test_quadratic_extrapolation(self):
        draft = ExtrapolationDraft(mode="quadratic")
        v0 = mx.ones((16, 1, 8, 8)) * 1.0
        v1 = mx.ones((16, 1, 8, 8)) * 3.0
        draft.record_verified(v0, 0.5)
        draft.record_verified(v1, 0.4)
        x = mx.random.normal((2, 16, 1, 8, 8))
        t = mx.array([0.3, 0.2])
        out = draft(x, t)
        # dv = v1 - v0 = 2.0; v_pred[j] = v1 + dv*(j+1)
        # j=0: 3.0 + 2.0*1 = 5.0
        # j=1: 3.0 + 2.0*2 = 7.0
        assert float(mx.max(mx.abs(out[0] - 5.0)).item()) < 1e-6
        assert float(mx.max(mx.abs(out[1] - 7.0)).item()) < 1e-6

    def test_quadratic_falls_back_to_linear_with_one_history(self):
        draft = ExtrapolationDraft(mode="quadratic")
        v1 = mx.ones((16, 1, 8, 8)) * 2.0
        draft.record_verified(v1, 0.5)
        x = mx.random.normal((1, 16, 1, 8, 8))
        t = mx.array([0.4])
        out = draft(x, t)
        # Only 1 history point -> linear (constant) fallback
        diff = float(mx.max(mx.abs(out[0] - v1)).item())
        assert diff < 1e-6

    def test_reset_clears_history(self):
        draft = ExtrapolationDraft(mode="linear")
        v = mx.ones((16, 1, 8, 8))
        draft.record_verified(v, 0.5)
        assert len(draft._history) == 1
        draft.reset()
        assert len(draft._history) == 0
        assert len(draft._t_history) == 0

    def test_create_extrap_draft_factory(self):
        draft = create_extrap_draft("linear")
        assert draft.mode == "linear"
        draft2 = create_extrap_draft("quadratic")
        assert draft2.mode == "quadratic"


class TestSpeculativeConfigDraftStrategy:
    def test_default_is_extrapolation(self):
        config = SpeculativeConfig()
        assert config.draft_strategy == "extrapolation"

    def test_from_env_default(self, monkeypatch):
        monkeypatch.delenv("FUSION_SPEC_DRAFT_STRATEGY", raising=False)
        config = SpeculativeConfig.from_env()
        assert config.draft_strategy == "extrapolation"

    def test_from_env_layer_pruned(self, monkeypatch):
        monkeypatch.setenv("FUSION_SPEC_DRAFT_STRATEGY", "layer_pruned")
        config = SpeculativeConfig.from_env()
        assert config.draft_strategy == "layer_pruned"

    def test_from_env_unknown_fallback(self, monkeypatch):
        monkeypatch.setenv("FUSION_SPEC_DRAFT_STRATEGY", "bogus")
        config = SpeculativeConfig.from_env()
        assert config.draft_strategy == "extrapolation"

    def test_from_env_case_insensitive(self, monkeypatch):
        monkeypatch.setenv("FUSION_SPEC_DRAFT_STRATEGY", "Extrapolation")
        config = SpeculativeConfig.from_env()
        assert config.draft_strategy == "extrapolation"


class TestSpecStatsDraftStrategy:
    def test_default_strategy(self):
        stats = SpecStats()
        assert stats.draft_strategy == "extrapolation"
        d = stats.to_dict()
        assert d["draft_strategy"] == "extrapolation"

    def test_custom_strategy(self):
        stats = SpecStats(draft_strategy="layer_pruned")
        d = stats.to_dict()
        assert d["draft_strategy"] == "layer_pruned"


class TestExtrapolationDenoiseIntegration:
    """Integration test: ExtrapolationDraft with speculative_denoise loop
    using a synthetic velocity field (no real DiT)."""

    @staticmethod
    def _make_synthetic_velocity(shape, scale=1.0):
        """Create a synthetic velocity function that produces smooth,
        slowly-varying velocities. This models the real-world property
        that DiT velocities change gradually across denoising steps."""

        def velocity(x_batch, t_batch):
            # Smooth velocity: depends on t only (not x), slowly varying
            k = t_batch.shape[0]
            batch = []
            for j in range(k):
                t = float(t_batch[j])
                # v(t) = scale * (1 + 0.1*sin(2*pi*t)) — smooth, near-constant
                v = (
                    mx.ones(shape)
                    * scale
                    * (1.0 + 0.1 * mx.sin(mx.array(2.0 * 3.14159 * t)))
                )
                batch.append(v)
            return mx.stack(batch, axis=0)

        return velocity

    def test_extrapolation_accepts_steps_with_smooth_velocity(self):
        """With smoothly varying velocity, extrapolation should accept
        multiple steps per macro step (constant-velocity is a good approx)."""
        mx.random.seed(42)
        shape = (16, 1, 8, 8)
        full_velocity = self._make_synthetic_velocity(shape, scale=1.0)
        latents = mx.random.normal(shape)
        timesteps = mx.array([1.0, 0.75, 0.5, 0.25, 0.0])

        config = SpeculativeConfig(
            K=4,
            epsilon=0.3,
            relative=True,
            eval_steps=True,
            draft_strategy="extrapolation",
        )
        out, stats = speculative_denoise(
            full_velocity, None, latents, timesteps, config
        )
        mx.eval(out)
        # With smooth velocity and epsilon=0.3, should accept some steps
        logger.info(
            "extrap integration: macro=%d accepted=%s avg=%.2f",
            stats.macro_steps,
            stats.accepted,
            stats.avg_accept,
        )
        # Extrapolation draft cost is 0 model forwards
        assert stats.draft_forwards == 0
        # Should have accepted at least 1 step somewhere (smooth velocity)
        assert stats.avg_accept > 0.0

    def test_extrapolation_vs_baseline_euler(self):
        """Extrapolation path should produce similar output to baseline Euler
        when velocity is smooth (both use 1st-order Euler integration)."""
        mx.random.seed(42)
        shape = (16, 1, 8, 8)
        full_velocity = self._make_synthetic_velocity(shape, scale=1.0)
        latents = mx.random.normal(shape)
        timesteps = mx.array([1.0, 0.75, 0.5, 0.25, 0.0])

        config = SpeculativeConfig(
            K=4,
            epsilon=0.5,
            relative=True,
            eval_steps=True,
            draft_strategy="extrapolation",
        )
        spec_out, stats = speculative_denoise(
            full_velocity, None, latents, timesteps, config
        )
        mx.eval(spec_out)

        base_out = baseline_euler(full_velocity, latents, timesteps)
        mx.eval(base_out)

        # Both paths use 1st-order Euler on the same velocity field.
        # When extrapolation is accurate, they should be close.
        diff = float(mx.max(mx.abs(spec_out - base_out)).item())
        logger.info(
            "extrap vs baseline: max_diff=%g avg_accept=%.2f",
            diff,
            stats.avg_accept,
        )
        # Allow some divergence (spec path may reject and re-step)
        # but overall should be in the same ballpark
        assert diff < 5.0, f"spec output diverged too far from baseline: {diff}"

    def test_extrapolation_with_rapid_velocity_change(self):
        """When velocity changes rapidly, extrapolation should reject more
        (constant-velocity assumption breaks), but still make progress."""
        mx.random.seed(42)
        shape = (16, 1, 8, 8)

        def rapid_velocity(x_batch, t_batch):
            k = t_batch.shape[0]
            batch = []
            for j in range(k):
                t = float(t_batch[j])
                # Rapidly varying: large swings
                v = mx.ones(shape) * (10.0 * mx.sin(mx.array(20.0 * t)))
                batch.append(v)
            return mx.stack(batch, axis=0)

        latents = mx.random.normal(shape)
        timesteps = mx.array([1.0, 0.75, 0.5, 0.25, 0.0])

        config = SpeculativeConfig(
            K=4,
            epsilon=0.3,
            relative=True,
            eval_steps=True,
            draft_strategy="extrapolation",
        )
        out, stats = speculative_denoise(
            rapid_velocity, None, latents, timesteps, config
        )
        mx.eval(out)
        logger.info(
            "rapid velocity: macro=%d accepted=%s avg=%.2f",
            stats.macro_steps,
            stats.accepted,
            stats.avg_accept,
        )
        # Even with rapid changes, should complete without error
        assert stats.macro_steps > 0
        assert stats.draft_forwards == 0


class TestExtrapolationWithTinyDiT:
    """Test extrapolation draft with a tiny SkyReels R2V DiT (no real weights)."""

    TINY_CFG = {
        "dim": 64,
        "ffn_dim": 128,
        "num_heads": 4,
        "num_kv_heads": 4,
        "num_layers": 4,
        "patch_size": (1, 2, 2),
        "in_dim": 16,
        "out_dim": 16,
        "text_dim": 64,
        "text_len": 32,
        "freq_dim": 32,
        "window_size": (-1, -1),
        "qk_norm": True,
        "cross_attn_norm": True,
        "eps": 1e-6,
    }

    def test_extrapolation_with_real_dit_structure(self):
        from fusion_mlx.video.skyreels_v3.transformer_r2v import SkyReelsR2VDiT

        mx.random.seed(7)
        dit = SkyReelsR2VDiT(dict(self.TINY_CFG))

        H = W = 8
        T = 1
        C = 16
        L_PATCH = T * (H // 2) * (W // 2)
        L_CTX = 8
        guidance = 5.0

        x = mx.random.normal((1, C, T, H, W))
        context = mx.random.normal((1, L_CTX, 64))
        seq_lens = [L_PATCH]
        grid_sizes = [(T, H // 2, W // 2)]
        latents = x[0]
        timesteps = mx.array([1.0, 0.75, 0.5, 0.25, 0.0])

        def _cfg_expand(x_batch, t_batch):
            k = x_batch.shape[0]
            x_2k = mx.concatenate([x_batch, x_batch], axis=0)
            t_2k = mx.concatenate([t_batch, t_batch], axis=0)
            ctx_2k = mx.concatenate([context] * (2 * k), axis=0)
            seq_2k = list(seq_lens) * (2 * k)
            grid_2k = list(grid_sizes) * (2 * k)
            return x_2k, t_2k, ctx_2k, seq_2k, grid_2k

        def full_velocity(x_batch, t_batch):
            x_2k, t_2k, ctx_2k, seq_2k, grid_2k = _cfg_expand(x_batch, t_batch)
            noise = dit(x_2k, t_2k, ctx_2k, seq_2k, grid_2k)
            return perform_guidance(noise, guidance)

        config = SpeculativeConfig(
            K=3, epsilon=0.5, eval_steps=True, draft_strategy="extrapolation"
        )
        out, stats = speculative_denoise(
            full_velocity, None, latents, timesteps, config
        )
        mx.eval(out)
        logger.info(
            "tiny DiT extrap: macro=%d accepted=%s avg=%.2f strategy=%s",
            stats.macro_steps,
            stats.accepted,
            stats.avg_accept,
            stats.draft_strategy,
        )
        assert stats.draft_strategy == "extrapolation"
        assert stats.draft_forwards == 0
        assert stats.full_forwards > 0
        assert tuple(out.shape) == (C, T, H, W)
