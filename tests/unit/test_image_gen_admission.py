# SPDX-License-Identifier: Apache-2.0
# OP-901 (B+C): tests for image-gen subprocess default + media admission gate.
import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from fusion_mlx.engines.image_gen import (
    ImageGenEngine,
    _estimate_activation_peak,
    _subprocess_enabled,
)


class TestSubprocessDefault:
    def test_subprocess_enabled_by_default(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("FUSION_IMAGE_SUBPROCESS", None)
            assert _subprocess_enabled() is True

    def test_subprocess_opt_out_explicit_zero(self):
        with patch.dict(os.environ, {"FUSION_IMAGE_SUBPROCESS": "0"}):
            assert _subprocess_enabled() is False

    def test_subprocess_opt_out_false(self):
        with patch.dict(os.environ, {"FUSION_IMAGE_SUBPROCESS": "false"}):
            assert _subprocess_enabled() is False

    def test_subprocess_opt_out_off(self):
        with patch.dict(os.environ, {"FUSION_IMAGE_SUBPROCESS": "off"}):
            assert _subprocess_enabled() is False

    def test_subprocess_explicit_on(self):
        with patch.dict(os.environ, {"FUSION_IMAGE_SUBPROCESS": "1"}):
            assert _subprocess_enabled() is True


class TestEstimateActivationPeak:
    def test_floor_at_512mb_for_tiny_image(self):
        peak = _estimate_activation_peak(
            64, 64, 4, 1, "flux1_schnell", subprocess_mode=True
        )
        assert peak >= 512 * 1024**2

    def test_subprocess_mode_uses_single_image_peak(self):
        sub = _estimate_activation_peak(
            2048, 2048, 28, 4, "flux1_dev", subprocess_mode=True
        )
        inproc = _estimate_activation_peak(
            2048, 2048, 28, 4, "flux1_dev", subprocess_mode=False
        )
        # subprocess peak = single; in-process = 4x single (no per-image clear)
        assert sub < inproc
        assert inproc >= sub * 4 - 1  # allow floor rounding

    def test_cfg_free_variant_lower_peak(self):
        cfg_free = _estimate_activation_peak(
            2048, 2048, 30, 1, "qwen_image", subprocess_mode=True
        )
        cfg_full = _estimate_activation_peak(
            2048, 2048, 30, 1, "flux1_dev", subprocess_mode=True
        )
        # qwen_image is CFG-free (1.0x), flux1_dev is CFG (1.8x)
        assert cfg_free < cfg_full

    def test_env_override_fixed_headroom(self):
        with patch.dict(os.environ, {"FUSION_IMAGE_ACTIVATION_HEADROOM_GB": "10"}):
            peak = _estimate_activation_peak(
                2048, 2048, 50, 8, "flux1_dev", subprocess_mode=True
            )
            assert peak == 10 * 1024**3

    def test_env_override_zero_disables(self):
        with patch.dict(os.environ, {"FUSION_IMAGE_ACTIVATION_HEADROOM_GB": "0"}):
            peak = _estimate_activation_peak(
                1024, 1024, 28, 4, "flux1_dev", subprocess_mode=True
            )
            assert peak == 0

    def test_larger_resolution_higher_peak(self):
        small = _estimate_activation_peak(
            1024, 1024, 28, 1, "flux1_dev", subprocess_mode=True
        )
        large = _estimate_activation_peak(
            2048, 2048, 28, 1, "flux1_dev", subprocess_mode=True
        )
        assert large > small


class TestAdmitGeneration:
    """_admit_generation drives pool.admit_media_job and raises 507 on failure."""

    def _make_engine(self):
        return ImageGenEngine(model_name="flux-schnell", variant="flux1_schnell")

    @pytest.mark.asyncio
    async def test_admission_skipped_when_pool_none(self):
        engine = self._make_engine()
        state = MagicMock()
        state.engine_pool = None
        with patch("fusion_mlx.server._server_state", state):
            # Should not raise — just skip the gate.
            await engine._admit_generation(
                width=1024,
                height=1024,
                steps=4,
                n_images=1,
                subprocess_mode=True,
            )

    @pytest.mark.asyncio
    async def test_admission_raises_on_failure(self):
        from fusion_mlx.exceptions import InsufficientMemoryError

        engine = self._make_engine()
        pool = MagicMock()
        pool.admit_media_job = AsyncMock(return_value=False)
        state = MagicMock()
        state.engine_pool = pool
        with patch("fusion_mlx.server._server_state", state):
            with patch(
                "fusion_mlx.media.job_manager.MediaJobManager._compute_lease_bytes",
                return_value=32 * 1024**3,
            ):
                with pytest.raises(InsufficientMemoryError):
                    await engine._admit_generation(
                        width=1024,
                        height=1024,
                        steps=4,
                        n_images=4,
                        subprocess_mode=True,
                    )

    @pytest.mark.asyncio
    async def test_admission_passes_when_ok(self):
        engine = self._make_engine()
        pool = MagicMock()
        pool.admit_media_job = AsyncMock(return_value=True)
        state = MagicMock()
        state.engine_pool = pool
        with patch("fusion_mlx.server._server_state", state):
            with patch(
                "fusion_mlx.media.job_manager.MediaJobManager._compute_lease_bytes",
                return_value=8 * 1024**3,
            ):
                await engine._admit_generation(
                    width=1024,
                    height=1024,
                    steps=4,
                    n_images=1,
                    subprocess_mode=True,
                )
        pool.admit_media_job.assert_awaited_once()
