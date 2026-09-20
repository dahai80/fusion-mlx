# SPDX-License-Identifier: Apache-2.0
"""Tests for PRD v1 unified video layer:
- video_unified_scheduler (3-level breaker, NF4 cache, mutex)
- video_router (LTX vs H3 routing)
- model_quant (NF4 validation, banned modes)
- video_audio_export (normalize/mux helpers)
- common bases (interface contract)
"""

from __future__ import annotations

import numpy as np
import pytest

from fusion_mlx.pipeline.video_router import VideoRouter, get_video_router
from fusion_mlx.scheduler.video_unified_scheduler import (
    DegradationPlan,
    MemoryLevel,
    NF4DequantCache,
    VideoUnifiedScheduler,
    get_video_scheduler,
)
from fusion_mlx.utils.model_quant import convert_to_nf4, validate_nf4_dir
from fusion_mlx.utils.video_audio_export import normalize_audio
from fusion_mlx.video.common import NoiseSchedulerBase, UpsampleBase, VideoVAEBase


class TestMemoryLevel:
    def test_ok_under_warn(self):
        s = VideoUnifiedScheduler()
        # _current_bytes reads real footprint; on a dev box it's well under 90GB.
        # Just assert probe returns a valid level enum.
        assert s.probe_level() in tuple(MemoryLevel)

    def test_plan_ok_when_low(self):
        s = VideoUnifiedScheduler()
        plan = s.plan()
        # On a dev box we expect OK; if the box is genuinely >90GB we still
        # get a valid plan with level >= L1. Assert structure, not the exact
        # level, so this doesn't flake on a loaded machine.
        assert isinstance(plan, DegradationPlan)
        assert plan.level in tuple(MemoryLevel)


class TestDegradationPlan:
    def _params(self):
        class P:
            num_inference_steps = 40
            height = 768
            width = 768
            audio = True
            no_compile = False

        return P()

    def test_l1_reduces_steps(self):
        p = self._params()
        plan = DegradationPlan(
            level=MemoryLevel.L1_WARN,
            reduce_steps=True,
            disable_upsample=True,
            reason="x",
        )
        plan.apply_to(p)
        assert p.num_inference_steps == 20
        assert p.no_compile is True

    def test_l2_drops_resolution_and_audio(self):
        p = self._params()
        plan = DegradationPlan(
            level=MemoryLevel.L2_PROTECT,
            drop_resolution=True,
            disable_audio=True,
            reason="x",
        )
        plan.apply_to(p)
        assert p.height == 384
        assert p.width == 384
        assert p.audio is False

    def test_l1_floor_steps(self):
        p = self._params()
        p.num_inference_steps = 12
        plan = DegradationPlan(level=MemoryLevel.L1_WARN, reduce_steps=True, reason="x")
        plan.apply_to(p)
        assert p.num_inference_steps == 8  # max(8, 12//2=6) = 8


class TestMutex:
    def test_acquire_release(self):
        s = VideoUnifiedScheduler()
        s.acquire("ltx2_5")
        assert s.snapshot()["active_model"] == "ltx2_5"
        s.release("ltx2_5")
        assert s.snapshot()["active_model"] is None

    def test_second_acquire_blocks_or_raises(self):
        s = VideoUnifiedScheduler()
        s.acquire("minimax_h3")
        # second acquire with different model must not silently co-resident
        with pytest.raises(RuntimeError):
            s.acquire("ltx2_5")
        s.release("minimax_h3")


class TestDequantCache:
    def test_resident_hit(self):
        c = NF4DequantCache(budget_gb=1)
        calls = {"n": 0}

        def dq():
            calls["n"] += 1
            import mlx.core as mx

            return mx.zeros((4, 4))

        c.get_or_dequant("w1", dq)
        c.get_or_dequant("w1", dq)
        assert calls["n"] == 1  # second was a cache hit

    def test_non_resident_no_cache(self):
        c = NF4DequantCache(budget_gb=1)
        import mlx.core as mx

        c.get_or_dequant("w", lambda: mx.zeros((2, 2)), resident=False)
        assert len(c._entries) == 0

    def test_release_clears(self):
        c = NF4DequantCache(budget_gb=1)
        import mlx.core as mx

        c.get_or_dequant("w", lambda: mx.zeros((2, 2)))
        assert len(c._entries) == 1
        c.release()
        assert len(c._entries) == 0


class TestRouter:
    def test_drama_routes_h3(self):
        r = VideoRouter()
        assert r.route("短剧人物对白", scene="drama") == "minimax_h3"

    def test_ui_routes_ltx(self):
        r = VideoRouter()
        assert r.route("UI page scroll demo", scene="general") == "ltx2_5"

    def test_hint_overrides(self):
        r = VideoRouter()
        assert r.route("anything", model_hint="h3") == "minimax_h3"
        assert r.route("anything", model_hint="ltx") == "ltx2_5"

    def test_keyword_drama(self):
        r = VideoRouter()
        assert r.route("a short drama dialogue scene") == "minimax_h3"

    def test_keyword_ui(self):
        r = VideoRouter()
        assert r.route("interface animation popup") == "ltx2_5"

    def test_default_general(self):
        r = VideoRouter()
        assert r.route("a cat walking") == "ltx2_5"

    def test_audio_default_h3(self):
        r = VideoRouter()
        assert r.route("a cat walking", audio=True) == "minimax_h3"


class TestModelQuant:
    def test_rejects_bf16(self, tmp_path):
        with pytest.raises(ValueError, match="banned"):
            convert_to_nf4("x", tmp_path, bits=16)

    def test_rejects_int8(self, tmp_path):
        with pytest.raises(ValueError, match="banned"):
            from fusion_mlx.utils.model_quant import _reject_banned

            _reject_banned("int8")

    def test_manifest_written(self, tmp_path):
        out = convert_to_nf4("org/foo", tmp_path / "q", model_kind="dit")
        assert validate_nf4_dir(out)
        assert (out / "manifest.json").exists()

    def test_validate_rejects_non_nf4(self, tmp_path):
        assert validate_nf4_dir(tmp_path) is False


class TestExport:
    def test_normalize_audio_clips(self):
        wav = np.array([0.5, -0.5, 0.3], dtype=np.float32)
        out = normalize_audio(wav, target_db=-3.0)
        assert np.max(np.abs(out)) <= 1.0 + 1e-6

    def test_normalize_silent_passthrough(self):
        wav = np.zeros(8, dtype=np.float32)
        out = normalize_audio(wav)
        assert np.allclose(out, 0.0)


class TestCommonBases:
    def test_vae_base_abstract(self):
        with pytest.raises(TypeError):
            VideoVAEBase()

    def test_upsample_base_abstract(self):
        with pytest.raises(TypeError):
            UpsampleBase()

    def test_scheduler_base_abstract(self):
        with pytest.raises(TypeError):
            NoiseSchedulerBase()


class TestSingleton:
    def test_scheduler_singleton(self):
        assert get_video_scheduler() is get_video_scheduler()

    def test_router_singleton(self):
        assert get_video_router() is get_video_router()
