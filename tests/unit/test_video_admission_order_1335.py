# SPDX-License-Identifier: Apache-2.0
# #1335: video admission must run BEFORE the model load; a rejection after a
# load must unload only the model that request loaded; the projection must
# scale with the requested duration/frames instead of a constant factor.

import base64
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import fusion_mlx.api.videos_routes as videos_routes
from fusion_mlx.admission import HeavyAdmissionDecision
from fusion_mlx.api.videos_routes import (
    VideoGenerateRequest,
    _entry_estimated_bytes,
    _execute_video_generation,
    set_videos_context,
)
from fusion_mlx.engines.video import (
    VideoGenEngine,
    video_activation_budget_bytes,
    video_admission_budget_bytes,
)
from fusion_mlx.pool.engine_pool import EngineEntry, EnginePool
from fusion_mlx.scheduler.video_unified_scheduler import (
    VideoAdmissionRejected,
    VideoMemoryPressureError,
)

GB = 2**30
MODEL = "ltx-2"
WEIGHT_BYTES = 60 * GB


class _RecordingGate:
    def __init__(self, admitted=True):
        self.admitted = admitted
        self.calls = []

    def admit_media(self, kind, budget_bytes):
        self.calls.append((kind, int(budget_bytes)))
        if self.admitted:
            return HeavyAdmissionDecision(admitted=True)
        return HeavyAdmissionDecision(
            admitted=False,
            reason="projected 120.0GB exceeds safety line 108.8GB",
        )


@pytest.fixture(autouse=True)
def _reset_pool():
    yield
    videos_routes._pool = None


def _entry(*, estimate=WEIGHT_BYTES, engine=None, in_use=0):
    return SimpleNamespace(
        estimated_size=estimate,
        last_observed_size=0,
        engine=engine,
        in_use=in_use,
    )


def _video_engine(*, active=False, generate=None):
    engine = MagicMock(spec=VideoGenEngine)
    engine.has_active_requests = MagicMock(return_value=active)
    if generate is not None:
        engine.generate = AsyncMock(side_effect=generate)
    else:
        engine.generate = AsyncMock(return_value=[b"MP4"])
    return engine


def _make_pool(*, entry_before, entry_after=None, generate=None):
    pool = MagicMock()
    if entry_after is None:
        pool.get_entry = MagicMock(return_value=entry_before)
    else:
        pool.get_entry = MagicMock(side_effect=[entry_before, entry_after])
    pool._resolve_hf_repo_id = MagicMock(return_value=MODEL)
    engine = _video_engine(generate=generate)
    pool.get_engine = AsyncMock(return_value=engine)
    pool.unload_engine_async = AsyncMock()
    return pool, engine


def _patch_gate(monkeypatch, gate):
    monkeypatch.setattr("fusion_mlx.admission.get_heavy_admission_gate", lambda: gate)


class TestAdmissionBeforeLoad:
    @pytest.mark.asyncio
    async def test_reject_before_load_never_awaits_get_engine(self, monkeypatch):
        gate = _RecordingGate(admitted=False)
        _patch_gate(monkeypatch, gate)
        pool, engine = _make_pool(entry_before=_entry(engine=None))
        set_videos_context(pool)

        with pytest.raises(VideoAdmissionRejected):
            await _execute_video_generation(
                VideoGenerateRequest(prompt="p", model=MODEL, duration=5.0)
            )

        pool.get_engine.assert_not_awaited()
        engine.generate.assert_not_awaited()
        assert gate.calls, "pre-load projection never reached the gate"
        assert gate.calls[0][0] == "video"
        expected = video_admission_budget_bytes(
            WEIGHT_BYTES, 121, 768, 512, 1, include_weights=True
        )
        assert gate.calls[0][1] == expected

    @pytest.mark.asyncio
    async def test_resident_model_projection_excludes_weights(self, monkeypatch):
        gate = _RecordingGate(admitted=True)
        _patch_gate(monkeypatch, gate)
        resident = _video_engine()
        pool, _engine = _make_pool(entry_before=_entry(engine=resident))
        set_videos_context(pool)

        response = await _execute_video_generation(
            VideoGenerateRequest(prompt="p", model=MODEL)
        )

        assert base64.b64decode(response.data[0].url.split(",", 1)[1]) == b"MP4"
        pool.get_engine.assert_awaited_once_with(MODEL)
        expected = video_admission_budget_bytes(
            WEIGHT_BYTES, 97, 768, 512, 1, include_weights=False
        )
        assert gate.calls[0][1] == expected


class TestProjectionScales:
    @pytest.mark.asyncio
    async def test_route_projection_responds_to_duration(self, monkeypatch):
        gate = _RecordingGate(admitted=True)
        _patch_gate(monkeypatch, gate)
        pool, _engine = _make_pool(entry_before=_entry(engine=None))
        set_videos_context(pool)

        await _execute_video_generation(
            VideoGenerateRequest(prompt="p", model=MODEL, duration=1.0)
        )
        await _execute_video_generation(
            VideoGenerateRequest(prompt="p", model=MODEL, duration=5.0)
        )

        assert [kind for kind, _budget in gate.calls] == ["video", "video"]
        assert gate.calls[1][1] > gate.calls[0][1], (
            "projected budget ignored the requested duration "
            f"(1s={gate.calls[0][1]}, 5s={gate.calls[1][1]})"
        )

    def test_projection_components(self):
        base = video_admission_budget_bytes(WEIGHT_BYTES, 97, 768, 512)
        assert base == WEIGHT_BYTES + video_activation_budget_bytes(
            97, 768, 512, 1, WEIGHT_BYTES
        )
        assert video_admission_budget_bytes(WEIGHT_BYTES, 169, 768, 512) > base
        assert video_admission_budget_bytes(WEIGHT_BYTES, 97, 1280, 720) > base
        assert video_admission_budget_bytes(WEIGHT_BYTES, 97, 768, 512, 2) > base
        assert (
            video_admission_budget_bytes(
                WEIGHT_BYTES, 97, 768, 512, 1, include_weights=False
            )
            < base
        )
        assert video_admission_budget_bytes(0, 97, 768, 512) == 0

    def test_activation_capped_at_legacy_constant(self):
        huge = video_admission_budget_bytes(
            WEIGHT_BYTES, 1024, 2048, 2048, 4, include_weights=False
        )
        assert huge == int(WEIGHT_BYTES * 0.40)

    def test_entry_estimate_prefers_observed_size(self):
        entry = SimpleNamespace(estimated_size=WEIGHT_BYTES, last_observed_size=70 * GB)
        assert _entry_estimated_bytes(entry) == 70 * GB
        assert _entry_estimated_bytes(None) == 0


class TestRejectAfterLoadUnloads:
    @pytest.mark.asyncio
    async def test_unloads_only_newly_loaded_engine(self, monkeypatch):
        gate = _RecordingGate(admitted=True)
        _patch_gate(monkeypatch, gate)
        loaded = _video_engine()
        pool, _engine = _make_pool(
            entry_before=None,
            entry_after=_entry(engine=loaded),
            generate=VideoAdmissionRejected("race: projected over line"),
        )
        set_videos_context(pool)

        with pytest.raises(VideoAdmissionRejected):
            await _execute_video_generation(
                VideoGenerateRequest(prompt="p", model=MODEL)
            )

        pool.get_engine.assert_awaited_once_with(MODEL)
        pool.unload_engine_async.assert_awaited_once_with(MODEL)

    @pytest.mark.asyncio
    async def test_resident_model_not_unloaded(self, monkeypatch):
        gate = _RecordingGate(admitted=True)
        _patch_gate(monkeypatch, gate)
        resident = _video_engine()
        pool, _engine = _make_pool(
            entry_before=_entry(engine=resident),
            generate=VideoAdmissionRejected("post-load activation over line"),
        )
        set_videos_context(pool)

        with pytest.raises(VideoAdmissionRejected):
            await _execute_video_generation(
                VideoGenerateRequest(prompt="p", model=MODEL)
            )

        pool.unload_engine_async.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_busy_engine_not_unloaded(self, monkeypatch):
        gate = _RecordingGate(admitted=True)
        _patch_gate(monkeypatch, gate)
        busy = _video_engine(active=True)
        pool, _engine = _make_pool(
            entry_before=None,
            entry_after=_entry(engine=busy),
            generate=VideoAdmissionRejected("race: projected over line"),
        )
        set_videos_context(pool)

        with pytest.raises(VideoAdmissionRejected):
            await _execute_video_generation(
                VideoGenerateRequest(prompt="p", model=MODEL)
            )

        pool.unload_engine_async.assert_not_awaited()

    def test_rejection_still_maps_to_memory_pressure(self):
        assert issubclass(VideoAdmissionRejected, VideoMemoryPressureError)


class TestSettleSkipAdmissionReadsLive:
    @pytest.mark.asyncio
    async def test_skipped_settle_decrements_counter_and_admission_reads_live(
        self,
    ):
        pool = EnginePool()
        pool._get_final_ceiling = lambda: 0
        entry = EngineEntry(
            model_id="ltx-2",
            model_path="/tmp/ltx-2",
            model_type="video",
            engine_type="video_gen",
            estimated_size=WEIGHT_BYTES,
        )
        pool._entries["ltx-2"] = entry
        pool._current_model_memory = 19 * GB + WEIGHT_BYTES
        live_bytes = 96 * GB
        phys_bytes = 36 * GB

        with (
            patch("fusion_mlx.pool.engine_pool.mx") as mock_mx,
            patch("fusion_mlx.pool.engine_pool.get_mlx_executor", return_value=None),
            patch(
                "fusion_mlx.pool.engine_pool.get_phys_footprint",
                return_value=phys_bytes,
            ),
            patch.object(pool, "_other_entries_serving", return_value=True),
        ):
            mock_mx.get_active_memory = MagicMock(return_value=live_bytes)
            mock_mx.synchronize = MagicMock()
            mock_mx.clear_cache = MagicMock()

            await pool._settle_unloaded_engine("ltx-2", pre_unload_active=live_bytes)

            assert pool._current_model_memory == 19 * GB
            assert pool._admission_current_usage() == live_bytes
