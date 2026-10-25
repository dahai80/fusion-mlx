# SPDX-License-Identifier: Apache-2.0
# #1407: the image job deadline was a total wall-clock budget, so a 30-step
# bf16 edit model (56GB, ~80s/step) was always killed at 600s even though the
# worker was emitting progress the whole time. The deadline is now an
# inactivity budget that every worker event pushes forward, bounded by an
# absolute FUSION_IMAGE_TOTAL_TIMEOUT cap.
import asyncio
import sys

import pytest

from fusion_mlx.engines import image_gen
from fusion_mlx.media import job_manager as jm

CHATTY = """
import json, time
for i in range(10):
    time.sleep(0.15)
    print(json.dumps({"type": "progress", "image": 0, "step": i, "total": 10}), flush=True)
print(json.dumps({"type": "result", "outputs": ["/tmp/out.png"]}), flush=True)
"""

SILENT = """
import time
time.sleep(30)
"""

SLOW_LOAD = """
import json, time
time.sleep(0.6)
print(json.dumps({"type": "result", "outputs": ["/tmp/slow.png"]}), flush=True)
"""


async def _spawn(script: str):
    return await asyncio.create_subprocess_exec(
        sys.executable,
        "-c",
        script,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )


async def _read(script, timeout, total_timeout=0.0, load_timeout=0.0):
    proc = await _spawn(script)
    mgr = jm.MediaJobManager()
    result = jm.MediaJobResult()
    try:
        await mgr._read_stdout(
            proc, timeout, None, result, load_timeout, None, total_timeout
        )
    finally:
        if proc.returncode is None:
            mgr._kill(proc)
        await proc.wait()
    return result


class TestTotalTimeoutGetter:
    def test_default(self, monkeypatch):
        monkeypatch.delenv("FUSION_IMAGE_TOTAL_TIMEOUT", raising=False)
        assert jm.get_image_total_timeout() == 7200.0

    def test_env_override(self, monkeypatch):
        monkeypatch.setenv("FUSION_IMAGE_TOTAL_TIMEOUT", "90")
        assert jm.get_image_total_timeout() == 90.0

    def test_non_positive_disables_cap(self, monkeypatch):
        monkeypatch.setenv("FUSION_IMAGE_TOTAL_TIMEOUT", "0")
        assert jm.get_image_total_timeout() == 0.0

    def test_invalid_falls_back(self, monkeypatch):
        monkeypatch.setenv("FUSION_IMAGE_TOTAL_TIMEOUT", "soon")
        assert jm.get_image_total_timeout() == 7200.0


class TestInactivityDeadline:
    def test_progressing_worker_outlives_timeout(self):
        # 10 events x 0.15s = 1.5s total, every gap well under 0.5s. The old
        # total-budget code raised TimeoutError at 0.5s.
        result = asyncio.run(_read(CHATTY, timeout=0.5))
        assert result.outputs == ["/tmp/out.png"]

    def test_silent_worker_still_dies(self):
        with pytest.raises(TimeoutError):
            asyncio.run(_read(SILENT, timeout=0.3))

    def test_total_cap_kills_chatty_worker(self):
        with pytest.raises(TimeoutError) as ei:
            asyncio.run(_read(CHATTY, timeout=0.5, total_timeout=0.4))
        assert "total timeout" in str(ei.value)

    def test_load_hang_still_fires(self):
        with pytest.raises(jm.MediaJobLoadHangError):
            asyncio.run(_read(SILENT, timeout=5.0, load_timeout=0.3))

    def test_first_event_waits_for_the_load_window(self):
        # Silent model load outlasts the inactivity budget but stays inside
        # the load window: a big checkpoint must not be killed as "silent".
        result = asyncio.run(_read(SLOW_LOAD, timeout=0.3, load_timeout=3.0))
        assert result.outputs == ["/tmp/slow.png"]

    def test_no_events_past_load_window_still_fails(self):
        with pytest.raises(jm.MediaJobLoadHangError):
            asyncio.run(_read(SILENT, timeout=0.2, load_timeout=0.4))


class TestStepProjection:
    @pytest.fixture(autouse=True)
    def _clear(self):
        image_gen._STEP_SECONDS.clear()
        yield
        image_gen._STEP_SECONDS.clear()

    def test_project_none_without_history(self):
        assert image_gen.project_image_seconds("qwen_image_edit", 30, 1) is None

    def test_projection_scales_with_steps_and_images(self):
        image_gen.record_step_seconds("qwen_image_edit", 80.0)
        assert image_gen.project_image_seconds("qwen_image_edit", 30, 1) == 2400.0
        assert image_gen.project_image_seconds("qwen_image_edit", 30, 2) == 4800.0

    def test_ema_moves_toward_new_samples(self):
        image_gen.record_step_seconds("v", 100.0)
        image_gen.record_step_seconds("v", 10.0)
        assert 10.0 < image_gen._STEP_SECONDS["v"] < 100.0

    def test_rejects_non_positive_and_absurd(self):
        image_gen.record_step_seconds("v", 0.0)
        image_gen.record_step_seconds("v", -1.0)
        image_gen.record_step_seconds("v", 10_000.0)
        assert image_gen._STEP_SECONDS == {}


class TestGuidanceFloor:
    def test_zero_guidance_accepted(self):
        from fusion_mlx.api.images import ImageGenerateRequest

        assert ImageGenerateRequest(prompt="x", guidance=0.0).guidance == 0.0

    def test_negative_guidance_rejected(self):
        from pydantic import ValidationError

        from fusion_mlx.api.images import ImageGenerateRequest

        with pytest.raises(ValidationError):
            ImageGenerateRequest(prompt="x", guidance=-0.1)

    def test_upper_bound_kept(self):
        from pydantic import ValidationError

        from fusion_mlx.api.images import ImageGenerateRequest

        with pytest.raises(ValidationError):
            ImageGenerateRequest(prompt="x", guidance=20.5)
