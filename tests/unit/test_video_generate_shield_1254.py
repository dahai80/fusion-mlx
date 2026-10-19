# SPDX-License-Identifier: Apache-2.0
"""#1254: sync /generate decoupled from the client connection.

The generation runs as a shielded job task: a client disconnect cancels only
the HTTP response, the GPU work completes server-side and stays retrievable
via the #1079 jobs registry.
"""

import asyncio
import base64
import os
import time

import pytest
from fastapi import HTTPException

from fusion_mlx.api import videos_routes as vr
from fusion_mlx.api.videos_routes import (
    VideoGenerateRequest,
    VideoGenerateResponse,
    VideoOutput,
    submit_video_job,
)


def _reset_jobs():
    vr._video_jobs.clear()


from unittest.mock import AsyncMock


def _make_request(**kw):
    base = {
        "model": "ltx2_5",
        "prompt": "test prompt",
        "num_frames": 97,
        "width": 768,
        "height": 512,
    }
    base.update(kw)
    return VideoGenerateRequest(**base)


def _fake_response():
    fake_mp4 = b"\x00\x00\x00\x18ftypisom\x00\x00\x02\x00"
    return VideoGenerateResponse(
        data=[
            VideoOutput(
                b64_json=base64.b64encode(fake_mp4).decode(),
                num_frames=97,
                fps=24,
                width=768,
                height=512,
                duration_seconds=4.0,
            )
        ],
        warnings=[],
    )


@pytest.fixture(autouse=True)
def _clean_registry():
    _reset_jobs()
    yield
    _reset_jobs()


def test_client_disconnect_leaves_job_running():
    req = _make_request()
    release = asyncio.Event()
    started = asyncio.Event()

    async def slow_gen(_request):
        started.set()
        await release.wait()
        return _fake_response()

    async def main():
        with pytest.MonkeyPatch().context() as m:
            m.setattr(vr, "_execute_video_generation", slow_gen)
            outer = asyncio.create_task(vr.generate_video(req))
            await started.wait()
            # Find the job that just went running, then cancel the request
            # task — the shielded job task must survive.
            running = [j for j in vr._video_jobs.values() if j["status"] == "running"]
            assert running, "job never started"
            job = running[0]
            outer.cancel()
            with pytest.raises(asyncio.CancelledError):
                await outer
            # The shielded generation is still running: finish it.
            assert job["status"] == "running"
            release.set()
            await asyncio.sleep(0.05)
            while job["status"] == "running":
                await asyncio.sleep(0.01)
        assert job["status"] == "completed"
        assert len(job["output_files"]) == 1
        for path in job["output_files"]:
            os.unlink(path)

    asyncio.run(main())


def test_sync_wait_timeout_returns_202_and_job_survives():
    req = _make_request()
    release = asyncio.Event()
    started = asyncio.Event()

    async def slow_gen(_request):
        started.set()
        await release.wait()
        return _fake_response()

    async def main():
        with pytest.MonkeyPatch().context() as m:
            m.setattr(vr, "_SYNC_VIDEO_WAIT_SECONDS", 0.05)
            m.setattr(vr, "_execute_video_generation", slow_gen)
            resp = await vr.generate_video(req)
            assert resp.status_code == 202
            assert resp.body and b"video_job_pending" in resp.body
            running = [j for j in vr._video_jobs.values() if j["status"] == "running"]
            assert running and running[0]["status"] == "running"
            release.set()
            job = running[0]
            while job["status"] == "running":
                await asyncio.sleep(0.01)
            assert job["status"] == "completed"

    asyncio.run(main())


def test_memory_guard_cancel_sets_error_code():
    req = _make_request()
    job_holder: dict = {}

    async def hang(_request):
        await asyncio.sleep(30)

    async def main():
        with pytest.MonkeyPatch().context() as m:
            m.setattr(vr, "_execute_video_generation", hang)
            task = asyncio.create_task(vr._run_video_job(_new_job(req), req))
            await asyncio.sleep(0.02)
            job = job_holder.get("job") or next(iter(vr._video_jobs.values()))
            job_holder["job"] = job
            from fusion_mlx.scheduler.video_unified_scheduler import (
                _VIDEO_ABORT_REASON,
            )

            _VIDEO_ABORT_REASON.append("enforcer: projected peak above red line")
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
        assert job["status"] == "failed"
        assert job["error_code"] == "memory_guard"
        assert "memory guard" in job["error"]

    asyncio.run(main())


def _new_job(req):
    job = vr._new_video_job(req)
    vr._video_jobs[job["job_id"]] = job
    return job


def test_happy_path_returns_encoded_response():
    req = _make_request()
    with pytest.MonkeyPatch().context() as m:
        m.setattr(
            vr, "_execute_video_generation", AsyncMock(return_value=_fake_response())
        )
        resp = asyncio.run(vr.generate_video(req))
    assert isinstance(resp, VideoGenerateResponse)
    assert len(resp.data) == 1
    assert resp.data[0].num_frames == 97
    job = next(iter(vr._video_jobs.values()))
    assert job["status"] == "completed"
    for path in job["output_files"]:
        os.unlink(path)


def test_submit_path_no_unretrieved_exception():
    # _run_video_job re-raises after recording the failure; the fire-and-forget
    # submit task must not spam "Task exception was never retrieved".
    req = _make_request()
    with pytest.MonkeyPatch().context() as m:
        m.setattr(
            vr,
            "_execute_video_generation",
            AsyncMock(side_effect=HTTPException(503, "mutex busy")),
        )
        resp = asyncio.run(submit_video_job(req))
        assert resp.job_id
        deadline = time.monotonic() + 2.0
        while True:
            job = vr._video_jobs.get(resp.job_id)
            assert job is not None
            if job["status"] == "failed":
                break
            assert time.monotonic() < deadline
            time.sleep(0.01)
        assert job["error_code"] == "http_error"


if __name__ == "__main__":  # pragma: no cover — convenience only
    pytest.main([__file__, "-v"])
