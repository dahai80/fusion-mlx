# SPDX-License-Identifier: Apache-2.0
"""Regression tests for issue #1079.

#1079: /v1/videos/generate was sync-blocking with no job status endpoints.
A client disconnect lost the result; the admin GUI had no video generation
entry point or progress display.

Fix (#1079): added POST /v1/videos/jobs (submit, returns job_id), GET
/v1/videos/jobs (list), GET /v1/videos/jobs/{id} (status), GET
/v1/videos/jobs/{id}/output/{index} (stream completed video file), DELETE
/v1/videos/jobs/{id} (cancel queued / delete terminal). Video output is
written to temp files (paths in job dict), not held in memory. TTL (30min)
+ cap (20 jobs) prune terminal jobs + unlink output files.
"""

from __future__ import annotations

import asyncio
import base64
import os
import tempfile
import time
from unittest.mock import AsyncMock

import pytest
from fastapi import HTTPException

from fusion_mlx.api import videos_routes as vr
from fusion_mlx.api.videos_routes import (
    VideoGenerateRequest,
    VideoGenerateResponse,
    VideoOutput,
)


def _reset_jobs():
    vr._video_jobs.clear()


def _make_request(**kwargs) -> VideoGenerateRequest:
    defaults = dict(prompt="test prompt", model="test-model")
    defaults.update(kwargs)
    return VideoGenerateRequest(**defaults)


def _make_completed_job(job_id: str = "j1") -> dict:
    fd, path = tempfile.mkstemp(prefix="vjob_test_", suffix=".mp4")
    with os.fdopen(fd, "wb") as f:
        f.write(b"\x00\x00\x00\x18ftypisom\x00\x00\x02\x00")
    now = time.time()
    return {
        "job_id": job_id,
        "status": "completed",
        "progress": 1.0,
        "model": "test-model",
        "prompt": "test",
        "n": 1,
        "output_files": [path],
        "output_meta": [
            {
                "num_frames": 97,
                "fps": 24,
                "width": 768,
                "height": 512,
                "duration_seconds": 4.0,
                "size_bytes": 20,
            }
        ],
        "warnings": [],
        "error": None,
        "created_at": now - 100,
        "updated_at": now,
    }


class TestNewVideoJob:
    def test_structure(self):
        req = _make_request(prompt="hello world")
        job = vr._new_video_job(req)
        assert job["status"] == "queued"
        assert job["progress"] == 0.0
        assert job["prompt"] == "hello world"
        assert job["output_files"] == []
        assert job["error"] is None
        assert len(job["job_id"]) == 16

    def test_prompt_truncated(self):
        req = _make_request(prompt="x" * 300)
        job = vr._new_video_job(req)
        assert len(job["prompt"]) == 200


class TestUnlinkVideoOutputs:
    def test_unlinks_existing_files(self):
        fd, path = tempfile.mkstemp(suffix=".mp4")
        os.close(fd)
        assert os.path.exists(path)
        vr._unlink_video_outputs({"output_files": [path]})
        assert not os.path.exists(path)

    def test_none_job_is_noop(self):
        vr._unlink_video_outputs(None)

    def test_missing_file_is_silent(self):
        vr._unlink_video_outputs({"output_files": ["/nonexistent/path.mp4"]})


class TestPruneVideoJobs:
    def _reset(self):
        _reset_jobs()

    def test_ttl_evicts_old_terminal(self):
        self._reset()
        old = _make_completed_job("old")
        old["updated_at"] = time.time() - vr._VIDEO_JOB_TTL_SECONDS - 60
        fresh = _make_completed_job("fresh")
        vr._video_jobs["old"] = old
        vr._video_jobs["fresh"] = fresh
        with vr._video_jobs_lock:
            vr._video_jobs_prune_locked()
        assert "old" not in vr._video_jobs
        assert "fresh" in vr._video_jobs

    def test_ttl_keeps_running_even_if_old(self):
        self._reset()
        running = _make_completed_job("run")
        running["status"] = "running"
        running["updated_at"] = time.time() - 99999
        vr._video_jobs["run"] = running
        with vr._video_jobs_lock:
            vr._video_jobs_prune_locked()
        assert "run" in vr._video_jobs

    def test_cap_drops_oldest_terminal(self):
        self._reset()
        now = time.time()
        for i in range(vr._MAX_VIDEO_JOBS + 5):
            j = _make_completed_job(f"cap-{i}")
            j["updated_at"] = now + i
            vr._video_jobs[f"cap-{i}"] = j
        with vr._video_jobs_lock:
            vr._video_jobs_prune_locked()
        assert len(vr._video_jobs) <= vr._MAX_VIDEO_JOBS
        assert "cap-0" not in vr._video_jobs


class TestDeleteVideoJob:
    def _reset(self):
        _reset_jobs()

    def test_delete_queued_cancels(self):
        self._reset()
        job = vr._new_video_job(_make_request())
        job["status"] = "queued"
        vr._video_jobs[job["job_id"]] = job
        import asyncio

        result = asyncio.run(vr.delete_video_job(job["job_id"], _auth=True))
        assert result["status"] == "cancelled"
        assert job["status"] == "cancelled"

    def test_delete_running_returns_409(self):
        self._reset()
        job = vr._new_video_job(_make_request())
        job["status"] = "running"
        vr._video_jobs[job["job_id"]] = job
        import asyncio

        with pytest.raises(HTTPException) as exc:
            asyncio.run(vr.delete_video_job(job["job_id"], _auth=True))
        assert exc.value.status_code == 409

    def test_delete_terminal_removes_and_unlinks(self):
        self._reset()
        job = _make_completed_job("del-me")
        output_path = job["output_files"][0]
        assert os.path.exists(output_path)
        vr._video_jobs["del-me"] = job
        import asyncio

        result = asyncio.run(vr.delete_video_job("del-me", _auth=True))
        assert result["status"] == "deleted"
        assert "del-me" not in vr._video_jobs
        assert not os.path.exists(output_path)

    def test_delete_not_found_returns_404(self):
        self._reset()
        import asyncio

        with pytest.raises(HTTPException) as exc:
            asyncio.run(vr.delete_video_job("nope", _auth=True))
        assert exc.value.status_code == 404


class TestGetVideoJobOutput:
    def _reset(self):
        _reset_jobs()

    def test_not_found_returns_404(self):
        self._reset()
        import asyncio

        with pytest.raises(HTTPException) as exc:
            asyncio.run(vr.get_video_job_output("nope", 0, _auth=True))
        assert exc.value.status_code == 404

    def test_not_completed_returns_409(self):
        self._reset()
        job = vr._new_video_job(_make_request())
        job["status"] = "running"
        vr._video_jobs[job["job_id"]] = job
        import asyncio

        with pytest.raises(HTTPException) as exc:
            asyncio.run(vr.get_video_job_output(job["job_id"], 0, _auth=True))
        assert exc.value.status_code == 409

    def test_out_of_range_returns_404(self):
        self._reset()
        job = _make_completed_job("oob")
        vr._video_jobs["oob"] = job
        import asyncio

        with pytest.raises(HTTPException) as exc:
            asyncio.run(vr.get_video_job_output("oob", 99, _auth=True))
        assert exc.value.status_code == 404

    def test_completed_returns_file(self):
        self._reset()
        job = _make_completed_job("ok")
        vr._video_jobs["ok"] = job
        import asyncio

        resp = asyncio.run(vr.get_video_job_output("ok", 0, _auth=True))
        assert resp.status_code == 200
        assert "video/mp4" in resp.media_type


class TestRunVideoJob:
    def _reset(self):
        _reset_jobs()

    def test_completed_writes_output_files(self):
        self._reset()
        req = _make_request()
        job = vr._new_video_job(req)
        vr._video_jobs[job["job_id"]] = job
        fake_mp4 = b"\x00\x00\x00\x18ftypisom\x00\x00\x02\x00"
        fake_b64 = base64.b64encode(fake_mp4).decode()
        fake_response = VideoGenerateResponse(
            data=[
                VideoOutput(
                    b64_json=fake_b64,
                    num_frames=97,
                    fps=24,
                    width=768,
                    height=512,
                    duration_seconds=4.0,
                )
            ],
            warnings=["test warning"],
        )
        with pytest.MonkeyPatch().context() as m:
            m.setattr(
                vr, "_execute_video_generation", AsyncMock(return_value=fake_response)
            )
            asyncio.run(vr._run_video_job(job, req))
        assert job["status"] == "completed"
        assert job["progress"] == 1.0
        assert len(job["output_files"]) == 1
        assert os.path.exists(job["output_files"][0])
        assert job["output_meta"][0]["num_frames"] == 97
        assert job["warnings"] == ["test warning"]
        for path in job["output_files"]:
            os.unlink(path)

    def test_failed_on_exception(self):
        self._reset()
        req = _make_request()
        job = vr._new_video_job(req)
        vr._video_jobs[job["job_id"]] = job
        with pytest.MonkeyPatch().context() as m:
            m.setattr(
                vr,
                "_execute_video_generation",
                AsyncMock(
                    side_effect=HTTPException(status_code=422, detail="bad params")
                ),
            )
            asyncio.run(vr._run_video_job(job, req))
        assert job["status"] == "failed"
        assert "bad params" in job["error"]

    def test_skips_when_cancelled(self):
        self._reset()
        req = _make_request()
        job = vr._new_video_job(req)
        job["status"] = "cancelled"
        vr._video_jobs[job["job_id"]] = job
        mock_exec = AsyncMock()
        with pytest.MonkeyPatch().context() as m:
            m.setattr(vr, "_execute_video_generation", mock_exec)
            asyncio.run(vr._run_video_job(job, req))
        assert job["status"] == "cancelled"
        assert job["output_files"] == []
        mock_exec.assert_not_called()


if __name__ == "__main__":  # pragma: no cover — convenience only
    pytest.main([__file__, "-v"])
