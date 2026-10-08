# SPDX-License-Identifier: Apache-2.0
"""Regression tests for issue #1081.

#1081: layered_quantize had POST (start) + GET (query) + DELETE (terminal)
but no cancel. A queued job backing up behind a long layered quantization
could not be aborted from the GUI.

Fix (#1081): added POST /v1/quantize/layered/jobs/{id}/cancel. Queued jobs
cancelled via future.cancel() + status flip; running jobs return 409; terminal
jobs idempotent. _run_layered_quantize checks cancelled status at entry and
skips if already cancelled.
"""

from __future__ import annotations

import time
from concurrent.futures import Future
from unittest.mock import MagicMock

import pytest
from fastapi import HTTPException

from fusion_mlx.api import layered_quantize_routes as lq


def _make_job(
    status: str = "queued",
    job_id: str | None = None,
    updated: float | None = None,
) -> dict:
    now = time.time()
    return {
        "job_id": job_id or f"lq-{now}",
        "kind": "layered-quantize",
        "model": "test-model",
        "status": status,
        "progress": 0.0,
        "output_path": None,
        "error": None,
        "created_at": now,
        "updated_at": updated if updated is not None else now,
    }


def _make_future() -> MagicMock:
    fut = MagicMock(spec=Future)
    fut.cancel.return_value = True
    return fut


class TestCancelLayeredJob:
    def _reset(self):
        lq._layered_jobs.clear()
        lq._layered_futures.clear()

    def test_cancel_queued_flips_status(self):
        self._reset()
        job = _make_job(status="queued", job_id="lq-q1")
        fut = _make_future()
        lq._layered_jobs[job["job_id"]] = job
        lq._layered_futures[job["job_id"]] = fut
        result = lq._cancel_layered_job(job["job_id"])
        assert result["status"] == "cancelled"
        assert job["status"] == "cancelled"
        fut.cancel.assert_called_once()
        assert job["job_id"] not in lq._layered_futures

    def test_cancel_running_returns_409(self):
        self._reset()
        job = _make_job(status="running", job_id="lq-r1")
        lq._layered_jobs[job["job_id"]] = job
        with pytest.raises(HTTPException) as exc:
            lq._cancel_layered_job(job["job_id"])
        assert exc.value.status_code == 409
        assert job["status"] == "running"

    def test_cancel_terminal_is_idempotent(self):
        self._reset()
        for terminal in ("completed", "failed", "interrupted", "cancelled"):
            job = _make_job(status=terminal, job_id=f"lq-t-{terminal}")
            lq._layered_jobs[job["job_id"]] = job
            result = lq._cancel_layered_job(job["job_id"])
            assert result["status"] == terminal
            assert job["status"] == terminal

    def test_cancel_not_found_returns_404(self):
        self._reset()
        with pytest.raises(HTTPException) as exc:
            lq._cancel_layered_job("nope")
        assert exc.value.status_code == 404

    def test_cancel_no_future_still_flips(self):
        self._reset()
        job = _make_job(status="queued", job_id="lq-nf")
        lq._layered_jobs[job["job_id"]] = job
        result = lq._cancel_layered_job(job["job_id"])
        assert result["status"] == "cancelled"
        assert job["status"] == "cancelled"

    def test_cancelled_in_terminal_statuses(self):
        assert "cancelled" in lq._LAYERED_TERMINAL_STATUSES


class TestRunLayeredSkipCancelled:
    def _reset(self):
        lq._layered_jobs.clear()
        lq._layered_futures.clear()

    def test_run_skips_when_cancelled(self):
        self._reset()
        job = _make_job(status="cancelled", job_id="lq-skip")
        lq._layered_jobs[job["job_id"]] = job
        req = MagicMock()
        lq._run_layered_quantize(job, req)
        assert job["status"] == "cancelled"
        assert job["progress"] == 0.0
        assert job["output_path"] is None

    def test_run_skips_queued_cancelled_race(self):
        self._reset()
        job = _make_job(status="queued", job_id="lq-race")
        lq._layered_jobs[job["job_id"]] = job
        with lq._layered_jobs_lock:
            job["status"] = "cancelled"
        req = MagicMock()
        lq._run_layered_quantize(job, req)
        assert job["status"] == "cancelled"
        assert job["output_path"] is None


class TestPruneLayeredCleansFutures:
    def _reset(self):
        lq._layered_jobs.clear()
        lq._layered_futures.clear()

    def test_prune_pops_stale_future(self):
        self._reset()
        now = time.time()
        job = _make_job(
            status="completed",
            job_id="lq-stale",
            updated=now - lq._LAYERED_JOB_TTL_SECONDS - 60,
        )
        lq._layered_jobs[job["job_id"]] = job
        lq._layered_futures[job["job_id"]] = _make_future()
        with lq._layered_jobs_lock:
            lq._prune_layered_jobs()
        assert job["job_id"] not in lq._layered_jobs
        assert job["job_id"] not in lq._layered_futures

    def test_delete_pops_future(self):
        self._reset()
        job = _make_job(status="completed", job_id="lq-del")
        lq._layered_jobs[job["job_id"]] = job
        lq._layered_futures[job["job_id"]] = _make_future()
        import asyncio

        result = asyncio.run(
            lq.delete_layered_quantize_job(job["job_id"], _is_admin=True)
        )
        assert result["status"] == "deleted"
        assert job["job_id"] not in lq._layered_futures


if __name__ == "__main__":  # pragma: no cover — convenience only
    pytest.main([__file__, "-v"])
