# SPDX-License-Identifier: Apache-2.0
"""Regression tests for issue #1078.

#1078: convert/quantize jobs had GET + DELETE but no cancel. A queued job
backing up behind a long conversion could not be aborted from the GUI.

Fix (#1078): added POST /v1/convert/jobs/{id}/cancel and
POST /v1/quantize/jobs/{id}/cancel. Queued jobs are cancelled via
future.cancel() + status flip; running jobs return 409 (mlx-lm convert()
has no cancel hook); terminal jobs are idempotent no-ops. _run_job checks
the cancelled status at entry and skips if already cancelled (race: the
executor may have just picked it up).
"""

from __future__ import annotations

import time
from concurrent.futures import Future
from unittest.mock import MagicMock

import pytest
from fastapi import HTTPException

from fusion_mlx.api import convert_routes as cr


def _make_job(
    kind: str = "convert",
    status: str = "queued",
    job_id: str | None = None,
    updated: float | None = None,
) -> dict:
    now = time.time()
    return {
        "job_id": job_id or f"job-{now}",
        "kind": kind,
        "model": "test-model",
        "status": status,
        "progress": 0.0,
        "output_path": None,
        "error": None,
        "created_at": now,
        "updated_at": updated if updated is not None else now,
    }


def _make_future(cancelled: bool = False) -> MagicMock:
    fut = MagicMock(spec=Future)
    fut.cancel.return_value = True
    if cancelled:
        fut.cancelled.return_value = True
    return fut


class TestCancelJob:
    def _reset(self):
        cr._jobs.clear()
        cr._futures.clear()

    def test_cancel_queued_flips_status(self):
        self._reset()
        job = _make_job(status="queued", job_id="q1")
        fut = _make_future()
        cr._jobs[job["job_id"]] = job
        cr._futures[job["job_id"]] = fut
        result = cr._cancel_job(job["job_id"], "convert")
        assert result["status"] == "cancelled"
        assert job["status"] == "cancelled"
        fut.cancel.assert_called_once()
        assert job["job_id"] not in cr._futures

    def test_cancel_running_returns_409(self):
        self._reset()
        job = _make_job(status="running", job_id="r1")
        cr._jobs[job["job_id"]] = job
        with pytest.raises(HTTPException) as exc:
            cr._cancel_job(job["job_id"], "convert")
        assert exc.value.status_code == 409
        assert job["status"] == "running"

    def test_cancel_terminal_is_idempotent(self):
        self._reset()
        for terminal in ("completed", "failed", "interrupted", "cancelled"):
            job = _make_job(status=terminal, job_id=f"t-{terminal}")
            cr._jobs[job["job_id"]] = job
            result = cr._cancel_job(job["job_id"], "convert")
            assert result["status"] == terminal
            assert job["status"] == terminal

    def test_cancel_not_found_returns_404(self):
        self._reset()
        with pytest.raises(HTTPException) as exc:
            cr._cancel_job("nope", "convert")
        assert exc.value.status_code == 404

    def test_cancel_wrong_kind_returns_404(self):
        self._reset()
        job = _make_job(kind="quantize", status="queued", job_id="qk1")
        cr._jobs[job["job_id"]] = job
        with pytest.raises(HTTPException) as exc:
            cr._cancel_job(job["job_id"], "convert")
        assert exc.value.status_code == 404

    def test_cancel_no_future_still_flips(self):
        self._reset()
        job = _make_job(status="queued", job_id="qnf")
        cr._jobs[job["job_id"]] = job
        result = cr._cancel_job(job["job_id"], "convert")
        assert result["status"] == "cancelled"
        assert job["status"] == "cancelled"

    def test_cancelled_in_terminal_statuses(self):
        assert "cancelled" in cr._TERMINAL_STATUSES

    def test_delete_cancels_cannot_delete_cancelled(self):
        self._reset()
        job = _make_job(status="cancelled", job_id="c1")
        cr._jobs[job["job_id"]] = job
        result = cr._delete_job(job["job_id"], "convert")
        assert result["status"] == "deleted"
        assert job["job_id"] not in cr._jobs


class TestRunJobSkipCancelled:
    def _reset(self):
        cr._jobs.clear()
        cr._futures.clear()

    def test_run_job_skips_when_cancelled(self):
        self._reset()
        job = _make_job(status="cancelled", job_id="skip1")
        cr._jobs[job["job_id"]] = job
        req = MagicMock()
        cr._run_job(job, req)
        assert job["status"] == "cancelled"
        assert job["progress"] == 0.0
        assert job["output_path"] is None

    def test_run_job_skips_when_queued_cancelled_race(self):
        self._reset()
        job = _make_job(status="queued", job_id="race1")
        cr._jobs[job["job_id"]] = job
        with cr._jobs_lock:
            job["status"] = "cancelled"
        req = MagicMock()
        cr._run_job(job, req)
        assert job["status"] == "cancelled"
        assert job["output_path"] is None


class TestPruneJobsCleansFutures:
    def _reset(self):
        cr._jobs.clear()
        cr._futures.clear()

    def test_prune_pops_stale_future(self):
        self._reset()
        now = time.time()
        job = _make_job(
            status="completed",
            job_id="stale-fut",
            updated=now - cr._JOB_TTL_SECONDS - 60,
        )
        cr._jobs[job["job_id"]] = job
        cr._futures[job["job_id"]] = _make_future()
        with cr._jobs_lock:
            cr._prune_jobs()
        assert job["job_id"] not in cr._jobs
        assert job["job_id"] not in cr._futures


if __name__ == "__main__":  # pragma: no cover — convenience only
    pytest.main([__file__, "-v"])
