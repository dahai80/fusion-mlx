# SPDX-License-Identifier: Apache-2.0
"""Regression tests for issue #1010.

#1010: convert/layered_quantize job dicts (_jobs / _layered_jobs) had only
POST + GET endpoints — no DELETE, no TTL, no cap. Terminal jobs (completed/
failed/interrupted, with output_path + metadata) accumulated forever in
memory across the server lifetime.

Fix (#1010): added TTL sweep (terminal jobs > 1h evicted on every submit) +
cap (max 200 retained, oldest terminal dropped first) + DELETE endpoints
for convert/quantize/layered-quantize jobs (terminal only; 409 for running).
"""

from __future__ import annotations

import time

import pytest
from fastapi import HTTPException

from fusion_mlx.api import convert_routes as cr
from fusion_mlx.api import layered_quantize_routes as lq


def _make_job(
    kind: str = "convert",
    status: str = "completed",
    updated: float | None = None,
    job_id: str | None = None,
) -> dict:
    now = time.time()
    return {
        "job_id": job_id or f"job-{now}",
        "kind": kind,
        "model": "test-model",
        "status": status,
        "progress": 1.0,
        "output_path": "/out/test",
        "error": None,
        "created_at": now - 100,
        "updated_at": updated if updated is not None else now,
    }


class TestConvertPruneJobs:
    def _reset(self):
        cr._jobs.clear()

    def test_ttl_evicts_old_terminal(self):
        self._reset()
        old = _make_job(
            updated=time.time() - cr._JOB_TTL_SECONDS - 60, job_id="old-job"
        )
        fresh = _make_job(updated=time.time(), job_id="fresh-job")
        cr._jobs[old["job_id"]] = old
        cr._jobs[fresh["job_id"]] = fresh
        with cr._jobs_lock:
            cr._prune_jobs()
        assert old["job_id"] not in cr._jobs
        assert fresh["job_id"] in cr._jobs

    def test_ttl_keeps_running_even_if_old(self):
        self._reset()
        running_old = _make_job(status="running", updated=time.time() - 99999)
        cr._jobs[running_old["job_id"]] = running_old
        with cr._jobs_lock:
            cr._prune_jobs()
        assert running_old["job_id"] in cr._jobs

    def test_cap_drops_oldest_terminal(self):
        self._reset()
        now = time.time()
        for i in range(cr._MAX_JOBS + 5):
            j = _make_job(updated=now + i, job_id=f"cap-{i}")
            cr._jobs[j["job_id"]] = j
        with cr._jobs_lock:
            cr._prune_jobs()
        assert len(cr._jobs) <= cr._MAX_JOBS
        assert "cap-0" not in cr._jobs
        assert "cap-9" in cr._jobs

    def test_cap_keeps_non_terminal_when_over(self):
        self._reset()
        now = time.time()
        for i in range(5):
            j = _make_job(status="running", updated=now + i, job_id=f"run-{i}")
            cr._jobs[j["job_id"]] = j
        for i in range(cr._MAX_JOBS):
            j = _make_job(updated=now + i, job_id=f"done-{i}")
            cr._jobs[j["job_id"]] = j
        with cr._jobs_lock:
            cr._prune_jobs()
        for i in range(5):
            assert f"run-{i}" in cr._jobs


class TestConvertDeleteJob:
    def _reset(self):
        cr._jobs.clear()

    def test_delete_terminal_returns_deleted(self):
        self._reset()
        job = _make_job(status="completed")
        cr._jobs[job["job_id"]] = job
        result = cr._delete_job(job["job_id"], "convert")
        assert result["status"] == "deleted"
        assert job["job_id"] not in cr._jobs

    def test_delete_running_returns_409(self):
        self._reset()
        job = _make_job(status="running")
        cr._jobs[job["job_id"]] = job
        with pytest.raises(HTTPException) as exc:
            cr._delete_job(job["job_id"], "convert")
        assert exc.value.status_code == 409

    def test_delete_not_found_returns_404(self):
        self._reset()
        with pytest.raises(HTTPException) as exc:
            cr._delete_job("nope", "convert")
        assert exc.value.status_code == 404

    def test_delete_wrong_kind_returns_404(self):
        self._reset()
        job = _make_job(kind="quantize", status="completed")
        cr._jobs[job["job_id"]] = job
        with pytest.raises(HTTPException) as exc:
            cr._delete_job(job["job_id"], "convert")
        assert exc.value.status_code == 404


class TestLayeredPruneJobs:
    def _reset(self):
        lq._layered_jobs.clear()

    def test_ttl_evicts_old_terminal(self):
        self._reset()
        old = _make_job(
            kind="layered-quantize",
            updated=time.time() - lq._LAYERED_JOB_TTL_SECONDS - 60,
            job_id="lq-old",
        )
        fresh = _make_job(
            kind="layered-quantize", updated=time.time(), job_id="lq-fresh"
        )
        lq._layered_jobs[old["job_id"]] = old
        lq._layered_jobs[fresh["job_id"]] = fresh
        with lq._layered_jobs_lock:
            lq._prune_layered_jobs()
        assert old["job_id"] not in lq._layered_jobs
        assert fresh["job_id"] in lq._layered_jobs

    def test_cap_drops_oldest_terminal(self):
        self._reset()
        now = time.time()
        for i in range(lq._MAX_LAYERED_JOBS + 5):
            j = _make_job(kind="layered-quantize", updated=now + i, job_id=f"lq-{i}")
            lq._layered_jobs[j["job_id"]] = j
        with lq._layered_jobs_lock:
            lq._prune_layered_jobs()
        assert len(lq._layered_jobs) <= lq._MAX_LAYERED_JOBS
        assert "lq-0" not in lq._layered_jobs
        assert "lq-9" in lq._layered_jobs


class TestLayeredDeleteJob:
    def _reset(self):
        lq._layered_jobs.clear()

    def test_delete_terminal_returns_deleted(self):
        self._reset()
        job = _make_job(kind="layered-quantize", status="completed")
        lq._layered_jobs[job["job_id"]] = job
        result = lq.delete_layered_quantize_job(job["job_id"], _is_admin=True)
        import asyncio

        resp = asyncio.run(result)
        assert resp["status"] == "deleted"
        assert job["job_id"] not in lq._layered_jobs

    def test_delete_running_returns_409(self):
        self._reset()
        job = _make_job(kind="layered-quantize", status="running")
        lq._layered_jobs[job["job_id"]] = job
        import asyncio

        with pytest.raises(HTTPException) as exc:
            asyncio.run(lq.delete_layered_quantize_job(job["job_id"], _is_admin=True))
        assert exc.value.status_code == 409

    def test_delete_not_found_returns_404(self):
        self._reset()
        import asyncio

        with pytest.raises(HTTPException) as exc:
            asyncio.run(lq.delete_layered_quantize_job("nope", _is_admin=True))
        assert exc.value.status_code == 404


if __name__ == "__main__":  # pragma: no cover — convenience only
    pytest.main([__file__, "-v"])
