# SPDX-License-Identifier: Apache-2.0
# Tests for /v1/convert + /v1/quantize async job API. _run_convert is mocked so
# no real model download / mlx_lm import happens; we assert the job lifecycle,
# request validation, kwargs hand-off to the convert CLI pipeline, and routing.

from __future__ import annotations

import time

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from fusion_mlx.api import convert_routes


@pytest.fixture(scope="session")
def client():
    app = FastAPI()
    app.include_router(convert_routes.router)
    with TestClient(app) as c:
        yield c


@pytest.fixture(autouse=True)
def _clear_jobs():
    with convert_routes._jobs_lock:
        convert_routes._jobs.clear()
    yield


def _wait(client, kind, job_id, timeout=5.0):
    deadline = time.perf_counter() + timeout
    while time.perf_counter() < deadline:
        r = client.get(f"/v1/{kind}/jobs/{job_id}")
        assert r.status_code == 200, r.text
        job = r.json()
        if job["status"] in ("done", "failed"):
            return job
        time.sleep(0.02)
    raise AssertionError(f"job {job_id} did not finish within {timeout}s")


def _fake_run_ok(captured=None):
    def _impl(model, **kwargs):
        if captured is not None:
            captured.update(kwargs)
        return kwargs["mlx_path"]

    return _impl


def test_convert_creates_job_and_completes(client, monkeypatch):
    monkeypatch.setattr("fusion_mlx.cli_convert._run_convert", _fake_run_ok())
    r = client.post("/v1/convert", json={"model": "test/repo"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["status"] == "queued"
    assert "job_id" in body

    job = _wait(client, "convert", body["job_id"])
    assert job["status"] == "done"
    assert job["progress"] == 1.0
    assert job["output_path"]
    assert job["error"] is None
    assert job["kind"] == "convert"
    assert job["model"] == "test/repo"


def test_convert_plain_does_not_quantize(client, monkeypatch):
    captured = {}
    monkeypatch.setattr("fusion_mlx.cli_convert._run_convert", _fake_run_ok(captured))
    r = client.post("/v1/convert", json={"model": "test/repo"})
    _wait(client, "convert", r.json()["job_id"])
    assert captured["quantize"] is False
    assert captured["q_bits"] is None


def test_convert_with_bits_quantizes(client, monkeypatch):
    captured = {}
    monkeypatch.setattr("fusion_mlx.cli_convert._run_convert", _fake_run_ok(captured))
    r = client.post(
        "/v1/convert",
        json={"model": "test/repo", "quant_bits": 4, "quant_group_size": 32},
    )
    _wait(client, "convert", r.json()["job_id"])
    assert captured["quantize"] is True
    assert captured["q_bits"] == 4
    assert captured["q_group_size"] == 32


def test_convert_float_mode_quantizes_without_bits(client, monkeypatch):
    captured = {}
    monkeypatch.setattr("fusion_mlx.cli_convert._run_convert", _fake_run_ok(captured))
    r = client.post("/v1/convert", json={"model": "test/repo", "quant_mode": "mxfp4"})
    _wait(client, "convert", r.json()["job_id"])
    assert captured["quantize"] is True
    assert captured["q_bits"] is None


def test_quantize_requires_quant_spec(client):
    r = client.post("/v1/quantize", json={"model": "test/repo"})
    assert r.status_code == 400
    assert "quant_bits" in r.json()["detail"]


def test_quantize_with_bits_accepted(client, monkeypatch):
    captured = {}
    monkeypatch.setattr("fusion_mlx.cli_convert._run_convert", _fake_run_ok(captured))
    r = client.post("/v1/quantize", json={"model": "test/repo", "quant_bits": 4})
    assert r.status_code == 200, r.text
    job = _wait(client, "quantize", r.json()["job_id"])
    assert job["status"] == "done"
    assert job["kind"] == "quantize"
    assert captured["quantize"] is True


def test_quantize_float_mode_accepted_without_bits(client, monkeypatch):
    monkeypatch.setattr("fusion_mlx.cli_convert._run_convert", _fake_run_ok())
    r = client.post("/v1/quantize", json={"model": "test/repo", "quant_mode": "nvfp4"})
    assert r.status_code == 200, r.text
    job = _wait(client, "quantize", r.json()["job_id"])
    assert job["status"] == "done"


def test_job_failure_recorded(client, monkeypatch):
    def boom(model, **kwargs):
        raise RuntimeError("disk full")

    monkeypatch.setattr("fusion_mlx.cli_convert._run_convert", boom)
    r = client.post("/v1/convert", json={"model": "test/repo"})
    job = _wait(client, "convert", r.json()["job_id"])
    assert job["status"] == "failed"
    assert "disk full" in job["error"]
    assert job["progress"] == 1.0
    assert job["output_path"] is None


def test_get_unknown_job_404(client):
    r = client.get("/v1/convert/jobs/does-not-exist")
    assert r.status_code == 404


def test_cross_kind_job_404(client, monkeypatch):
    monkeypatch.setattr("fusion_mlx.cli_convert._run_convert", _fake_run_ok())
    r = client.post("/v1/convert", json={"model": "test/repo"})
    job_id = r.json()["job_id"]
    _wait(client, "convert", job_id)
    # A convert job must NOT be reachable under the /v1/quantize/jobs prefix.
    r2 = client.get(f"/v1/quantize/jobs/{job_id}")
    assert r2.status_code == 404


def test_list_convert_jobs(client, monkeypatch):
    monkeypatch.setattr("fusion_mlx.cli_convert._run_convert", _fake_run_ok())
    ids = []
    for i in range(2):
        r = client.post("/v1/convert", json={"model": f"test/repo{i}"})
        ids.append(r.json()["job_id"])
    for jid in ids:
        _wait(client, "convert", jid)
    r = client.get("/v1/convert/jobs")
    assert r.status_code == 200
    listed = r.json()
    listed_ids = {j["job_id"] for j in listed}
    assert set(ids).issubset(listed_ids)
    assert all(j["kind"] == "convert" for j in listed)


def test_invalid_quant_bits_rejected(client):
    r = client.post("/v1/convert", json={"model": "test/repo", "quant_bits": 5})
    assert r.status_code == 422


def test_invalid_quant_mode_rejected(client):
    r = client.post("/v1/convert", json={"model": "test/repo", "quant_mode": "bogus"})
    assert r.status_code == 422
