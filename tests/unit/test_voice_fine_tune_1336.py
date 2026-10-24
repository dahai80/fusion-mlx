# Tests for the voice fine-tune backend (#1336): dataset loading, job
# service, admin routes, and `voice_ref` resolution on /v1/audio/speech.
# No model is loaded — training itself is verified by a real-weights E2E run.

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from fusion_mlx.admin import fine_tune_route
from fusion_mlx.admin.auth import require_admin
from fusion_mlx.admin.routes import router as admin_router
from fusion_mlx.api.audio_models import AudioSpeechRequest
from fusion_mlx.api.audio_routes import router as audio_router
from fusion_mlx.middleware.auth import check_rate_limit, verify_api_key
from fusion_mlx.training import voice_service as vs
from fusion_mlx.training.service import JobStatus
from fusion_mlx.training.voice import (
    SUPPORTED_BACKENDS,
    VoiceDatasetError,
    VoiceTrainConfig,
    _rms,
    _spectrum_stats,
    load_dataset,
)


def _write_wav(path: Path, seconds: float = 1.5, freq: float = 220.0, sr: int = 24000):
    import soundfile as sf

    t = np.arange(int(seconds * sr), dtype=np.float32) / sr
    sf.write(str(path), 0.2 * np.sin(2 * np.pi * freq * t), sr)


@pytest.fixture()
def voice_dirs(tmp_path, monkeypatch):
    base = tmp_path / "voices"
    base.mkdir()
    monkeypatch.setattr(vs, "VOICE_BASE_DIR", base)
    monkeypatch.setattr(vs, "VOICE_JOBS_FILE", tmp_path / "voice_jobs.json")
    from fusion_mlx.training import voice as voice_mod

    monkeypatch.setattr(voice_mod, "VOICE_BASE_DIR", base)
    return base


@pytest.fixture()
def dataset_dir(tmp_path):
    root = tmp_path / "dataset"
    root.mkdir()
    for i in range(4):
        _write_wav(root / f"sample{i}.wav", freq=180.0 + i * 40)
        (root / f"sample{i}.txt").write_text(f"sentence number {i}", encoding="utf-8")
    return root


def test_kokoro_backend_declared():
    assert "kokoro" in SUPPORTED_BACKENDS
    assert SUPPORTED_BACKENDS["kokoro"]["artifact"] == "voice.safetensors"


def test_spectrum_stats_level_invariant():
    rng = np.random.default_rng(0)
    mag = rng.random((20, 64)) + 0.1
    mean_a, spread_a = _spectrum_stats(mag)
    mean_b, spread_b = _spectrum_stats(mag * 4.0)
    np.testing.assert_allclose(mean_a, mean_b, atol=1e-5)
    np.testing.assert_allclose(spread_a, spread_b, atol=1e-5)
    assert abs(float(mean_a.mean())) < 1e-5


def test_rms_rejects_silence_division():
    assert _rms(np.zeros(64, dtype=np.float32)) > 0.0


def test_load_dataset_wav_txt(dataset_dir):
    samples = load_dataset(dataset_dir)
    assert len(samples) == 4
    assert all(sample.text.startswith("sentence") for sample in samples)
    assert all(1.0 <= sample.duration <= 1.6 for sample in samples)


def test_load_dataset_max_samples(dataset_dir):
    assert len(load_dataset(dataset_dir, max_samples=2)) == 2


def test_load_dataset_metadata_csv(dataset_dir):
    (dataset_dir / "metadata.csv").write_text(
        "sample0.wav|first line\nsample1.wav|second line\n", encoding="utf-8"
    )
    samples = load_dataset(dataset_dir)
    assert [sample.text for sample in samples] == ["first line", "second line"]


def test_load_dataset_metadata_jsonl(dataset_dir):
    lines = [
        json.dumps({"audio": "sample0.wav", "text": "jsonl one"}),
        json.dumps({"file": "sample1.wav", "transcript": "jsonl two"}),
    ]
    (dataset_dir / "metadata.jsonl").write_text("\n".join(lines), encoding="utf-8")
    samples = load_dataset(dataset_dir)
    assert [sample.text for sample in samples] == ["jsonl one", "jsonl two"]


def test_load_dataset_missing_dir(tmp_path):
    with pytest.raises(VoiceDatasetError, match="not found"):
        load_dataset(tmp_path / "nope")


def test_load_dataset_empty_dir(tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(VoiceDatasetError, match="no training pairs"):
        load_dataset(empty)


def test_load_dataset_rejects_short_audio(tmp_path):
    root = tmp_path / "short"
    root.mkdir()
    _write_wav(root / "a.wav", seconds=0.2)
    (root / "a.txt").write_text("too short", encoding="utf-8")
    with pytest.raises(VoiceDatasetError, match="shorter than"):
        load_dataset(root)


def test_load_dataset_rejects_thin_total(tmp_path):
    root = tmp_path / "thin"
    root.mkdir()
    _write_wav(root / "a.wav", seconds=1.2, freq=200)
    (root / "a.txt").write_text("one", encoding="utf-8")
    with pytest.raises(VoiceDatasetError, match="at least 3s"):
        load_dataset(root)


def test_resolve_voice_ref(voice_dirs):
    package = voice_dirs / "myvoice"
    package.mkdir()
    pack = package / "voice.safetensors"
    pack.write_bytes(b"stub")
    assert vs.resolve_voice_ref("myvoice") == str(pack)
    assert vs.resolve_voice_ref("missing") is None
    assert vs.resolve_voice_ref("../etc") is None
    assert vs.resolve_voice_ref("a/b") is None
    assert vs.resolve_voice_ref(".hidden") is None
    assert vs.resolve_voice_ref("") is None


def test_list_trained_voices(voice_dirs):
    ready = voice_dirs / "ready-voice"
    ready.mkdir()
    (ready / "voice.safetensors").write_bytes(b"stub")
    (ready / "manifest.json").write_text(
        json.dumps({"voice_id": "ready-voice", "family": "kokoro"}), encoding="utf-8"
    )
    broken = voice_dirs / "broken-voice"
    broken.mkdir()
    (broken / "manifest.json").write_text(
        json.dumps({"voice_id": "broken-voice"}), encoding="utf-8"
    )

    voices = {item["voice_id"]: item for item in vs.list_trained_voices()}
    assert voices["ready-voice"]["ready"] is True
    assert voices["broken-voice"]["ready"] is False
    assert voices["ready-voice"]["path"].endswith("voice.safetensors")


def test_service_create_and_persist(voice_dirs, dataset_dir, tmp_path):
    svc = vs.VoiceService()
    job = svc.create_job("mlx-community/Kokoro-82M-bf16", str(dataset_dir), "voice-a")
    assert job.job_id in svc._jobs
    assert job.config.steps == VoiceTrainConfig().steps
    assert vs.VOICE_JOBS_FILE.is_file()

    reloaded = vs.VoiceService()
    assert reloaded.get_job(job.job_id) is not None
    assert reloaded.get_job(job.job_id).status == JobStatus.CANCELLED


def test_service_rejects_bad_voice_id(voice_dirs, dataset_dir):
    svc = vs.VoiceService()
    with pytest.raises(ValueError, match="alphanumeric"):
        svc.create_job("kokoro", str(dataset_dir), "bad id!")


def test_service_rejects_duplicate_voice_id(voice_dirs, dataset_dir):
    svc = vs.VoiceService()
    existing = voice_dirs / "dupe"
    existing.mkdir()
    (existing / "voice.safetensors").write_bytes(b"stub")
    with pytest.raises(ValueError, match="already exists"):
        svc.create_job("kokoro", str(dataset_dir), "dupe")


def test_service_cancel_queued(voice_dirs, dataset_dir):
    svc = vs.VoiceService()
    job = svc.create_job("kokoro", str(dataset_dir), "voice-b")
    assert svc.cancel_job(job.job_id) is True
    assert job.status == JobStatus.CANCELLED
    assert job.terminal is True
    assert svc.cancel_job(job.job_id) is False


def test_service_delete_running_refused(voice_dirs, dataset_dir):
    svc = vs.VoiceService()
    job = svc.create_job("kokoro", str(dataset_dir), "voice-c")
    job.status = JobStatus.RUNNING
    assert svc.delete_job(job.job_id) is False
    job.status = JobStatus.FAILED
    job.terminal = True
    assert svc.delete_job(job.job_id) is True


def _build_admin_app(dataset_dir: Path, voice_dirs: Path):
    app = FastAPI()
    svc = vs.VoiceService()
    svc.start_processing = lambda *a, **kw: None
    svc._process_queue = lambda *a, **kw: None
    fine_tune_route.set_voice_context(None, svc)
    app.include_router(admin_router)
    app.dependency_overrides[require_admin] = lambda: True
    return app, svc


def test_route_backends(voice_dirs, dataset_dir):
    app, _ = _build_admin_app(dataset_dir, voice_dirs)
    client = TestClient(app)
    resp = client.get("/admin/api/fine-tune/voice/backends")
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert "kokoro" in body["backends"]
    assert isinstance(body["voices"], list)


def test_route_create_job(voice_dirs, dataset_dir):
    app, svc = _build_admin_app(dataset_dir, voice_dirs)
    client = TestClient(app)
    resp = client.post(
        "/admin/api/fine-tune/voice/jobs",
        json={
            "model_id": "mlx-community/Kokoro-82M-bf16",
            "dataset_dir": str(dataset_dir),
            "voice_id": "route-voice",
            "config": {"steps": 5},
        },
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["voice_id"] == "route-voice"
    assert body["config"]["steps"] == 5
    assert body["dataset_samples"] == 4
    assert body["status"] == "queued"

    listed = client.get("/admin/api/fine-tune/voice/jobs").json()
    assert any(job["job_id"] == body["job_id"] for job in listed)


def test_route_create_missing_model_id(voice_dirs, dataset_dir):
    app, _ = _build_admin_app(dataset_dir, voice_dirs)
    resp = TestClient(app).post(
        "/admin/api/fine-tune/voice/jobs", json={"dataset_dir": str(dataset_dir)}
    )
    assert resp.status_code == 400
    assert "model_id" in resp.json()["detail"]


def test_route_rejects_non_kokoro(voice_dirs, dataset_dir):
    app, _ = _build_admin_app(dataset_dir, voice_dirs)
    resp = TestClient(app).post(
        "/admin/api/fine-tune/voice/jobs",
        json={"model_id": "mlx-community/Llama-3-8B", "dataset_dir": str(dataset_dir)},
    )
    assert resp.status_code == 400
    assert "kokoro" in resp.json()["detail"]


def test_route_rejects_bad_dataset(voice_dirs, tmp_path):
    app, _ = _build_admin_app(tmp_path, voice_dirs)
    resp = TestClient(app).post(
        "/admin/api/fine-tune/voice/jobs",
        json={"model_id": "kokoro", "dataset_dir": str(tmp_path / "nope")},
    )
    assert resp.status_code == 400
    assert "not found" in resp.json()["detail"]


def test_route_get_unknown_job(voice_dirs, dataset_dir):
    app, _ = _build_admin_app(dataset_dir, voice_dirs)
    resp = TestClient(app).get("/admin/api/fine-tune/voice/jobs/deadbeef")
    assert resp.status_code == 404


def test_route_cancel_and_delete(voice_dirs, dataset_dir):
    app, _ = _build_admin_app(dataset_dir, voice_dirs)
    client = TestClient(app)
    created = client.post(
        "/admin/api/fine-tune/voice/jobs",
        json={"model_id": "kokoro", "dataset_dir": str(dataset_dir)},
    ).json()
    job_id = created["job_id"]

    cancelled = client.post(f"/admin/api/fine-tune/voice/jobs/{job_id}/cancel")
    assert cancelled.status_code == 200
    assert cancelled.json()["status"] == "cancelled"

    deleted = client.delete(f"/admin/api/fine-tune/voice/jobs/{job_id}")
    assert deleted.status_code == 200
    assert client.get(f"/admin/api/fine-tune/voice/jobs/{job_id}").status_code == 404


def test_speech_request_accepts_voice_ref():
    request = AudioSpeechRequest(model="kokoro", input="hi", voice_ref="my-voice")
    assert request.voice_ref == "my-voice"
    assert request.voice is None


def _build_audio_app():
    app = FastAPI()
    app.include_router(audio_router)
    app.dependency_overrides[verify_api_key] = lambda: True
    app.dependency_overrides[check_rate_limit] = lambda: True
    return app


def test_speech_unknown_voice_ref_is_404(voice_dirs):
    client = TestClient(_build_audio_app())
    resp = client.post(
        "/v1/audio/speech",
        json={"model": "kokoro", "input": "hello", "voice_ref": "ghost-voice"},
    )
    assert resp.status_code == 404
    detail = resp.json()["detail"]
    assert "ghost-voice" in detail
    assert "/admin/api/fine-tune/voice/jobs" in detail


def test_voices_lists_trained_pack_quality(voice_dirs):
    package = voice_dirs / "listed-voice"
    package.mkdir()
    (package / "voice.safetensors").write_bytes(b"stub")
    (package / "manifest.json").write_text(
        json.dumps(
            {
                "voice_id": "listed-voice",
                "family": "kokoro",
                "quality": "improved",
                "base_voice": "af_heart",
            }
        ),
        encoding="utf-8",
    )
    resp = TestClient(_build_audio_app()).get("/v1/audio/voices")
    assert resp.status_code == 200, resp.text
    trained = resp.json()["trained_voices"]
    assert [item["voice_ref"] for item in trained] == ["listed-voice"]
    assert trained[0]["quality"] == "improved"
