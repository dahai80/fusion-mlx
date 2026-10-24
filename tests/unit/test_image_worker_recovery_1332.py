# SPDX-License-Identifier: Apache-2.0
# #1332: image worker self-recovery. A killed subprocess worker (hang deadline
# or load-hang fast-fail) must not poison the image subsystem for later
# requests, and an external-signal death must be logged with the signal and
# elapsed time so the sender can be diagnosed.

from __future__ import annotations

import asyncio
import logging
import signal
import time

import pytest

from fusion_mlx import engine_core
from fusion_mlx.engines import image_gen as image_gen_mod
from fusion_mlx.engines.image_gen import ImageGenEngine
from fusion_mlx.media import job_manager as jm
from fusion_mlx.media.job_manager import MediaJobManager, MediaJobResult

_SPEC = {
    "variant": "txt2img",
    "model_path": "fake-flux",
    "quantize": None,
    "config_label": None,
    "output_dir": "/tmp/fake",
    "output_format": "PNG",
    "n_images": 1,
    "gen_params": {"seed": 0},
    "lease_bytes": 1024,
}


@pytest.fixture(autouse=True)
def _clean_poison():
    engine_core.reset_image_executor_poison()
    yield
    engine_core.reset_image_executor_poison()


class _NeverStdout:
    async def readline(self):
        # Never resolves: a worker that emits no event at all.
        await asyncio.Event().wait()


class _EofStdout:
    async def readline(self):
        return b""


class _Stderr:
    def __init__(self, data: bytes = b""):
        self._data = data

    async def read(self):
        return self._data


class _FakeProc:
    def __init__(
        self,
        *,
        exit_code: int | None = None,
        stdout=None,
        stderr: bytes = b"",
        on_signal_returncode: int | None = None,
    ):
        self.pid = 4242
        self.returncode = exit_code
        self.stdout = stdout if stdout is not None else _NeverStdout()
        self.stderr = _Stderr(stderr)
        self.signals: list[int] = []
        self._on_signal_returncode = on_signal_returncode

    def send_signal(self, sig: int) -> None:
        self.signals.append(sig)
        if self.returncode is None and self._on_signal_returncode is not None:
            self.returncode = self._on_signal_returncode

    async def wait(self):
        return self.returncode


class _FakeManager:
    def __init__(self, items):
        self._items = list(items)
        self.calls = 0

    async def run_image_job(self, **kwargs):
        self.calls += 1
        item = self._items.pop(0)
        if isinstance(item, BaseException):
            raise item
        return item


def _patch_fake_subprocess(monkeypatch, proc: _FakeProc) -> None:
    async def _fake_exec(*args, **kwargs):
        return proc

    monkeypatch.setattr(asyncio, "create_subprocess_exec", _fake_exec)


def _make_manager(monkeypatch, tmp_path) -> MediaJobManager:
    mgr = MediaJobManager()
    monkeypatch.setattr(
        mgr, "_build_spec", lambda *a, **k: (str(tmp_path), dict(_SPEC))
    )
    return mgr


async def _run_image_job(mgr: MediaJobManager, timeout: float = 1.0):
    return await mgr.run_image_job(
        variant="txt2img",
        model_path="fake-flux",
        quantize=None,
        config_label=None,
        output_format="PNG",
        n_images=1,
        gen_params={"seed": 0},
        timeout=timeout,
    )


def _patch_generate_prologue(monkeypatch) -> None:
    # Keep generate() hermetic: no admission gate, no pacing throttle, no
    # LLM-idle defer. The manager fake supplies the job itself.
    async def _admit(self, **kwargs):
        return None

    async def _pace_ok(cls, timeout_s=0.0):
        return True

    monkeypatch.setattr(ImageGenEngine, "_admit_generation", _admit)
    monkeypatch.setattr("fusion_mlx.pacing.pace_acquire_async", _pace_ok)
    monkeypatch.setattr(image_gen_mod, "_MEDIA_LLM_IDLE_WAIT_S", 0.0)
    monkeypatch.setenv("FUSION_IMAGE_SUBPROCESS", "1")


@pytest.mark.asyncio
async def test_killed_worker_does_not_poison_next_job(monkeypatch, tmp_path):
    # (a) #1332: a hang-deadline kill must clear the poison state so the NEXT
    # request runs — and in subprocess mode must NOT reload the model
    # in-process (pre-fix the reload burned the request and kept the
    # subsystem dead until restart).
    job_dir = tmp_path / "job0"
    job_dir.mkdir()
    out = job_dir / "0.png"
    out.write_bytes(b"PNGDATA")
    mgr = _FakeManager(
        [
            TimeoutError("worker exceeded 600s timeout (killed)"),
            MediaJobResult(outputs=[str(out)]),
        ]
    )
    monkeypatch.setattr(ImageGenEngine, "_media_mgr", mgr, raising=False)
    _patch_generate_prologue(monkeypatch)

    start_calls: list[bool] = []

    async def _fake_start(self):
        start_calls.append(True)

    monkeypatch.setattr(ImageGenEngine, "start", _fake_start)

    engine = ImageGenEngine("fake-flux", variant="txt2img")
    # Prior hung job left the subsystem poisoned.
    engine_core.poison_executor("image")
    assert engine_core.is_image_executor_poisoned()

    with pytest.raises(RuntimeError) as excinfo:
        await engine.generate(prompt="p", width=64, height=64, steps=1, guidance=1.0)

    assert "exceeded the hang deadline" in str(excinfo.value)
    assert "restart fusion-mlx" not in str(excinfo.value)
    assert engine_core.is_image_executor_poisoned() is False
    assert start_calls == []

    images = await engine.generate(
        prompt="p", width=64, height=64, steps=1, guidance=1.0
    )
    assert images == [b"PNGDATA"]
    assert mgr.calls == 2
    assert engine_core.is_image_executor_poisoned() is False


@pytest.mark.asyncio
async def test_load_hang_fast_fails_with_load_error(monkeypatch, tmp_path):
    # (b) #1332: no worker event within the short load window = load hang.
    # Must fail fast with the load-specific error, kill the worker, and leave
    # the executor unpoisoned.
    monkeypatch.setenv("FUSION_IMAGE_LOAD_TIMEOUT", "0.2")
    proc = _FakeProc(on_signal_returncode=-signal.SIGTERM)
    _patch_fake_subprocess(monkeypatch, proc)
    mgr = _make_manager(monkeypatch, tmp_path)

    caught: BaseException | None = None
    started = time.monotonic()
    try:
        await _run_image_job(mgr, timeout=5.0)
    except TimeoutError:
        pytest.fail("no load-hang fast-fail window: burned the overall deadline")
    except Exception as exc:
        caught = exc
    else:
        pytest.fail("expected a load-hang failure")
    elapsed = time.monotonic() - started

    assert type(caught).__name__ == "MediaJobLoadHangError"
    assert isinstance(caught, RuntimeError)
    msg = str(caught)
    assert "model load produced no events" in msg
    assert "load hang, not a generation hang" in msg
    assert "not poisoned" in msg
    assert elapsed < 3.0
    assert proc.signals and proc.signals[0] == signal.SIGTERM
    assert engine_core.is_image_executor_poisoned() is False


@pytest.mark.asyncio
async def test_load_window_disabled_by_env(monkeypatch, tmp_path):
    # FUSION_IMAGE_LOAD_TIMEOUT <= 0 restores the plain overall deadline.
    monkeypatch.setenv("FUSION_IMAGE_LOAD_TIMEOUT", "0")
    proc = _FakeProc(on_signal_returncode=-signal.SIGTERM)
    _patch_fake_subprocess(monkeypatch, proc)
    mgr = _make_manager(monkeypatch, tmp_path)

    with pytest.raises(TimeoutError):
        await _run_image_job(mgr, timeout=0.3)


def test_get_image_load_timeout_env(monkeypatch):
    monkeypatch.delenv("FUSION_IMAGE_LOAD_TIMEOUT", raising=False)
    assert jm.get_image_load_timeout() == 120.0
    monkeypatch.setenv("FUSION_IMAGE_LOAD_TIMEOUT", "45")
    assert jm.get_image_load_timeout() == 45.0
    monkeypatch.setenv("FUSION_IMAGE_LOAD_TIMEOUT", "not-a-number")
    assert jm.get_image_load_timeout() == 120.0


@pytest.mark.asyncio
async def test_external_signal_death_logs_signal_and_elapsed(
    monkeypatch, tmp_path, caplog
):
    # (c) #1332: a negative return code that none of our kill paths caused
    # must be logged as external with the signal number and elapsed time.
    proc = _FakeProc(
        exit_code=-signal.SIGTERM,
        stdout=_EofStdout(),
        stderr=b"terminated by peer",
    )
    _patch_fake_subprocess(monkeypatch, proc)
    mgr = _make_manager(monkeypatch, tmp_path)

    with caplog.at_level(logging.ERROR, logger="fusion_mlx.media.job_manager"):
        with pytest.raises(RuntimeError) as excinfo:
            await _run_image_job(mgr, timeout=5.0)

    msg = str(excinfo.value)
    assert "-15" in msg
    assert "SIGTERM" in msg
    assert "external signal" in msg
    assert "next request will spawn a fresh worker" in msg

    external = [r for r in caplog.records if "EXTERNAL signal" in r.getMessage()]
    assert external, [r.getMessage() for r in caplog.records]
    logged = external[0].getMessage()
    assert "pid=4242" in logged
    assert "signal 15" in logged
    assert "SIGTERM" in logged
    assert "after " in logged and "s —" in logged
