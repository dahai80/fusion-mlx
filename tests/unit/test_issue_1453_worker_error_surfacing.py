# SPDX-License-Identifier: Apache-2.0
"""#1453: a crashed image worker's real error must not be swallowed.

Prod evidence (2026-10-30): 3 concurrent FLUX.1-schnell requests, worker
died at 0% with HTTP 500 body

    worker exited 1: mx.metal.set_cache_limit is deprecated ... 0%| | 0/4 [

The worker emits a structured {"type":"error"} event carrying the real
exception type + message, but MediaJobManager discarded it and rebuilt the
error from the raw stderr tail (deprecation warning + tqdm bar). stderr was
also only read AFTER process exit, so a chatty worker could fill the 64KB
pipe and block mid-write.
"""

import asyncio
import json
from types import SimpleNamespace

import pytest

from fusion_mlx.media import job_manager as jm
from fusion_mlx.media.job_manager import MediaJobManager


def _make_proc(stdout: asyncio.StreamReader, stderr: asyncio.StreamReader, rc: int):
    async def wait():
        proc.returncode = rc
        return rc

    proc = SimpleNamespace(
        pid=4242,
        stdout=stdout,
        stderr=stderr,
        returncode=None,
        send_signal=lambda *a: None,
        wait=wait,
    )
    return proc


@pytest.fixture()
def fake_spawn(monkeypatch, tmp_path):
    out_dir = tmp_path / "job"
    out_dir.mkdir()
    monkeypatch.setattr(jm.tempfile, "mkdtemp", lambda prefix="", **kw: str(out_dir))
    holder = {}

    async def _spawn(*args, **kwargs):
        return holder["proc"]

    monkeypatch.setattr(asyncio, "create_subprocess_exec", _spawn)
    return holder


async def test_worker_error_event_surfaced_with_stderr(fake_spawn):
    stdout, stderr = asyncio.StreamReader(), asyncio.StreamReader()
    stdout.feed_data(b'{"type":"error","message":"RuntimeError: boom"}\n')
    stdout.feed_eof()
    stderr.feed_data(b"mx.metal.set_cache_limit is deprecated\n")
    stderr.feed_eof()
    fake_spawn["proc"] = _make_proc(stdout, stderr, 1)

    mgr = MediaJobManager()
    with pytest.raises(RuntimeError) as ei:
        await mgr.run_image_job(
            variant="txt2img",
            model_path="/models/flux",
            quantize=None,
            config_label=None,
            output_format="PNG",
            n_images=1,
            gen_params={},
            timeout=5.0,
        )
    msg = str(ei.value)
    assert "worker exited 1" in msg
    assert "RuntimeError: boom" in msg
    assert "deprecated" in msg


async def test_large_stderr_does_not_block(fake_spawn):
    stdout, stderr = asyncio.StreamReader(), asyncio.StreamReader()
    stdout.feed_data(b'{"type":"error","message":"ValueError: nope"}\n')
    stdout.feed_eof()
    stderr.feed_data(b"x" * (200 * 1024))
    stderr.feed_eof()
    fake_spawn["proc"] = _make_proc(stdout, stderr, 1)

    mgr = MediaJobManager()
    with pytest.raises(RuntimeError) as ei:
        await mgr.run_image_job(
            variant="txt2img",
            model_path="/models/flux",
            quantize=None,
            config_label=None,
            output_format="PNG",
            n_images=1,
            gen_params={},
            timeout=5.0,
        )
    assert "ValueError: nope" in str(ei.value)
    # no lingering drain task
    pending = [
        t
        for t in asyncio.all_tasks()
        if t is not asyncio.current_task() and not t.done()
    ]
    assert pending == []


async def test_success_path_returns_outputs(fake_spawn):
    stdout, stderr = asyncio.StreamReader(), asyncio.StreamReader()
    stdout.feed_data(
        json.dumps({"type": "result", "outputs": ["/tmp/fusion_x/0.png"]}).encode()
        + b"\n"
    )
    stdout.feed_eof()
    stderr.feed_data(b"tqdm noise\r")
    stderr.feed_eof()
    fake_spawn["proc"] = _make_proc(stdout, stderr, 0)

    mgr = MediaJobManager()
    result = await mgr.run_image_job(
        variant="txt2img",
        model_path="/models/flux",
        quantize=None,
        config_label=None,
        output_format="PNG",
        n_images=1,
        gen_params={},
        timeout=5.0,
    )
    assert result.outputs == ["/tmp/fusion_x/0.png"]
    pending = [
        t
        for t in asyncio.all_tasks()
        if t is not asyncio.current_task() and not t.done()
    ]
    assert pending == []
