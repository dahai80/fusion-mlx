# SPDX-License-Identifier: Apache-2.0
# Tests for #1265: summary metrics measured in the correct direction —
# ttft = time TO first token, prefill rates over actually-prefilled tokens.
import logging
from types import SimpleNamespace

import fusion_mlx.scheduler.sched_response as sr
from fusion_mlx.scheduler.sched_response import _log_completion_summary


def _run(caplog, monkeypatch, req, out, now):
    # Patch sr's time REFERENCE, not sr.time.monotonic: sr.time IS the
    # stdlib time module — mutating it freezes time.monotonic() for the
    # whole pytest process, and every later test's asyncio.sleep() timer
    # never fires (the macOS rest-shard hang: tests that run
    # alphabetically after this file wedged for 600 s each under
    # pytest-timeout, 3 consecutive CI runs).
    monkeypatch.setattr(sr, "time", SimpleNamespace(monotonic=lambda: now))
    with caplog.at_level(logging.INFO, logger="fusion_mlx.scheduler.sched_response"):
        _log_completion_summary(req, out)
    return caplog.records[-1].getMessage()


def _req(arrival=100.0, gen_start=105.0, first_token=189.0, n_out=4):
    return SimpleNamespace(
        arrival_time=arrival,
        generation_started_at=gen_start,
        first_token_at=first_token,
        num_output_tokens=n_out,
    )


def _out(prompt=68808, cached=45568, completion=4):
    return SimpleNamespace(
        prompt_tokens=prompt,
        completion_tokens=completion,
        cached_tokens=cached,
        finish_reason="stop",
    )


def test_ttft_is_time_to_first_token(caplog, monkeypatch):
    line = _run(caplog, monkeypatch, _req(), _out(), now=189.28)
    assert "ttft=84.00s" in line
    assert "prompt_eval=3.6ms/token" in line
    assert "completion_eval=93.3ms/token" in line
    assert "prompt_tps=276.7" in line
    assert "completion_tps=14.3" in line
    assert "total=89.3s" in line
    assert "finish=stop" in line


def test_no_first_token_fallbacks(caplog, monkeypatch):
    line = _run(
        caplog,
        monkeypatch,
        _req(first_token=None),
        _out(prompt=100, cached=0, completion=0),
        now=190.0,
    )
    assert "ttft=0.00s" in line
    assert "prompt_eval=0.0ms/token" in line
    assert "total=90.0s" in line


def test_full_resume_no_div_zero(caplog, monkeypatch):
    line = _run(
        caplog,
        monkeypatch,
        _req(first_token=189.0),
        _out(prompt=100, cached=100, completion=2),
        now=189.1,
    )
    assert "prompt_eval" in line
    assert "nan" not in line.lower()


def test_gen_start_falls_back_to_arrival(caplog, monkeypatch):
    line = _run(
        caplog,
        monkeypatch,
        _req(gen_start=None),
        _out(prompt=100, cached=0, completion=2),
        now=190.0,
    )
    # gen_start=arrival=100, first_token=189 -> ttft=89s over 100 tokens
    assert "ttft=89.00s" in line
    assert "prompt_eval=890.0ms/token" in line
