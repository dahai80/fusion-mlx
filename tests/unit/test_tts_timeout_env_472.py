# SPDX-License-Identifier: Apache-2.0
"""Tests for configurable TTS timeout (#472).

Verifies TTSEngine.synthesize reads FUSION_TTS_TIMEOUT (seconds) and
applies it to asyncio.wait_for, and that an invalid value falls back to
180s. No real model is loaded and no GPU is touched — the model and
audio executor are stubbed.
"""

import asyncio
from unittest.mock import MagicMock, patch

import pytest


def _make_engine():
    from fusion_mlx.engines.tts import TTSEngine

    eng = TTSEngine(model_name="qwen3-tts")
    eng._model = MagicMock()
    return eng


async def _run_synthesize_capturing_timeout(eng):
    """Run synthesize with run_in_executor patched to a pending future so
    asyncio.wait_for's timeout fires. Returns the timeout value passed to
    wait_for (raised via asyncio.TimeoutError)."""
    captured = {}

    async def _pending_coro(*_a, **_kw):
        await asyncio.sleep(3600)

    loop = asyncio.get_running_loop()

    def _run_in_executor(_executor, _fn):
        return asyncio.ensure_future(_pending_coro())

    real_wait_for = asyncio.wait_for

    async def _spy_wait_for(coro, timeout):
        captured["timeout"] = timeout
        return await real_wait_for(coro, timeout)

    with (
        patch("fusion_mlx.engines.tts.get_executor", lambda _name: None),
        patch.object(loop, "run_in_executor", side_effect=_run_in_executor),
        patch("fusion_mlx.engines.tts.asyncio.wait_for", _spy_wait_for),
    ):
        with pytest.raises((asyncio.TimeoutError, TimeoutError)):
            await eng.synthesize("hello")

    return captured["timeout"]


@pytest.mark.asyncio
async def test_timeout_env_applied(monkeypatch):
    monkeypatch.setenv("FUSION_TTS_TIMEOUT", "0.05")
    eng = _make_engine()
    timeout = await _run_synthesize_capturing_timeout(eng)
    assert timeout == pytest.approx(
        0.05
    ), f"expected timeout=0.05 from env, got {timeout}"


@pytest.mark.asyncio
async def test_timeout_env_invalid_falls_back(monkeypatch):
    monkeypatch.setenv("FUSION_TTS_TIMEOUT", "not-a-number")
    eng = _make_engine()
    timeout = await _run_synthesize_capturing_timeout(eng)
    assert timeout == 180.0, f"expected fallback timeout=180.0, got {timeout}"


@pytest.mark.asyncio
async def test_timeout_env_default_when_unset(monkeypatch):
    monkeypatch.delenv("FUSION_TTS_TIMEOUT", raising=False)
    eng = _make_engine()
    timeout = await _run_synthesize_capturing_timeout(eng)
    assert timeout == 180.0, f"expected default timeout=180.0, got {timeout}"
