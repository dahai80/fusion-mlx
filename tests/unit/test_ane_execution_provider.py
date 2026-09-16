# SPDX-License-Identifier: Apache-2.0
"""Unit tests for ANEExecutionProvider trait (P0底座) + BaseEngine abort surface (#903)."""

import pytest

from fusion_mlx.engines.base import ANEExecutionProvider, BaseEngine


class _PlainEngine(BaseEngine):
    """Minimal concrete engine that does NOT inherit ANEExecutionProvider."""

    @property
    def model_name(self):
        return "plain"

    @property
    def model_type(self):
        return None

    @property
    def tokenizer(self):
        return None

    async def start(self):
        pass

    async def stop(self):
        pass

    async def generate(self, prompt, **kwargs):
        pass

    async def stream_generate(self, prompt, **kwargs):
        pass

    async def chat(self, messages, **kwargs):
        pass

    async def stream_chat(self, messages, **kwargs):
        pass


class _AneEngine(ANEExecutionProvider):
    """Standalone mixin instance (no BaseEngine needed for trait tests)."""


class _AneOverrideEngine(ANEExecutionProvider):
    def ane_supported(self):
        return True


def test_ane_default_disabled():
    p = _AneEngine()
    assert p.ane_supported() is False
    assert p.ane_warmup() is False
    assert p.ane_resident_memory() == 0


def test_ane_pinned_and_not_pageable():
    p = _AneEngine()
    assert p.ane_pinned is True
    assert p.ane_pageable is False


def test_ane_fallback_flips_enabled_off():
    p = _AneEngine()
    p._ane_enabled = True
    p._ane_resident_bytes = 1024
    p.ane_fallback_to_metal()
    assert p._ane_enabled is False


def test_ane_fallback_when_already_off_no_warn(caplog):
    p = _AneEngine()
    p._ane_enabled = False
    with caplog.at_level("WARNING"):
        p.ane_fallback_to_metal()
    assert not any("ANE fallback" in r.message for r in caplog.records)


def test_plain_engine_has_no_ane_attrs():
    """Engines not inheriting the mixin must not expose ane_* surface."""
    e = _PlainEngine()
    assert not hasattr(e, "ane_supported")
    assert not hasattr(e, "ane_pinned")


def test_ane_override_resident_memory():
    e = _AneOverrideEngine()
    e._ane_resident_bytes = 2 * 1024**3
    assert e.ane_resident_memory() == 2 * 1024**3
    assert e.ane_supported() is True


# --- #903: BaseEngine.abort_request unified surface ---


class _FakeEngineCoreEngine:
    """Innermost engine (AsyncEngineCore.engine shape)."""

    def __init__(self):
        self.aborted = []

    async def abort_request(self, request_id):
        self.aborted.append(request_id)
        return True


class _FakeEngineCore:
    def __init__(self):
        self.engine = _FakeEngineCoreEngine()


class _EngineWithCore(_PlainEngine):
    """Engine shape mirroring BatchedEngine/VLMBatchedEngine (_engine attr)."""

    def __init__(self):
        self._engine = _FakeEngineCore()


class _EngineWithoutCore(_PlainEngine):
    def __init__(self):
        self._engine = None


@pytest.mark.asyncio
async def test_abort_request_delegates_to_enginecore():
    e = _EngineWithCore()
    ok = await e.abort_request("req-1")
    assert ok is True
    assert e._engine.engine.aborted == ["req-1"]


@pytest.mark.asyncio
async def test_abort_request_no_core_returns_false():
    e = _EngineWithoutCore()
    assert await e.abort_request("req-1") is False


@pytest.mark.asyncio
async def test_vlm_shape_engine_has_abort_request():
    """The exact #903 repro shape: engine with only _engine + no local
    abort_request previously raised AttributeError from _disconnect_guard."""
    e = _EngineWithCore()
    assert hasattr(e, "abort_request")
    await e.abort_request("req-42")
    assert e._engine.engine.aborted == ["req-42"]
