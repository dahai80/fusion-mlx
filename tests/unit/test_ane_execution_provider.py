# SPDX-License-Identifier: Apache-2.0
"""Unit tests for ANEExecutionProvider trait (P0底座)."""

from fusion_mlx.engine.base import ANEExecutionProvider, BaseEngine


class _PlainEngine(BaseEngine):
    """Minimal concrete engine that does NOT inherit ANEExecutionProvider."""

    @property
    def model_name(self):
        return "plain"

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
