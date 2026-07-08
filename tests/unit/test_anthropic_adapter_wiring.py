"""Wiring tests for Anthropic route adapter_path threading (#410).

Mirrors test_lora_adapter_wiring.py for the Anthropic Messages API. The
``adapters`` field was added to ``api/anthropic_models.MessagesRequest``
and both ``_run_anthropic_messages`` and ``_stream_anthropic_generator``
thread it through to ``EnginePool.get_engine``/``release_engine`` as
``adapter_path``. These tests assert the field survives and reaches
``get_engine`` with the right keyword, short-circuiting the route via a
boom pool before any engine.chat work runs.
"""

from __future__ import annotations

import sys
import types

import pytest

import fusion_mlx.api.anthropic_routes as routes
from fusion_mlx.api.anthropic_models import (
    MessagesRequest as AnthropicMessagesRequest,
)


class _BoomPool:
    def __init__(self) -> None:
        self.captured: tuple[str, dict] | None = None
        self.released: list[tuple[str, dict]] = []

    async def get_engine(self, model_id: str, **kwargs):
        self.captured = (model_id, kwargs)
        raise RuntimeError("boom-stop")

    async def release_engine(self, model_id: str, **kwargs):
        self.released.append((model_id, kwargs))
        return None


def _stub_resolve_model_id(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = types.ModuleType("fusion_mlx.server")
    fake.resolve_model_id = lambda model_id: model_id  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "fusion_mlx.server", fake)


def _make_req(adapter: str | None = "/lora") -> AnthropicMessagesRequest:
    return AnthropicMessagesRequest(
        model="m",
        max_tokens=8,
        messages=[{"role": "user", "content": "hi"}],
        adapters=adapter,
    )


def test_messages_request_adapters_field_present() -> None:
    req = _make_req("/lora")
    assert hasattr(req, "adapters")
    assert req.adapters == "/lora"


def test_messages_request_adapters_defaults_none() -> None:
    req = AnthropicMessagesRequest(
        model="m", max_tokens=8, messages=[{"role": "user", "content": "hi"}]
    )
    assert req.adapters is None


async def test_run_anthropic_messages_threads_adapters_to_get_engine(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _stub_resolve_model_id(monkeypatch)
    pool = _BoomPool()
    monkeypatch.setattr(routes, "_pool", pool)

    req = _make_req("/lora")
    with pytest.raises(RuntimeError, match="boom-stop"):
        await routes._run_anthropic_messages(req)

    assert pool.captured is not None
    _model_id, kwargs = pool.captured
    assert kwargs.get("adapter_path") == "/lora"
    assert kwargs.get("_lease") is True


async def test_stream_anthropic_threads_adapters_to_get_engine(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _stub_resolve_model_id(monkeypatch)
    pool = _BoomPool()
    monkeypatch.setattr(routes, "_pool", pool)

    req = _make_req("/lora")
    gen = routes._stream_anthropic_generator(req)
    with pytest.raises(RuntimeError, match="boom-stop"):
        await anext(gen)

    assert pool.captured is not None
    _model_id, kwargs = pool.captured
    assert kwargs.get("adapter_path") == "/lora"
    assert kwargs.get("_lease") is True


async def test_run_anthropic_messages_none_adapter_passes_none(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _stub_resolve_model_id(monkeypatch)
    pool = _BoomPool()
    monkeypatch.setattr(routes, "_pool", pool)

    req = _make_req(adapter=None)
    with pytest.raises(RuntimeError, match="boom-stop"):
        await routes._run_anthropic_messages(req)

    assert pool.captured is not None
    _model_id, kwargs = pool.captured
    assert kwargs.get("adapter_path") is None
