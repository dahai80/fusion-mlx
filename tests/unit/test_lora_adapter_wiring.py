"""Wiring tests for runtime LoRA adapter hot-swap (B.2 path A').

These guard the request-model -> route -> engine-pool wiring. An earlier
commit added the ``adapters`` field to the *dead* ``api/models.py`` request
classes (used by legacy unregistered routes) instead of the *active*
``api/openai_models.py`` classes imported by ``api/openai_routes.py``. The
result: ``/v1/chat/completions`` silently dropped ``adapters``
(getattr -> None) and ``/v1/completions`` raised AttributeError -> 500.
These tests construct the active request classes with ``adapters`` and assert
the field survives and reaches ``EnginePool.get_engine`` as ``adapter_path``.
"""

from __future__ import annotations

import sys
import types

import pytest

import fusion_mlx.api.openai_routes as routes
from fusion_mlx.api.openai_models import ChatCompletionRequest, CompletionRequest


class _BoomPool:
    # Records get_engine kwargs then raises, short-circuiting the route before
    # any engine.chat work so we can assert adapter threading in isolation.
    def __init__(self) -> None:
        self.captured: tuple[str, dict] | None = None

    async def get_engine(self, model_id: str, **kwargs):
        self.captured = (model_id, kwargs)
        raise RuntimeError("boom-stop")

    async def release_engine(self, model_id: str, **kwargs):
        return None


def _stub_resolve_model_id(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = types.ModuleType("fusion_mlx.server")
    fake.resolve_model_id = lambda model_id: model_id  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "fusion_mlx.server", fake)


def test_chat_request_adapters_field_present() -> None:
    req = ChatCompletionRequest(
        model="m", messages=[{"role": "user", "content": "hi"}], adapters="/lora"
    )
    assert hasattr(req, "adapters")
    assert req.adapters == "/lora"


def test_completion_request_adapters_field_present() -> None:
    req = CompletionRequest(model="m", prompt="hi", adapters="/lora")
    assert hasattr(req, "adapters")
    assert req.adapters == "/lora"


def test_completions_conversion_threads_adapters() -> None:
    # Mirrors the conversion in openai_routes.completions(): before the fix,
    # request.adapters raised AttributeError -> 500 on /v1/completions.
    request = CompletionRequest(model="m", prompt="hi", adapters="/lora")
    chat_req = ChatCompletionRequest(
        model=request.model,
        adapters=request.adapters,
        messages=[{"role": "user", "content": request.prompt}],
        max_tokens=request.max_tokens,
        temperature=request.temperature,
        top_p=request.top_p,
        stream=request.stream,
        stop=request.stop,
    )
    assert chat_req.adapters == "/lora"


async def test_run_chat_threads_adapters_to_get_engine(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _stub_resolve_model_id(monkeypatch)
    pool = _BoomPool()
    monkeypatch.setattr(routes, "_pool", pool)
    monkeypatch.setattr(routes, "_request_router", object())

    req = ChatCompletionRequest(
        model="m", messages=[{"role": "user", "content": "hi"}], adapters="/lora"
    )
    with pytest.raises(RuntimeError, match="boom-stop"):
        await routes._run_chat(req)

    assert pool.captured is not None
    _model_id, kwargs = pool.captured
    assert kwargs.get("adapter_path") == "/lora"
