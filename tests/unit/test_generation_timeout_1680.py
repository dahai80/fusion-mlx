import asyncio

import pytest

from fusion_mlx.exceptions import GenerationTimeoutError
from fusion_mlx.request import SamplingParams


def test_timeout_error_attributes_and_message():
    err = GenerationTimeoutError("req-123", 1638.4)
    assert err.request_id == "req-123"
    assert err.timeout_s == 1638.4
    assert "FUSION_MLX_GENERATE_TIMEOUT_S" in str(err)
    assert isinstance(err, RuntimeError)


def test_timeout_scales_with_max_tokens(monkeypatch):
    # #1680: effective floor = max(env_base, max_tokens / 5). Without an
    # explicit max_tokens the env base stands.
    import inspect

    from fusion_mlx.engine_core import EngineCore

    src = inspect.getsource(EngineCore.generate)
    assert "max(_gen_timeout" in src, "per-request scaling missing"
    assert "/ 5.0" in src


def test_zero_env_disables_scaling(monkeypatch):
    # 0 = unbounded must stay unbounded regardless of max_tokens.
    import inspect

    from fusion_mlx.engine_core import EngineCore

    src = inspect.getsource(EngineCore.generate)
    assert "_gen_timeout > 0 and sampling_params" in src


@pytest.mark.asyncio
async def test_timeout_raises_generation_timeout_error(monkeypatch):
    # End-to-end through generate(): a finished_event that never sets with
    # a tiny env timeout must surface GenerationTimeoutError (not a bare
    # RuntimeError), after aborting the request.
    from fusion_mlx.engine_core import EngineCore

    core = EngineCore.__new__(EngineCore)

    async def fake_add_request(**kwargs):
        return "req-42"

    async def fake_abort(rid):
        fake_abort.called = True

    def fake_cleanup(rid):
        pass

    ctx = type("Ctx", (), {})()
    ctx.finished_event = asyncio.Event()  # never set
    ctx.collector = None
    core._active_contexts = {"req-42": ctx}
    core.add_request = fake_add_request
    core.abort_request = fake_abort
    core._cleanup_request = fake_cleanup
    monkeypatch.setenv("FUSION_MLX_GENERATE_TIMEOUT_S", "0.05")

    with pytest.raises(GenerationTimeoutError) as exc_info:
        await core.generate("hi", SamplingParams(max_tokens=16))
    assert exc_info.value.request_id == "req-42"
    assert exc_info.value.timeout_s >= 0.05
    assert getattr(fake_abort, "called", False) is True


@pytest.mark.asyncio
async def test_timeout_floor_uses_max_tokens(monkeypatch):
    # A max_tokens=8192 request must get >= 1638s even when env is tiny —
    # the whole point of #1680.
    from fusion_mlx.engine_core import EngineCore

    core = EngineCore.__new__(EngineCore)
    captured = {}

    async def fake_add_request(**kwargs):
        return "req-7"

    async def fake_abort(rid):
        pass

    ctx = type("Ctx", (), {})()
    ctx.finished_event = asyncio.Event()  # never set; spy raises first
    ctx.collector = None
    core._active_contexts = {"req-7": ctx}
    core.add_request = fake_add_request
    core.abort_request = fake_abort
    core._cleanup_request = lambda rid: None

    real_wait_for = asyncio.wait_for

    async def spy_wait_for(fut, timeout):
        captured["timeout"] = timeout
        fut.set() if hasattr(fut, "set") else None
        raise TimeoutError()

    monkeypatch.setenv("FUSION_MLX_GENERATE_TIMEOUT_S", "600")
    monkeypatch.setattr(asyncio, "wait_for", spy_wait_for)

    with pytest.raises(GenerationTimeoutError):
        await core.generate("hi", SamplingParams(max_tokens=8192))
    assert captured["timeout"] == pytest.approx(8192 / 5.0)


@pytest.mark.asyncio
async def test_handler_returns_504_generation_timeout():
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from fusion_mlx.middleware.exception_handlers import (
        install_exception_handlers,
    )

    app = FastAPI()
    install_exception_handlers(app)

    @app.post("/v1/chat/completions")
    async def chat():
        raise GenerationTimeoutError("req-9", 600.0)

    client = TestClient(app, raise_server_exceptions=False)
    resp = client.post("/v1/chat/completions", json={})
    assert resp.status_code == 504
    body = resp.json()
    assert body["error"]["type"] == "generation_timeout"
    assert body["error"]["code"] == "generation_timeout"
    assert "req-9" in body["error"]["message"]
