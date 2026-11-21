import asyncio

import pytest

from fusion_mlx.api.ollama_routes import (
    _apply_ollama_keep_alive,
    _parse_ollama_keep_alive,
    set_ollama_context,
)


@pytest.mark.parametrize(
    "raw,expected",
    [
        (None, None),
        ("", None),
        (0, 0.0),
        ("0", 0.0),
        (10, 10.0),
        ("10", 10.0),
        ("5m", 300.0),
        ("1h30m", 5400.0),
        ("500ms", 0.5),
        ("-1", -1.0),
        (1.5, 1.5),
        ("abc", None),
        ("5x", None),
        ("1h30", 3630.0),
    ],
)
def test_parse_ollama_keep_alive(raw, expected):
    assert _parse_ollama_keep_alive(raw) == expected


class _FakePool:
    def __init__(self, result=True, exc=None):
        self.calls = []
        self._result = result
        self._exc = exc

    async def unload_if_idle_unpinned(self, model_id):
        self.calls.append(model_id)
        if self._exc is not None:
            raise self._exc
        return self._result


@pytest.fixture
def fake_resolve(monkeypatch):
    import fusion_mlx.server as server_mod

    monkeypatch.setattr(
        server_mod,
        "resolve_model_with_profile",
        lambda model: (f"resolved::{model}", {}),
    )


@pytest.mark.asyncio
async def test_keep_alive_zero_unloads_async(monkeypatch, fake_resolve):
    pool = _FakePool()
    set_ollama_context(pool)
    _apply_ollama_keep_alive("mymodel", "0")
    tasks = [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]
    if tasks:
        await asyncio.gather(*tasks)
    assert pool.calls == ["resolved::mymodel"]


@pytest.mark.asyncio
async def test_keep_alive_zero_unload_failure_logged(monkeypatch, fake_resolve):
    pool = _FakePool(exc=RuntimeError("pool busy"))
    set_ollama_context(pool)
    _apply_ollama_keep_alive("mymodel", 0)
    tasks = [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]
    if tasks:
        await asyncio.gather(*tasks)
    assert pool.calls == ["resolved::mymodel"]


@pytest.mark.asyncio
async def test_keep_alive_positive_duration_no_unload(fake_resolve):
    pool = _FakePool()
    set_ollama_context(pool)
    _apply_ollama_keep_alive("mymodel", "5m")
    _apply_ollama_keep_alive("mymodel", -1)
    _apply_ollama_keep_alive("mymodel", None)
    _apply_ollama_keep_alive("mymodel", "garbage")
    tasks = [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]
    if tasks:
        await asyncio.gather(*tasks)
    assert pool.calls == []


@pytest.mark.asyncio
async def test_keep_alive_no_pool_bound(fake_resolve):
    set_ollama_context(None)
    # Must not raise.
    _apply_ollama_keep_alive("mymodel", 0)
    tasks = [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]
    if tasks:
        await asyncio.gather(*tasks)
