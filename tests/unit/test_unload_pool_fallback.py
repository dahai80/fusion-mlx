from __future__ import annotations

import pytest


def _make_entry(engine):
    # Minimal duck-typed stand-in for EngineEntry.
    class _Entry:
        def __init__(self, eng):
            self.engine = eng

    return _Entry(engine)


def _patch_server(monkeypatch, pool, resolved=None):
    # _unload_pool_model imports get_server/resolve_model_id from
    # fusion_mlx.server at call time, so patch them on that module.
    import fusion_mlx.server as srv

    class _Server:
        pass

    s = _Server()
    s.pool = pool
    monkeypatch.setattr(srv, "get_server", lambda: s, raising=False)
    monkeypatch.setattr(
        srv,
        "resolve_model_id",
        lambda m: resolved if resolved is not None else m,
        raising=False,
    )
    return s


@pytest.mark.asyncio
async def test_unload_pool_fallback_unloads_loaded_engine(monkeypatch):
    # Issue #631: model loaded in the engine pool but absent from the gui
    # database -> the fallback must find it in the pool and unload it.
    calls = []

    class _Pool:
        def get_entry(self, mid):
            calls.append(("get_entry", mid))
            return _make_entry(engine=object())

        async def unload_engine_async(self, mid):
            calls.append(("unload", mid))

    _patch_server(monkeypatch, _Pool())
    import fusion_mlx.gui_compat.server as gcs

    result = await gcs._unload_pool_model("Qwen3-0.6B-4bit")
    assert result is True
    assert ("unload", "Qwen3-0.6B-4bit") in calls


@pytest.mark.asyncio
async def test_unload_pool_fallback_not_loaded_returns_false(monkeypatch):
    class _Pool:
        def get_entry(self, mid):
            return _make_entry(engine=None)

        async def unload_engine_async(self, mid):
            pytest.fail("should not unload an engine-less entry")

    _patch_server(monkeypatch, _Pool())
    import fusion_mlx.gui_compat.server as gcs

    result = await gcs._unload_pool_model("Qwen3-0.6B-4bit")
    assert result is False


@pytest.mark.asyncio
async def test_unload_pool_fallback_not_in_pool_returns_none(monkeypatch):
    # Model neither in db nor pool -> keep the original 404 (return None).
    class _Pool:
        def get_entry(self, mid):
            return None

    _patch_server(monkeypatch, _Pool())
    import fusion_mlx.gui_compat.server as gcs

    result = await gcs._unload_pool_model("does-not-exist")
    assert result is None


@pytest.mark.asyncio
async def test_unload_pool_fallback_no_server_returns_none(monkeypatch):
    import fusion_mlx.server as srv

    monkeypatch.setattr(srv, "get_server", lambda: None, raising=False)
    import fusion_mlx.gui_compat.server as gcs

    result = await gcs._unload_pool_model("anything")
    assert result is None
