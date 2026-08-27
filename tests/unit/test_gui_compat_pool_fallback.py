# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest


def _make_srv(slash_resolved, hyphen_entry):
    pool = MagicMock()
    entry_map = {slash_resolved: None, slash_resolved.replace("/", "-"): hyphen_entry}
    pool.get_entry.side_effect = lambda mid: entry_map.get(mid)
    pool.get_engine = AsyncMock(return_value=MagicMock())
    pool.unload_engine_async = AsyncMock(return_value=None)
    srv = SimpleNamespace(pool=pool)
    return srv, pool


@pytest.mark.asyncio
async def test_resolve_pool_model_slash_id_falls_back_to_hyphen(monkeypatch):
    from fusion_mlx.gui_compat import server as gui

    slash_id = "mlx-community/Llama-3.2"
    hyphen_entry = SimpleNamespace(engine=None)
    srv, pool = _make_srv(slash_id, hyphen_entry)
    monkeypatch.setattr("fusion_mlx.server.get_server", lambda: srv)
    monkeypatch.setattr("fusion_mlx.server.resolve_model_id", lambda mid: mid)
    result = await gui._resolve_pool_model(slash_id)
    assert result is not None
    assert result["status"] == "ok"
    assert result["model_id"] == slash_id
    assert pool.get_entry.call_count == 2
    assert pool.get_entry.call_args_list[0].args[0] == slash_id
    assert pool.get_entry.call_args_list[1].args[0] == slash_id.replace("/", "-")


@pytest.mark.asyncio
async def test_unload_pool_model_slash_id_falls_back_to_hyphen(monkeypatch):
    from fusion_mlx.gui_compat import server as gui

    slash_id = "mlx-community/Llama-3.2"
    hyphen_entry = SimpleNamespace(engine=MagicMock())
    srv, pool = _make_srv(slash_id, hyphen_entry)
    monkeypatch.setattr("fusion_mlx.server.get_server", lambda: srv)
    monkeypatch.setattr("fusion_mlx.server.resolve_model_id", lambda mid: mid)
    result = await gui._unload_pool_model(slash_id)
    assert result is True
    assert pool.get_entry.call_count == 2
    assert pool.get_entry.call_args_list[1].args[0] == slash_id.replace("/", "-")
    pool.unload_engine_async.assert_awaited_once_with(slash_id.replace("/", "-"))


@pytest.mark.asyncio
async def test_resolve_pool_model_no_slash_no_retry(monkeypatch):
    from fusion_mlx.gui_compat import server as gui

    plain_id = "qwen2.5-0.5b"
    pool = MagicMock()
    pool.get_entry.return_value = None
    srv = SimpleNamespace(pool=pool)
    monkeypatch.setattr("fusion_mlx.server.get_server", lambda: srv)
    monkeypatch.setattr("fusion_mlx.server.resolve_model_id", lambda mid: mid)
    result = await gui._resolve_pool_model(plain_id)
    assert result is None
    assert pool.get_entry.call_count == 1
