# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import threading
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient

from fusion_mlx.pool.engine_pool import EngineEntry, EnginePool, UnloadResult


def _make_pool() -> EnginePool:
    pool = EnginePool()
    pool._get_final_ceiling = lambda: 0
    return pool


def _make_engine():
    eng = MagicMock(spec=["safe_evict", "stop", "has_active_requests"])
    eng.safe_evict = AsyncMock()
    eng.stop = AsyncMock()
    eng.has_active_requests = MagicMock(return_value=False)
    return eng


def _make_entry(model_id: str, engine=None, repo_id: str | None = None) -> EngineEntry:
    return EngineEntry(
        model_id=model_id,
        model_path=f"/tmp/fusion-test/{model_id}",
        model_type="llm",
        engine_type="batched",
        estimated_size=1024,
        engine=engine,
        source_repo_id=repo_id,
    )


CANONICAL = "mlx-community--Qwen3.5-9B-4bit"
REPO_ID = "mlx-community/Qwen3.5-9B-4bit"
BARE = "Qwen3.5-9B-4bit"


class TestUnloadResultContract:
    @pytest.mark.asyncio
    async def test_not_loaded_when_entry_absent(self):
        pool = _make_pool()
        result = await pool.unload_engine_async("ghost-model")
        assert isinstance(result, UnloadResult)
        assert result.status == "not_loaded"
        assert result.unloaded is False
        assert result.model_id == "ghost-model"
        assert result.reason

    @pytest.mark.asyncio
    async def test_not_loaded_when_engine_absent(self):
        pool = _make_pool()
        pool._entries["model-a"] = _make_entry("model-a", engine=None)
        result = await pool.unload_engine_async("model-a")
        assert result.status == "not_loaded"
        assert pool._entries["model-a"].engine is None

    @pytest.mark.asyncio
    async def test_deferred_when_engine_leased(self):
        pool = _make_pool()
        engine = _make_engine()
        entry = _make_entry("model-a", engine=engine)
        entry.in_use = 2
        pool._entries["model-a"] = entry

        result = await pool.unload_engine_async("model-a")

        assert result.status == "deferred"
        assert result.unloaded is False
        assert "in_use=2" in (result.reason or "")
        assert entry.engine is engine
        assert entry.pending_unload_reason
        assert entry.is_unloading is False
        engine.safe_evict.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_unloaded_reports_freed_bytes(self):
        pool = _make_pool()
        engine = _make_engine()
        entry = _make_entry("model-a", engine=engine)
        pool._entries["model-a"] = entry

        with (
            patch("fusion_mlx.pool.engine_pool.mx") as mock_mx,
            patch("fusion_mlx.pool.engine_pool.get_mlx_executor", return_value=None),
        ):
            mock_mx.get_active_memory = MagicMock(side_effect=[9000, 6000])
            mock_mx.synchronize = MagicMock()
            mock_mx.clear_cache = MagicMock()

            result = await pool.unload_engine_async("model-a", with_settle=False)

        assert result.status == "unloaded"
        assert result.unloaded is True
        assert result.freed_bytes == 3000
        assert entry.engine is None
        engine.safe_evict.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_freed_bytes_never_negative(self):
        pool = _make_pool()
        entry = _make_entry("model-a", engine=_make_engine())
        pool._entries["model-a"] = entry

        with (
            patch("fusion_mlx.pool.engine_pool.mx") as mock_mx,
            patch("fusion_mlx.pool.engine_pool.get_mlx_executor", return_value=None),
        ):
            mock_mx.get_active_memory = MagicMock(side_effect=[1000, 5000])
            mock_mx.synchronize = MagicMock()
            mock_mx.clear_cache = MagicMock()

            result = await pool.unload_engine_async("model-a", with_settle=False)

        assert result.status == "unloaded"
        assert result.freed_bytes == 0

    @pytest.mark.asyncio
    async def test_unload_resolves_bare_repo_name(self):
        pool = _make_pool()
        entry = _make_entry(CANONICAL, engine=_make_engine(), repo_id=REPO_ID)
        pool._entries[CANONICAL] = entry

        with (
            patch("fusion_mlx.pool.engine_pool.mx") as mock_mx,
            patch("fusion_mlx.pool.engine_pool.get_mlx_executor", return_value=None),
        ):
            mock_mx.get_active_memory = MagicMock(side_effect=[5000, 4000])
            mock_mx.synchronize = MagicMock()
            mock_mx.clear_cache = MagicMock()

            result = await pool.unload_engine_async(BARE, with_settle=False)

        assert result.status == "unloaded"
        assert result.model_id == CANONICAL
        assert entry.engine is None


class TestResolveRegistryKey:
    def test_slash_and_bare_forms_resolve_to_key(self):
        pool = _make_pool()
        pool._entries[CANONICAL] = _make_entry(CANONICAL, repo_id=REPO_ID)
        assert pool.resolve_registry_key(REPO_ID) == CANONICAL
        assert pool.resolve_registry_key(BARE) == CANONICAL
        assert pool.resolve_registry_key(CANONICAL) == CANONICAL

    def test_unknown_name_passes_through(self):
        pool = _make_pool()
        pool._entries[CANONICAL] = _make_entry(CANONICAL, repo_id=REPO_ID)
        assert pool.resolve_registry_key("not-a-model") == "not-a-model"

    @pytest.mark.asyncio
    async def test_release_engine_decrements_resolved_lease(self):
        pool = _make_pool()
        entry = _make_entry(CANONICAL, engine=_make_engine())
        entry.in_use = 1
        pool._entries[CANONICAL] = entry

        await pool.release_engine(BARE)

        assert entry.in_use == 0
        assert entry.engine is not None


class TestServerUnloadHelpers:
    def test_resolve_unload_key_uses_pool_resolver(self):
        from fusion_mlx.server import resolve_unload_key

        pool = _make_pool()
        pool._entries[CANONICAL] = _make_entry(CANONICAL, repo_id=REPO_ID)
        assert resolve_unload_key(pool, BARE) == CANONICAL
        assert resolve_unload_key(pool, REPO_ID) == CANONICAL

    def test_resolve_unload_key_duck_pool_hyphen_fallback(self):
        from fusion_mlx.server import resolve_unload_key

        class _Pool:
            def __init__(self):
                self.entries = {CANONICAL: object()}

            def get_entry(self, mid):
                return self.entries.get(mid)

        pool = _Pool()
        assert resolve_unload_key(pool, REPO_ID) == CANONICAL
        assert resolve_unload_key(pool, BARE) == BARE

    def test_payload_unloaded_includes_freed_bytes(self):
        from fusion_mlx.server import unload_result_payload

        result = UnloadResult(status="unloaded", model_id=CANONICAL, freed_bytes=2048)
        payload = unload_result_payload(BARE, result)
        assert payload["status"] == "unloaded"
        assert payload["model_id"] == BARE
        assert payload["freed_bytes"] == 2048

    def test_payload_deferred_raises_409(self):
        from fusion_mlx.server import unload_result_payload

        result = UnloadResult(
            status="deferred", model_id=CANONICAL, reason="in_use=1 busy=False"
        )
        with pytest.raises(HTTPException) as exc:
            unload_result_payload(BARE, result)
        assert exc.value.status_code == 409
        assert "in_use=1" in exc.value.detail

    def test_payload_not_loaded_raises_404(self):
        from fusion_mlx.server import unload_result_payload

        result = UnloadResult(status="not_loaded", model_id=CANONICAL, reason="none")
        with pytest.raises(HTTPException) as exc:
            unload_result_payload(BARE, result)
        assert exc.value.status_code == 404
        assert "not loaded" in exc.value.detail.lower()

    def test_payload_legacy_none_treated_as_unloaded(self):
        from fusion_mlx.server import unload_result_payload

        payload = unload_result_payload(BARE, None)
        assert payload["status"] == "unloaded"
        assert "freed_bytes" not in payload


def _make_gui_app(db_first):
    from fusion_mlx.gui_compat import server as gui

    app = FastAPI()
    app.include_router(gui.get_gui_compat_router())
    app.dependency_overrides[gui.verify_api_key] = lambda: "test-key"
    app.dependency_overrides[gui.require_admin] = lambda: True
    db = MagicMock()
    db.query.return_value.filter.return_value.first.return_value = db_first
    app.dependency_overrides[gui.get_db_session] = lambda: db
    return app


class TestGuiUnloadRouteStatus:
    def _pool_for(self, engine_present: bool, result):
        pool = MagicMock()
        entry = SimpleNamespace(engine=MagicMock() if engine_present else None)
        pool.get_entry.return_value = entry
        pool.unload_engine_async = AsyncMock(return_value=result)
        return pool

    def test_pool_deferred_returns_409(self, monkeypatch):
        pool = self._pool_for(
            True,
            UnloadResult(status="deferred", model_id=BARE, reason="in_use=1"),
        )
        monkeypatch.setattr(
            "fusion_mlx.server.get_server", lambda: SimpleNamespace(pool=pool)
        )
        monkeypatch.setattr("fusion_mlx.server.resolve_model_id", lambda mid: mid)
        app = _make_gui_app(None)
        resp = TestClient(app).post(f"/v1/models/{BARE}/unload")
        assert resp.status_code == 409
        assert "deferred" in resp.json()["detail"]

    def test_pool_unloaded_returns_200(self, monkeypatch):
        pool = self._pool_for(
            True,
            UnloadResult(status="unloaded", model_id=BARE, freed_bytes=4096),
        )
        monkeypatch.setattr(
            "fusion_mlx.server.get_server", lambda: SimpleNamespace(pool=pool)
        )
        monkeypatch.setattr("fusion_mlx.server.resolve_model_id", lambda mid: mid)
        app = _make_gui_app(None)
        resp = TestClient(app).post(f"/v1/models/{BARE}/unload")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "unloaded"
        assert body["freed_bytes"] == 4096

    def test_pool_not_loaded_reports_status(self, monkeypatch):
        pool = self._pool_for(False, None)
        monkeypatch.setattr(
            "fusion_mlx.server.get_server", lambda: SimpleNamespace(pool=pool)
        )
        monkeypatch.setattr("fusion_mlx.server.resolve_model_id", lambda mid: mid)
        app = _make_gui_app(None)
        resp = TestClient(app).post(f"/v1/models/{BARE}/unload")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "not_loaded"
        assert "unloaded" not in body["message"]
        pool.unload_engine_async.assert_not_awaited()

    def test_pool_absent_returns_404(self, monkeypatch):
        pool = MagicMock()
        pool.get_entry.return_value = None
        monkeypatch.setattr(
            "fusion_mlx.server.get_server", lambda: SimpleNamespace(pool=pool)
        )
        monkeypatch.setattr("fusion_mlx.server.resolve_model_id", lambda mid: mid)
        app = _make_gui_app(None)
        resp = TestClient(app).post("/v1/models/ghost/unload")
        assert resp.status_code == 404

    def test_db_model_deferred_returns_409(self, monkeypatch):
        from fusion_mlx.gui_compat import server as gui

        manager = MagicMock()
        manager.unload_model_async = AsyncMock(
            return_value=UnloadResult(
                status="deferred", model_id="db-model", reason="in_use=2"
            )
        )
        monkeypatch.setattr(gui, "get_model_manager", lambda: manager)
        app = _make_gui_app(MagicMock())
        resp = TestClient(app).post("/v1/models/db-model/unload")
        assert resp.status_code == 409
        assert "in_use=2" in resp.json()["detail"]
        manager.unload_model_async.assert_awaited_once_with("db-model")

    def test_db_model_unloaded_returns_200(self, monkeypatch):
        from fusion_mlx.gui_compat import server as gui

        manager = MagicMock()
        manager.unload_model_async = AsyncMock(
            return_value=UnloadResult(status="unloaded", model_id="db-model")
        )
        monkeypatch.setattr(gui, "get_model_manager", lambda: manager)
        app = _make_gui_app(MagicMock())
        resp = TestClient(app).post("/v1/models/db-model/unload")
        assert resp.status_code == 200
        assert resp.json()["status"] == "unloaded"


def _make_manager(pool):
    from fusion_mlx.gui_compat.model_manager import LoadingStatus, ModelManager

    mm = ModelManager.__new__(ModelManager)
    mm._loaded_models = {"model-a": object()}
    mm._loading_status = {"model-a": LoadingStatus.LOADED}
    mm._lock = threading.RLock()
    mm._engine_pool = pool
    mm._engine_pool_lock = threading.Lock()
    return mm


class TestModelManagerTruthfulUnload:
    def _dbm_patch(self):
        dbm = MagicMock()
        session = MagicMock()
        dbm.get_session.return_value.__enter__.return_value = session
        return patch(
            "fusion_mlx.gui_compat.model_manager.get_database_manager",
            return_value=dbm,
        )

    @pytest.mark.asyncio
    async def test_deferred_keeps_model_tracked(self):
        from fusion_mlx.gui_compat.model_manager import LoadingStatus

        pool = MagicMock()
        pool.unload_engine_async = AsyncMock(
            return_value=UnloadResult(
                status="deferred", model_id="model-a", reason="in_use=1"
            )
        )
        mm = _make_manager(pool)
        with (
            self._dbm_patch(),
            patch("fusion_mlx.server.get_server", lambda: None),
        ):
            result = await mm.unload_model_async("model-a")

        assert result.status == "deferred"
        assert "model-a" in mm._loaded_models
        assert mm._loading_status["model-a"] == LoadingStatus.IDLE

    @pytest.mark.asyncio
    async def test_unloaded_drops_model_tracking(self):
        from fusion_mlx.gui_compat.model_manager import LoadingStatus

        pool = MagicMock()
        pool.unload_engine_async = AsyncMock(
            return_value=UnloadResult(
                status="unloaded", model_id="model-a", freed_bytes=8192
            )
        )
        mm = _make_manager(pool)
        with (
            self._dbm_patch(),
            patch("fusion_mlx.server.get_server", lambda: None),
        ):
            result = await mm.unload_model_async("model-a")

        assert result.status == "unloaded"
        assert "model-a" not in mm._loaded_models
        assert mm._loading_status["model-a"] == LoadingStatus.IDLE

    def test_sync_unload_returns_false_when_deferred(self):
        pool = MagicMock()
        pool.unload_engine_async = AsyncMock(
            return_value=UnloadResult(status="deferred", model_id="model-a")
        )
        mm = _make_manager(pool)
        with (
            self._dbm_patch(),
            patch("fusion_mlx.server.get_server", lambda: None),
        ):
            assert mm.unload_model("model-a") is False

    def test_sync_unload_returns_true_when_unloaded(self):
        pool = MagicMock()
        pool.unload_engine_async = AsyncMock(
            return_value=UnloadResult(status="unloaded", model_id="model-a")
        )
        mm = _make_manager(pool)
        with (
            self._dbm_patch(),
            patch("fusion_mlx.server.get_server", lambda: None),
        ):
            assert mm.unload_model("model-a") is True

    @pytest.mark.asyncio
    async def test_sync_unload_inside_loop_fails_visibly(self):
        pool = MagicMock()
        pool.unload_engine_async = AsyncMock(
            return_value=UnloadResult(status="unloaded", model_id="model-a")
        )
        mm = _make_manager(pool)
        with (
            self._dbm_patch(),
            patch("fusion_mlx.server.get_server", lambda: None),
        ):
            assert mm.unload_model("model-a") is False
        pool.unload_engine_async.assert_not_awaited()
