# SPDX-License-Identifier: Apache-2.0

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import fusion_mlx.admin.helpers as admin_helpers


def _setup_mocks(engine_pool=None, settings_manager=None, global_settings=None):
    originals = {
        "pool": admin_helpers._get_engine_pool,
        "settings_manager": admin_helpers._get_settings_manager,
        "global_settings": admin_helpers._get_global_settings,
    }
    admin_helpers._get_engine_pool = lambda: engine_pool
    admin_helpers._get_settings_manager = lambda: settings_manager
    admin_helpers._get_global_settings = lambda: global_settings
    return originals


def _restore_mocks(originals):
    admin_helpers._get_engine_pool = originals["pool"]
    admin_helpers._get_settings_manager = originals["settings_manager"]
    admin_helpers._get_global_settings = originals["global_settings"]


def _mock_server_state(engine_pool=None, settings_manager=None):
    state = MagicMock()
    state.engine_pool = engine_pool
    state.settings_manager = settings_manager
    return state


class TestReloadModels:
    def test_reload_success(self):
        pool = MagicMock()
        pool.model_count = 5
        pool.preload_pinned_models = AsyncMock()

        settings_manager = MagicMock()
        settings_manager._load = MagicMock()
        settings_manager.get_pinned_model_ids = MagicMock(return_value=[])

        global_settings = MagicMock()
        global_settings.model.model_dirs = ["/path/to/models"]
        global_settings.model.model_dir = "/path/to/models"
        global_settings.get_effective_model_dirs = MagicMock(
            return_value=["/path/to/models"]
        )

        originals = _setup_mocks(pool, settings_manager, global_settings)

        mock_server_state = _mock_server_state(pool, settings_manager)

        try:
            with patch("fusion_mlx.server._server_state", mock_server_state):
                with patch(
                    "fusion_mlx.admin.helpers._apply_model_dirs_runtime",
                    new_callable=AsyncMock,
                    return_value=(True, "Re-discovered 5 models from 1 directory"),
                ) as mock_apply:
                    success, msg = asyncio.run(admin_helpers._reload_models())

                    assert success is True
                    assert "5 models" in msg
                    settings_manager._load.assert_called_once()
                    mock_apply.assert_called_once_with(["/path/to/models"])
                    pool.preload_pinned_models.assert_called_once()
        finally:
            _restore_mocks(originals)

    def test_reload_engine_pool_none(self):
        mock_server_state = _mock_server_state(engine_pool=None)

        originals = _setup_mocks(
            engine_pool=None,
            settings_manager=None,
            global_settings=MagicMock(),
        )

        try:
            with patch("fusion_mlx.server._server_state", mock_server_state):
                success, msg = asyncio.run(admin_helpers._reload_models())
                assert success is False
                assert "not initialized" in msg
        finally:
            _restore_mocks(originals)

    def test_reload_global_settings_none(self):
        pool = MagicMock()

        mock_server_state = _mock_server_state(engine_pool=pool)

        originals = _setup_mocks(
            engine_pool=pool,
            settings_manager=None,
            global_settings=None,
        )

        try:
            with patch("fusion_mlx.server._server_state", mock_server_state):
                success, msg = asyncio.run(admin_helpers._reload_models())
                assert success is False
                assert "not initialized" in msg
        finally:
            _restore_mocks(originals)

    def test_reload_apply_dirs_fails(self):
        pool = MagicMock()
        pool.preload_pinned_models = AsyncMock()

        settings_manager = MagicMock()
        settings_manager._load = MagicMock()

        global_settings = MagicMock()
        global_settings.model.model_dirs = ["/bad/path"]
        global_settings.model.model_dir = "/bad/path"
        global_settings.get_effective_model_dirs = MagicMock(return_value=["/bad/path"])

        originals = _setup_mocks(pool, settings_manager, global_settings)

        mock_server_state = _mock_server_state(engine_pool=pool)

        try:
            with patch("fusion_mlx.server._server_state", mock_server_state):
                with patch(
                    "fusion_mlx.admin.helpers._apply_model_dirs_runtime",
                    new_callable=AsyncMock,
                    return_value=(False, "Model directory does not exist: /bad/path"),
                ):
                    success, msg = asyncio.run(admin_helpers._reload_models())
                    assert success is False
                    assert "does not exist" in msg
                    pool.preload_pinned_models.assert_not_called()
        finally:
            _restore_mocks(originals)

    def test_reload_fallback_to_model_dir(self):
        pool = MagicMock()
        pool.preload_pinned_models = AsyncMock()

        settings_manager = MagicMock()
        settings_manager._load = MagicMock()

        global_settings = MagicMock()
        global_settings.model.model_dirs = []
        global_settings.model.model_dir = "/fallback/path"
        global_settings.get_effective_model_dirs = MagicMock(
            return_value=["/fallback/path"]
        )

        originals = _setup_mocks(pool, settings_manager, global_settings)

        mock_server_state = _mock_server_state(engine_pool=pool)

        try:
            with patch("fusion_mlx.server._server_state", mock_server_state):
                with patch(
                    "fusion_mlx.admin.helpers._apply_model_dirs_runtime",
                    new_callable=AsyncMock,
                    return_value=(True, "Re-discovered 3 models from 1 directory"),
                ) as mock_apply:
                    success, msg = asyncio.run(admin_helpers._reload_models())
                    assert success is True
                    mock_apply.assert_called_once_with(["/fallback/path"])
        finally:
            _restore_mocks(originals)


class TestApplyModelDirsRuntime:
    def test_rejects_nonexistent_model_dir(self, tmp_path):
        nonexistent = tmp_path / "nonexistent"

        mock_server_state = _mock_server_state(
            engine_pool=MagicMock(), settings_manager=None
        )

        with patch("fusion_mlx.server._server_state", mock_server_state):
            success, msg = asyncio.run(
                admin_helpers._apply_model_dirs_runtime([str(nonexistent)])
            )

        assert success is False
        assert "does not exist" in msg

    def test_rejects_non_directory_path(self, tmp_path):
        file_path = tmp_path / "afile.txt"
        file_path.write_text("not a dir")

        mock_server_state = _mock_server_state(
            engine_pool=MagicMock(), settings_manager=None
        )

        with patch("fusion_mlx.server._server_state", mock_server_state):
            success, msg = asyncio.run(
                admin_helpers._apply_model_dirs_runtime([str(file_path)])
            )

        assert success is False
        assert "not a directory" in msg

    def test_skips_unreadable_secondary_model_dir(self, tmp_path, monkeypatch):
        primary = tmp_path / "primary"
        secondary = tmp_path / "secondary"
        primary.mkdir()
        secondary.mkdir()

        original_iterdir = Path.iterdir

        def fake_iterdir(path):
            if path == secondary.resolve():
                raise PermissionError("Operation not permitted")
            return original_iterdir(path)

        monkeypatch.setattr(Path, "iterdir", fake_iterdir)
        monkeypatch.setattr(admin_helpers, "_hf_downloader", None)
        monkeypatch.setattr(admin_helpers, "_ms_downloader", None)
        monkeypatch.setattr(admin_helpers, "_oq_manager", None)
        monkeypatch.setattr(admin_helpers, "_hf_uploader", None)

        pool = MagicMock()
        pool.get_loaded_model_ids.return_value = []
        pool.model_count = 0
        mock_server_state = _mock_server_state(engine_pool=pool, settings_manager=None)

        with patch("fusion_mlx.server._server_state", mock_server_state):
            success, msg = asyncio.run(
                admin_helpers._apply_model_dirs_runtime(
                    [str(primary), str(secondary)]
                )
            )

        assert success is True
        assert "from 2 directories" in msg
        pool.discover_models.assert_called_once_with(
            [str(primary.resolve()), str(secondary.resolve())], []
        )

    def test_applies_existing_model_dir(self, tmp_path, monkeypatch):
        model_dir = tmp_path / "models"
        model_dir.mkdir()

        monkeypatch.setattr(admin_helpers, "_hf_downloader", None)
        monkeypatch.setattr(admin_helpers, "_ms_downloader", None)
        monkeypatch.setattr(admin_helpers, "_oq_manager", None)
        monkeypatch.setattr(admin_helpers, "_hf_uploader", None)

        pool = MagicMock()
        pool.get_loaded_model_ids.return_value = []
        pool.model_count = 0
        mock_server_state = _mock_server_state(engine_pool=pool, settings_manager=None)

        with patch("fusion_mlx.server._server_state", mock_server_state):
            success, msg = asyncio.run(
                admin_helpers._apply_model_dirs_runtime([str(model_dir)])
            )

        assert success is True
        assert "from 1 directory" in msg
        pool.discover_models.assert_called_once_with([str(model_dir.resolve())], [])
