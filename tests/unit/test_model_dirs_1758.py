# SPDX-License-Identifier: Apache-2.0
# #1758: (1) model_dirs from settings.json must be honored by boot/rescan
# discovery (ServerConfig resolution), (2) unsupported-architecture load
# failures must surface the loader's message instead of a generic 500.

import json
import unittest
from unittest.mock import MagicMock, patch

from fastapi import FastAPI
from fastapi.security import HTTPAuthorizationCredentials
from fastapi.testclient import TestClient

from fusion_mlx.config import ServerConfig


class TestServerConfigModelDirs(unittest.TestCase):
    def _write_settings(self, tmp_path, dirs):
        (tmp_path / "settings.json").write_text(
            json.dumps({"model": {"model_dirs": dirs}})
        )

    def test_model_dirs_read_from_settings(self, tmp_path=None):
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            self._write_settings(tmp, [str(tmp / "primary"), str(tmp / "secondary")])
            cfg = ServerConfig(settings_dir=str(tmp))
            self.assertEqual(
                cfg.model_dirs, [str(tmp / "primary"), str(tmp / "secondary")]
            )
            # model_dir stays the primary root for downloads
            self.assertEqual(cfg.model_dir, str(tmp / "primary"))

    def test_explicit_model_dir_beats_settings(self):
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            self._write_settings(tmp, ["/settings/primary", "/settings/second"])
            cfg = ServerConfig(model_dir="/cli/explicit", settings_dir=str(tmp))
            self.assertEqual(cfg.model_dirs, ["/cli/explicit"])

    def test_explicit_model_dirs_arg_wins(self):
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            self._write_settings(tmp, ["/settings/primary"])
            cfg = ServerConfig(model_dirs=["/a", " ", "/b"], settings_dir=str(tmp))
            self.assertEqual(cfg.model_dirs, ["/a", "/b"])

    def test_missing_settings_falls_back_to_model_dir(self):
        import tempfile

        with tempfile.TemporaryDirectory() as td:
            cfg = ServerConfig(settings_dir=td)
            self.assertEqual(cfg.model_dirs, [cfg.model_dir])

    def test_malformed_settings_falls_back(self):
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            (tmp / "settings.json").write_text("{broken json")
            cfg = ServerConfig(settings_dir=str(tmp))
            self.assertEqual(cfg.model_dirs, [cfg.model_dir])


async def _fake_scoped_key(
    request=None,
    credentials: HTTPAuthorizationCredentials = None,
    required_role: str = "model_manager",
) -> str:
    return "model_manager"


class TestLoadErrorSurfacing(unittest.TestCase):
    def _model_manager_app(self):
        from fusion_mlx.admin.model_manager_route import (
            router,
            verify_scoped_api_key,
        )

        app = FastAPI()
        app.include_router(router)
        app.dependency_overrides[verify_scoped_api_key] = _fake_scoped_key
        return app

    def _admin_models_app(self):
        from fusion_mlx.admin.models_route import router, verify_scoped_api_key

        app = FastAPI()
        app.include_router(router)
        app.dependency_overrides[verify_scoped_api_key] = _fake_scoped_key
        return app

    def _mock_pool(self, exc):
        mock_pool = MagicMock()
        entry = MagicMock()
        entry.engine = None
        entry.is_loading = False
        mock_pool.get_entry.return_value = entry
        mock_pool.get_engine = MagicMock(side_effect=exc)
        return mock_pool

    def test_model_manager_unsupported_arch_is_422(self):
        app = self._model_manager_app()
        pool = self._mock_pool(ValueError("Model type inkling2_loop not supported."))
        with patch(
            "fusion_mlx.admin.model_manager_route._get_engine_pool",
            return_value=pool,
        ):
            resp = TestClient(app).post("/api/model-manager/models/m/load")
        self.assertEqual(resp.status_code, 422)
        self.assertIn("inkling2_loop", resp.json()["detail"])

    def test_model_manager_generic_error_keeps_500_with_detail(self):
        app = self._model_manager_app()
        pool = self._mock_pool(RuntimeError("weights checksum mismatch"))
        with patch(
            "fusion_mlx.admin.model_manager_route._get_engine_pool",
            return_value=pool,
        ):
            resp = TestClient(app).post("/api/model-manager/models/m/load")
        self.assertEqual(resp.status_code, 500)
        self.assertIn("checksum mismatch", resp.json()["detail"])

    def test_admin_models_unsupported_arch_is_422(self):
        app = self._admin_models_app()
        pool = self._mock_pool(ValueError("Model type inkling2_loop not supported."))
        with patch(
            "fusion_mlx.admin.models_route._get_engine_pool",
            return_value=pool,
        ):
            resp = TestClient(app).post("/api/models/m/load")
        self.assertEqual(resp.status_code, 422)
        self.assertIn("inkling2_loop", resp.json()["detail"])

    def test_admin_models_generic_error_keeps_500_with_detail(self):
        app = self._admin_models_app()
        pool = self._mock_pool(RuntimeError("disk full"))
        with patch(
            "fusion_mlx.admin.models_route._get_engine_pool",
            return_value=pool,
        ):
            resp = TestClient(app).post("/api/models/m/load")
        self.assertEqual(resp.status_code, 500)
        self.assertIn("disk full", resp.json()["detail"])


if __name__ == "__main__":
    unittest.main()
