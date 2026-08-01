# SPDX-License-Identifier: Apache-2.0
import unittest
from unittest.mock import MagicMock, patch

from fastapi import FastAPI
from fastapi.security import HTTPAuthorizationCredentials
from fastapi.testclient import TestClient

from fusion_mlx.admin.model_manager_route import router


async def _fake_scoped_key(
    request=None,
    credentials: HTTPAuthorizationCredentials = None,
    required_role: str = "model_manager",
) -> str:
    return "model_manager"


def _make_app():
    app = FastAPI()
    app.include_router(router)
    from fusion_mlx.admin.model_manager_route import verify_scoped_api_key
    app.dependency_overrides[verify_scoped_api_key] = _fake_scoped_key
    return app


class TestModelManagerListModels(unittest.TestCase):
    def test_list_models_no_pool(self):
        app = _make_app()
        with patch("fusion_mlx.admin.model_manager_route._get_engine_pool", return_value=None):
            client = TestClient(app)
            resp = client.get("/api/model-manager/models")
            self.assertEqual(resp.status_code, 200)
            self.assertEqual(resp.json()["models"], [])

    def test_list_models_with_entries(self):
        mock_pool = MagicMock()
        mock_pool.get_status.return_value = {
            "models": [
                {"id": "test-model", "loaded": True, "is_loading": False,
                 "estimated_size": 1024, "pinned": False,
                 "engine_type": "batched", "model_type": "llm"},
            ]
        }
        app = _make_app()
        with patch("fusion_mlx.admin.model_manager_route._get_engine_pool", return_value=mock_pool):
            client = TestClient(app)
            resp = client.get("/api/model-manager/models")
            self.assertEqual(resp.status_code, 200)
            data = resp.json()
            self.assertEqual(len(data["models"]), 1)
            self.assertEqual(data["models"][0]["id"], "test-model")


class TestModelManagerLoadUnload(unittest.TestCase):
    def test_load_model_not_found(self):
        mock_pool = MagicMock()
        mock_pool.get_entry.return_value = None
        app = _make_app()
        with patch("fusion_mlx.admin.model_manager_route._get_engine_pool", return_value=mock_pool):
            client = TestClient(app)
            resp = client.post("/api/model-manager/models/no-such-model/load")
            self.assertEqual(resp.status_code, 404)

    def test_unload_model_not_loaded(self):
        mock_pool = MagicMock()
        entry = MagicMock()
        entry.engine = None
        mock_pool.get_entry.return_value = entry
        app = _make_app()
        with patch("fusion_mlx.admin.model_manager_route._get_engine_pool", return_value=mock_pool):
            client = TestClient(app)
            resp = client.post("/api/model-manager/models/test-model/unload")
            self.assertEqual(resp.status_code, 400)

    def test_model_status(self):
        mock_pool = MagicMock()
        entry = MagicMock()
        entry.engine = MagicMock()
        entry.is_loading = False
        entry.pinned = True
        mock_pool.get_entry.return_value = entry
        app = _make_app()
        with patch("fusion_mlx.admin.model_manager_route._get_engine_pool", return_value=mock_pool):
            client = TestClient(app)
            resp = client.get("/api/model-manager/models/test-model/status")
            self.assertEqual(resp.status_code, 200)
            data = resp.json()
            self.assertTrue(data["loaded"])
            self.assertTrue(data["pinned"])


if __name__ == "__main__":
    unittest.main()
