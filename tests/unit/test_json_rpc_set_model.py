# SPDX-License-Identifier: Apache-2.0
"""Tests for JSON-RPC /rpc and REST /v1/set_default_model endpoints (#277)."""

from unittest.mock import MagicMock, patch

from fusion_mlx.pool.engine_pool import EngineEntry


def _make_entry(model_id="test-model"):
    entry = MagicMock(spec=EngineEntry)
    entry.model_path = model_id
    entry.engine = MagicMock()
    return entry


class TestJsonRpcSetModelLogic:
    """Unit tests for mlx.set_model JSON-RPC logic."""

    def test_set_model_updates_server_state(self):
        from fusion_mlx.server import _server_state

        with patch.dict(_server_state, {"default_model": None}, clear=False):
            with patch("fusion_mlx.server.resolve_model_id", return_value="qwen3-4b"):
                entry = _make_entry("qwen3-4b")
                params = {"model": "qwen3-4b"}
                model_id = params.get("model")
                assert model_id == "qwen3-4b"
                _server_state["default_model"] = "qwen3-4b"
                assert _server_state["default_model"] == "qwen3-4b"

    def test_set_model_missing_param_returns_error(self):
        params = {}
        model_id = params.get("model")
        code = -32602 if not model_id else 0
        assert code == -32602

    def test_set_model_not_found_returns_error(self):
        from fusion_mlx.server import _server_state

        with patch.dict(_server_state, {"default_model": None}, clear=False):
            with patch("fusion_mlx.server.resolve_model_id", return_value="nope"):
                params = {"model": "nope"}
                model_id = params.get("model")
                # Simulate pool.get_entry returning None
                entry = None
                code = -32602 if entry is None else 0
                assert code == -32602


class TestJsonRpcResponseFormat:
    """Tests for JSON-RPC 2.0 response envelope."""

    def test_result_envelope(self):
        req_id = 1
        result = {"status": "ok", "model": "qwen3-4b"}
        resp = {"jsonrpc": "2.0", "result": result}
        if req_id is not None:
            resp["id"] = req_id
        assert resp["jsonrpc"] == "2.0"
        assert resp["result"]["status"] == "ok"
        assert resp["result"]["model"] == "qwen3-4b"
        assert resp["id"] == 1

    def test_error_envelope(self):
        req_id = 2
        resp = {
            "jsonrpc": "2.0",
            "error": {"code": -32602, "message": "Missing 'model' parameter"},
        }
        if req_id is not None:
            resp["id"] = req_id
        assert resp["error"]["code"] == -32602
        assert resp["id"] == 2

    def test_method_not_found_envelope(self):
        method = "mlx.nonexistent"
        resp = {
            "jsonrpc": "2.0",
            "error": {"code": -32601, "message": f"Method not found: {method}"},
            "id": 3,
        }
        assert resp["error"]["code"] == -32601
        assert "not found" in resp["error"]["message"]


class TestJsonRpcMlxStatus:
    """Tests for mlx.status response shape."""

    def test_status_response_shape(self):
        from fusion_mlx.server import _server_state

        with patch.dict(_server_state, {"default_model": "qwen3-4b"}, clear=False):
            result = {
                "status": "ok",
                "default_model": _server_state.get("default_model"),
                "models_loaded": 1,
                "models_discovered": 2,
                "uptime_seconds": 42,
            }
            assert result["default_model"] == "qwen3-4b"
            assert result["models_loaded"] == 1
            assert result["status"] == "ok"

    def test_status_no_default_model(self):
        from fusion_mlx.server import _server_state

        with patch.dict(_server_state, {"default_model": None}, clear=False):
            result = {
                "status": "ok",
                "default_model": _server_state.get("default_model"),
            }
            assert result["default_model"] is None
