# SPDX-License-Identifier: Apache-2.0
"""Unit tests for admin/auth_routes.py /api/setup-api-key endpoint.

Covers:
- Key validation (length, ASCII, confirmation match) — via unit-level tests
- Missing fields → 422 — via TestClient
- Localhost enforcement — tested via integration tests only
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import FastAPI, Request
from fastapi.testclient import TestClient

from fusion_mlx.admin.auth_routes import router as auth_router


@pytest.fixture
def app():
    _app = FastAPI()
    _app.include_router(auth_router)
    return _app


@pytest.fixture
def client(app):
    return TestClient(app, raise_server_exceptions=False)


class TestSetupApiKeyUnit:
    """Unit-level tests: call the underlying auth_routes functions directly."""

    @pytest.mark.asyncio
    async def test_key_confirmation_mismatch(self):
        from fusion_mlx.admin.auth_routes import setup_api_key
        from fusion_mlx.admin.models import SetupApiKeyRequest

        request_data = SetupApiKeyRequest(api_key="key-one", api_key_confirm="key-two")
        response = MagicMock()
        fastapi_request = AsyncMock(spec=Request)
        fastapi_request.client.host = "127.0.0.1"

        with patch("fusion_mlx.admin.auth_routes._get_global_settings") as mock_gs:
            settings = MagicMock()
            settings.auth.api_key = None
            mock_gs.return_value = settings

            from fastapi import HTTPException

            with pytest.raises(HTTPException) as exc:
                await setup_api_key(request_data, response, fastapi_request)
            assert exc.value.status_code == 400
            assert "do not match" in exc.value.detail.lower()

    @pytest.mark.asyncio
    async def test_key_too_short(self):
        from fusion_mlx.admin.auth_routes import setup_api_key
        from fusion_mlx.admin.models import SetupApiKeyRequest

        request_data = SetupApiKeyRequest(api_key="ab", api_key_confirm="ab")
        response = MagicMock()
        fastapi_request = AsyncMock(spec=Request)
        fastapi_request.client.host = "127.0.0.1"

        with patch("fusion_mlx.admin.auth_routes._get_global_settings") as mock_gs:
            settings = MagicMock()
            settings.auth.api_key = None
            mock_gs.return_value = settings

            from fastapi import HTTPException

            with pytest.raises(HTTPException) as exc:
                await setup_api_key(request_data, response, fastapi_request)
            assert exc.value.status_code == 400
            assert "at least 4" in exc.value.detail.lower()

    @pytest.mark.asyncio
    async def test_key_non_ascii(self):
        from fusion_mlx.admin.auth_routes import setup_api_key
        from fusion_mlx.admin.models import SetupApiKeyRequest

        request_data = SetupApiKeyRequest(
            api_key="\u5bc6\u94a5\u5bc6\u94a5\u5bc6\u94a5",
            api_key_confirm="\u5bc6\u94a5\u5bc6\u94a5\u5bc6\u94a5",
        )
        response = MagicMock()
        fastapi_request = AsyncMock(spec=Request)
        fastapi_request.client.host = "127.0.0.1"

        with patch("fusion_mlx.admin.auth_routes._get_global_settings") as mock_gs:
            settings = MagicMock()
            settings.auth.api_key = None
            mock_gs.return_value = settings

            from fastapi import HTTPException

            with pytest.raises(HTTPException) as exc:
                await setup_api_key(request_data, response, fastapi_request)
            assert exc.value.status_code == 400
            assert "ascii" in exc.value.detail.lower()

    @pytest.mark.asyncio
    async def test_key_already_configured(self):
        from fusion_mlx.admin.auth_routes import setup_api_key
        from fusion_mlx.admin.models import SetupApiKeyRequest

        request_data = SetupApiKeyRequest(
            api_key="new-key-123", api_key_confirm="new-key-123"
        )
        response = MagicMock()
        fastapi_request = AsyncMock(spec=Request)
        fastapi_request.client.host = "127.0.0.1"

        with patch("fusion_mlx.admin.auth_routes._get_global_settings") as mock_gs:
            settings = MagicMock()
            settings.auth.api_key = "already-set"
            mock_gs.return_value = settings

            from fastapi import HTTPException

            with pytest.raises(HTTPException) as exc:
                await setup_api_key(request_data, response, fastapi_request)
            assert exc.value.status_code == 400
            assert "already" in exc.value.detail.lower()

    @pytest.mark.asyncio
    async def test_success(self):
        from fusion_mlx.admin.auth_routes import setup_api_key
        from fusion_mlx.admin.models import SetupApiKeyRequest

        request_data = SetupApiKeyRequest(
            api_key="valid-key-1234", api_key_confirm="valid-key-1234"
        )
        response = MagicMock()
        fastapi_request = AsyncMock(spec=Request)
        fastapi_request.client.host = "127.0.0.1"

        with (
            patch("fusion_mlx.admin.auth_routes._get_global_settings") as mock_gs,
            patch(
                "fusion_mlx.server._server_state",
                {},
            ),
        ):
            settings = MagicMock()
            settings.auth.api_key = None
            settings.save.return_value = None
            mock_gs.return_value = settings

            result = await setup_api_key(request_data, response, fastapi_request)
            assert result["success"] is True
            assert settings.auth.api_key == "valid-key-1234"
            settings.save.assert_called_once()

    @pytest.mark.asyncio
    async def test_non_localhost_returns_403(self):
        from fusion_mlx.admin.auth_routes import setup_api_key
        from fusion_mlx.admin.models import SetupApiKeyRequest

        request_data = SetupApiKeyRequest(api_key="any-key", api_key_confirm="any-key")
        response = MagicMock()
        fastapi_request = AsyncMock(spec=Request)
        fastapi_request.client.host = "192.168.1.1"  # non-loopback

        from fastapi import HTTPException

        with pytest.raises(HTTPException) as exc:
            await setup_api_key(request_data, response, fastapi_request)
        assert exc.value.status_code == 403
        assert "localhost" in exc.value.detail.lower()


class TestSetupApiKeyViaTestClient:
    """Endpoint-level tests via TestClient (validation, schema)."""

    def test_missing_fields_return_422(self, client):
        resp = client.post("/api/setup-api-key", json={})
        assert resp.status_code == 422
