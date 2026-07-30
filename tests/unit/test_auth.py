# SPDX-License-Identifier: Apache-2.0
import pytest
from unittest.mock import MagicMock
from fastapi import HTTPException

from fastapi import Request as FastAPIRequest

from fusion_mlx.admin.auth import (
    create_session_token,
    extract_session_token,
    require_admin,
    set_api_key,
    validate_api_key,
    verify_api_key,
    verify_session,
)


class TestSetApiKey:
    def test_set_and_retrieve(self):
        set_api_key("Test1234key")
        assert verify_api_key("Test1234key", "Test1234key") is True
        assert verify_api_key("wrong", "Test1234key") is False


class TestValidateApiKey:
    def test_valid_key_returns_tuple(self):
        is_valid, error = validate_api_key("Valid1key")
        assert is_valid is True
        assert error == ""

    def test_too_short(self):
        is_valid, error = validate_api_key("Ab")
        assert is_valid is False
        assert "4 characters" in error

    def test_non_ascii(self):
        is_valid, error = validate_api_key("éééé")
        assert is_valid is False
        assert "ASCII" in error


class TestVerifyApiKey:
    def test_matching_keys(self):
        assert verify_api_key("abcDEF12", "abcDEF12") is True

    def test_different_keys(self):
        assert verify_api_key("abcDEF12", "xyzDEF12") is False

    def test_none_key(self):
        assert verify_api_key(None, "something") is False

    def test_empty_expected(self):
        assert verify_api_key("something", "") is False


class TestVerifySession:
    def test_valid_session(self):
        token = create_session_token()
        assert verify_session(token) is True

    def test_invalid_token(self):
        assert verify_session("nonexistent") is False


class TestExtractSessionToken:
    def test_extract_from_request(self):
        mock_req = MagicMock()
        mock_req.cookies.get.return_value = "my-token-123"
        assert extract_session_token(mock_req) == "my-token-123"

    def test_missing_token(self):
        mock_req = MagicMock()
        mock_req.cookies.get.return_value = None
        assert extract_session_token(mock_req) is None


class _MockRequest(FastAPIRequest):
    """Minimal Request subclass that passes isinstance check without ASGI setup."""

    def __init__(self, token=None, auth_header=""):
        self.scope = {"query_string": b"", "type": "http"}
        self._cookies = MagicMock()
        self._cookies.get.return_value = token
        self._headers = MagicMock()
        self._headers.get.return_value = auth_header

    @property
    def cookies(self):
        return self._cookies

    @property
    def headers(self):
        return self._headers


class TestRequireAdmin:
    import asyncio

    def test_allows_valid_session(self):
        set_api_key("Test1234key")
        token = create_session_token()
        mock_req = _MockRequest(token=token)
        result = self.asyncio.run(require_admin(mock_req))
        assert result is True

    def test_allows_valid_api_key(self):
        set_api_key("Test1234key")
        mock_req = _MockRequest(auth_header="Bearer Test1234key")
        result = self.asyncio.run(require_admin(mock_req))
        assert result is True

    def test_rejects_no_auth(self):
        set_api_key("Test1234key")
        mock_req = _MockRequest()
        from fastapi import HTTPException

        try:
            self.asyncio.run(require_admin(mock_req))
            assert False, "Should have raised"
        except HTTPException as e:
            assert e.status_code == 401


class TestMiddlewareAuth:
    def test_no_configured_key_allows_anonymous(self):
        from unittest.mock import patch

        from fusion_mlx.middleware.auth import _verify_api_key_values

        with patch(
            "fusion_mlx.middleware.auth._get_configured_api_key",
            return_value=None,
        ):
            assert _verify_api_key_values() is True

    def test_configured_key_requires_matching_key(self):
        import secrets
        from unittest.mock import patch

        from fusion_mlx.middleware.auth import _verify_api_key_values

        key = secrets.token_hex(16)
        with patch(
            "fusion_mlx.middleware.auth._get_configured_api_key",
            return_value=key,
        ):
            assert _verify_api_key_values(key) is True

    def test_configured_key_rejects_wrong_key(self):
        from unittest.mock import patch

        from fusion_mlx.middleware.auth import _verify_api_key_values

        with patch(
            "fusion_mlx.middleware.auth._get_configured_api_key",
            return_value="correct",
        ):
            with pytest.raises(HTTPException) as exc_info:
                _verify_api_key_values("wrong")
            assert exc_info.value.status_code == 401

    def test_configured_key_rejects_no_key(self):
        from unittest.mock import patch

        from fusion_mlx.middleware.auth import _verify_api_key_values

        with patch(
            "fusion_mlx.middleware.auth._get_configured_api_key",
            return_value="secret",
        ):
            with pytest.raises(HTTPException) as exc_info:
                _verify_api_key_values()
            assert exc_info.value.status_code == 401
