from unittest.mock import MagicMock

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
        is_valid, error = validate_api_key("Ab1")
        assert is_valid is False
        assert "8 characters" in error

    def test_no_uppercase(self):
        is_valid, error = validate_api_key("abcdefg1")
        assert is_valid is False
        assert "uppercase" in error

    def test_no_lowercase(self):
        is_valid, error = validate_api_key("ABCDEFG1")
        assert is_valid is False
        assert "lowercase" in error

    def test_no_digit(self):
        is_valid, error = validate_api_key("Abcdefgh")
        assert is_valid is False
        assert "digit" in error


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
        self.scope = {}
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
    def test_allows_valid_session(self):
        set_api_key("Test1234key")
        token = create_session_token()

        @require_admin
        def protected(request):
            return "ok"

        mock_req = _MockRequest(token=token)
        assert protected(mock_req) == "ok"

    def test_allows_valid_api_key(self):
        set_api_key("Test1234key")

        @require_admin
        def protected(request):
            return "ok"

        mock_req = _MockRequest(auth_header="Bearer Test1234key")
        assert protected(mock_req) == "ok"

    def test_rejects_no_auth(self):
        set_api_key("Test1234key")

        @require_admin
        def protected(request):
            return "ok"

        mock_req = _MockRequest()
        from fastapi import HTTPException
        try:
            protected(mock_req)
            assert False, "Should have raised"
        except HTTPException as e:
            assert e.status_code == 401
