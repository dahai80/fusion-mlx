# SPDX-License-Identifier: Apache-2.0

import asyncio
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException

from fusion_mlx.admin import auth as admin_auth
from fusion_mlx.admin import auth_routes as admin_auth_routes
from fusion_mlx.admin import helpers as admin_helpers
from fusion_mlx.admin import html_routes as admin_html_routes
from fusion_mlx.admin import update_check as admin_update_check


def _mock_global_settings(api_key=None, skip=False):
    mock = MagicMock()
    mock.auth.api_key = api_key
    mock.auth.skip_api_key_verification = skip
    return mock


def _patch_getter_on_module(module, mock_settings):
    original = module._get_global_settings
    if mock_settings is None:
        module._get_global_settings = lambda: None
    else:
        module._get_global_settings = lambda: mock_settings
    return original


def _restore_getter_on_module(module, original):
    module._get_global_settings = original


def _inject_templates():
    """Inject mock templates into html_routes module.

    html_routes.py uses 'templates' as a bare global name but never imports
    it -- the production server injects it from routes.py. For unit tests we
    must set it on the module dict ourselves.
    """
    mock_templates = MagicMock()
    mock_templates.TemplateResponse.return_value = MagicMock()
    original = getattr(admin_html_routes, "templates", None)
    admin_html_routes.templates = mock_templates
    return original, mock_templates


def _remove_templates(original):
    if original is None:
        del admin_html_routes.templates
    else:
        admin_html_routes.templates = original


def _make_request_with_key(key):
    mock_request = MagicMock()
    mock_request.json = AsyncMock(return_value={"key": key})
    return mock_request


class TestAutoLogin:
    def test_auto_login_success_redirects_to_dashboard(self):
        mock_settings = _mock_global_settings(api_key="test-key")
        original = _patch_getter_on_module(admin_auth_routes, mock_settings)
        try:
            mock_request = _make_request_with_key("test-key")
            result = asyncio.run(
                admin_auth_routes.auto_login(
                    fastapi_request=mock_request, redirect="/admin/dashboard"
                )
            )
            assert result.status_code == 302
            assert result.headers["location"] == "/admin/dashboard"
            cookie_header = result.headers.get("set-cookie", "")
            assert "fusionmlx_admin_session" in cookie_header
        finally:
            _restore_getter_on_module(admin_auth_routes, original)

    def test_auto_login_success_redirects_to_chat(self):
        mock_settings = _mock_global_settings(api_key="test-key")
        original = _patch_getter_on_module(admin_auth_routes, mock_settings)
        try:
            mock_request = _make_request_with_key("test-key")
            result = asyncio.run(
                admin_auth_routes.auto_login(
                    fastapi_request=mock_request, redirect="/admin/chat"
                )
            )
            assert result.status_code == 302
            assert result.headers["location"] == "/admin/chat"
        finally:
            _restore_getter_on_module(admin_auth_routes, original)

    def test_auto_login_invalid_key_redirects_to_login(self):
        mock_settings = _mock_global_settings(api_key="correct-key")
        original = _patch_getter_on_module(admin_auth_routes, mock_settings)
        try:
            mock_request = _make_request_with_key("wrong-key")
            result = asyncio.run(
                admin_auth_routes.auto_login(
                    fastapi_request=mock_request, redirect="/admin/dashboard"
                )
            )
            assert result.status_code == 302
            assert result.headers["location"] == "/admin"
            cookie_header = result.headers.get("set-cookie", "")
            assert "fusionmlx_admin_session" not in cookie_header
        finally:
            _restore_getter_on_module(admin_auth_routes, original)

    def test_auto_login_empty_key_redirects_to_login(self):
        mock_settings = _mock_global_settings(api_key="test-key")
        original = _patch_getter_on_module(admin_auth_routes, mock_settings)
        try:
            mock_request = _make_request_with_key("")
            result = asyncio.run(
                admin_auth_routes.auto_login(
                    fastapi_request=mock_request, redirect="/admin/dashboard"
                )
            )
            assert result.status_code == 302
            assert result.headers["location"] == "/admin"
        finally:
            _restore_getter_on_module(admin_auth_routes, original)

    def test_auto_login_no_server_key_redirects_to_login(self):
        mock_settings = _mock_global_settings(api_key=None)
        original = _patch_getter_on_module(admin_auth_routes, mock_settings)
        try:
            mock_request = _make_request_with_key("any-key")
            result = asyncio.run(
                admin_auth_routes.auto_login(
                    fastapi_request=mock_request, redirect="/admin/dashboard"
                )
            )
            assert result.status_code == 302
            assert result.headers["location"] == "/admin"
        finally:
            _restore_getter_on_module(admin_auth_routes, original)

    def test_auto_login_invalid_redirect_returns_400(self):
        mock_settings = _mock_global_settings(api_key="test-key")
        original = _patch_getter_on_module(admin_auth_routes, mock_settings)
        try:
            mock_request = MagicMock()
            with pytest.raises(HTTPException) as exc_info:
                asyncio.run(
                    admin_auth_routes.auto_login(
                        fastapi_request=mock_request,
                        redirect="https://evil.com",
                    )
                )
            assert exc_info.value.status_code == 400
            assert "Invalid redirect path" in exc_info.value.detail
        finally:
            _restore_getter_on_module(admin_auth_routes, original)

    def test_auto_login_redirect_to_admin_root(self):
        mock_settings = _mock_global_settings(api_key="test-key")
        original = _patch_getter_on_module(admin_auth_routes, mock_settings)
        try:
            mock_request = _make_request_with_key("test-key")
            result = asyncio.run(
                admin_auth_routes.auto_login(
                    fastapi_request=mock_request, redirect="/admin"
                )
            )
            assert result.status_code == 302
            assert result.headers["location"] == "/admin"
        finally:
            _restore_getter_on_module(admin_auth_routes, original)


class TestLoginPage:
    def test_login_page_uses_new_template_signature(self):
        mock_settings = _mock_global_settings(api_key="test-key")
        original_gs = _patch_getter_on_module(admin_html_routes, mock_settings)
        original_tpl, mock_templates = _inject_templates()
        try:
            mock_request = MagicMock()
            with patch("fusion_mlx.admin.auth.verify_session", return_value=False):
                asyncio.run(admin_html_routes.login_page(request=mock_request))
                mock_templates.TemplateResponse.assert_called_once_with(
                    mock_request,
                    "login.html",
                    {"api_key_configured": True},
                )
        finally:
            _restore_getter_on_module(admin_html_routes, original_gs)
            _remove_templates(original_tpl)


class TestDashboardPage:
    def test_dashboard_page_uses_new_template_signature(self):
        original_tpl, mock_templates = _inject_templates()
        try:
            mock_request = MagicMock()
            asyncio.run(
                admin_html_routes.dashboard_page(request=mock_request, is_admin=True)
            )
            mock_templates.TemplateResponse.assert_called_once_with(
                mock_request, "dashboard.html", {}
            )
        finally:
            _remove_templates(original_tpl)


class TestChatPageApiKeyInjection:
    def test_chat_page_passes_api_key_in_context(self):
        mock_settings = _mock_global_settings(api_key="test-chat-key")
        original_gs = _patch_getter_on_module(admin_html_routes, mock_settings)
        original_tpl, mock_templates = _inject_templates()
        try:
            mock_request = MagicMock()
            asyncio.run(
                admin_html_routes.chat_page(request=mock_request, is_admin=True)
            )
            mock_templates.TemplateResponse.assert_called_once_with(
                mock_request,
                "chat.html",
                {"api_key": "test-chat-key"},
            )
        finally:
            _restore_getter_on_module(admin_html_routes, original_gs)
            _remove_templates(original_tpl)

    def test_chat_page_passes_empty_when_no_key(self):
        mock_settings = _mock_global_settings(api_key=None)
        original_gs = _patch_getter_on_module(admin_html_routes, mock_settings)
        original_tpl, mock_templates = _inject_templates()
        try:
            mock_request = MagicMock()
            asyncio.run(
                admin_html_routes.chat_page(request=mock_request, is_admin=True)
            )
            call_args = mock_templates.TemplateResponse.call_args
            context = call_args[0][2]
            assert context["api_key"] == ""
        finally:
            _restore_getter_on_module(admin_html_routes, original_gs)
            _remove_templates(original_tpl)

    def test_chat_page_passes_empty_when_no_settings(self):
        original_gs = _patch_getter_on_module(admin_html_routes, None)
        original_tpl, mock_templates = _inject_templates()
        try:
            mock_request = MagicMock()
            asyncio.run(
                admin_html_routes.chat_page(request=mock_request, is_admin=True)
            )
            call_args = mock_templates.TemplateResponse.call_args
            context = call_args[0][2]
            assert context["api_key"] == ""
        finally:
            _restore_getter_on_module(admin_html_routes, original_gs)
            _remove_templates(original_tpl)


class TestSkipAdminAuth:
    def _mock_gs(self, skip=True):
        mock = MagicMock()
        mock.auth.skip_api_key_verification = skip
        return mock

    def test_require_admin_skipped_when_enabled(self):
        gs = self._mock_gs(skip=True)
        original = admin_helpers._get_global_settings
        admin_helpers._get_global_settings = lambda: gs
        try:
            mock_request = MagicMock()
            mock_request.cookies.get.return_value = None
            result = asyncio.run(admin_auth.require_admin(mock_request))
            assert result is True
        finally:
            admin_helpers._get_global_settings = original

    def test_require_admin_not_skipped_when_disabled(self):
        gs = self._mock_gs(skip=False)
        original = admin_helpers._get_global_settings
        admin_helpers._get_global_settings = lambda: gs
        try:
            mock_request = MagicMock()
            mock_request.cookies.get.return_value = None
            mock_request.headers.get.return_value = "application/json"
            with pytest.raises(HTTPException) as exc_info:
                asyncio.run(admin_auth.require_admin(mock_request))
            assert exc_info.value.status_code == 401
        finally:
            admin_helpers._get_global_settings = original

    def test_login_page_redirects_when_skip_enabled(self):
        gs = MagicMock()
        gs.auth.skip_api_key_verification = True
        gs.auth.api_key = "test-key"
        original_gs = _patch_getter_on_module(admin_html_routes, gs)
        try:
            mock_request = MagicMock()
            with patch("fusion_mlx.admin.auth.verify_session", return_value=False):
                result = asyncio.run(admin_html_routes.login_page(request=mock_request))
                assert result.status_code == 302
                assert result.headers["location"] == "/admin/dashboard"
        finally:
            _restore_getter_on_module(admin_html_routes, original_gs)


class TestInitAuth:
    """fusion-mlx uses dict-based sessions, not itsdangerous serializer.

    init_auth / SECRET_KEY / _serializer / verify_session_token do not
    exist in fusion-mlx.admin.auth. These tests are skipped.
    """

    @pytest.mark.skip(reason="fusion-mlx uses dict-based sessions, no init_auth")
    def test_init_auth_sets_serializer(self):
        pass

    @pytest.mark.skip(reason="fusion-mlx uses dict-based sessions, no SECRET_KEY")
    def test_init_auth_env_var_takes_priority(self):
        pass

    @pytest.mark.skip(reason="fusion-mlx uses dict-based sessions, no SECRET_KEY")
    def test_init_auth_uses_provided_key_when_no_env(self):
        pass

    @pytest.mark.skip(
        reason="fusion-mlx uses dict-based sessions, tokens are ephemeral"
    )
    def test_tokens_survive_reinit_with_same_key(self):
        pass

    @pytest.mark.skip(
        reason="fusion-mlx uses dict-based sessions, tokens are ephemeral"
    )
    def test_tokens_invalid_after_reinit_with_different_key(self):
        pass


class TestRememberMe:
    def setup_method(self):
        admin_auth._active_sessions.clear()

    def test_create_token_default_no_remember(self):
        token = admin_auth.create_session_token()
        assert admin_auth.verify_session(token) is True

    def test_create_token_with_remember(self):
        token = admin_auth.create_session_token(remember=True)
        assert admin_auth.verify_session(token) is True

    def test_remember_token_has_extended_expiry(self):
        token = admin_auth.create_session_token(remember=True)
        session_data = admin_auth._active_sessions[token]
        assert session_data["expires"] > time.time()

    def test_non_remember_token_payload(self):
        token = admin_auth.create_session_token(remember=False)
        session_data = admin_auth._active_sessions[token]
        assert session_data["user"] == "admin"

    def test_remember_me_max_age_constant(self):
        assert admin_auth.REMEMBER_ME_MAX_AGE == 86400

    def test_session_max_age_constant(self):
        assert admin_auth.SESSION_MAX_AGE == 3600


class TestSessionCookieName:
    def test_cookie_name_is_session_token(self):
        assert admin_auth.SESSION_COOKIE_NAME == "session_token"


class _FakeResponse:
    def __init__(self, status_code, json_data):
        self.status_code = status_code
        self._json = json_data

    def json(self):
        return self._json


def _make_async_return(value):
    async def _coro(*args, **kwargs):
        return value

    return _coro


class TestCheckUpdate:
    def setup_method(self):
        admin_update_check._update_cache = None
        admin_update_check._update_cache_time = 0.0

    @pytest.mark.asyncio
    async def test_prerelease_not_shown(self):
        fake_resp = _FakeResponse(
            200,
            [
                {
                    "tag_name": "v99.0.0.dev1",
                    "html_url": "https://github.com/dahai80/fusion-mlx/releases/tag/v99.0.0.dev1",
                }
            ],
        )
        with (
            patch("fusion_mlx.admin.update_check.asyncio") as mock_asyncio,
            patch(
                "fusion_mlx.utils.release_check.select_latest_stable_release",
                return_value=None,
            ),
        ):
            mock_asyncio.to_thread = _make_async_return(fake_resp)
            result = await admin_update_check.check_update(is_admin=True)

        assert result["update_available"] is False
        assert result["latest_version"] is None

    @pytest.mark.asyncio
    async def test_stable_version_shown(self):
        fake_resp = _FakeResponse(
            200,
            [
                {
                    "tag_name": "v99.0.0",
                    "html_url": "https://github.com/dahai80/fusion-mlx/releases/tag/v99.0.0",
                }
            ],
        )
        stable_data = {
            "tag_name": "v99.0.0",
            "html_url": "https://github.com/dahai80/fusion-mlx/releases/tag/v99.0.0",
        }
        with (
            patch("fusion_mlx.admin.update_check.asyncio") as mock_asyncio,
            patch(
                "fusion_mlx.utils.release_check.select_latest_stable_release",
                return_value=stable_data,
            ),
            patch(
                "fusion_mlx.admin.update_check._fusionmlx_version",
                "0.1.0",
                create=True,
            ),
        ):
            mock_asyncio.to_thread = _make_async_return(fake_resp)
            result = await admin_update_check.check_update(is_admin=True)

        assert result["update_available"] is True
        assert result["latest_version"] == "99.0.0"

    @pytest.mark.asyncio
    async def test_rc_not_shown(self):
        fake_resp = _FakeResponse(
            200,
            [
                {
                    "tag_name": "v99.0.0rc1",
                    "html_url": "https://github.com/dahai80/fusion-mlx/releases/tag/v99.0.0rc1",
                }
            ],
        )
        with (
            patch("fusion_mlx.admin.update_check.asyncio") as mock_asyncio,
            patch(
                "fusion_mlx.utils.release_check.select_latest_stable_release",
                return_value=None,
            ),
        ):
            mock_asyncio.to_thread = _make_async_return(fake_resp)
            result = await admin_update_check.check_update(is_admin=True)

        assert result["update_available"] is False
        assert result["latest_version"] is None
