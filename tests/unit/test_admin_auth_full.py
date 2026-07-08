"""admin/auth 完整测试（importlib 绕过 mlx 链，覆盖 require_admin 全分支）。

覆盖 require_admin 的 6 条路径 + extract_session_token +
_get_settings_api_key/_get_settings_sub_keys/_is_skip_api_key_verification。
"""

from __future__ import annotations

import importlib.util
import sys
import time
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch


def _load_module():
    # 先注册一个空 fusion_mlx 包，避免触发 __init__.py 的 mlx 链
    import types

    if "fusion_mlx" not in sys.modules:
        pkg = types.ModuleType("fusion_mlx")
        pkg.__path__ = ["fusion_mlx"]
        sys.modules["fusion_mlx"] = pkg
    if "fusion_mlx.admin" not in sys.modules:
        sub = types.ModuleType("fusion_mlx.admin")
        sub.__path__ = ["fusion_mlx/admin"]
        sys.modules["fusion_mlx.admin"] = sub
    # helpers 桩：require_admin 内部 `from .helpers import _get_global_settings`
    if "fusion_mlx.admin.helpers" not in sys.modules:
        helpers = types.ModuleType("fusion_mlx.admin.helpers")
        helpers._get_global_settings = lambda: None
        sys.modules["fusion_mlx.admin.helpers"] = helpers
    # settings 桩：admin/auth 不直接 import，但 middleware/auth 可能
    if "fusion_mlx.settings" not in sys.modules:
        settings = types.ModuleType("fusion_mlx.settings")
        sys.modules["fusion_mlx.settings"] = settings
    spec = importlib.util.spec_from_file_location(
        "fusion_mlx.admin.auth", "fusion_mlx/admin/auth.py"
    )
    m = importlib.util.module_from_spec(spec)
    _prev_auth = sys.modules.get("fusion_mlx.admin.auth")
    sys.modules["fusion_mlx.admin.auth"] = m
    spec.loader.exec_module(m)
    # Restore sys.modules so this standalone load does not leak a duplicate
    # module instance into the rest of the pytest session (it broke telemetry
    # and admin-profiles tests via identity/import mismatches).
    if _prev_auth is not None:
        sys.modules["fusion_mlx.admin.auth"] = _prev_auth
    else:
        sys.modules.pop("fusion_mlx.admin.auth", None)
    return m


auth = _load_module()


def _make_request(cookies=None, auth_header=None):
    req = MagicMock()
    req.cookies = cookies or {}
    if auth_header is not None:
        req.headers = {"authorization": auth_header}
    else:
        req.headers = {}
    req.query_params = {}
    return req


class TestRequireAdmin(unittest.IsolatedAsyncioTestCase):

    def setUp(self):
        auth._active_sessions.clear()
        auth._api_key = ""

    async def test_skip_api_key_verification_returns_true(self):
        gs = SimpleNamespace(
            auth=SimpleNamespace(skip_api_key_verification=True, api_key="k")
        )
        with patch.object(auth, "_is_skip_api_key_verification", return_value=True):
            result = await auth.require_admin(_make_request())
        self.assertTrue(result)

    async def test_valid_session_cookie_returns_true(self):
        token = auth.create_session_token()
        req = _make_request(cookies={auth.SESSION_COOKIE_NAME: token})
        result = await auth.require_admin(req)
        self.assertTrue(result)

    async def test_bearer_matches_api_key_returns_true(self):
        auth.set_api_key("my-secret-key")
        req = _make_request(auth_header="Bearer my-secret-key")
        result = await auth.require_admin(req)
        self.assertTrue(result)

    async def test_bearer_matches_settings_api_key(self):
        gs = SimpleNamespace(
            auth=SimpleNamespace(
                api_key="settings-key", sub_keys=[], skip_api_key_verification=False
            )
        )
        with patch("fusion_mlx.admin.helpers._get_global_settings", return_value=gs):
            auth._api_key = ""  # 不走 _api_key 路径
            req = _make_request(auth_header="Bearer settings-key")
            result = await auth.require_admin(req)
        self.assertTrue(result)

    async def test_bearer_matches_sub_key(self):
        sub_key = SimpleNamespace(key="sub-key-123")
        gs = SimpleNamespace(
            auth=SimpleNamespace(
                api_key="main-key", sub_keys=[sub_key], skip_api_key_verification=False
            )
        )
        with patch("fusion_mlx.admin.helpers._get_global_settings", return_value=gs):
            auth._api_key = ""
            req = _make_request(auth_header="Bearer sub-key-123")
            result = await auth.require_admin(req)
        self.assertTrue(result)

    async def test_no_auth_raises_401(self):
        with self.assertRaises(Exception) as cm:
            await auth.require_admin(_make_request())
        # HTTPException
        self.assertEqual(getattr(cm.exception, "status_code", None), 401)

    async def test_invalid_bearer_raises_401(self):
        auth.set_api_key("correct-key")
        with patch("fusion_mlx.admin.helpers._get_global_settings", return_value=None):
            req = _make_request(auth_header="Bearer wrong-key")
            with self.assertRaises(Exception) as cm:
                await auth.require_admin(req)
        self.assertEqual(getattr(cm.exception, "status_code", None), 401)

    async def test_non_bearer_auth_header_raises_401(self):
        auth.set_api_key("key")
        req = _make_request(auth_header="Basic abc123")
        with self.assertRaises(Exception) as cm:
            await auth.require_admin(req)
        self.assertEqual(getattr(cm.exception, "status_code", None), 401)


class TestExtractSessionToken(unittest.TestCase):

    def test_returns_cookie_value(self):
        req = MagicMock()
        req.cookies = {auth.SESSION_COOKIE_NAME: "token123"}
        self.assertEqual(auth.extract_session_token(req), "token123")

    def test_returns_none_when_no_cookie(self):
        req = MagicMock()
        req.cookies = {}
        self.assertIsNone(auth.extract_session_token(req))


class TestSettingsHelpers(unittest.TestCase):

    def test_get_settings_api_key_none(self):
        self.assertEqual(auth._get_settings_api_key(None), "")

    def test_get_settings_api_key_from_auth_attr(self):
        gs = SimpleNamespace(auth=SimpleNamespace(api_key="from-auth"))
        self.assertEqual(auth._get_settings_api_key(gs), "from-auth")

    def test_get_settings_api_key_from_flat_attr(self):
        gs = SimpleNamespace(api_key="flat-key")
        gs.auth = None
        self.assertEqual(auth._get_settings_api_key(gs), "flat-key")

    def test_get_settings_api_key_empty(self):
        gs = SimpleNamespace(auth=SimpleNamespace(api_key=""))
        self.assertEqual(auth._get_settings_api_key(gs), "")

    def test_get_settings_sub_keys_none(self):
        self.assertEqual(auth._get_settings_sub_keys(None), [])

    def test_get_settings_sub_keys_from_auth(self):
        sk = [SimpleNamespace(key="k1"), SimpleNamespace(key="k2")]
        gs = SimpleNamespace(auth=SimpleNamespace(sub_keys=sk))
        self.assertEqual(auth._get_settings_sub_keys(gs), sk)

    def test_get_settings_sub_keys_from_flat(self):
        sk = [SimpleNamespace(key="k1")]
        gs = SimpleNamespace(sub_keys=sk)
        gs.auth = None
        self.assertEqual(auth._get_settings_sub_keys(gs), sk)

    def test_is_skip_api_key_verification_none_gs(self):
        self.assertFalse(auth._is_skip_api_key_verification(None))

    def test_is_skip_api_key_verification_true(self):
        gs = SimpleNamespace(auth=SimpleNamespace(skip_api_key_verification=True))
        self.assertTrue(auth._is_skip_api_key_verification(gs))

    def test_is_skip_api_key_verification_false(self):
        gs = SimpleNamespace(auth=SimpleNamespace(skip_api_key_verification=False))
        self.assertFalse(auth._is_skip_api_key_verification(gs))

    def test_is_skip_from_global_settings_dict(self):
        gs = SimpleNamespace()
        gs.auth = None
        gs.global_settings = {"skip_api_key_verification": True}
        self.assertTrue(auth._is_skip_api_key_verification(gs))


class TestValidateApiKey(unittest.TestCase):

    def test_valid_key(self):
        self.assertEqual(auth.validate_api_key("strong-key-123"), (True, ""))

    def test_too_short_rejected(self):
        is_valid, msg = auth.validate_api_key("abc")
        self.assertFalse(is_valid)
        self.assertIn("at least 4", msg)

    def test_empty_rejected(self):
        self.assertFalse(auth.validate_api_key("")[0])

    def test_non_ascii_short_rejected_for_length(self):
        is_valid, _ = auth.validate_api_key("密钥")
        self.assertFalse(is_valid)

    def test_non_ascii_long_rejected(self):
        is_valid, msg = auth.validate_api_key("密钥密钥密钥")
        self.assertFalse(is_valid)
        self.assertIn("ASCII", msg)

    def test_min_length_boundary(self):
        self.assertTrue(auth.validate_api_key("abcd")[0])

    def test_special_chars_allowed(self):
        self.assertTrue(auth.validate_api_key("sk-12345!@#")[0])


class TestVerifyApiKey(unittest.TestCase):

    def test_matching_keys(self):
        self.assertTrue(auth.verify_api_key("secret", "secret"))

    def test_non_matching_keys(self):
        self.assertFalse(auth.verify_api_key("secret", "wrong"))

    def test_none_input_rejected(self):
        self.assertFalse(auth.verify_api_key(None, "key"))
        self.assertFalse(auth.verify_api_key("key", None))

    def test_empty_input_rejected(self):
        self.assertFalse(auth.verify_api_key("", "key"))

    def test_constant_time_no_leak(self):
        self.assertTrue(auth.verify_api_key("a" * 32, "a" * 32))
        self.assertFalse(auth.verify_api_key("a" * 31 + "b", "a" * 32))


class TestSessionToken(unittest.TestCase):

    def setUp(self):
        auth._active_sessions.clear()

    def test_create_returns_hex(self):
        token = auth.create_session_token()
        self.assertEqual(len(token), 64)
        int(token, 16)

    def test_remember_uses_longer_expiry(self):
        short = auth.create_session_token(remember=False)
        long_ = auth.create_session_token(remember=True)
        self.assertTrue(auth.verify_session(short))
        self.assertTrue(auth.verify_session(long_))

    def test_verify_invalid_token(self):
        self.assertFalse(auth.verify_session("nonexistent"))

    def test_verify_expired_deletes_session(self):
        token = auth.create_session_token()
        auth._active_sessions[token]["expires"] = time.time() - 1
        self.assertFalse(auth.verify_session(token))
        self.assertNotIn(token, auth._active_sessions)

    def test_tokens_unique(self):
        self.assertNotEqual(auth.create_session_token(), auth.create_session_token())


class TestVerifySessionFromRequest(unittest.TestCase):

    def setUp(self):
        auth._active_sessions.clear()

    def test_no_cookie_false(self):
        self.assertFalse(auth.verify_session_from_request(_make_request()))

    def test_invalid_cookie_false(self):
        req = _make_request(cookies={auth.SESSION_COOKIE_NAME: "bad"})
        self.assertFalse(auth.verify_session_from_request(req))

    def test_valid_cookie_true(self):
        token = auth.create_session_token()
        req = _make_request(cookies={auth.SESSION_COOKIE_NAME: token})
        self.assertTrue(auth.verify_session_from_request(req))


class TestSetters(unittest.TestCase):

    def test_set_api_key(self):
        auth.set_api_key("my-key")
        self.assertEqual(auth._api_key, "my-key")

    def test_set_global_settings_getter(self):
        g = lambda: None
        auth.set_global_settings_getter(g)
        self.assertIs(auth._global_settings_getter, g)


class TestConstants(unittest.TestCase):

    def test_cookie_name(self):
        self.assertEqual(auth.SESSION_COOKIE_NAME, "fusionmlx_admin_session")

    def test_max_ages(self):
        self.assertEqual(auth.SESSION_MAX_AGE, 3600)
        self.assertEqual(auth.REMEMBER_ME_MAX_AGE, 86400)


if __name__ == "__main__":
    unittest.main()
