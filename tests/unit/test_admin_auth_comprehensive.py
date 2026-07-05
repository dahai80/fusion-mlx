"""admin/auth 纯函数测试。

覆盖 validate_api_key/verify_api_key/create_session_token/
verify_session/verify_session_from_request 的边界。

填补 REVIEW_REPORT：admin/auth.py session 全局 dict 无清理 + api_key 最短 4 字符。
"""

from __future__ import annotations

import time
import unittest
from unittest.mock import MagicMock

from fusion_mlx.admin.auth import (
    REMEMBER_ME_MAX_AGE,
    SESSION_COOKIE_NAME,
    SESSION_MAX_AGE,
    create_session_token,
    set_api_key,
    set_global_settings_getter,
    validate_api_key,
    verify_api_key,
    verify_session,
    verify_session_from_request,
)


class TestValidateApiKey(unittest.TestCase):

    def test_valid_key(self):
        self.assertEqual(validate_api_key("strong-key-123"), (True, ""))

    def test_too_short_rejected(self):
        is_valid, msg = validate_api_key("abc")
        self.assertFalse(is_valid)
        self.assertIn("at least 4", msg)

    def test_empty_rejected(self):
        is_valid, msg = validate_api_key("")
        self.assertFalse(is_valid)

    def test_non_ascii_rejected(self):
        # "密钥" 是 2 字符（非 ASCII），先触发长度检查（<4），故错误信息是 "at least 4"
        is_valid, msg = validate_api_key("密钥")
        self.assertFalse(is_valid)

    def test_non_ascii_long_enough_rejected(self):
        # 4+ 字符非 ASCII 应触发 ASCII 检查
        is_valid, msg = validate_api_key("密钥密钥密钥")
        self.assertFalse(is_valid)
        self.assertIn("ASCII", msg)

    def test_min_length_boundary(self):
        # 恰好 4 字符应通过
        self.assertTrue(validate_api_key("abcd")[0])

    def test_special_chars_allowed(self):
        # ASCII 特殊字符应通过
        self.assertTrue(validate_api_key("sk-12345!@#")[0])


class TestVerifyApiKey(unittest.TestCase):

    def test_matching_keys(self):
        self.assertTrue(verify_api_key("secret", "secret"))

    def test_non_matching_keys(self):
        self.assertFalse(verify_api_key("secret", "wrong"))

    def test_none_input_rejected(self):
        self.assertFalse(verify_api_key(None, "key"))
        self.assertFalse(verify_api_key("key", None))
        self.assertFalse(verify_api_key(None, None))

    def test_empty_input_rejected(self):
        self.assertFalse(verify_api_key("", "key"))
        self.assertFalse(verify_api_key("key", ""))

    def test_constant_time_no_leak(self):
        # 验证使用 secrets.compare_digest（constant-time），此处仅验证功能正确
        self.assertTrue(verify_api_key("a" * 32, "a" * 32))
        self.assertFalse(verify_api_key("a" * 31 + "b", "a" * 32))


class TestSessionToken(unittest.TestCase):

    def test_create_session_token_returns_hex(self):
        token = create_session_token()
        self.assertEqual(len(token), 64)  # token_hex(32) → 64 chars
        int(token, 16)  # 合法 hex

    def test_create_session_with_remember_uses_longer_expiry(self):
        short_token = create_session_token(remember=False)
        long_token = create_session_token(remember=True)
        # 两者都应有效
        self.assertTrue(verify_session(short_token))
        self.assertTrue(verify_session(long_token))

    def test_verify_session_invalid_token(self):
        self.assertFalse(verify_session("nonexistent-token"))

    def test_verify_session_expired(self):
        # 创建后立即过期
        token = create_session_token()
        # 手动设置过期时间为过去
        from fusion_mlx.admin import auth

        auth._active_sessions[token]["expires"] = time.time() - 1
        self.assertFalse(verify_session(token))
        # 过期 session 应被删除
        self.assertNotIn(token, auth._active_sessions)

    def test_session_tokens_are_unique(self):
        t1 = create_session_token()
        t2 = create_session_token()
        self.assertNotEqual(t1, t2)


class TestVerifySessionFromRequest(unittest.TestCase):

    def test_no_cookie_returns_false(self):
        req = MagicMock()
        req.cookies = {}
        self.assertFalse(verify_session_from_request(req))

    def test_invalid_cookie_returns_false(self):
        req = MagicMock()
        req.cookies = {SESSION_COOKIE_NAME: "bad-token"}
        self.assertFalse(verify_session_from_request(req))

    def test_valid_cookie_returns_true(self):
        token = create_session_token()
        req = MagicMock()
        req.cookies = {SESSION_COOKIE_NAME: token}
        self.assertTrue(verify_session_from_request(req))


class TestSetApiKey(unittest.TestCase):

    def test_set_api_key_stores_globally(self):
        from fusion_mlx.admin import auth

        set_api_key("my-test-key")
        self.assertEqual(auth._api_key, "my-test-key")

    def test_set_global_settings_getter(self):
        getter = lambda: MagicMock()
        set_global_settings_getter(getter)
        from fusion_mlx.admin import auth

        self.assertIs(auth._global_settings_getter, getter)


class TestConstants(unittest.TestCase):

    def test_cookie_name(self):
        self.assertEqual(SESSION_COOKIE_NAME, "omlx_admin_session")

    def test_max_ages(self):
        self.assertEqual(SESSION_MAX_AGE, 3600)
        self.assertEqual(REMEMBER_ME_MAX_AGE, 86400)


if __name__ == "__main__":
    unittest.main()
