# SPDX-License-Identifier: Apache-2.0
"""Session cleanup and boundary tests for admin/auth.py.

Covers:
- Session expiration and cleanup behavior
- Remember-me vs default TTL
- Session dict growth under idle/expired tokens
"""

from __future__ import annotations

import time
from unittest.mock import MagicMock

from fusion_mlx.admin.auth import (
    REMEMBER_ME_MAX_AGE,
    SESSION_MAX_AGE,
    create_session_token,
    verify_session,
    verify_session_from_request,
)

# Direct access for state inspection
from fusion_mlx.admin.auth import _active_sessions as _orig_active_sessions


def _clear_sessions():
    _orig_active_sessions.clear()


class TestSessionCleanup:
    """Session expiration and cleanup behavior."""

    def setup_method(self):
        _clear_sessions()

    def test_expired_session_removed_on_verify(self, monkeypatch):
        """verify_session cleans the checked token if expired."""
        token = create_session_token()
        monkeypatch.setattr(time, "time", lambda: 999999999999.0)  # far future
        assert verify_session(token) is False  # expired
        assert token not in _orig_active_sessions  # cleaned

    def test_active_session_preserved(self):
        token = create_session_token()
        assert verify_session(token) is True
        assert token in _orig_active_sessions

    def test_unknown_token_returns_false(self):
        assert verify_session("nonexistent-token") is False

    def test_no_state_mutation_for_unknown_token(self):
        original_len = len(_orig_active_sessions)
        verify_session("nonexistent-token")
        assert len(_orig_active_sessions) == original_len

    def test_expired_session_removed_from_request(self, monkeypatch):
        token = create_session_token()
        monkeypatch.setattr(time, "time", lambda: 999999999999.0)
        mock_req = MagicMock()
        mock_req.cookies.get.return_value = token
        assert verify_session_from_request(mock_req) is False
        assert token not in _orig_active_sessions

    def test_active_session_from_request(self):
        token = create_session_token()
        mock_req = MagicMock()
        mock_req.cookies.get.return_value = token
        assert verify_session_from_request(mock_req) is True

    def test_no_cookie_returns_false(self):
        mock_req = MagicMock()
        mock_req.cookies.get.return_value = None
        assert verify_session_from_request(mock_req) is False

    def test_remember_me_uses_longer_ttl(self, monkeypatch):
        fake_now = 1000000.0
        monkeypatch.setattr(time, "time", lambda: fake_now)
        token = create_session_token(remember=True)
        expected_expiry = fake_now + REMEMBER_ME_MAX_AGE
        assert _orig_active_sessions[token]["expires"] == expected_expiry

    def test_default_session_uses_1h(self, monkeypatch):
        fake_now = 2000000.0
        monkeypatch.setattr(time, "time", lambda: fake_now)
        token = create_session_token()  # remember=False
        expected_expiry = fake_now + SESSION_MAX_AGE
        assert _orig_active_sessions[token]["expires"] == expected_expiry

    def test_session_expiry_strict_greater(self, monkeypatch):
        """Session is valid when time == expires (strictly greater check)."""
        fake_now = 3000000.0
        monkeypatch.setattr(time, "time", lambda: fake_now)
        token = create_session_token()
        _orig_active_sessions[token]["expires"] = fake_now  # exactly now
        # Not expired because now > expires is False (strict >)
        assert verify_session(token) is True

    def test_session_not_cleaned_by_other_session_verify(self, monkeypatch):
        """verify_session only cleans the token being checked, not all expired."""
        fake_now = 1000000.0
        monkeypatch.setattr(time, "time", lambda: fake_now)
        expired_token = create_session_token()
        _orig_active_sessions[expired_token]["expires"] = fake_now - 1

        active_token = create_session_token()

        # Verifying the active token should NOT clean the expired one
        assert verify_session(active_token) is True
        assert expired_token in _orig_active_sessions  # still there!

        # Only verifying the expired token cleans it
        monkeypatch.setattr(time, "time", lambda: fake_now)
        assert verify_session(expired_token) is False
        assert expired_token not in _orig_active_sessions
