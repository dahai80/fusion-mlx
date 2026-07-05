# SPDX-License-Identifier: Apache-2.0
"""Unit tests for middleware/auth.py API key helper functions.

Covers:
- _get_configured_api_key core logic (resilience to missing/invalid settings)
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

from fusion_mlx.middleware.auth import _get_configured_api_key


class TestGetConfiguredApiKey:
    """_get_configured_api_key reads the configured key via admin helpers."""

    def test_returns_none_when_settings_unavailable(self):
        """No global settings registered -> None."""
        with patch("fusion_mlx.admin.helpers._get_global_settings", return_value=None):
            result = _get_configured_api_key()
        assert result is None

    def test_returns_configured_key(self):
        """Real key on settings.auth.api_key is returned."""
        settings = SimpleNamespace(auth=SimpleNamespace(api_key="sk-real-1234"))
        with patch("fusion_mlx.admin.helpers._get_global_settings", return_value=settings):
            result = _get_configured_api_key()
        assert result == "sk-real-1234"

    def test_returns_none_when_auth_missing(self):
        """settings.auth is None -> None."""
        settings = SimpleNamespace(auth=None)
        with patch("fusion_mlx.admin.helpers._get_global_settings", return_value=settings):
            result = _get_configured_api_key()
        assert result is None

    def test_exception_safety(self):
        """Exception inside the helper does not propagate."""
        with patch(
            "fusion_mlx.admin.helpers._get_global_settings",
            side_effect=RuntimeError("boom"),
        ):
            result = _get_configured_api_key()
        assert result is None
