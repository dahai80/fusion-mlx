# SPDX-License-Identifier: Apache-2.0
"""Regression for G-8/T-2 (#0912 audit): API key three-source divergence.

Three key sources could drift:
1. admin layer — ``global_settings.auth.api_key`` (mutated by admin
   initial-setup route without syncing others)
2. module global — ``admin/auth._api_key`` (set via ``set_api_key``)
3. config layer — ``get_config().api_key`` (set at boot, not updated by
   admin runtime writes)

Pre-fix ``_get_configured_api_key`` read the admin layer first then fell
back to config — no consistency check. If they diverged (admin setup
mutated admin layer only), same-machine instances recognized different
keys → 401 with no hint about the effective key source.

Fix: ``_get_configured_api_key`` treats config as source of truth (matches
boot priority CLI > env > settings.json). If admin layer disagrees, it
logs ERROR (fail-visible) and re-aligns the admin layer to config. The
admin initial-setup route now syncs all three sources.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest


def _make_admin_settings(api_key: str | None):
    return SimpleNamespace(auth=SimpleNamespace(api_key=api_key))


@pytest.fixture
def _clean_key_state():
    """Reset all key sources before + after each test.

    Hard-reset to None on teardown (not restore-to-saved) so a key set in
    one test never leaks into a later test that expects anonymous access
    (the unit-test conftest sets FUSION_ALLOW_ANONYMOUS=true, but a
    non-None configured_key short-circuits the anonymous branch).
    """
    from fusion_mlx.config import get_config

    cfg = get_config()
    cfg.api_key = None
    try:
        from fusion_mlx.admin import helpers as admin_helpers

        admin_helpers._admin_getters["global_settings"] = None
    except Exception:
        pass
    yield cfg
    cfg.api_key = None
    try:
        from fusion_mlx.admin import helpers as admin_helpers

        admin_helpers._admin_getters["global_settings"] = None
    except Exception:
        pass


def _set_admin_layer(api_key: str | None):
    """Register a stable admin settings object (production getter returns
    the same ``self.settings`` instance each call, not a fresh object)."""
    from fusion_mlx.admin import helpers as admin_helpers

    settings = _make_admin_settings(api_key)
    admin_helpers._admin_getters["global_settings"] = lambda: settings


def test_config_is_source_of_truth_when_both_set(_clean_key_state):
    from fusion_mlx.middleware.auth import _get_configured_api_key

    _clean_key_state.api_key = "config-key-123456"
    _set_admin_layer("admin-key-123456")
    result = _get_configured_api_key()
    assert result == "config-key-123456"


def test_admin_realigned_to_config_on_conflict(_clean_key_state):
    from fusion_mlx.middleware.auth import _get_configured_api_key

    _clean_key_state.api_key = "config-key-123456"
    _set_admin_layer("admin-key-123456")
    _get_configured_api_key()
    # Admin layer should now hold the config value.
    from fusion_mlx.admin.helpers import _get_global_settings

    gs = _get_global_settings()
    assert gs.auth.api_key == "config-key-123456"


def test_admin_only_used_when_config_unset(_clean_key_state):
    from fusion_mlx.middleware.auth import _get_configured_api_key

    _clean_key_state.api_key = None
    _set_admin_layer("admin-key-123456")
    result = _get_configured_api_key()
    assert result == "admin-key-123456"


def test_none_when_both_unset(_clean_key_state):
    from fusion_mlx.middleware.auth import _get_configured_api_key

    _clean_key_state.api_key = None
    _set_admin_layer(None)
    result = _get_configured_api_key()
    assert result is None


def test_no_drift_when_admin_absent(_clean_key_state):
    """Admin getter not registered (partial startup) — config still works."""
    from fusion_mlx.admin import helpers as admin_helpers

    admin_helpers._admin_getters["global_settings"] = None
    _clean_key_state.api_key = "config-only-123456"
    from fusion_mlx.middleware.auth import _get_configured_api_key

    result = _get_configured_api_key()
    assert result == "config-only-123456"


def test_conflict_logs_error(_clean_key_state, caplog):
    import logging

    from fusion_mlx.middleware.auth import _get_configured_api_key

    _clean_key_state.api_key = "config-key-123456"
    _set_admin_layer("admin-key-123456")
    with caplog.at_level(logging.ERROR, logger="fusion_mlx.middleware.auth"):
        _get_configured_api_key()
    assert any("API key conflict" in r.message for r in caplog.records)
