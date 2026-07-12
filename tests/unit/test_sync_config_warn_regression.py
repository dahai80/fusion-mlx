# SPDX-License-Identifier: Apache-2.0
# #69 regression: _sync_config setattr failures must WARN, not debug.
# Locks in the behavior added in 63dc5c3: when a staged server global cannot be
# written onto the ServerConfig singleton (frozen dataclass / read-only
# property), the failure is logged at WARNING so config drift is visible,
# instead of buried at DEBUG.

import logging

import fusion_mlx.config as config_mod
import fusion_mlx.server as server_mod


class _DriftyConfig:
    # Read-only property: hasattr(cfg, "model_name") is True, but setattr
    # raises AttributeError (no setter). All other _sync_config target attrs
    # are absent -> hasattr False -> skipped, so only model_name hits the
    # warn path.
    @property
    def model_name(self):
        return "drift"


def test_sync_config_warns_on_setattr_failure(monkeypatch, caplog):
    # A setattr failure must surface as a WARNING so drift is visible.
    monkeypatch.setattr(config_mod, "get_config", lambda: _DriftyConfig())
    # _sync_config reads module globals (_api_key etc.) and runs an auth
    # propagation block when _api_key is set. Pin it falsy to isolate the
    # test to the setattr-warn path.
    monkeypatch.setattr(server_mod, "_api_key", None, raising=False)

    with caplog.at_level(logging.WARNING, logger=server_mod.logger.name):
        server_mod._sync_config()

    msgs = [r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING]
    assert any("setattr model_name failed" in m for m in msgs), msgs


def test_sync_config_no_warn_when_all_attrs_settable(monkeypatch, caplog):
    # No warning when every target attr is writable (baseline sanity).

    class _CleanConfig:
        pass

    clean = _CleanConfig()
    monkeypatch.setattr(config_mod, "get_config", lambda: clean)
    monkeypatch.setattr(server_mod, "_api_key", None, raising=False)

    with caplog.at_level(logging.WARNING, logger=server_mod.logger.name):
        server_mod._sync_config()

    drift_msgs = [
        r.getMessage()
        for r in caplog.records
        if r.levelno >= logging.WARNING and "setattr" in r.getMessage()
    ]
    assert drift_msgs == [], drift_msgs
