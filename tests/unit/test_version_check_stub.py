# SPDX-License-Identifier: Apache-2.0
"""Unit tests for _version_check.py stub module.

Covers:
- Stub interface contract (no-op functions)
- Logging behavior
"""

from __future__ import annotations

import importlib.util
import logging


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "_version_check", "fusion_mlx/_version_check.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


vmod = _load_module()


class TestVersionCheckStub:
    """Version check is a no-op stub in this build."""

    def test_check_for_update_does_not_raise(self):
        vmod.check_for_update()
        vmod.check_for_update("any-args", kw="test")

    def test_print_staleness_warning_does_not_raise(self):
        vmod.print_staleness_warning_if_any()

    def test_prompt_upgrade_does_not_raise(self):
        vmod.prompt_upgrade_if_available()

    def test_check_for_update_logs_debug(self, caplog):
        caplog.set_level(logging.DEBUG)
        vmod.check_for_update()
        assert any("Version check skipped" in r.message for r in caplog.records)
