# SPDX-License-Identifier: Apache-2.0
"""Unit tests for fusion_mlx.utils.install — install-method detection + CLI prefix."""

from __future__ import annotations

import shlex
import sys
from pathlib import Path

from fusion_mlx.utils import install as install_mod


class TestAppBundleDetection:
    def test_is_app_bundle_returns_bool(self):
        assert isinstance(install_mod.is_app_bundle(), bool)

    def test_is_app_bundle_false_in_dev_venv(self):
        # __file__ lives under .venv, not inside a .app/Contents/ tree.
        assert install_mod.is_app_bundle() is False


class TestHomebrewDetection:
    def test_is_homebrew_returns_bool(self):
        assert isinstance(install_mod.is_homebrew(), bool)

    def test_is_homebrew_false_in_dev_venv(self):
        # This venv is .venv/ under the repo, not a /Cellar/ or /homebrew/ path.
        assert install_mod.is_homebrew() is False


class TestInstallMethod:
    def test_get_install_method_in_known_set(self):
        assert install_mod.get_install_method() in {"dmg", "homebrew", "pip"}

    def test_get_install_method_pip_in_dev_venv(self):
        assert install_mod.get_install_method() == "pip"


class TestBundlePaths:
    def test_app_bundle_default_constant(self):
        assert str(install_mod._APP_BUNDLE_DEFAULT) == "/Applications/FusionMLX.app"

    def test_get_app_bundle_path_default_when_not_in_bundle(self):
        # Not running inside a bundle → defaults to /Applications/FusionMLX.app.
        assert install_mod.get_app_bundle_path() == Path("/Applications/FusionMLX.app")

    def test_get_app_bundle_cli_path_default_when_not_in_bundle(self):
        cli = install_mod.get_app_bundle_cli_path()
        assert cli == Path("/Applications/FusionMLX.app/Contents/MacOS/fusion-cli")

    def test_get_app_bundle_path_is_cli_grandparent(self):
        # get_app_bundle_path() == get_app_bundle_cli_path().parents[2].
        cli = install_mod.get_app_bundle_cli_path()
        assert install_mod.get_app_bundle_path() == cli.parents[2]

    def test_get_user_cli_shim_path(self):
        expected = Path.home() / ".fusion-mlx" / "bin" / "fusion"
        assert install_mod.get_user_cli_shim_path() == expected


class TestCliPrefix:
    def test_get_cli_prefix_nonempty(self):
        assert install_mod.get_cli_prefix()

    def test_get_cli_prefix_dev_form_when_no_console_script(self):
        # In the dev venv either fusion-mlx is on PATH or we fall back to
        # ``<python> -m fusion_mlx``. Both are acceptable; assert the form.
        prefix = install_mod.get_cli_prefix()
        assert prefix == "fusion-mlx" or prefix == f"{sys.executable} -m fusion_mlx"

    def test_get_cli_command_prefix_is_quoted_cli_prefix(self):
        assert install_mod.get_cli_command_prefix() == shlex.quote(install_mod.get_cli_prefix())
