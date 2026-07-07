# SPDX-License-Identifier: Apache-2.0
"""Installation method detection for fusion-mlx.

Detects whether the running CLI lives inside the macOS .app bundle, a
Homebrew virtualenv, or a plain pip/dev install, and resolves the CLI
command prefix appropriate for each. Used by the lifecycle commands
(start/stop/restart) and by integration launch helpers.
"""

import os
import shlex
import shutil
import sys
from pathlib import Path

# Binary inside FusionMLX.app/Contents/MacOS/
_APP_BUNDLE_CLI_NAME = "fusion-cli"
# Canonical pip console-script name (pyproject [project.scripts]).
_PATH_CLI = "fusion-mlx"
# The macOS app installs a PATH shim at ~/.fusion-mlx/bin/fusion (named
# ``fusion``, NOT ``fusion-mlx``) that points back at the bundle CLI.
_APP_PATH_CLI = "fusion"
_USER_CLI_SHIM = Path(".fusion-mlx") / "bin" / "fusion"
_APP_BUNDLE_DEFAULT = Path("/Applications/FusionMLX.app")


def is_app_bundle() -> bool:
    """Return True if running inside the macOS .app bundle."""
    here = Path(__file__).resolve()
    return ".app/Contents/" in str(here)


def get_app_bundle_cli_path() -> Path:
    """Return the app-bundle CLI path for the currently running bundle."""
    here = Path(__file__).resolve()
    marker = ".app/Contents/"
    path = str(here)
    idx = path.find(marker)
    if idx == -1:
        return _APP_BUNDLE_DEFAULT / "Contents" / "MacOS" / _APP_BUNDLE_CLI_NAME
    app_root = Path(path[: idx + len(".app")])
    return app_root / "Contents" / "MacOS" / _APP_BUNDLE_CLI_NAME


def get_app_bundle_path() -> Path:
    """Return the .app directory for the currently running bundle."""
    cli_path = get_app_bundle_cli_path()
    try:
        return cli_path.parents[2]
    except IndexError:
        return _APP_BUNDLE_DEFAULT


def get_user_cli_shim_path() -> Path:
    """Return the user PATH shim installed by the macOS app."""
    return Path.home() / _USER_CLI_SHIM


def _is_executable(path: Path) -> bool:
    return path.exists() and os.access(path, os.X_OK)


def _same_resolved_path(left: Path, right: Path) -> bool:
    return left.resolve() == right.resolve()


def _is_app_managed_cli(path: Path) -> bool:
    """Return True when path points at the app-managed shim or wrapper."""
    if not _is_executable(path):
        return False
    user_shim = get_user_cli_shim_path()
    if _is_executable(user_shim) and _same_resolved_path(path, user_shim):
        return True
    app_cli = get_app_bundle_cli_path()
    return _is_executable(app_cli) and _same_resolved_path(path, app_cli)


def _path_resolves_to_app_managed_cli() -> bool:
    resolved = shutil.which(_APP_PATH_CLI)
    return bool(resolved) and _is_app_managed_cli(Path(resolved))


def is_homebrew() -> bool:
    """Return True if running inside a Homebrew-installed virtualenv."""
    prefix = sys.prefix
    return "/Cellar/" in prefix or "/homebrew/" in prefix


def get_install_method() -> str:
    """Return the installation method: 'dmg', 'homebrew', or 'pip'."""
    if is_app_bundle():
        return "dmg"
    if is_homebrew():
        return "homebrew"
    return "pip"


def get_cli_prefix() -> str:
    """Return the CLI command prefix for the current installation.

    App-bundle: the ``fusion`` shim (if on PATH) or the full bundle CLI path.
    Pip: ``fusion-mlx`` when the console script is installed.
    Dev/other: ``<python> -m fusion_mlx`` so printed commands still run.
    """
    if is_app_bundle():
        if _path_resolves_to_app_managed_cli():
            return _APP_PATH_CLI
        return str(get_app_bundle_cli_path())
    if shutil.which(_PATH_CLI):
        return _PATH_CLI
    return f"{sys.executable} -m fusion_mlx"


def get_cli_command_prefix() -> str:
    """Return a shell-safe CLI command prefix for display/copy-paste."""
    return shlex.quote(get_cli_prefix())
