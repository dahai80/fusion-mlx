# SPDX-License-Identifier: Apache-2.0
"""Version-check + upgrade helper for the fusion-mlx CLI.

Two surfaces:
  - ``print_staleness_warning_if_any`` / ``staleness_warning``: opt-in
    (TTY + non-CI + not disabled), cache-backed, fail-silent banner
    printed on stderr when the running install is >=2 patch releases
    behind the latest GitHub release within the SAME minor line.
  - ``prompt_upgrade_if_available`` + ``upgrade_command`` (cli_commands):
    interactive one-shot auto-upgrade prompt before serve boots, plus the
    ``fusion-mlx upgrade`` subcommand. Detects brew / install.sh / pip and
    runs the right command.

Everything network-facing is fail-silent so an offline laptop never sees a
broken CLI. The cache lives under ``~/.fusion-mlx/version_check.json``.
"""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

_BINARY_NAME = "fusion-mlx"
_GITHUB_REPO = "fusion-mlx/fusion-mlx"
_GITHUB_TAGS_URL = f"https://api.github.com/repos/{_GITHUB_REPO}/git/refs/tags"
_BREW_TAP_FORMULA = "dahai80/homebrew-fusion-mlx/fusion-mlx"

_DISABLE_ENV = "FUSION_MLX_DISABLE_VERSION_CHECK"


def _parse_version(raw):
    """Parse ``X.Y.Z`` (optionally ``v``-prefixed, dev/rc suffix tolerated).

    Returns ``(major, minor, patch)`` int tuple or ``None`` on garbage.
    Dev/alpha/beta/rc suffixes are tolerated: the patch number is taken
    as-is so a dev build on ``0.6.14.dev3`` parses to ``(0, 6, 14)``.
    """
    if not isinstance(raw, str):
        return None
    cleaned = raw.strip().lstrip("v")
    m = re.match(r"^(\d+)\.(\d+)\.(\d+)", cleaned)
    if not m:
        return None
    try:
        return (int(m.group(1)), int(m.group(2)), int(m.group(3)))
    except ValueError:
        return None


def _installed_version():
    """Return the installed package version string, or ``None`` if unknown."""
    try:
        from ._version import __version__

        return __version__
    except Exception as e:  # noqa: BLE001
        logger.debug("_installed_version: could not read version: %s", e)
        return None


def _disabled():
    """True when the version check is opted out (env) or running in CI."""
    if os.environ.get(_DISABLE_ENV, "").strip() not in ("", "0", "false", "no"):
        return True
    if os.environ.get("CI", "").strip():
        return True
    return False


def _cache_path():
    """Path to the staleness cache file."""
    try:
        return Path.home() / ".fusion-mlx" / "version_check.json"
    except Exception:  # noqa: BLE001
        return Path("/tmp") / "fusion-mlx-version-check.json"


def _fetch_latest_from_github():
    """Fetch the latest release tag from GitHub. Returns tag string or None."""
    try:
        import urllib.request

        req = urllib.request.Request(
            _GITHUB_TAGS_URL,
            headers={"Accept": "application/vnd.github+json"},
        )
        with urllib.request.urlopen(req, timeout=5) as resp:  # noqa: S310
            data = json.loads(resp.read().decode("utf-8"))
    except Exception as e:  # noqa: BLE001
        logger.debug("_fetch_latest_from_github: fetch failed: %s", e)
        return None
    if not isinstance(data, list) or not data:
        return None
    tags = []
    for entry in data:
        ref = entry.get("ref", "") if isinstance(entry, dict) else ""
        if ref.startswith("refs/tags/v"):
            tags.append(ref[len("refs/tags/v") :])
        elif ref.startswith("refs/tags/"):
            tags.append(ref[len("refs/tags/") :])
    parsed = [(t, _parse_version(t)) for t in tags]
    parsed = [(t, v) for t, v in parsed if v is not None]
    if not parsed:
        return None
    parsed.sort(key=lambda tv: tv[1])
    return parsed[-1][0]


def _read_cache():
    try:
        path = _cache_path()
        if not path.exists():
            return None
        data = json.loads(path.read_text())
        if not isinstance(data, dict):
            return None
        return data
    except Exception as e:  # noqa: BLE001
        logger.debug("_read_cache: read failed: %s", e)
        return None


def _write_cache(latest):
    try:
        path = _cache_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"latest": latest, "ts": time.time()}))
    except Exception as e:  # noqa: BLE001
        logger.debug("_write_cache: write failed: %s", e)


def get_latest_version(force_refresh=False):
    """Return cached or freshly-fetched latest version string, or None.

    A cache file carrying a ``latest`` value is treated as fresh. The
    CLI startup banner (``staleness_warning``) reads the cache so a
    laptop boot never hits GitHub; the explicit ``fusion-mlx upgrade``
    subcommand passes ``force_refresh=True`` to bypass the cache and
    always check GitHub live. The cache is written after every
    successful fetch, so a single online run seeds it for subsequent
    offline boots.
    """
    if not force_refresh:
        cached = _read_cache()
        if cached is not None:
            latest = cached.get("latest")
            if latest:
                return latest
    latest = _fetch_latest_from_github()
    if latest is not None:
        _write_cache(latest)
    return latest


def staleness_warning():
    """Return a warning string if the install is stale, else None.

    Only nags within the same minor line and only when >=2 patch releases
    behind (1-patch lag is normal bug-fix noise). Cross-minor bumps are
    silent — the user may be intentionally pinning the line. Never raises.
    """
    if _disabled():
        return None
    current = _installed_version()
    cur = _parse_version(current) if current else None
    if cur is None:
        return None
    latest = get_latest_version()
    lat = _parse_version(latest) if latest else None
    if lat is None:
        return None
    if cur[0] != lat[0] or cur[1] != lat[1]:
        return None
    if lat[2] - cur[2] < 2:
        return None
    return (
        f"A new fusion-mlx release is available: {current} -> {latest}.\n"
        f"Run `fusion-mlx upgrade` to update."
    )


@dataclass
class InstallInfo:
    method: str  # "brew" | "install_sh" | "pip" | "unknown"
    binary_path: str | None = None
    upgrade_command: str = ""
    upgrade_argv: list[str] = field(default_factory=list)


def _detect_brew(realpath):
    if realpath is None:
        return None
    if "/Cellar/" in realpath:
        return _BREW_TAP_FORMULA
    return None


def detect_install_method():
    """Detect how fusion-mlx was installed and return the upgrade command."""
    binary = shutil.which(_BINARY_NAME)
    realpath = None
    if binary:
        try:
            realpath = os.path.realpath(binary)
        except Exception:  # noqa: BLE001
            realpath = binary

    brew_formula = _detect_brew(realpath) if realpath else None
    if brew_formula:
        argv = ["brew", "upgrade", brew_formula]
        return InstallInfo(
            method="brew",
            binary_path=binary,
            upgrade_command=" ".join(argv),
            upgrade_argv=argv,
        )

    home = Path.home()
    is_install_sh = False
    if realpath and ".fusion-mlx" in realpath:
        is_install_sh = True
    if binary and str(home / ".local" / "bin") in binary:
        is_install_sh = True
    if is_install_sh:
        cmd = f"curl -fsSL https://raw.githubusercontent.com/{_GITHUB_REPO}/main/install.sh | bash"
        return InstallInfo(
            method="install_sh",
            binary_path=binary,
            upgrade_command=cmd,
            upgrade_argv=["bash", "-c", cmd],
        )

    if binary:
        argv = [sys.executable, "-m", "pip", "install", "--upgrade", _BINARY_NAME]
        return InstallInfo(
            method="pip",
            binary_path=binary,
            upgrade_command=" ".join(argv),
            upgrade_argv=argv,
        )

    argv = [sys.executable, "-m", "pip", "install", "--upgrade", _BINARY_NAME]
    return InstallInfo(
        method="pip",
        binary_path=None,
        upgrade_command=" ".join(argv),
        upgrade_argv=argv,
    )


def check_for_update(*args, **kwargs):
    """Legacy entry point — prints a staleness warning if any."""
    msg = staleness_warning()
    if msg:
        print(msg, file=sys.stderr)


def print_staleness_warning_if_any():
    """Print the staleness banner to stderr if stale. Never raises."""
    try:
        msg = staleness_warning()
        if msg:
            print(msg, file=sys.stderr)
    except Exception as e:  # noqa: BLE001
        logger.debug("print_staleness_warning_if_any: swallowed: %s", e)


def _is_dev_build(version):
    """True if the version string is a PEP 440 non-final release
    (dev/rc/alpha/beta/post/local). Devs running pre-release builds
    should never get a downgrade prompt."""
    if not version:
        return True
    v = version.strip().lstrip("v")
    if re.search(r"(dev|rc|a|b|post)\d", v):
        return True
    if "+" in v:
        return True
    return False


def prompt_upgrade_if_available():
    """Interactive auto-upgrade prompt before serve boots.

    Returns True if an upgrade was run (caller should exit so the new
    binary takes over), False otherwise. Honors disabled / non-TTY /
    already-current / dev-build / offline guards. Never raises.
    """
    try:
        if _disabled():
            return False
        if not sys.stdin.isatty() or not sys.stderr.isatty():
            return False
        current = _installed_version()
        if _is_dev_build(current):
            return False
        cur = _parse_version(current) if current else None
        latest = get_latest_version()
        lat = _parse_version(latest) if latest else None
        if cur is None or lat is None:
            return False
        if cur >= lat:
            return False
        info = detect_install_method()
        if info.method == "unknown":
            return False
        print(
            f"\n  fusion-mlx {current} -> {latest} is available.",
            file=sys.stderr,
        )
        print(f"  Run:  {info.upgrade_command}\n", file=sys.stderr)
        try:
            answer = input("  Upgrade now? [Y/n] ").strip().lower()
        except EOFError:
            answer = ""
        except KeyboardInterrupt:
            print(file=sys.stderr)
            return False
        if answer in {"n", "no"}:
            return False
        result = subprocess.run(info.upgrade_argv, check=False)
        return result.returncode == 0
    except Exception as e:  # noqa: BLE001
        logger.debug("prompt_upgrade_if_available: swallowed: %s", e)
        return False


__all__ = [
    "InstallInfo",
    "check_for_update",
    "detect_install_method",
    "get_latest_version",
    "print_staleness_warning_if_any",
    "prompt_upgrade_if_available",
    "staleness_warning",
]
