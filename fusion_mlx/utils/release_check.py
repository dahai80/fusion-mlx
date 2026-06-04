"""Release checking utilities for fusion-mlx."""

from typing import Optional


def select_latest_stable_release(version_list: Optional[list[str]] = None) -> str:
    """Select the latest stable release from a list of versions.

    Args:
        version_list: Optional list of version strings. If None, returns default.

    Returns:
        The latest stable version string.
    """
    if not version_list:
        return "0.1.0"
    # Simple semver sort — filter out pre-releases
    stable = [v for v in version_list if "-" not in v and "." in v]
    if not stable:
        stable = version_list
    return sorted(stable, key=_version_key)[-1]


def _version_key(v: str) -> tuple:
    """Parse version string into comparable tuple."""
    try:
        parts = v.lstrip("v").split(".")
        return tuple(int(p) for p in parts)
    except (ValueError, AttributeError):
        return (0, 0, 0)
