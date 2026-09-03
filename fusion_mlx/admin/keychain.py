# SPDX-License-Identifier: Apache-2.0
"""macOS Keychain backend for the admin API key.

Stores the API key in the macOS Keychain instead of settings.json plaintext.
Gated by FUSION_KEYCHAIN=on (default off, matching repo convention). Uses the
shipped `security` CLI so there is no extra dependency. On non-macOS or when
the CLI is absent, every call fails visibly and callers fall back to the
plaintext settings.json path.
"""

import logging
import platform
import subprocess

logger = logging.getLogger(__name__)

_SERVICE = "fusion-mlx.api-key"
_ACCOUNT = "fusion-mlx"


def is_enabled() -> bool:
    return __import__("os").environ.get("FUSION_KEYCHAIN", "").lower() == "on"


def is_available() -> bool:
    if platform.system() != "Darwin":
        return False
    try:
        subprocess.run(
            ["security", "which-keychain"],
            capture_output=True,
            check=False,
            timeout=5,
        )
        return True
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def _run(args: list[str], input_bytes: bytes | None = None) -> tuple[int, bytes, bytes]:
    cmd = ["security", *args]
    proc = subprocess.run(
        cmd,
        input=input_bytes,
        capture_output=True,
        check=False,
        timeout=15,
    )
    return proc.returncode, proc.stdout, proc.stderr


def get_key() -> str | None:
    if not is_available():
        logger.debug("keychain: get_key skipped (unavailable)")
        return None
    rc, out, err = _run(["find-generic-password", "-s", _SERVICE, "-a", _ACCOUNT, "-w"])
    if rc != 0:
        if "could not be found" not in (err or b"").decode("utf-8", "ignore").lower():
            logger.warning("keychain: find-generic-password rc=%d err=%s", rc, err)
        return None
    return out.decode("utf-8").strip() or None


def set_key(key: str) -> bool:
    if not is_available():
        logger.warning("keychain: set_key failed (unavailable)")
        return False
    if not key:
        return delete_key()
    # delete first so re-setting does not collide with an existing item
    _run(["delete-generic-password", "-s", _SERVICE, "-a", _ACCOUNT])
    rc, out, err = _run(
        ["add-generic-password", "-s", _SERVICE, "-a", _ACCOUNT, "-w", key, "-U"]
    )
    if rc != 0:
        logger.error("keychain: add-generic-password rc=%d err=%s", rc, err)
        return False
    logger.info("keychain: api key stored in Keychain (service=%s)", _SERVICE)
    return True


def delete_key() -> bool:
    if not is_available():
        return False
    rc, out, err = _run(["delete-generic-password", "-s", _SERVICE, "-a", _ACCOUNT])
    if (
        rc != 0
        and "could not be found" not in (err or b"").decode("utf-8", "ignore").lower()
    ):
        logger.warning("keychain: delete-generic-password rc=%d err=%s", rc, err)
        return False
    return True
