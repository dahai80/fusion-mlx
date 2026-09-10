"""G5 dependency lock — vendor code integrity verification.

Generates and verifies a SHA256 checksum manifest for all files in
``fusion_mlx/patches/`` (upstream-derived vendor code). The manifest is
committed alongside the source so any tampering or drift is detected at
startup (``doctor``) or explicitly (``verify``).

This closes the C3 audit finding: vendor code + monkeypatches had no
version lock or checksum — a reproducible-builds gap that breaks the
fault-reproduction chain.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)

_PATCHES_DIR = Path(__file__).parent / "patches"
_MANIFEST_PATH = _PATCHES_DIR.parent / "patches.checksums.json"
_BLOCKSIZE = 65536


def _iter_vendor_files(root: Path) -> list[Path]:
    files: list[Path] = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [
            d for d in dirnames if not d.startswith(".") and d != "__pycache__"
        ]
        for fname in sorted(filenames):
            if fname.startswith("."):
                continue
            if fname.endswith((".pyc", ".pyo")):
                continue
            fp = Path(dirpath) / fname
            if fp.is_file():
                files.append(fp)
    return sorted(files)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(_BLOCKSIZE)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def generate_manifest(patches_dir: Path | None = None) -> dict[str, str]:
    root = patches_dir or _PATCHES_DIR
    if not root.is_dir():
        logger.warning("G5: patches dir %s does not exist", root)
        return {}
    manifest: dict[str, str] = {}
    for fp in _iter_vendor_files(root):
        rel = fp.relative_to(root.parent).as_posix()
        manifest[rel] = _sha256(fp)
    return manifest


def write_manifest(manifest: dict[str, str] | None = None) -> Path:
    if manifest is None:
        manifest = generate_manifest()
    with open(_MANIFEST_PATH, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, sort_keys=True, ensure_ascii=False)
        f.write("\n")
    logger.info("G5: wrote %d checksums to %s", len(manifest), _MANIFEST_PATH)
    return _MANIFEST_PATH


def verify_manifest(manifest: dict[str, str] | None = None) -> list[str]:
    if manifest is None:
        if not _MANIFEST_PATH.is_file():
            return [f"manifest missing: {_MANIFEST_PATH}"]
        with open(_MANIFEST_PATH, encoding="utf-8") as f:
            manifest = json.load(f)
    errors: list[str] = []
    for rel, expected in sorted(manifest.items()):
        fp = _PATCHES_DIR.parent / rel
        if not fp.is_file():
            errors.append(f"MISSING: {rel}")
            continue
        actual = _sha256(fp)
        if actual != expected:
            errors.append(
                f"MISMATCH: {rel} (expected {expected[:12]}…, got {actual[:12]}…)"
            )
    current = generate_manifest()
    for rel in current:
        if rel not in manifest:
            errors.append(f"UNTRACKED: {rel} (in patches/ but not in manifest)")
    return errors


def verify_or_raise(manifest: dict[str, str] | None = None) -> None:
    errors = verify_manifest(manifest)
    if errors:
        msg = "G5 vendor checksum verification failed:\n  " + "\n  ".join(errors)
        logger.error(msg)
        raise RuntimeError(msg)
    logger.info("G5: vendor checksums verified OK (%d files)", len(generate_manifest()))
