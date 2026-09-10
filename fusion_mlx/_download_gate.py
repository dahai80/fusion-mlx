# SPDX-License-Identifier: Apache-2.0
"""Download confirmation gate.

Checks whether a model repo is already cached locally before triggering
a potentially multi-GB download. If not cached, estimates the repo size
via the HuggingFace API (or mirror) and prompts the user for confirmation.
"""

import logging
import os
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

_LARGE_DOWNLOAD_THRESHOLD = 2 * 1024**3


def _hf_cache_root() -> Path:
    try:
        from huggingface_hub.constants import HF_HUB_CACHE

        return Path(HF_HUB_CACHE)
    except Exception:
        return Path.home() / ".cache" / "huggingface" / "hub"


def _repo_snapshot_dir(model_name: str) -> Path:
    repo_dir = "models--" + model_name.replace("/", "--")
    base = _hf_cache_root() / repo_dir / "snapshots"
    if not base.exists():
        return base
    refs_dir = _hf_cache_root() / repo_dir / "refs"
    rev = "main"
    if refs_dir.exists():
        ref_file = refs_dir / "main"
        if ref_file.exists():
            try:
                rev = ref_file.read_text().strip()
            except Exception:
                pass
    snap = base / rev
    if snap.exists():
        return snap
    snaps = list(base.iterdir())
    return snaps[0] if snaps else base


def is_repo_cached(model_name: str) -> bool:
    snap = _repo_snapshot_dir(model_name)
    if not snap.exists():
        return False
    has_weights = any(
        f.suffix in (".safetensors", ".gguf", ".npz", ".bin", ".mlx")
        for f in snap.rglob("*")
        if f.is_file()
    )
    has_config = (snap / "config.json").exists()
    return has_weights or has_config


def estimate_repo_size_bytes(model_name: str) -> int | None:
    if is_repo_cached(model_name):
        snap = _repo_snapshot_dir(model_name)
        total = sum(f.stat().st_size for f in snap.rglob("*") if f.is_file())
        return total if total > 0 else None

    try:
        from huggingface_hub import HfApi

        api = HfApi()
        info = api.repo_info(model_name, files_metadata=True)
        total = sum(
            sibling.rfilename and getattr(sibling, "size", 0) or 0
            for sibling in getattr(info, "siblings", [])
        )
        total = sum(
            getattr(s, "size", 0) or 0 for s in getattr(info, "siblings", [])
        )
        return total if total > 0 else None
    except Exception as e:
        logger.debug("estimate_repo_size_bytes: HF API query failed: %s", e)
        return None


def _format_size(num_bytes: int) -> str:
    if num_bytes >= 1024**3:
        return f"{num_bytes / 1024**3:.1f} GB"
    if num_bytes >= 1024**2:
        return f"{num_bytes / 1024**2:.1f} MB"
    if num_bytes >= 1024:
        return f"{num_bytes / 1024:.1f} KB"
    return f"{num_bytes} B"


def confirm_or_abort(model_name: str, estimated_bytes: int | None = None) -> None:
    if is_repo_cached(model_name):
        logger.debug("download gate: %s already cached, skipping", model_name)
        return

    if estimated_bytes is None:
        estimated_bytes = estimate_repo_size_bytes(model_name)

    if estimated_bytes is not None and estimated_bytes > 0:
        size_str = _format_size(estimated_bytes)
        print(
            f"\nModel '{model_name}' is not cached locally. "
            f"Estimated download size: {size_str}.",
            file=sys.stderr,
        )
        if estimated_bytes >= _LARGE_DOWNLOAD_THRESHOLD:
            print(
                "This is a large download. Proceed? [y/N] ",
                file=sys.stderr,
                end="",
                flush=True,
            )
            try:
                if not sys.stdin.isatty():
                    logger.info(
                        "download gate: non-interactive session, auto-proceeding "
                        "for %s (%s)",
                        model_name,
                        size_str,
                    )
                    return
                response = input().strip().lower()
                if response not in ("y", "yes"):
                    print("Aborted.", file=sys.stderr)
                    raise SystemExit(1)
            except (EOFError, KeyboardInterrupt):
                print("Aborted.", file=sys.stderr)
                raise SystemExit(1)
    else:
        print(
            f"\nModel '{model_name}' is not cached locally. "
            "Download size unknown (will fetch from mirror).",
            file=sys.stderr,
        )
