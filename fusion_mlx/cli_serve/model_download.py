# SPDX-License-Identifier: Apache-2.0
"""Mirror prefetch + ensure-model-downloaded."""

import logging
import os
import sys

from .preflight import _check_disk_space

logger = logging.getLogger(__name__)


def _try_mirror_prefetch(model_name: str) -> bool:
    """Pre-fetch a HuggingFace repo via R2-first / HF-fallback (per file).

    Delegates to :func:`fusion_mlx._mirror.download_with_mirror_fallback`.
    Returns ``True`` if the snapshot is fully populated (any mix of R2
    and HF). Returns ``False`` if the caller should fall through to the
    plain ``snapshot_download(repo_id)`` path (catalog unavailable for
    catalog-only paths, or one or more files failed both R2 and HF).

    Set ``FUSION_MLX_MODEL_MIRROR=""`` to disable R2 entirely and force
    HuggingFace.

    Codex round-6 BLOCKING #2: the mirror module already returns
    ``False`` on every recoverable network/cache error, so the only
    catch worth doing here is ``ImportError`` (mirror module disabled
    or missing in a minimal install). Programmer errors propagate so
    bugs in the mirror module surface as real stack traces instead of
    silently routing to ``snapshot_download``.
    """
    try:
        from fusion_mlx._mirror import download_with_mirror_fallback
    except ImportError:
        # Mirror module not available (minimal-deps install or
        # deliberately removed). Use the legacy HF path.
        return False
    return download_with_mirror_fallback(model_name)


def _ensure_model_downloaded(model_name: str) -> None:
    """Pre-fetch a model in the foreground so HF's tqdm progress is visible.

    Used by ``fusion-mlx chat``: the chat REPL spawns ``serve`` as a
    subprocess with stdout/stderr redirected to a log file. If the model
    isn't cached, the user sees a silent multi-minute hang while several
    GB downloads behind the log. Calling ``snapshot_download`` here first
    surfaces the standard HF progress bars on the user's terminal, then
    the spawned server starts as a cache hit.

    No-op when the model is already cached, when ``model_name`` is a local
    path, or when the HF lookup fails (let the loader's own error paths
    handle it).
    """
    if os.path.exists(model_name):
        return
    # Bare names (e.g. ``Qwen3.5-9B-4bit``) already cached in the fusion-mlx /
    # fusion-mlx / HF-snapshot model dirs need no HuggingFace fetch. Reuse
    # the same resolver the server's load_model uses so ``serve --model
    # <bare-name>`` does not attempt a doomed HF lookup for a model that
    # is already on disk locally. Slash-names and genuine HF repos fall
    # through unchanged (resolver returns them as-is when no local path
    # exists).
    try:
        from .. import server as _server_mod

        resolved = _server_mod._resolve_single_model_path(model_name)
        if os.path.exists(resolved):
            return
    except Exception:
        pass
    # Reuse the same weight-file-presence probe as ``is_repo_cached``:
    # the older ``try_to_load_from_cache('config.json')`` check
    # short-circuits on a partial cache (metadata downloaded, weight
    # shards still in flight), letting the spawned ``serve`` quietly
    # finish the download inside its logfile. Codex round-3 BLOCKING #2.
    try:
        from fusion_mlx._download_gate import is_repo_cached

        if is_repo_cached(model_name):
            return
    except Exception:
        # Probe failed (filesystem permission error, unexpected layout) —
        # fall through to the heavy snapshot_download path; HF will
        # short-circuit on its own cache check if the repo really is
        # fully present.
        pass

    # Disk-space gate: a 20 GB partial download that fails on the last
    # shard wastes the user's time. ``_check_disk_space`` queries HF for
    # the repo size and aborts with a clear message + exit(1) if there
    # isn't enough room on the resolved HF cache filesystem.
    _check_disk_space(model_name)

    # User-configured mirror path (R2/S3/any HTTP host). When the mirror
    # serves every file the repo declares, populate the HF cache layout
    # ourselves and skip snapshot_download. On any miss we fall through
    # to the normal HuggingFace download below.
    if _try_mirror_prefetch(model_name):
        return

    try:
        from huggingface_hub import model_info, snapshot_download

        size_gb = 0.0
        try:
            info = model_info(model_name, files_metadata=True)
            size_bytes = sum(
                (s.size or 0)
                for s in (getattr(info, "siblings", None) or [])
                if hasattr(s, "size")
            )
            size_gb = size_bytes / (1024**3)
        except Exception:
            pass

        is_tty = sys.stdout.isatty() and "NO_COLOR" not in os.environ
        BOLD = "\x1b[1m" if is_tty else ""
        DIM = "\x1b[2m" if is_tty else ""
        RESET = "\x1b[0m" if is_tty else ""
        if size_gb > 0:
            print(
                f"\n  {BOLD}First-time download{RESET} — "
                f"fetching {model_name} {DIM}(~{size_gb:.1f} GB){RESET} "
                "from HuggingFace ..."
            )
        else:
            print(
                f"\n  {BOLD}First-time download{RESET} — "
                f"fetching {model_name} from HuggingFace ..."
            )

        snapshot_download(model_name)
        print()
    except SystemExit:
        # _check_disk_space aborts via sys.exit(1) — let it through.
        raise
    except Exception as e:
        # Definitive 404s are surfaced so callers (e.g. ``/model bogus``)
        # can refuse fast instead of spawning a doomed serve subprocess
        # that fails after ``--ready-timeout``. Other transient errors
        # (network, auth) fall through silently — the spawned server's
        # own loader will retry and surface a real error if needed.
        from huggingface_hub.utils import RepositoryNotFoundError

        if isinstance(e, RepositoryNotFoundError) or "404" in str(e):
            raise RuntimeError(f"Model {model_name!r} not found on HuggingFace") from e
        print(f"\n  Pre-download skipped ({type(e).__name__}); server will retry.")
