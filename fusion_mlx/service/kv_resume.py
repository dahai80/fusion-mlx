# SPDX-License-Identifier: Apache-2.0
"""Disconnect KV two-end sync: persist on abort, resume on request.

Bridges the client-disconnect / scheduler-abort path to the disk-KV
checkpoint writer (``fusion_mlx.runtime.disk_kv_checkpoint``), and the
resume API to the loader. Until this module the checkpoint writer had
zero live callers — a long-context request that disconnected mid-decode
lost its KV tail and had to re-prefill on resume.

Two ends:
- **Persist end** (disconnect/abort): ``persist_request_kv`` snapshots
  the request's live ``prompt_cache`` at the current token offset so the
  KV tail survives the abort. Best-effort: a write failure is logged and
  swallowed (the abort still succeeds).
- **Resume end** (new request): ``load_resumable_kv`` reads the most
  recent checkpoint for a prior request_id and returns the cache +
  token offset so the caller can seed a fresh request with it instead
  of re-prefilling.

Both ends are gated on the same env flag the disk-KV writer honours
(``FUSION_MLX_KV_CHECKPOINT_INTERVAL``); when checkpointing is disabled
the persist end is a no-op and the resume end returns None.
"""

from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)

_RESUME_ENV = "FUSION_MLX_KV_CHECKPOINT_INTERVAL"


def _interval() -> int:
    raw = os.environ.get(_RESUME_ENV)
    if raw is None:
        return 0
    try:
        return max(0, int(raw))
    except (TypeError, ValueError):
        logger.warning("[kv_resume] bad %s=%r, disabling", _RESUME_ENV, raw)
        return 0


def persist_request_kv(
    request_id: str,
    prompt_cache,
    num_tokens: int,
    model_name: str | None = None,
    kv_dtype: str = "bf16",
) -> str | None:
    """Persist a live KV cache to disk on disconnect/abort.

    Called from the disconnect hook BEFORE the scheduler tears down the
    request (the cache references are nulled in ``_do_abort_request``).
    Writes a single checkpoint at the current token offset regardless of
    the boundary cadence — a disconnect is an unscheduled event and the
    last partial boundary is exactly the state worth resuming from.

    Returns the safetensors path on success, None when disabled or the
    write failed. Never raises: a persistence bug must not block abort.
    """
    interval = _interval()
    if interval <= 0:
        return None
    if not prompt_cache:
        logger.debug("[kv_resume] no prompt_cache for %s; skip persist", request_id)
        return None
    if num_tokens <= 0:
        return None
    try:
        from ..runtime.disk_kv_checkpoint import (
            get_default_root,
            model_requires_full_checkpoint,
            request_hash,
            write_checkpoint,
        )

        req_hash = request_hash(request_id, model_name)
        root = get_default_root()
        requires_full = model_requires_full_checkpoint(model_name)
        path = write_checkpoint(
            list(prompt_cache),
            root=root,
            req_hash=req_hash,
            token_offset=int(num_tokens),
            kv_dtype=kv_dtype,
            requires_full_checkpoint=requires_full,
            model_name=model_name,
            extra_metadata={"source": "disconnect_persist", "request_id": request_id},
        )
        if path is not None:
            logger.info(
                "[kv_resume] persisted KV for %s at %d tokens -> %s",
                request_id,
                num_tokens,
                path,
            )
        return path
    except Exception as e:
        logger.warning("[kv_resume] persist failed for %s: %s", request_id, e)
        return None


def load_resumable_kv(
    request_id: str,
    model_name: str | None = None,
):
    """Load the most recent persisted KV checkpoint for a request_id.

    Returns a ``LoadedCheckpoint`` (cache + token_offset + metadata) the
    caller seeds a fresh request with, or None when disabled / no
    checkpoint exists / the load failed. Never raises.
    """
    interval = _interval()
    if interval <= 0:
        return None
    try:
        from ..runtime.disk_kv_checkpoint import (
            get_default_root,
            load_checkpoint,
            request_hash,
            scan_checkpoints,
        )

        req_hash = request_hash(request_id, model_name)
        root = get_default_root()
        entries = scan_checkpoints(root)
        # Pick the newest checkpoint whose path lives under this req_hash.
        candidates = [
            (path, mtime, size)
            for path, mtime, size in entries
            if os.path.join(root, req_hash) in path
        ]
        if not candidates:
            logger.debug("[kv_resume] no checkpoint for %s (hash=%s)", request_id, req_hash)
            return None
        candidates.sort(key=lambda t: t[1], reverse=True)
        newest_path = candidates[0][0]
        loaded = load_checkpoint(newest_path)
        if loaded is None:
            logger.warning("[kv_resume] load_checkpoint returned None for %s", newest_path)
            return None
        logger.info(
            "[kv_resume] resumed KV for %s at %d tokens from %s",
            request_id,
            loaded.token_offset,
            newest_path,
        )
        return loaded
    except Exception as e:
        logger.warning("[kv_resume] load failed for %s: %s", request_id, e)
        return None


def cleanup_resumable_kv(request_id: str, model_name: str | None = None) -> None:
    """Drop persisted checkpoints for a request after a successful resume.

    Best-effort: a cleanup failure is logged, not raised.
    """
    interval = _interval()
    if interval <= 0:
        return
    try:
        from ..runtime.disk_kv_checkpoint import (
            cleanup_request,
            get_default_root,
            request_hash,
        )

        req_hash = request_hash(request_id, model_name)
        root = get_default_root()
        cleanup_request(root, req_hash)
        logger.debug("[kv_resume] cleaned up checkpoints for %s", request_id)
    except Exception as e:
        logger.warning("[kv_resume] cleanup failed for %s: %s", request_id, e)
