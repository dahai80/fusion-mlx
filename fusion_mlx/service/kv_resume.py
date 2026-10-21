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

import hashlib
import logging
import os

logger = logging.getLogger(__name__)

_RESUME_ENV = "FUSION_MLX_KV_CHECKPOINT_INTERVAL"


def _prompt_hash(prompt_token_ids: list[int], num_tokens: int) -> str:
    h = hashlib.sha256()
    h.update(",".join(map(str, prompt_token_ids[:num_tokens])).encode())
    return h.hexdigest()


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
    prompt_token_ids: list[int] | None = None,
) -> str | None:
    """Persist a live KV cache to disk on disconnect/abort.

    Called from the disconnect hook BEFORE the scheduler tears down the
    request (the cache references are nulled in ``_do_abort_request``).
    Writes a single checkpoint at the current token offset regardless of
    the boundary cadence — a disconnect is an unscheduled event and the
    last partial boundary is exactly the state worth resuming from.

    ``prompt_token_ids`` records a content hash of the cached prefix so a
    RETRY (new request_id, same prompt) can find this checkpoint via
    ``find_resumable_kv_for_prompt`` — request_id-keyed lookup cannot
    match retries.

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
        extra_metadata = {"source": "disconnect_persist", "request_id": request_id}
        if prompt_token_ids:
            try:
                extra_metadata["prompt_hash"] = _prompt_hash(
                    list(prompt_token_ids), int(num_tokens)
                )
            except Exception as hash_exc:
                logger.debug(
                    "[kv_resume] prompt_hash skipped for %s: %s", request_id, hash_exc
                )
        path = write_checkpoint(
            list(prompt_cache),
            root=root,
            req_hash=req_hash,
            token_offset=int(num_tokens),
            kv_dtype=kv_dtype,
            requires_full_checkpoint=requires_full,
            model_name=model_name,
            extra_metadata=extra_metadata,
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
            logger.debug(
                "[kv_resume] no checkpoint for %s (hash=%s)", request_id, req_hash
            )
            return None
        candidates.sort(key=lambda t: t[1], reverse=True)
        newest_path = candidates[0][0]
        loaded = load_checkpoint(newest_path)
        if loaded is None:
            logger.warning(
                "[kv_resume] load_checkpoint returned None for %s", newest_path
            )
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


def find_resumable_kv_for_prompt(
    prompt_token_ids: list[int],
    model_name: str | None = None,
):
    """Find a persisted checkpoint whose cached prefix matches this prompt.

    Abort-persisted checkpoints are keyed by the ABORTED request_id, but a
    client retry arrives with a NEW request_id — request_id-keyed lookup
    (``load_resumable_kv``) can never match it. This finder matches by
    content instead: the sidecar's ``prompt_hash`` (sha256 of the cached
    token prefix, recorded at persist time) plus model_name.

    Scans sidecar JSON only (cheap); loads the safetensors body of the
    newest match. Returns a ``LoadedCheckpoint`` with token_offset a
    strict prefix of ``prompt_token_ids``, or None. Never raises.
    """
    interval = _interval()
    if interval <= 0 or not prompt_token_ids:
        return None
    try:
        import json

        from ..runtime.disk_kv_checkpoint import (
            _CHECKPOINT_EXT,
            _METADATA_EXT,
            get_default_root,
            load_checkpoint,
            scan_checkpoints,
        )

        root = get_default_root()
        entries = scan_checkpoints(root)
        # #1308: a prompt hash matches every strict prefix of the prompt, so
        # the largest matching offset always reuses the most KV. Scanning
        # newest-first returned whichever checkpoint happened to be written
        # last (usually a small early one, e.g. 2048 tokens for a 69845-token
        # prompt), discarding a larger valid prefix. Sidecars are cheap to
        # read, so collect every valid candidate and sort by offset desc.
        candidates: list[tuple[int, float, str, dict]] = []
        for path, mtime, _size in entries:
            sidecar_path = path.replace(_CHECKPOINT_EXT, _METADATA_EXT)
            try:
                with open(sidecar_path, encoding="utf-8") as fh:
                    meta = json.load(fh)
            except Exception:
                continue
            if meta.get("source") != "disconnect_persist":
                continue
            stored_model = meta.get("model_name")
            if model_name and stored_model and str(stored_model) != str(model_name):
                continue
            offset = meta.get("token_offset")
            if not isinstance(offset, int) or offset <= 0:
                continue
            if offset > len(prompt_token_ids):
                continue
            if meta.get("prompt_hash") != _prompt_hash(prompt_token_ids, offset):
                continue
            candidates.append((offset, mtime, path, meta))
        if not candidates:
            return None
        candidates.sort(key=lambda row: (-row[0], -row[1]))
        for offset, _mtime, path, meta in candidates:
            loaded = load_checkpoint(path)
            if loaded is None:
                continue
            # Trust the body's own offset over the sidecar copy.
            final_offset = int(loaded.token_offset) or offset
            if final_offset <= 0 or final_offset > len(prompt_token_ids):
                continue
            if meta.get("prompt_hash") != _prompt_hash(prompt_token_ids, final_offset):
                logger.warning(
                    "[kv_resume] body offset %d != sidecar offset %d for %s; "
                    "skipping mismatched checkpoint",
                    final_offset,
                    offset,
                    path,
                )
                continue
            logger.info(
                "[kv_resume] retry re-seed match: %d cached tokens for a "
                "%d-token prompt (from %s)",
                final_offset,
                len(prompt_token_ids),
                path,
            )
            return loaded
        return None
    except Exception as e:
        logger.warning("[kv_resume] prompt-based find failed: %s", e)
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
