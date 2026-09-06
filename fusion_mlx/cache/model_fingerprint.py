# SPDX-License-Identifier: Apache-2.0
"""CS-4 / CS-5 / CS-6 (#811 audit 0906): model/adapter content fingerprint.

The KV prefix cache, boundary snapshot store, and response cache all isolate
entries by a model identity string. That string is the resolved weight *path*
(for paged/boundary caches) or the request *alias* (for the response cache) —
neither is a content revision. Re-pulling, re-quantizing, or retraining weights
*at the same path/alias* leaves the cache key unchanged, so the next request
silently restores stale KV / returns a stale completion built from the OLD
weights. For LoRA adapters the same hole applies: overwriting the adapter
weights at the same path reuses completions computed against the old adapter.

This module computes a cheap filesystem signal — total size of the weight
files plus their max mtime — and folds it into the cache key. A content
change at the same path then yields a different key, so stale entries are
simply never matched (they age out by TTL / eviction rather than poisoning
new requests). When the path cannot be stat'd (non-local id, missing dir,
no weight files) the raw path/alias is returned unchanged so behavior never
regresses relative to the pre-fix keying.
"""

import hashlib
import logging
import os

logger = logging.getLogger(__name__)

_WEIGHT_SUFFIXES = (".safetensors", ".gguf", ".npz", ".bin")


def model_path_signature(path: str) -> str:
    """Return a content-aware signature for a model weight directory.

    ``sha256(path + "|" + total_weight_bytes + "|" + max_weight_mtime)`` when
    weight files are found under ``path``; otherwise the raw ``path`` (no
    regression vs. the old path-only key). The walk runs once per cache init,
    not per request.
    """
    if not path:
        return ""
    total, max_mtime = _scan_weight_files(path)
    if total == 0:
        return path
    sig = hashlib.sha256(f"{path}|{total}|{int(max_mtime)}".encode()).hexdigest()
    logger.debug(
        "CS-4/CS-6 model signature %s -> %s (bytes=%d mtime=%d)",
        path,
        sig[:12],
        total,
        int(max_mtime),
    )
    return sig


def adapter_path_signature(adapter_path: str) -> str:
    """Return a content-aware signature for a LoRA adapter path.

    Same idea as :func:`model_path_signature` but for adapter weights, called
    per cached request (adapter dirs are tiny, so the walk is cheap). Returns
    the raw path when no weight files are found.
    """
    if not adapter_path:
        return ""
    total, max_mtime = _scan_weight_files(adapter_path)
    if total == 0:
        return adapter_path
    return hashlib.sha256(
        f"{adapter_path}|{total}|{int(max_mtime)}".encode()
    ).hexdigest()


def _scan_weight_files(path: str) -> tuple[int, float]:
    total = 0
    max_mtime = 0.0
    try:
        for root, _dirs, files in os.walk(path):
            for fn in files:
                if fn.endswith(_WEIGHT_SUFFIXES):
                    try:
                        st = os.stat(os.path.join(root, fn))
                    except OSError:
                        continue
                    total += st.st_size
                    if st.st_mtime > max_mtime:
                        max_mtime = st.st_mtime
    except OSError:
        return 0, 0.0
    return total, max_mtime
