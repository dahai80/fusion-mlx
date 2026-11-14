# SPDX-License-Identifier: Apache-2.0
"""Warm-list: bounded local store of hot prompt heads for sleep-time
prefix pre-warming (Lever-3, docs/lever3-sleeptime-warmlist-design-1114).

At request end the scheduler records the same prompt head slice that
``prompt_shape_hashes`` hashed (first 1024 chars of a text prompt), keyed
by its head hash. An idle-time warm path later replays prefill-only on
these heads so the radix/prefix cache is hot before real traffic arrives.

Storage is a single JSON file in the runtime data dir, bounded by both
entry count and total bytes (LRU by last_seen). The head text never
leaves the machine — this is a local-first ecosystem, but the store is
still content on disk, so it ships with a kill switch
(FUSION_WARMLIST_DISABLE=1) and caps. Every failure path is non-fatal:
warm-list bookkeeping must never break inference.
"""

import json
import logging
import os
import threading
import time

logger = logging.getLogger(__name__)

_ENV_DISABLE = "FUSION_WARMLIST_DISABLE"
_ENV_MAX_ENTRIES = "FUSION_WARMLIST_MAX_ENTRIES"
_ENV_MAX_BYTES = "FUSION_WARMLIST_MAX_BYTES"

DEFAULT_MAX_ENTRIES = 256
DEFAULT_MAX_BYTES = 2 * 1024 * 1024
HEAD_CHARS = 1024
FLUSH_EVERY = 32


class WarmList:
    def __init__(self, path: str):
        self._path = path
        self._lock = threading.Lock()
        self._entries: dict[str, dict] = {}
        self._dirty = 0
        self._loaded = False
        self._disabled = os.environ.get(_ENV_DISABLE, "") == "1"

    def _load(self) -> None:
        if self._loaded:
            return
        self._loaded = True
        try:
            with open(self._path, encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                self._entries = {
                    k: v
                    for k, v in data.items()
                    if isinstance(v, dict) and isinstance(v.get("head"), str)
                }
        except FileNotFoundError:
            pass
        except Exception as e:
            logger.debug("warm-list: load failed, starting empty: %s", e)

    def _flush_locked(self) -> None:
        try:
            os.makedirs(os.path.dirname(self._path), exist_ok=True)
            tmp = self._path + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(self._entries, f, ensure_ascii=False)
            os.replace(tmp, self._path)
            self._dirty = 0
        except Exception as e:
            logger.debug("warm-list: flush failed (non-fatal): %s", e)

    def _evict_locked(self) -> None:
        max_entries = int(os.environ.get(_ENV_MAX_ENTRIES, DEFAULT_MAX_ENTRIES))
        max_bytes = int(os.environ.get(_ENV_MAX_BYTES, DEFAULT_MAX_BYTES))
        entries = self._entries
        while len(entries) > max_entries:
            oldest = min(entries, key=lambda k: entries[k].get("last_seen", 0))
            del entries[oldest]
        total = sum(len(v.get("head", "")) for v in entries.values())
        while total > max_bytes and entries:
            oldest = min(entries, key=lambda k: entries[k].get("last_seen", 0))
            total -= len(entries[oldest].get("head", ""))
            del entries[oldest]

    def record(self, head_hash: str, prompt, prompt_tokens: int | None) -> None:
        if self._disabled or not head_hash:
            return
        if not isinstance(prompt, str) or not prompt:
            return
        head = prompt[:HEAD_CHARS]
        now = time.time()
        try:
            with self._lock:
                self._load()
                prev = self._entries.get(head_hash)
                entry = {
                    "head": head,
                    "prompt_tokens": prompt_tokens,
                    "last_seen": now,
                    "hits": (prev.get("hits", 0) + 1) if prev else 1,
                }
                self._entries[head_hash] = entry
                self._evict_locked()
                self._dirty += 1
                if self._dirty >= FLUSH_EVERY:
                    self._flush_locked()
        except Exception as e:
            logger.debug("warm-list: record failed (non-fatal): %s", e)

    def top(self, k: int) -> list[dict]:
        with self._lock:
            self._load()
            ranked = sorted(
                self._entries.items(),
                key=lambda kv: (kv[1].get("hits", 0), kv[1].get("last_seen", 0)),
                reverse=True,
            )
            return [
                {
                    "head_hash": h,
                    **{kk: vv for kk, vv in v.items() if kk != "head"},
                    "head_len": len(v.get("head", "")),
                }
                for h, v in ranked[:k]
            ]

    def head_for(self, head_hash: str) -> str | None:
        with self._lock:
            self._load()
            entry = self._entries.get(head_hash)
            return entry.get("head") if entry else None

    def flush(self) -> None:
        with self._lock:
            self._load()
            if self._dirty:
                self._flush_locked()

    def stats(self) -> dict:
        with self._lock:
            self._load()
            return {
                "entries": len(self._entries),
                "dirty": self._dirty,
                "disabled": self._disabled,
                "path": self._path,
            }


_default: WarmList | None = None
_default_lock = threading.Lock()


def get_warm_list(path: str | None = None) -> WarmList:
    global _default
    with _default_lock:
        if _default is None:
            if path is None:
                base = os.environ.get(
                    "FUSION_WARMLIST_DATA_DIR",
                    os.path.expanduser("~/.fusion-mlx"),
                )
                path = os.path.join(base, "warm_list.json")
            _default = WarmList(path)
        return _default
