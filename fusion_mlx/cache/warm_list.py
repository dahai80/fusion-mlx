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
_ENV_IDLE_S = "FUSION_WARMLIST_IDLE_S"
_ENV_TOPK = "FUSION_WARMLIST_TOPK"
_ENV_MIN_TOKENS = "FUSION_WARMLIST_MIN_TOKENS"

DEFAULT_MAX_ENTRIES = 256
DEFAULT_MAX_BYTES = 2 * 1024 * 1024
DEFAULT_IDLE_S = 600
DEFAULT_TOPK = 3
# Warm replay only pays off when the stored prefix is long enough for a
# cache hit to skip real work. Production observation (2026-11-16): with
# no filter the store fills with 4-10 token smoke-test prompts ("warmup:")
# that prefill in microseconds and warm nothing. 256 matches the tail of
# the stage-1 concentration segment (>=512 showed 67.9% top-5 coverage).
DEFAULT_MIN_TOKENS = 256
HEAD_CHARS = 1024
# Token-id prompts: head slice matches prompt_shape_hashes' 256-id window
# so a stored head replays exactly the prefix the head hash identifies.
HEAD_TOKEN_IDS = 256
FLUSH_EVERY = 32
FLUSH_INTERVAL_S = 60


def warm_idle_threshold_s() -> float:
    try:
        return float(os.environ.get(_ENV_IDLE_S, DEFAULT_IDLE_S))
    except ValueError:
        return float(DEFAULT_IDLE_S)


def warm_topk() -> int:
    try:
        return max(0, int(os.environ.get(_ENV_TOPK, DEFAULT_TOPK)))
    except ValueError:
        return DEFAULT_TOPK


def warm_min_tokens() -> int:
    try:
        return max(0, int(os.environ.get(_ENV_MIN_TOKENS, DEFAULT_MIN_TOKENS)))
    except ValueError:
        return DEFAULT_MIN_TOKENS


class WarmList:
    def __init__(self, path: str):
        self._path = path
        self._lock = threading.Lock()
        self._entries: dict[str, dict] = {}
        self._dirty = 0
        self._loaded = False
        self._last_flush_ts = 0.0
        self._disabled = os.environ.get(_ENV_DISABLE, "") == "1"

    def _is_disabled(self) -> bool:
        # Read the kill switch at call time: env may be set after process
        # start (operator hot-disable) and must take effect immediately.
        return os.environ.get(_ENV_DISABLE, "") == "1"

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
                    if isinstance(v, dict)
                    and (
                        isinstance(v.get("head"), str)
                        or (
                            isinstance(v.get("head"), list)
                            and bool(v.get("head"))
                            and isinstance(v.get("head")[0], int)
                        )
                    )
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
            self._last_flush_ts = time.time()
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
        if self._is_disabled() or not head_hash:
            return
        # Accept both prompt shapes the scheduler takes: rendered text
        # (str, /v1/completions-style) and token ids (list[int], the chat
        # path — production observation showed chat requests, the majority
        # of traffic, were silently dropped by a str-only guard).
        if isinstance(prompt, str):
            if not prompt:
                return
            head = prompt[:HEAD_CHARS]
        elif isinstance(prompt, list) and prompt and isinstance(prompt[0], int):
            # Same slice prompt_head_hash covers for token-id prompts.
            head = prompt[:HEAD_TOKEN_IDS]
        else:
            return
        # Size filter: prompts below the threshold prefill too fast for a
        # warm cache hit to matter. Unknown token count is treated as too
        # small rather than guessed.
        min_tokens = warm_min_tokens()
        if prompt_tokens is None or prompt_tokens < min_tokens:
            logger.debug(
                "warm-list: skipping short prompt (tokens=%s, min=%d)",
                prompt_tokens,
                min_tokens,
            )
            return
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
                if self._dirty >= FLUSH_EVERY or (
                    now - self._last_flush_ts >= FLUSH_INTERVAL_S
                ):
                    # Time-based flush keeps small traffic days durable:
                    # a lone record must not wait for 31 siblings.
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

    def top_heads(self, k: int) -> list[tuple[str, str]]:
        # Top-k head texts for the idle warm path, ranked by hits then
        # recency. Returns (head_hash, head_text) pairs; empty head text
        # entries and below-threshold entries (stored before the size
        # filter existed) are skipped, so legacy tiny heads age out of
        # the replay path without rewriting the store.
        min_tokens = warm_min_tokens()
        with self._lock:
            self._load()
            ranked = sorted(
                self._entries.items(),
                key=lambda kv: (kv[1].get("hits", 0), kv[1].get("last_seen", 0)),
                reverse=True,
            )
            return [
                (h, v["head"])
                for h, v in ranked[:k]
                if v.get("head") and (v.get("prompt_tokens") or 0) >= min_tokens
            ]

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
                "disabled": self._is_disabled(),
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
