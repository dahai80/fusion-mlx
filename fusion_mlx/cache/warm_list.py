# SPDX-License-Identifier: Apache-2.0
"""Warm-list: bounded local store of hot prompt heads for sleep-time
prefix pre-warming (Lever-3, docs/lever3-sleeptime-warmlist-design-1114).

At request end the scheduler records a prompt head slice (block-aligned
window, default 2048 token ids / 8192 chars), keyed by its head hash. An
idle-time warm path later replays prefill-only on
these heads so the radix/prefix cache is hot before real traffic arrives.

Storage is a single JSON file in the runtime data dir, bounded by both
entry count and total bytes (LRU by last_seen). The head text never
leaves the machine — this is a local-first ecosystem, but the store is
still content on disk, so it ships with a kill switch
(FUSION_WARMLIST_DISABLE=1) and caps. Every failure path is non-fatal:
warm-list bookkeeping must never break inference.
"""

import atexit
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
_ENV_HEAD_IDS = "FUSION_WARMLIST_HEAD_IDS"
_ENV_HEAD_CHARS = "FUSION_WARMLIST_HEAD_CHARS"

DEFAULT_MAX_ENTRIES = 256
# 256 entries x up to 2048 token ids (or 8192 chars) of JSON is tens of
# MB; the old 2MiB cap would LRU-thrash the store down to a few long
# heads (#1634).
DEFAULT_MAX_BYTES = 48 * 1024 * 1024
DEFAULT_IDLE_S = 600
DEFAULT_TOPK = 3
# Warm replay only pays off when the stored prefix is long enough for a
# cache hit to skip real work. Production observation (2026-11-16): with
# no filter the store fills with 4-10 token smoke-test prompts ("warmup:")
# that prefill in microseconds and warm nothing. 256 matches the tail of
# the stage-1 concentration segment (>=512 showed 67.9% top-5 coverage).
DEFAULT_MIN_TOKENS = 256
# The block-aware prefix cache matches complete paged-cache blocks
# (block_size=2048 in production), so a replayed head must reach at
# least one full block before it can produce a hit (#1634: the old
# 256-id slice always stayed partial and bought zero benefit — measured
# ttft parity with an unwarmed repeat). 2048 = one block; larger values
# warm more blocks per replay.
DEFAULT_HEAD_IDS = 2048
# Text prompts: ~4 chars/token, so 8192 chars ≈ the same one-block
# target as DEFAULT_HEAD_IDS.
DEFAULT_HEAD_CHARS = 8192
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


def warm_head_ids() -> int:
    try:
        return max(0, int(os.environ.get(_ENV_HEAD_IDS, DEFAULT_HEAD_IDS)))
    except ValueError:
        return DEFAULT_HEAD_IDS


def warm_head_chars() -> int:
    try:
        return max(0, int(os.environ.get(_ENV_HEAD_CHARS, DEFAULT_HEAD_CHARS)))
    except ValueError:
        return DEFAULT_HEAD_CHARS


def _head_size(head) -> int:
    # Serialized size for byte-cap accounting. Lists of ids need json
    # length, not len() (which would count token count, ~5x smaller than
    # their serialized footprint).
    try:
        return len(json.dumps(head, ensure_ascii=False))
    except Exception:
        return len(str(head))


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

    def _entry_size(self, v: dict) -> int:
        b = v.get("bytes")
        if isinstance(b, int) and b > 0:
            return b
        return _head_size(v.get("head", ""))

    def _evict_locked(self) -> None:
        max_entries = int(os.environ.get(_ENV_MAX_ENTRIES, DEFAULT_MAX_ENTRIES))
        max_bytes = int(os.environ.get(_ENV_MAX_BYTES, DEFAULT_MAX_BYTES))
        entries = self._entries
        while len(entries) > max_entries:
            oldest = min(entries, key=lambda k: entries[k].get("last_seen", 0))
            del entries[oldest]
        total = sum(self._entry_size(v) for v in entries.values())
        while total > max_bytes and entries:
            oldest = min(entries, key=lambda k: entries[k].get("last_seen", 0))
            total -= self._entry_size(entries[oldest])
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
            head = prompt[: warm_head_chars()]
        elif isinstance(prompt, list) and prompt and isinstance(prompt[0], int):
            # Block-aligned replay slice (#1634): the block-aware cache
            # only serves complete paged-cache blocks, so the head must
            # reach at least one block for replay to produce a hit.
            # prompt_head_hash telemetry keeps its own 256-id window.
            head = prompt[: warm_head_ids()]
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
                    "bytes": _head_size(head),
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
        # the replay path without rewriting the store. Filtering runs
        # BEFORE the top-k slice: rank-then-filter lets a few high-hit
        # legacy short heads crowd every eligible long head out of the
        # window entirely (observed in production: store top-3 by hits
        # were 4-10 token warmup prompts, so top_heads() returned []
        # while a genuine 509-token chat head sat at rank 16).
        min_tokens = warm_min_tokens()
        with self._lock:
            self._load()
            eligible = (
                (h, v)
                for h, v in self._entries.items()
                if v.get("head") and (v.get("prompt_tokens") or 0) >= min_tokens
            )
            ranked = sorted(
                eligible,
                key=lambda kv: (kv[1].get("hits", 0), kv[1].get("last_seen", 0)),
                reverse=True,
            )
            return [(h, v["head"]) for h, v in ranked[:k]]

    def flush(self) -> None:
        with self._lock:
            self._load()
            if self._dirty:
                self._flush_locked()

    def flush_if_stale(self, now: float) -> None:
        # Time-driven flush for callers with a heartbeat (engine loop tick):
        # the in-record() timer only fires while traffic keeps coming, so a
        # quiet day + restart would lose every entry but the first. Cheap
        # no-lock check first; the tick runs every engine step. Uses wall
        # clock internally — `now` may be time.monotonic() at the call site
        # and must never be compared against the epoch-based _last_flush_ts.
        if not self._dirty:
            return
        if time.time() - self._last_flush_ts < FLUSH_INTERVAL_S:
            return
        self.flush()

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


def _flush_default_at_exit() -> None:
    # The 60s flush timer is evaluated opportunistically inside record()
    # only — with no further traffic it never fires, so a low-traffic day
    # followed by a restart loses every entry after the first record of
    # each process. Flush whatever is dirty at interpreter shutdown.
    try:
        if _default is not None:
            _default.flush()
    except Exception as e:
        logger.debug("warm-list: atexit flush failed (non-fatal): %s", e)


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
            atexit.register(_flush_default_at_exit)
        return _default
