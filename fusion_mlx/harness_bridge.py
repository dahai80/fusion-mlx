# SPDX-License-Identifier: Apache-2.0
"""fusion-harness bridge objects (#1559).

Consumer-side adapters satisfying the fusion-harness CompactionConfig
injection seams (evolution plan P1-2 / P3-1):

- RemoteTokenCounter  -> CompactionConfig.token_counter
                          (TokenCounter protocol: count_text(text) -> int)
- PrefixCacheAdvisor  -> CompactionConfig.cache_advisor
                          (suggest_keep_rounds(messages) -> int)

Both talk to the engine's /harness/v1 routes over HTTP. Consumers import
this module and wire the objects into their CompactionConfig; the
stdlib-only harness kernel stays engine-free.

Advisor semantics (v1): raise the keep window so a compaction drops at
least one round while keeping the post-compaction prompt at or below
max_keep_fraction of the current prompt — fewer compaction events over a
session lifetime, and each surviving cache region (system prefix blocks)
is wiped less often. The advisor only ever RAISES (harness enforces the
same invariant); every failure path returns 0, which the harness treats
as "no opinion" and falls back to the configured keep_recent_rounds.
"""

import logging
import time

import httpx

logger = logging.getLogger(__name__)

_DEFAULT_TIMEOUT_S = 10.0


class _BaseRemote:
    def __init__(
        self,
        base_url: str,
        model_id: str | None = None,
        api_key: str | None = None,
        timeout_s: float = _DEFAULT_TIMEOUT_S,
        transport: httpx.BaseTransport | None = None,
    ):
        self._base_url = base_url.rstrip("/")
        self._model_id = model_id
        headers = {}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        self._client = httpx.Client(
            base_url=self._base_url,
            headers=headers,
            timeout=timeout_s,
            transport=transport,
        )

    def close(self) -> None:
        self._client.close()

    def _post(self, path: str, payload: dict) -> dict:
        response = self._client.post(path, json=payload)
        response.raise_for_status()
        return response.json()


class RemoteTokenCounter(_BaseRemote):
    """TokenCounter protocol backed by the engine's real tokenizer."""

    def count_text(self, text: str) -> int:
        return self.count_texts([text])[0]

    def count_texts(self, texts: list[str]) -> list[int]:
        if not texts:
            return []
        payload: dict = {"texts": texts}
        if self._model_id:
            payload["model_id"] = self._model_id
        data = self._post("/harness/v1/tokens/count", payload)
        return [int(c) for c in data["counts"]]


class PrefixCacheAdvisor(_BaseRemote):
    """CacheAdvisor protocol backed by the engine's cache-probe route.

    Policy: suggest a keep window that halves the current prompt (kept
    tokens <= max_keep_fraction * total) whenever the engine reports a
    cached prefix worth protecting (>= min_cached_tokens). When nothing
    is cached, or any probe/count call fails, returns 0 — the harness
    then uses its configured keep_recent_rounds unchanged.
    """

    def __init__(
        self,
        base_url: str,
        model_id: str | None = None,
        api_key: str | None = None,
        timeout_s: float = _DEFAULT_TIMEOUT_S,
        transport: httpx.BaseTransport | None = None,
        min_cached_tokens: int = 1024,
        max_keep_fraction: float = 0.5,
    ):
        super().__init__(
            base_url=base_url,
            model_id=model_id,
            api_key=api_key,
            timeout_s=timeout_s,
            transport=transport,
        )
        self._min_cached_tokens = min_cached_tokens
        self._max_keep_fraction = max_keep_fraction

    def _probe(self, messages: list[dict]) -> dict:
        payload: dict = {"messages": messages}
        if self._model_id:
            payload["model_id"] = self._model_id
        return self._post("/harness/v1/cache/probe", payload)

    def _round_start_indexes(self, messages: list[dict]) -> list[int]:
        # Mirror harness Compactor._partition + _split_rounds exactly:
        # leading system messages stay fixed; the remainder splits into
        # rounds where each round starts at a user message, and any
        # non-user messages before the first user message form the first
        # round.
        system_end = 0
        for m in messages:
            if m.get("role") == "system":
                system_end += 1
            else:
                break
        rest = messages[system_end:]
        if not rest:
            return []
        starts = [system_end]
        for i, m in enumerate(rest, start=system_end):
            if m.get("role") == "user" and i != system_end:
                starts.append(i)
        return starts

    def suggest_keep_rounds(self, messages: list[dict]) -> int:
        t0 = time.perf_counter()
        try:
            probe = self._probe(messages)
            cached = int(probe.get("cached_prefix_tokens", 0))
            total = int(probe.get("total_tokens", 0))
        except Exception as exc:
            logger.warning("cache advisor probe failed, no opinion: %s", exc)
            return 0
        if cached < self._min_cached_tokens or total <= 0:
            logger.debug(
                "cache advisor: nothing worth protecting (cached=%d total=%d)",
                cached,
                total,
            )
            return 0

        try:
            counts = self._count_per_message(messages)
        except Exception as exc:
            logger.warning("cache advisor count failed, no opinion: %s", exc)
            return 0
        if counts is None:
            return 0

        starts = self._round_start_indexes(messages)
        n_rounds = len(starts)
        if n_rounds <= 1:
            return 0

        # Cumulative token offset for each message index (approximate:
        # per-message counts, template overhead spread evenly).
        overhead = max(0, total - sum(counts)) // max(1, len(messages))
        offsets = [0] * (len(messages) + 1)
        running = 0
        for i, m in enumerate(messages):
            offsets[i] = running
            running += counts[i] + overhead
        offsets[len(messages)] = running

        budget = int(total * self._max_keep_fraction)
        # Minimal keep k whose kept window (rounds[n-k:]) fits the budget.
        # keep=k keeps rounds[n-k:], so the kept window starts at
        # message index starts[n-k].
        keep = n_rounds - 1
        for k in range(1, n_rounds):
            kept_start = starts[n_rounds - k]
            kept = total - offsets[kept_start]
            if kept <= budget:
                keep = k
                break

        elapsed_ms = (time.perf_counter() - t0) * 1000
        logger.info(
            "cache advisor: cached=%d total=%d rounds=%d suggested_keep=%d (%.1f ms)",
            cached,
            total,
            n_rounds,
            keep,
            elapsed_ms,
        )
        return keep

    def _count_per_message(self, messages: list[dict]) -> list[int] | None:
        texts = []
        for m in messages:
            content = m.get("content") or ""
            if not isinstance(content, str):
                content = str(content)
            texts.append(content)
        payload: dict = {"texts": texts}
        if self._model_id:
            payload["model_id"] = self._model_id
        data = self._post("/harness/v1/tokens/count", payload)
        return [int(c) for c in data["counts"]]
