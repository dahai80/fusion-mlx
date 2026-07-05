# SPDX-License-Identifier: Apache-2.0
"""Regression for issue #214 — prefix cache misses on every turn for hybrid
attention models in growing multi-turn conversations (migrated from Rapid-MLX).

We isolate the cache-lookup layer (no model load) and assert the multi-turn
growing pattern hits the prefix path on hybrid layouts the same way it does
on dense.
"""

from unittest.mock import MagicMock

import pytest

from fusion_mlx.memory_cache import MemoryAwarePrefixCache, MemoryCacheConfig


class _MockArray:
    def __init__(self, nbytes: int):
        self.nbytes = nbytes


class TrimmableLayer:

    def __init__(self, nbytes: int = 200, offset: int = 0):
        self.keys = _MockArray(nbytes // 2)
        self.values = _MockArray(nbytes // 2)
        self._offset = offset

    @property
    def offset(self) -> int:
        return self._offset

    @offset.setter
    def offset(self, val: int) -> None:
        self._offset = val

    def is_trimmable(self) -> bool:
        return True


class NonTrimmableLayer:

    def __init__(self, nbytes: int = 200):
        self.keys = _MockArray(nbytes // 2)
        self.values = _MockArray(nbytes // 2)

    def is_trimmable(self) -> bool:
        return False


def _hybrid_cache(n_trimmable: int = 10, n_non_trimmable: int = 30):
    return [TrimmableLayer() for _ in range(n_trimmable)] + [
        NonTrimmableLayer() for _ in range(n_non_trimmable)
    ]


@pytest.fixture
def cache():
    config = MemoryCacheConfig(max_memory_mb=10, max_entries=64)
    return MemoryAwarePrefixCache(MagicMock(), config)


def test_dense_growing_conversation_hits_prefix(cache):
    prompt = list(range(1000, 1100))
    response_1 = [9001, 9002]
    new_msg = list(range(2000, 2050))

    cache.store(prompt, _hybrid_cache(n_trimmable=10, n_non_trimmable=0))
    cache.store(prompt + response_1, _hybrid_cache(n_trimmable=10, n_non_trimmable=0))

    turn_2 = prompt + response_1 + new_msg
    result, remaining = cache.fetch(turn_2)

    assert result is not None, "Dense growing conversation should hit prefix"
    assert remaining == new_msg


def test_hybrid_growing_conversation_hits_prefix(cache):
    prompt = list(range(1000, 1100))
    response_1 = [9001, 9002]
    new_msg = list(range(2000, 2050))

    cache.store(prompt, _hybrid_cache())
    cache.store(prompt + response_1, _hybrid_cache())

    turn_2 = prompt + response_1 + new_msg
    result, remaining = cache.fetch(turn_2)

    assert result is not None, (
        "Hybrid growing conversation MISSED prefix cache — this is issue #214. "
        "Stored [P + R1] is a strict prefix of request [P + R1 + M2]; no "
        "trim is required, so non-trimmable RNN layers should not block."
    )
    assert remaining == new_msg, (
        f"Expected remaining = M2 ({len(new_msg)} tokens, picking [P+R1]); "
        f"got {len(remaining)} tokens"
    )


def test_hybrid_only_prompt_stored_still_hits(cache):
    prompt = list(range(1000, 1100))
    response_1 = [9001, 9002]
    new_msg = list(range(2000, 2050))

    cache.store(prompt, _hybrid_cache())

    turn_2 = prompt + response_1 + new_msg
    result, remaining = cache.fetch(turn_2)

    assert result is not None, "Stored [P] must hit as prefix of [P + R1 + M2]"
    assert remaining == response_1 + new_msg


def test_hybrid_three_turn_growth(cache):
    prompt = list(range(1000, 1100))
    r1, r2 = [9001, 9002], [9003, 9004]
    m2, m3 = list(range(2000, 2050)), list(range(3000, 3030))

    cache.store(prompt, _hybrid_cache())
    cache.store(prompt + r1, _hybrid_cache())

    turn_2 = prompt + r1 + m2
    res, rem = cache.fetch(turn_2)
    assert res is not None
    assert rem == m2, "Turn 2 should pick [P + R1] (longest prefix)"

    cache.store(prompt + r1 + m2 + r2, _hybrid_cache())

    turn_3 = prompt + r1 + m2 + r2 + m3
    res, rem = cache.fetch(turn_3)
    assert res is not None
    assert rem == m3, "Turn 3 should pick [P + R1 + M2 + R2] (longest prefix)"


def test_hybrid_supersequence_still_skipped(cache):
    long_stored = list(range(1000, 1200))
    cache.store(long_stored, _hybrid_cache())

    short_request = list(range(1000, 1100))
    result, remaining = cache.fetch(short_request)

    assert (
        result is None
    ), "Trim-required match on non-trimmable hybrid layers must still skip"
    assert remaining == short_request


def test_hybrid_lcp_with_divergence_still_skipped(cache):
    stored = list(range(1000, 1100)) + [5000, 5001, 5002]
    cache.store(stored, _hybrid_cache())

    request = list(range(1000, 1100)) + [6000, 6001, 6002]
    result, remaining = cache.fetch(request)

    assert result is None
    assert remaining == request
