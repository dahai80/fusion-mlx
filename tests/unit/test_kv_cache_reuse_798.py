# SPDX-License-Identifier: Apache-2.0
"""Verify server-side KV-cache reuse across requests (#798).

The core claim of #798: a long-context session reuses the prefix KV cache
computed by a prior request, instead of recomputing it. BlockAwarePrefixCache
is the default cache (scheduler/sched_init.py); its fetch_cache/store_cache
build a chain-hash block index so a second request sharing a token prefix
hits the cached blocks and only processes the suffix.

These tests run headless (no real MLX): store_cache with non-tensor cache_data
registers block hashes in PagedCacheManager without touching SSD, and
fetch_cache resolves the shared prefix via the same chain hash.
"""

from __future__ import annotations

import logging

from fusion_mlx.cache.paged_cache import PagedCacheManager
from fusion_mlx.cache.prefix_cache import BlockAwarePrefixCache

logger = logging.getLogger(__name__)


def _make_cache(block_size: int = 8, max_blocks: int = 64) -> BlockAwarePrefixCache:
    # A trivial stand-in model: _get_model_num_layers falls back to 0 when no
    # layers/cache attrs exist, which disables layer-count validation.
    class _FakeModel:
        pass

    paged = PagedCacheManager(
        block_size=block_size,
        max_blocks=max_blocks,
        enable_caching=True,
        model_name="test-model-798",
    )
    return BlockAwarePrefixCache(_FakeModel(), paged)


def test_cross_request_prefix_hit_reuses_blocks():
    # Request 1 processes a full prefix; request 2 shares that prefix and
    # extends it. fetch_cache on request 2 must return the cached blocks
    # (a hit) and leave only the suffix as remaining tokens.
    cache = _make_cache(block_size=8)
    prefix = list(range(100, 132))  # 32 tokens = 4 full blocks of 8
    suffix = list(range(200, 208))  # 8 tokens = 1 extra block

    # Request 1: store the prefix. Non-tensor cache_data skips SSD persistence
    # and only registers block metadata + the chain-hash prefix index.
    bt1 = cache.store_cache("req-1", prefix, [None])
    assert bt1 is not None
    assert len(bt1.block_ids) == 4
    assert bt1.num_tokens == 32

    # Request 2: same prefix + a suffix. fetch_cache should find the 4 shared
    # prefix blocks and return only the 8 suffix tokens as remaining.
    bt2, remaining = cache.fetch_cache("req-2", prefix + suffix)
    assert bt2 is not None, "expected a prefix-cache hit on the shared prefix"
    assert len(bt2.block_ids) == 4
    assert bt2.num_tokens == 32
    assert remaining == suffix

    # Hit/miss counters prove the reuse was counted, not silently skipped.
    assert cache._hits == 1
    assert cache._misses == 0
    assert cache._tokens_saved == 32


def test_distinct_prefix_is_a_miss():
    cache = _make_cache(block_size=8)
    cache.store_cache("req-1", list(range(100, 132)), [None])

    # A completely different token sequence must miss.
    bt, remaining = cache.fetch_cache("req-2", list(range(500, 540)))
    assert bt is None
    assert remaining == list(range(500, 540))
    assert cache._misses == 1
    assert cache._hits == 0


def test_partial_shared_prefix_hits_up_to_block_boundary():
    cache = _make_cache(block_size=8)
    first = list(range(100, 132))  # 32 tokens, 4 blocks
    cache.store_cache("req-1", first, [None])

    # Request 2 shares the first 2 blocks (16 tokens) then diverges.
    shared = first[:16] + list(range(900, 920))
    bt, remaining = cache.fetch_cache("req-2", shared)
    assert bt is not None
    assert bt.num_tokens == 16
    # Remaining is everything after the matched 16-token prefix.
    assert remaining == shared[16:]
    assert cache._hits == 1


def test_repeated_identical_requests_all_hit():
    cache = _make_cache(block_size=8)
    tokens = list(range(100, 132))
    cache.store_cache("req-1", tokens, [None])

    for i in range(2, 6):
        bt, remaining = cache.fetch_cache(f"req-{i}", tokens)
        assert bt is not None, f"req-{i} should hit"
        assert bt.num_tokens == 32
        assert remaining == []

    assert cache._hits == 4
