# SPDX-License-Identifier: Apache-2.0
# #1354 audit H3: fetch_cache's hash-map branch looked blocks up with
# find_shared_prefix (which takes no reference) and incremented ref_count
# afterwards. A block freed + reallocated in that gap was pinned under a
# stale id with different content — a silent KV hole. claim=True takes the
# reference inside the same paged-lock hold as the hash lookup. The same
# change transfers the lazy-restore pin instead of stacking a second ref
# on it (M11: restored blocks were pinned forever).

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from fusion_mlx.cache.paged_cache import PagedCacheManager, compute_block_hash
from fusion_mlx.cache.prefix_cache import BlockAwarePrefixCache

BLOCK = 4


class _Model:
    num_layers = 4
    args = None

    def make_cache(self):
        return []


def _manager() -> PagedCacheManager:
    return PagedCacheManager(
        block_size=BLOCK,
        max_blocks=100,
        model_name="test-model",
        initial_blocks=100,
    )


def _register_chain(manager, tokens):
    blocks = []
    parent = None
    for i in range(len(tokens) // BLOCK):
        block = manager.allocate_block()
        assert block is not None
        block.token_count = BLOCK
        manager.register_block_hash(block, tokens[i * BLOCK : (i + 1) * BLOCK], parent)
        parent = block.block_hash
        blocks.append(block)
    return blocks


def test_claim_increments_hits_inside_the_lookup():
    manager = _manager()
    tokens = list(range(8))
    blocks = _register_chain(manager, tokens)
    assert [b.ref_count for b in blocks] == [1, 1]

    claimed, num_tokens = manager.get_computed_blocks(tokens, claim=True)

    assert num_tokens == 8
    assert [b.block_id for b in claimed] == [b.block_id for b in blocks]
    assert [b.ref_count for b in claimed] == [2, 2]


def test_unclaimed_lookup_leaves_refs_untouched():
    manager = _manager()
    blocks = _register_chain(manager, list(range(8)))

    looked_up, num_tokens = manager.get_computed_blocks(list(range(8)))

    assert num_tokens == 8
    assert [b.ref_count for b in looked_up] == [1, 1]
    assert [b.ref_count for b in blocks] == [1, 1]


def test_ssd_restore_claim_transfers_the_pin():
    # M11: the lazy-restore allocation pins the block with ref_count=1 so it
    # cannot be evicted before the caller claims it. A claiming caller must
    # take over that pin (ref stays 1), not stack a second reference on it.
    manager = _manager()
    tokens = list(range(8))
    hash1 = compute_block_hash(None, [0, 1, 2, 3], model_name="test-model")
    hash2 = compute_block_hash(hash1, [4, 5, 6, 7], model_name="test-model")

    ssd = MagicMock(spec=[])
    ssd.has_block = MagicMock(side_effect=lambda h: h in (hash1, hash2))
    manager._paged_ssd_cache_manager = ssd

    claimed, num_tokens = manager.get_computed_blocks(tokens, claim=True)

    assert num_tokens == 8
    assert [b.ref_count for b in claimed] == [1, 1]
    assert all(b.block_hash is not None for b in claimed)


def test_fetch_cache_claims_shared_blocks_for_the_request():
    manager = _manager()
    cache = BlockAwarePrefixCache(model=_Model(), paged_cache_manager=manager)
    tokens = list(range(8))
    blocks = _register_chain(manager, tokens)

    table, remaining = cache.fetch_cache("req-1", tokens)

    assert table is not None
    assert remaining == []
    assert table.num_tokens == 8
    assert [b.ref_count for b in blocks] == [2, 2]

    # A second request shares the same blocks (ref 2 -> 3).
    table2, remaining2 = cache.fetch_cache("req-2", tokens)
    assert table2 is not None and remaining2 == []
    assert [b.ref_count for b in blocks] == [3, 3]


def test_index_win_releases_shared_claims():
    # The index branch must hand back the hash-map claims it does not use,
    # otherwise those blocks stay pinned for a request that never reads them.
    manager = _manager()
    cache = BlockAwarePrefixCache(model=_Model(), paged_cache_manager=manager)
    tokens = list(range(24))
    shared = _register_chain(manager, tokens[:12])

    tail = []
    parent = shared[-1].block_hash
    for i in range(3, 6):
        block = manager.allocate_block()
        block.token_count = BLOCK
        manager.register_block_hash(block, tokens[i * BLOCK : (i + 1) * BLOCK], parent)
        parent = block.block_hash
        tail.append(block)
    cache._update_prefix_index(tokens, [b.block_id for b in shared + tail])

    table, remaining = cache.fetch_cache("req-index", tokens)

    assert table is not None
    assert table.num_tokens == 24
    assert remaining == []
    # Every block in the returned table carries exactly one reference for
    # this request. The shared blocks were claimed twice (hash-map probe, then
    # the index claim), so the unused hash-map claim must have been released —
    # without the release they would sit at 3.
    assert [b.ref_count for b in shared] == [2, 2, 2]
    assert [b.ref_count for b in tail] == [2, 2, 2]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
