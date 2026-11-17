# SPDX-License-Identifier: Apache-2.0
"""M9 KV drop/pin primitives: drop_request_segments detaches an active
request's cached blocks from the prefix indexes without freeing them
(arXiv 2610.28577). Content stays valid for the owning request's decode;
no future fetch_cache may match dropped or downstream blocks."""

import logging
from unittest.mock import MagicMock

import pytest

from fusion_mlx.cache.paged_cache import PagedCacheManager
from fusion_mlx.cache.prefix_cache import BlockAwarePrefixCache

logger = logging.getLogger(__name__)


class MockModel:
    def __init__(self, num_layers: int = 4):
        self._num_layers = num_layers
        self.layers = [MagicMock() for _ in range(num_layers)]

    @property
    def args(self):
        mock_args = MagicMock()
        mock_args.num_hidden_layers = self._num_layers
        return mock_args


@pytest.fixture
def paged_cache():
    return PagedCacheManager(
        block_size=4,
        max_blocks=100,
        model_name="test-model",
        initial_blocks=100,
    )


@pytest.fixture
def prefix_cache(paged_cache):
    return BlockAwarePrefixCache(
        model=MockModel(num_layers=4),
        paged_cache_manager=paged_cache,
        paged_ssd_cache_manager=None,
    )


def _store_blocks(prefix_cache, paged_cache, request_id, tokens):
    """Allocate + register one block per block_size chunk and index them
    under request_id, mirroring the admission/store path."""
    bs = prefix_cache.block_size
    table = paged_cache.get_or_create_block_table(request_id)
    parent = None
    for i in range(0, len(tokens) - len(tokens) % bs, bs):
        block = paged_cache.allocate_block()
        block.token_count = bs
        paged_cache.register_block_hash(block, tokens[i : i + bs], parent)
        parent = block.block_hash
        table.add_block(block.block_id, bs)
    prefix_cache._update_prefix_index(tokens, table.block_ids)


class TestDropRequestSegments:
    def test_interior_drop_detaches_and_keeps_content(self, prefix_cache, paged_cache):
        tokens = list(range(24))
        _store_blocks(prefix_cache, paged_cache, "req-a", tokens)

        table, remaining = prefix_cache.fetch_cache("other", tokens)
        assert table is not None and remaining == []

        detached = prefix_cache.drop_request_segments("req-a", 4, 12)
        assert detached == 5  # blocks 1,2,3 fully inside + tail blocks 4,5

        # Any future fetch must not match beyond the untouched first block.
        table2, remaining2 = prefix_cache.fetch_cache("other2", tokens)
        matched = 0 if table2 is None else table2.num_tokens
        assert matched <= 4, f"dropped blocks must not match, matched {matched}"
        # The owning request's blocks keep content + table membership.
        t = paged_cache.get_block_table("req-a")
        assert t is not None and len(t.block_ids) == 6

    def test_partial_coverage_drops_nothing(self, prefix_cache, paged_cache):
        tokens = list(range(16))
        _store_blocks(prefix_cache, paged_cache, "req-b", tokens)
        before = prefix_cache.fetch_cache("x", tokens)
        assert before[0] is not None
        # Range smaller than one full block (spans within block 1 only).
        assert prefix_cache.drop_request_segments("req-b", 5, 7) == 0
        after = prefix_cache.fetch_cache("x2", tokens)
        assert after[0] is not None

    def test_tail_only_drop(self, prefix_cache, paged_cache):
        tokens = list(range(16))
        _store_blocks(prefix_cache, paged_cache, "req-c", tokens)
        # Drop last block (tokens 12..16).
        assert prefix_cache.drop_request_segments("req-c", 12, 16) == 1
        table, remaining = prefix_cache.fetch_cache("y", tokens)
        # First 12 tokens may still match; tail 4 must be recomputed.
        matched = 0 if table is None else table.num_tokens
        assert matched <= 12

    def test_degenerate_inputs(self, prefix_cache, paged_cache):
        assert prefix_cache.drop_request_segments("missing", 0, 8) == 0
        tokens = list(range(8))
        _store_blocks(prefix_cache, paged_cache, "req-d", tokens)
        assert prefix_cache.drop_request_segments("req-d", 4, 4) == 0
        assert prefix_cache.drop_request_segments("req-d", 8, 4) == 0
        # Out-of-range end is clamped to table length.
        assert prefix_cache.drop_request_segments("req-d", 4, 999) == 1

    def test_owning_request_decode_view_unchanged(self, prefix_cache, paged_cache):
        tokens = list(range(24))
        _store_blocks(prefix_cache, paged_cache, "req-e", tokens)
        ids_before = list(paged_cache.get_block_table("req-e").block_ids)
        prefix_cache.drop_request_segments("req-e", 4, 20)
        ids_after = list(paged_cache.get_block_table("req-e").block_ids)
        assert ids_before == ids_after
        for bid in ids_after:
            block = paged_cache.allocated_blocks.get(bid)
            assert block is not None
            assert block.ref_count >= 1
