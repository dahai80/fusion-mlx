# SPDX-License-Identifier: Apache-2.0
# Tests for #1264: paged-cache block cap scaled to the model's real context
# length (context-derived floor over max_cache_blocks) + OOB observability.
from types import SimpleNamespace

import pytest

from fusion_mlx.cache.paged_cache import PagedCacheManager
from fusion_mlx.scheduler.sched_init import (
    _effective_max_blocks,
    _model_context_length,
)
from fusion_mlx.server_metrics import get_server_metrics


class TestModelContextLength:
    def test_direct_max_position_embeddings(self):
        model = SimpleNamespace(args=SimpleNamespace(max_position_embeddings=262144))
        assert _model_context_length(model) == 262144

    def test_text_config_nesting_for_vlm(self):
        model = SimpleNamespace(
            args=SimpleNamespace(
                max_position_embeddings=None,
                text_config=SimpleNamespace(max_position_embeddings=131072),
            )
        )
        assert _model_context_length(model) == 131072

    def test_no_args_returns_none(self):
        assert _model_context_length(SimpleNamespace()) is None

    def test_garbage_value_returns_none(self):
        model = SimpleNamespace(args=SimpleNamespace(max_position_embeddings="abc"))
        assert _model_context_length(model) is None


class TestEffectiveMaxBlocks:
    def test_floor_raises_configured_cap(self):
        # Qwen3.8-27B-4bit shape: 262144 ctx / 64 block = 4096 needed > 1000
        model = SimpleNamespace(args=SimpleNamespace(max_position_embeddings=262144))
        assert _effective_max_blocks(1000, model, 64) == 4096

    def test_configured_above_floor_wins(self):
        model = SimpleNamespace(args=SimpleNamespace(max_position_embeddings=262144))
        assert _effective_max_blocks(8192, model, 64) == 8192

    def test_exact_boundary_not_raised(self):
        # 4096 blocks exactly cover 262144 tokens — no raise needed.
        model = SimpleNamespace(args=SimpleNamespace(max_position_embeddings=262144))
        assert _effective_max_blocks(4096, model, 64) == 4096

    def test_unknown_context_keeps_configured(self):
        assert _effective_max_blocks(1000, SimpleNamespace(), 64) == 1000

    def test_non_exact_context_ceils(self):
        # 100000 ctx / 64 = 1562.5 -> 1563
        model = SimpleNamespace(args=SimpleNamespace(max_position_embeddings=100000))
        assert _effective_max_blocks(1000, model, 64) == 1563

    def test_zero_block_size_keeps_configured(self):
        model = SimpleNamespace(args=SimpleNamespace(max_position_embeddings=262144))
        assert _effective_max_blocks(1000, model, 0) == 1000


class TestOobObservability:
    def test_paged_oob_increments_counter(self):
        mgr = PagedCacheManager(block_size=8, max_blocks=2, model_name="m")
        before = get_server_metrics().kv_block_oob_total
        # Allocate until exhaustion (cap=2 metadata blocks); the exact pool
        # growth schedule is internal — the contract is that at least one
        # exhausted allocation returns None AND bumps the counter.
        results = [mgr.allocate_block() for _ in range(8)]
        assert any(b is None for b in results)
        after = get_server_metrics().kv_block_oob_total
        assert after > before
        assert get_server_metrics().kv_block_oob_by_scope.get("paged", 0) >= 1

    @pytest.mark.parametrize("scope", ["paged", "prefix"])
    def test_record_kv_block_oob_breakdown(self, scope):
        sm = get_server_metrics()
        before = sm.kv_block_oob_by_scope.get(scope, 0)
        sm.record_kv_block_oob(scope)
        assert sm.kv_block_oob_by_scope.get(scope, 0) == before + 1
