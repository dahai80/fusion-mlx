"""#1532: terminal partial-block persistence and tail match.

store_cache persists the trailing partial block and get_computed_blocks
matches it (hash-verified), so a repeated prompt with a length that is not a
multiple of block_size reuses its tail instead of re-prefilling up to
block_size-1 tokens on every request.
"""

from unittest.mock import MagicMock, patch

import pytest

from fusion_mlx.cache.paged_cache import PagedCacheManager
from fusion_mlx.cache.prefix_cache import BlockAwarePrefixCache


class MockModel:
    def __init__(self, num_layers: int = 1):
        self._num_layers = num_layers
        self.layers = [MagicMock() for _ in range(num_layers)]

    @property
    def args(self):
        mock_args = MagicMock()
        mock_args.num_hidden_layers = self._num_layers
        return mock_args


@pytest.fixture
def mx():
    try:
        import mlx.core as mx

        return mx
    except ImportError:
        pytest.skip("MLX not available")


def _kv_cache_data(mx, num_tokens):
    return [
        {
            "state": (
                mx.ones((1, 8, num_tokens, 64)),
                mx.ones((1, 8, num_tokens, 64)),
            ),
            "cache_type": "KVCache",
            "class_name": "KVCache",
        }
    ]


def _make_cache(block_size=4, max_blocks=100):
    paged_cache = PagedCacheManager(
        block_size=block_size,
        max_blocks=max_blocks,
        model_name="test-model",
        initial_blocks=max_blocks,
    )
    model = MockModel(num_layers=1)
    return BlockAwarePrefixCache(model=model, paged_cache_manager=paged_cache)


class TestPartialTailRoundtrip:
    def test_exact_repeat_matches_through_partial(self, mx):
        cache = _make_cache()
        tokens = list(range(10))  # 2 full blocks + 2 partial tokens
        cache.store_cache("req-a", tokens, _kv_cache_data(mx, 10))

        block_table, remaining = cache.fetch_cache("req-b", tokens)
        assert block_table is not None
        assert block_table.num_tokens == 10
        assert remaining == []

    def test_extension_matches_through_partial(self, mx):
        cache = _make_cache()
        prompt = list(range(10))
        cache.store_cache("req-a", prompt, _kv_cache_data(mx, 10))

        extended = prompt + [90, 91, 92]
        block_table, remaining = cache.fetch_cache("req-b", extended)
        assert block_table is not None
        assert block_table.num_tokens == 10
        assert remaining == [90, 91, 92]

    def test_different_tail_does_not_match_partial(self, mx):
        cache = _make_cache()
        prompt = list(range(10))
        cache.store_cache("req-a", prompt, _kv_cache_data(mx, 10))

        other = list(range(8)) + [77, 78]
        block_table, remaining = cache.fetch_cache("req-b", other)
        # Full blocks still match; the partial must not.
        assert block_table is not None
        assert block_table.num_tokens == 8
        assert remaining == [77, 78]

    def test_short_prompt_partial_roundtrip(self, mx):
        cache = _make_cache()
        tokens = [1, 2, 3]  # shorter than block_size=4
        cache.store_cache("req-a", tokens, _kv_cache_data(mx, 3))

        block_table, remaining = cache.fetch_cache("req-b", tokens)
        assert block_table is not None
        assert block_table.num_tokens == 3
        assert remaining == []

    def test_evicted_partial_falls_back_to_full_blocks(self, mx):
        cache = _make_cache(max_blocks=8)
        prompt = list(range(10))
        cache.store_cache("req-a", prompt, _kv_cache_data(mx, 10))

        # Evict every cached block (refs are zero after store detaches).
        for block in list(cache.paged_cache.allocated_blocks.values()):
            cache.paged_cache.free_block(block.block_id)

        block_table, remaining = cache.fetch_cache("req-b", prompt)
        assert block_table is None
        assert remaining == prompt

    def test_index_hit_extends_partial_when_request_one_token_longer(self, mx):
        # E2E shape from #1532: store persists the prompt-end state at N-1
        # (10 tokens), the repeat request arrives at N (11 tokens). The
        # prefix index only keys block-chunk-aligned chain hashes, so the
        # last chunk (3 tokens vs stored 2) hash-mismatches and the index
        # hit stops at the last full block. fetch_cache must still extend
        # the hit with the stored terminal partial.
        cache = _make_cache()
        stored = list(range(10))
        cache.store_cache("req-a", stored, _kv_cache_data(mx, 10))

        prompt = stored + [99]
        block_table, remaining = cache.fetch_cache("req-b", prompt)
        assert block_table is not None
        assert block_table.num_tokens == 10
        assert remaining == [99]

    def test_try_claim_terminal_partial_rejects_bad_parent(self, mx):
        cache = _make_cache()
        stored = list(range(10))
        cache.store_cache("req-a", stored, _kv_cache_data(mx, 10))

        prompt = stored + [99]
        wrong_parent = b"not-a-real-chain-hash"
        ext_len, ext_block = cache.paged_cache.try_claim_terminal_partial(
            prompt, 8, wrong_parent
        )
        assert ext_len == 0
        assert ext_block is None

    def test_try_claim_terminal_partial_claims_and_verifies(self, mx):
        cache = _make_cache()
        stored = list(range(10))
        cache.store_cache("req-a", stored, _kv_cache_data(mx, 10))

        prompt = stored + [99]
        # Derive parent hash the same way store did: chain over full blocks.
        from fusion_mlx.cache.prefix_cache import compute_block_hash

        parent_hash = compute_block_hash(
            compute_block_hash(None, prompt[0:4], model_name="test-model"),
            prompt[4:8],
            model_name="test-model",
        )
        ext_len, ext_block = cache.paged_cache.try_claim_terminal_partial(
            prompt, 8, parent_hash
        )
        assert ext_len == 2
        assert ext_block is not None
        assert ext_block.token_count == 2
        assert ext_block.ref_count >= 1

    def test_stateful_exact_hit_is_truncated_by_admission_helper(self, mx):
        """Admission helper drops the terminal partial for stateful models."""
        from fusion_mlx.cache.hybrid_cache import ModelCacheConfig

        paged_cache = PagedCacheManager(
            block_size=4,
            max_blocks=100,
            model_name="test-model",
            initial_blocks=100,
        )
        model = MockModel(num_layers=1)
        cache = BlockAwarePrefixCache(model=model, paged_cache_manager=paged_cache)

        prompt = list(range(7))  # 1 full block + 3 partial tokens
        arrays_data = [
            {
                "state": (mx.ones((1, 3, 64)), mx.ones((1, 32, 128, 128))),
                "cache_type": "ArraysCache",
                "class_name": "ArraysCache",
            }
        ]
        model_cache_config = ModelCacheConfig.from_type_list(
            ["ArraysCache"], model_name="test-model"
        )
        cache.store_cache(
            "req-a", prompt, arrays_data, model_cache_config=model_cache_config
        )

        block_table, remaining = cache.fetch_cache("req-b", prompt)
        assert block_table is not None
        assert block_table.num_tokens == 7
        assert remaining == []

        # The admission-side helper removes the partial so a stateful cache
        # reconstructs at the last full boundary instead of falling back to
        # a full prefill.
        class _FakeAdmission:
            paged_cache_manager = paged_cache
            _stateful_non_sliceable_model = True

            def _model_has_stateful_non_sliceable_cache(self):
                return True

        from fusion_mlx.scheduler.sched_admission import (
            _drop_terminal_partial_for_stateful,
        )

        kept = _drop_terminal_partial_for_stateful(
            _FakeAdmission(), block_table, prompt
        )
        assert kept.num_tokens == 4
        assert len(kept.block_ids) == 1


class TestPromptEndBoundarySnapshot:
    """#1532: prompt-end (N-1) snapshot capture and store-override preference."""

    def test_override_prefers_prompt_end_snapshot(self, mock_model, mock_tokenizer):

        from fusion_mlx.config import SchedulerConfig
        from fusion_mlx.scheduler.core import Scheduler

        scheduler = Scheduler(
            model=mock_model,
            tokenizer=mock_tokenizer,
            config=SchedulerConfig(paged_cache_block_size=2048),
        )
        extracted = [{"state": ("prompt-end",)}]
        scheduler._boundary_cache_snapshots["req-pe"] = {
            6144: object(),
            8033: object(),
        }
        scheduler._boundary_prompt_end_tcs["req-pe"] = {8033}
        with patch.object(
            scheduler, "_extract_cache_states", return_value=(extracted, "cfg")
        ):
            result = scheduler._get_boundary_store_override("req-pe", list(range(8066)))
        assert result is not None
        token_sequence, cache_to_store, model_config, _provider = result
        assert len(token_sequence) == 8033
        assert cache_to_store is extracted
        assert model_config == "cfg"

    def test_on_prefill_boundary_snapshot_records_prompt_end(
        self, mock_model, mock_tokenizer
    ):
        from fusion_mlx.config import SchedulerConfig
        from fusion_mlx.scheduler.core import Scheduler

        scheduler = Scheduler(
            model=mock_model,
            tokenizer=mock_tokenizer,
            config=SchedulerConfig(paged_cache_block_size=2048),
        )
        scheduler.block_aware_cache = MagicMock()
        scheduler._boundary_snapshot_required = True
        scheduler._boundary_snapshot_store = None
        layer = MagicMock()
        type(layer).__name__ = "ArraysCache"

        scheduler._on_prefill_boundary_snapshot(
            "req-rec", [layer], 8033, require_aligned=False, prompt_end=True
        )
        assert 8033 in scheduler._boundary_cache_snapshots["req-rec"]
        assert 8033 in scheduler._boundary_prompt_end_tcs["req-rec"]

        # Default (aligned) path still rejects non-aligned token counts.
        scheduler._on_prefill_boundary_snapshot("req-rec2", [layer], 8033)
        assert "req-rec2" not in scheduler._boundary_cache_snapshots
        assert "req-rec2" not in scheduler._boundary_prompt_end_tcs

    def test_emit_final_boundary_emits_prompt_end_when_unaligned(
        self, mock_model, mock_tokenizer
    ):
        from types import SimpleNamespace

        from fusion_mlx.config import SchedulerConfig
        from fusion_mlx.scheduler.core import Scheduler

        scheduler = Scheduler(
            model=mock_model,
            tokenizer=mock_tokenizer,
            config=SchedulerConfig(paged_cache_block_size=2048),
        )
        captured = []
        with patch.object(
            scheduler,
            "_emit_prefill_boundary_snapshot",
            side_effect=lambda request, cache, tc, **kw: captured.append(kw),
        ):
            state = SimpleNamespace(
                boundary_enabled=True,
                block_size=2048,
                base_size=6144,
                tokens_processed=1889,
                emitted_boundaries={},
                request=SimpleNamespace(request_id="req-final"),
                cache=[object()],
            )
            scheduler._emit_final_boundary_if_needed(state)
        assert captured == [{"require_aligned": False, "prompt_end": True}]

        # Nothing prefilled in this state (prefix hit reused the tail) —
        # no duplicate prompt-end snapshot.
        captured.clear()
        state_reuse = SimpleNamespace(
            boundary_enabled=True,
            block_size=2048,
            base_size=8033,
            tokens_processed=0,
            emitted_boundaries={},
            request=SimpleNamespace(request_id="req-reuse"),
            cache=[object()],
        )
        scheduler._emit_final_boundary_if_needed(state_reuse)
        assert captured == []


class TestChunkClampArraysCache:
    def _fake_sched(self, prefill_step_size):
        from types import SimpleNamespace

        fake = SimpleNamespace(
            config=SimpleNamespace(
                paged_ssd_cache_dir="/tmp/ssd-test",
                paged_cache_block_size=64,
                prefill_step_size=prefill_step_size,
            ),
            model=SimpleNamespace(make_cache=lambda: [object()]),
        )
        fake._detect_rotating_window_sizes = lambda: None
        fake._cache_tree_has_arrays_cache = lambda obj: True
        fake._ARRAYS_CACHE_BLOCK_SIZE = 2048
        from fusion_mlx.scheduler.sched_cache import (
            _clamp_prefill_step_to_block_size,
        )

        fake._clamp_prefill_step_to_block_size = lambda: _clamp_prefill_step_to_block_size(
            fake
        )
        return fake

    def test_enlarge_clamps_prefill_step_size(self):
        from fusion_mlx.scheduler.sched_cache import (
            _enlarge_block_size_for_arrays_cache,
        )

        fake = self._fake_sched(4096)
        _enlarge_block_size_for_arrays_cache(fake)
        assert fake.config.paged_cache_block_size == 2048
        assert fake.config.prefill_step_size == 2048

    def test_enlarge_keeps_smaller_step_size(self):
        from fusion_mlx.scheduler.sched_cache import (
            _enlarge_block_size_for_arrays_cache,
        )

        fake = self._fake_sched(1024)
        _enlarge_block_size_for_arrays_cache(fake)
        assert fake.config.paged_cache_block_size == 2048
        assert fake.config.prefill_step_size == 1024

    def test_user_tier_pins_block_size_and_clamps_chunk(self):
        # P2-3 downshift: explicit tier (e.g. 512) is never auto-enlarged to
        # 2048; prefill chunk still clamps down to the tier so boundary
        # snapshots align with the smaller blocks.
        from fusion_mlx.scheduler.sched_cache import (
            _enlarge_block_size_for_arrays_cache,
        )

        fake = self._fake_sched(4096)
        fake.config.paged_cache_block_size = 512
        fake.config.paged_cache_block_size_override = 512
        _enlarge_block_size_for_arrays_cache(fake)
        assert fake.config.paged_cache_block_size == 512
        assert fake.config.prefill_step_size == 512


class TestResolveBlockSizeTier:
    def _settings(self, tmp_path, cache_section):
        import json

        path = tmp_path / "settings.json"
        path.write_text(json.dumps({"cache": cache_section}), encoding="utf-8")
        return str(tmp_path)

    def test_env_wins_over_settings(self, tmp_path, monkeypatch):
        from fusion_mlx.cli_serve.config_resolve import _resolve_block_size_tier

        monkeypatch.setenv("FUSION_MLX_PAGED_BLOCK_SIZE", "512")
        assert _resolve_block_size_tier({"paged_cache_block_size": 2048}) == 512

    def test_settings_fallback(self, tmp_path, monkeypatch):
        from fusion_mlx.cli_serve.config_resolve import _resolve_block_size_tier

        monkeypatch.delenv("FUSION_MLX_PAGED_BLOCK_SIZE", raising=False)
        assert _resolve_block_size_tier({"paged_cache_block_size": 256}) == 256

    def test_unset_returns_zero(self, tmp_path, monkeypatch):
        from fusion_mlx.cli_serve.config_resolve import _resolve_block_size_tier

        monkeypatch.delenv("FUSION_MLX_PAGED_BLOCK_SIZE", raising=False)
        assert _resolve_block_size_tier({}) == 0

    def test_invalid_values_ignored(self, tmp_path, monkeypatch):
        from fusion_mlx.cli_serve.config_resolve import _resolve_block_size_tier

        monkeypatch.setenv("FUSION_MLX_PAGED_BLOCK_SIZE", "abc")
        assert _resolve_block_size_tier({}) == 0
        monkeypatch.setenv("FUSION_MLX_PAGED_BLOCK_SIZE", "4")
        assert _resolve_block_size_tier({}) == 0
        monkeypatch.setenv("FUSION_MLX_PAGED_BLOCK_SIZE", "100000")
        assert _resolve_block_size_tier({}) == 0

    def test_env_only_applies_without_settings_section(self, tmp_path, monkeypatch):
        from fusion_mlx.cli_serve.serve_command import _apply_settings_cache_section
        from fusion_mlx.config import SchedulerConfig

        monkeypatch.setenv("FUSION_MLX_PAGED_BLOCK_SIZE", "512")
        empty = tmp_path / "settings.json"
        empty.write_text("{}", encoding="utf-8")
        cfg = SchedulerConfig()
        assert _apply_settings_cache_section(cfg, str(tmp_path)) is True
        assert cfg.paged_cache_block_size == 512
        assert cfg.paged_cache_block_size_override == 512

    def test_settings_tier_sets_override(self, tmp_path, monkeypatch):
        from fusion_mlx.cli_serve.serve_command import _apply_settings_cache_section
        from fusion_mlx.config import SchedulerConfig

        monkeypatch.delenv("FUSION_MLX_PAGED_BLOCK_SIZE", raising=False)
        d = self._settings(tmp_path, {"paged_cache_block_size": 512})
        cfg = SchedulerConfig()
        assert _apply_settings_cache_section(cfg, d) is True
        assert cfg.paged_cache_block_size == 512
        assert cfg.paged_cache_block_size_override == 512
