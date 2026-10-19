# SPDX-License-Identifier: Apache-2.0
"""#1259: boundary-aligned chunked prefill kept every chunk at 64 tokens.

The old chunk sizing clamped each chunk DOWN to the next KV-block
boundary (min(n, delta), block=64), so with boundary snapshots enabled
every chunk was exactly 64 tokens — the #1204 throttle floor (>=512)
never applied and a 56k-token prompt took ~875 steps at ~0.6s each
(~9min TTFT, Claude Code stream timeout).

Fix under test: chunks are ROUNDED UP to a block_size multiple instead.
Chunk starts are boundary-aligned (base_size and prior chunks are block
multiples), so chunk ends still land exactly on boundaries and boundary
snapshots fire at chunk end — 8x less often, which also calms the
per-step snapshot disk-eviction storm from the same issue.
"""

from unittest.mock import MagicMock

import mlx.core as mx
import pytest

import fusion_mlx.scheduler.sched_batch as sched_batch_mod
from fusion_mlx.request import Request, SamplingParams
from fusion_mlx.scheduler import Scheduler, SchedulerConfig, _PrefillState


def _make_scheduler(
    prefill_step_size: int = 2048,
    memory_limit_bytes: int = 0,
    memory_hard_limit_bytes: int = 0,
) -> Scheduler:
    model = MagicMock()
    model.layers = []
    tokenizer = MagicMock()
    tokenizer.eos_token_id = 2
    config = SchedulerConfig(
        max_num_seqs=8,
        prefill_step_size=prefill_step_size,
        chunked_prefill=True,
        paged_cache_block_size=64,
        max_num_batched_tokens=8192,
    )
    scheduler = Scheduler(model=model, tokenizer=tokenizer, config=config)
    scheduler.batch_generator = MagicMock()
    scheduler._current_sampler_params = ()
    scheduler._prefill_memory_guard = False
    scheduler._memory_limit_bytes = memory_limit_bytes
    scheduler._memory_hard_limit_bytes = memory_hard_limit_bytes
    scheduler._emit_prefill_boundary_snapshot = MagicMock()
    return scheduler


def _make_state(n_remaining: int, tokens_processed: int = 0) -> _PrefillState:
    req = Request(
        request_id="t1259",
        prompt=list(range(n_remaining + 1)),
        sampling_params=SamplingParams(max_tokens=8),
    )
    req.prompt_token_ids = list(range(n_remaining + 1))
    req.num_prompt_tokens = n_remaining + 1
    req.remaining_tokens = list(range(n_remaining + 1))
    return _PrefillState(
        request=req,
        cache=[],
        tokens_remaining=mx.zeros((1, n_remaining), dtype=mx.int32),
        last_token=[99],
        tokens_processed=tokens_processed,
        base_size=0,
        emitted_boundaries={},
        boundary_enabled=True,
        block_size=64,
        total_length=n_remaining + 1,
        sampler=MagicMock(),
        sm=MagicMock(),
        per_row_lps=[],
    )


def _chunk_n(scheduler: Scheduler) -> int:
    model = scheduler.model
    assert model.call_count == 1
    chunk = model.call_args[0][0]
    return int(chunk.shape[1])


class TestChunkAlignment1259:
    def test_full_step_not_truncated_to_block(self):
        # Old behavior: chunk = 64 (next boundary). Fixed: 2048.
        scheduler = _make_scheduler(prefill_step_size=2048)
        state = _make_state(n_remaining=10000)
        scheduler._step_prefill_chunk(state)
        assert _chunk_n(scheduler) == 2048

    def test_small_step_rounds_up_to_block_multiple(self):
        scheduler = _make_scheduler(prefill_step_size=1000)
        state = _make_state(n_remaining=10000)
        scheduler._step_prefill_chunk(state)
        assert _chunk_n(scheduler) == 1024

    def test_last_chunk_clamped_to_remaining(self):
        scheduler = _make_scheduler(prefill_step_size=2048)
        state = _make_state(n_remaining=300)
        scheduler._step_prefill_chunk(state)
        assert _chunk_n(scheduler) == 300

    def test_throttle_floor_applies_with_boundaries(self):
        # The #1259 headline: with boundary snapshots enabled the throttle
        # floor (512) must still bound the chunk — old code returned 64.
        scheduler = _make_scheduler(
            prefill_step_size=2048,
            memory_limit_bytes=102 * 1024**3,
            memory_hard_limit_bytes=120 * 1024**3,
        )
        state = _make_state(n_remaining=56042)
        # 110GB > soft watermark (102*0.8=81.6GB) → tier 256 → floor 512.
        over = 110 * 1024**3
        with pytest.MonkeyPatch().context() as m:
            m.setattr(mx, "get_active_memory", lambda: over)
            m.setattr(sched_batch_mod, "get_phys_footprint", lambda: over)
            scheduler._step_prefill_chunk(state)
        assert _chunk_n(scheduler) == 512

    def test_snapshot_fires_at_aligned_chunk_end(self):
        scheduler = _make_scheduler(prefill_step_size=2048)
        state = _make_state(n_remaining=10000)
        scheduler._step_prefill_chunk(state)
        # Chunk end = base_size(0) + processed(2048) — on a 64 boundary.
        scheduler._emit_prefill_boundary_snapshot.assert_called_once()
        args = scheduler._emit_prefill_boundary_snapshot.call_args
        assert args[0][2] == 2048  # total_tokens position

    def test_misaligned_base_size_warns(self, caplog):
        import logging as _logging

        scheduler = _make_scheduler(prefill_step_size=2048)
        # Directly exercise the _begin_prefill alignment warning via a
        # cached_tokens mismatch: base_size = cached_tokens = 100 (not
        # block-aligned). Patch the model setup so no forward runs.
        req = Request(
            request_id="misalign",
            prompt=list(range(10)),
            sampling_params=SamplingParams(max_tokens=4),
        )
        req.prompt_token_ids = list(range(10))
        req.num_prompt_tokens = 10
        req.remaining_tokens = list(range(10))
        req.cached_tokens = 100
        scheduler.block_aware_cache = MagicMock()
        with pytest.MonkeyPatch().context() as m:
            m.setattr(sched_batch_mod, "_prompt_cache_needs_snapshots", lambda c: True)
            with caplog.at_level(_logging.WARNING):
                try:
                    scheduler._begin_prefill(req, list(range(10)), existing_cache=None)
                except Exception:
                    pass  # model mocks may explode past the warning we assert
        assert any("not block-aligned" in rec.message for rec in caplog.records)


if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__, "-v"])
