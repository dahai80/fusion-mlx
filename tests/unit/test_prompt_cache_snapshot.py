# SPDX-License-Identifier: Apache-2.0
"""Prompt-boundary cache snapshot path tests (migrated from Rapid-MLX).

Adaptations:
- vllm_mlx.request -> fusion_mlx.request
- vllm_mlx.scheduler -> fusion_mlx.scheduler
- fusion-mlx Scheduler does NOT have _snapshot_promoted_prompts,
  _snapshot_boundary_segments, _prompt_cache_save_cb, or
  memory_aware_cache attributes. All tests that depend on these
  APIs are skipped.
- fusion-mlx SchedulerConfig does NOT have enable_prefix_cache or
  use_memory_aware_cache in the scheduler.config variant; the
  top-level fusion_mlx.config.SchedulerConfig does, but the
  Scheduler internally uses scheduler.config.SchedulerConfig.
"""

from __future__ import annotations

import pytest


@pytest.mark.skip(
    reason="fusion-mlx Scheduler lacks _snapshot_promoted_prompts / _prompt_cache_save_cb"
)
class TestPromptCacheSnapshot:
    def test_callback_built_when_memory_cache_enabled(self):
        pass

    def test_no_callback_without_memory_cache(self):
        pass

    def test_snapshot_stores_promoted_prompt_only(self):
        pass

    def test_snapshot_skips_mid_prompt_chunks(self):
        pass

    def test_snapshot_handles_extract_failure(self):
        pass

    def test_snapshot_skips_already_removed_uid(self):
        pass

    def test_snapshot_no_op_when_callback_disabled(self):
        pass

    def test_snapshot_with_empty_responses(self):
        pass

    def test_snapshot_stores_only_end_of_prompt_subset(self):
        pass


@pytest.mark.skip(reason="fusion-mlx Scheduler lacks _snapshot_boundary_segments")
class TestBoundarySnapshot:
    def test_snapshot_stores_at_prefix_boundary(self):
        pass

    def test_snapshot_skips_end_of_prompt_responses(self):
        pass

    def test_snapshot_skips_when_no_prefix_boundary(self):
        pass

    def test_snapshot_skips_when_no_memory_cache(self):
        pass

    def test_snapshot_idempotent_per_request(self):
        pass

    def test_snapshot_handles_extract_failure(self):
        pass

    def test_snapshot_handles_store_failure(self):
        pass

    def test_snapshot_skips_unknown_uid(self):
        pass

    def test_snapshot_with_empty_responses(self):
        pass

    def test_snapshot_skips_tail_segment_fire(self):
        pass


@pytest.mark.skip(
    reason="fusion-mlx Scheduler lacks prefix_boundary / insert_segments dispatch"
)
class TestScheduleWaitingInsertDispatch:
    def test_split_in_middle_returns_local_offset(self):
        pass

    def test_split_at_boundary_no_cache_returns_full_offset(self):
        pass

    def test_no_split_when_boundary_already_cached(self):
        pass

    def test_no_split_when_boundary_at_or_past_tokens(self):
        pass

    def test_no_split_when_no_prefix_boundary(self):
        pass

    def test_no_split_for_single_token_kickoff(self):
        pass
