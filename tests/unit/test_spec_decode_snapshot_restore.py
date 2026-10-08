# SPDX-License-Identifier: Apache-2.0
"""#1055: spec_decode_step must restore/trim prompt_cache when verify or
replay raises. Without it, the K draft KV tokens written by the verify
forward permanently pollute the request's prompt_cache."""

from unittest.mock import MagicMock, patch

import pytest

from fusion_mlx.scheduler import spec_decode as spec_decode_module
from fusion_mlx.scheduler.spec_decode import spec_decode_step


def _make_scheduler(draft_tokens, prompt_cache, *, sampled_from_regular=None):
    spec_state = MagicMock()
    spec_state.draft_model = MagicMock()  # truthy
    spec_state._last_request_id = "req-1"
    spec_state.should_speculate.return_value = True
    spec_state.get_drafts.return_value = draft_tokens
    spec_state.steps_since_start = 10
    spec_state.total_spec_steps = 0

    request = MagicMock()
    request.prompt_token_ids = [1, 2, 3]

    gen = MagicMock()
    gen.prompt_cache = prompt_cache
    gen.model = MagicMock()
    gen._next_tokens = MagicMock()
    gen._next_tokens.item.return_value = sampled_from_regular

    bg = MagicMock()
    bg._generation_batch = gen

    sched = MagicMock()
    sched._spec_decode_state = spec_state
    sched._stream = MagicMock()
    sched.running = {"req-1": request}
    sched.batch_generator = bg
    return sched, spec_state, gen


def test_verify_exception_restores_non_trimmable_and_trims():
    """When _run_spec_verify raises after the snapshot, the except block
    restores non-trimmable caches (deepcopy) and trims trimmable by K."""
    trimmable = MagicMock()
    trimmable.is_trimmable.return_value = True
    trimmable.trim = MagicMock()
    non_trimmable_orig = MagicMock()
    non_trimmable_orig.is_trimmable.return_value = False
    prompt_cache = [trimmable, non_trimmable_orig]

    sched, spec_state, gen = _make_scheduler(
        [10, 20, 30], prompt_cache, sampled_from_regular=10
    )

    with patch.object(
        spec_decode_module, "_run_spec_verify", side_effect=RuntimeError("verify boom")
    ):
        with pytest.raises(RuntimeError, match="verify boom"):
            spec_decode_step(sched, output=None, current_token=10, request_id="req-1")

    # Trimmable layer trimmed by K=3 (the draft tokens verify would have written).
    trimmable.trim.assert_called_once_with(3)
    # Non-trimmable layer restored from the deepcopy snapshot (replaced in
    # prompt_cache). The original object is no longer at index 1.
    assert prompt_cache[1] is not non_trimmable_orig


def test_replay_exception_restores_and_trims():
    """When the replay forward (inside the rollback branch) raises, the
    except block still restores + trims before re-raising."""
    trimmable = MagicMock()
    trimmable.is_trimmable.return_value = True
    trimmable.trim = MagicMock()
    non_trimmable_orig = MagicMock()
    non_trimmable_orig.is_trimmable.return_value = False
    prompt_cache = [trimmable, non_trimmable_orig]

    sched, spec_state, gen = _make_scheduler(
        [10, 20, 30], prompt_cache, sampled_from_regular=10
    )

    # Verify returns a partial acceptance (n_accepted=1 < K=3) so the
    # rollback+replay branch runs; the replay model() call raises.
    verified = [10, 99]
    n_accepted = 1
    cache_tokens_processed = 3

    def fake_verify(*args, **kwargs):
        return verified, n_accepted, cache_tokens_processed

    gen.model.side_effect = RuntimeError("replay boom")

    with patch.object(spec_decode_module, "_run_spec_verify", side_effect=fake_verify):
        with pytest.raises(RuntimeError, match="replay boom"):
            spec_decode_step(sched, output=None, current_token=10, request_id="req-1")

    trimmable.trim.assert_called_once_with(3)
    assert prompt_cache[1] is not non_trimmable_orig
