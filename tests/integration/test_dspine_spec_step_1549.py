# SPDX-License-Identifier: Apache-2.0
"""#1549 Stage 2: DSpine runtime + hidden-window alignment tests.

Loads the real Qwen3-8B bf16 target and the real DSpine-b16 checkpoint
(both must exist locally; skipped otherwise, per the real-weights rule).
Exercises the exact plumbing dspine_spec_step uses: capture install,
prefill bootstrap window, draft round, verify capture rows, next-window
assembly [anchor_row + D1..D_a rows].
"""

from __future__ import annotations

import os

import mlx.core as mx
import pytest

from fusion_mlx.speculative.dspine import DSpineDrafter

DRAFT_PATH = os.path.expanduser("~/.fusion-mlx/models/tencent--Qwen3-8B-DSpine-b16")
TARGET_PATH = os.path.expanduser("~/.fusion-mlx/models/Qwen--Qwen3-8B")

pytestmark = [
    pytest.mark.skipif(
        not os.path.isdir(DRAFT_PATH), reason=f"{DRAFT_PATH} not present"
    ),
    pytest.mark.skipif(
        not os.path.isdir(TARGET_PATH), reason=f"{TARGET_PATH} not present"
    ),
]


def _load_target():
    import sys

    sys.path.insert(0, TARGET_PATH)
    from mlx_lm import load

    model, tokenizer = load(TARGET_PATH)
    return model, tokenizer


def test_spec_state_lifecycle():
    from fusion_mlx.scheduler.spec_decode import DSpineSpecState
    from fusion_mlx.speculative.dspine.runtime import DSpineRuntime

    class _Drafter:
        resets = 0

        def reset_cache(self):
            type(self).resets += 1

    rt = DSpineRuntime(drafter=_Drafter(), draft_repo=DRAFT_PATH, block_size=16)
    st = DSpineSpecState(rt)
    st.on_new_request("r1")
    assert st.should_speculate() is False
    for _ in range(2):
        st.add_token(1)
    assert st.should_speculate() is True
    st.record_result(3, 16)
    assert st.total_draft_proposed == 16
    assert st.total_draft_accepted == 3
    st.on_new_request("r2")
    assert st.steps_since_start == 0
    assert st._window is None and st._caches is None
    assert _Drafter.resets == 2


def test_capture_window_alignment_real_weights():
    """Two draft rounds through the real target: window grows by
    [anchor_row] + [accepted rows] each round; drafts stay in-vocab."""

    from mlx_lm.models.cache import make_prompt_cache

    from fusion_mlx.scheduler.spec_decode import _dspine_concat_captured
    from fusion_mlx.speculative.dspine.runtime import load_runtime

    model, tokenizer = _load_target()
    runtime = load_runtime(DRAFT_PATH, model, block_size=16)
    drafter: DSpineDrafter = runtime.drafter
    capture = runtime.capture
    assert capture.installed
    assert capture.layer_ids == drafter.target_layer_ids

    prompt_ids = tokenizer.encode(
        "The capital of France is Paris. The capital of Japan is",
        add_special_tokens=True,
    )
    tokens = mx.array([prompt_ids], dtype=mx.uint32)
    cache = make_prompt_cache(model)
    logits = model(tokens, cache=cache)
    mx.eval(logits)

    # Prefill capture: multi-token forward must populate the prefill store.
    prefill = capture.get_prefill_captured()
    assert all(lid in prefill for lid in drafter.target_layer_ids)
    min_sl = min(int(prefill[lid].shape[1]) for lid in drafter.target_layer_ids)
    assert min_sl >= 1
    window = mx.concatenate(
        [prefill[lid][:, -min_sl:, :] for lid in drafter.target_layer_ids],
        axis=-1,
    )
    capture.clear_prefill_captured()

    sampled = int(mx.argmax(logits[0, -1]).item())
    vocab = drafter.config.vocab_size
    caches = drafter.make_caches()

    # Round 1: draft from prefill window, anchor = sampled token.
    drafts1 = drafter.draft_block(window, sampled, caches)
    assert len(drafts1) == runtime.block_size - 1
    assert all(0 <= t < vocab for t in drafts1)

    # Verify: feed [D1..DK] (cache already past the anchor), greedy chain.
    verify_in = mx.array(drafts1, dtype=mx.uint32)
    vlogits = model(verify_in[None], cache=cache)
    mx.eval(vlogits)
    vs = mx.argmax(vlogits[0], axis=-1).tolist()
    accepted = 1  # D1 verified against the anchor's own prediction here
    for i in range(len(drafts1) - 1):
        if vs[i] == drafts1[i + 1]:
            accepted += 1
        else:
            break

    # Next window: [anchor_row] + first `accepted` verify capture rows.
    anchor_row = _dspine_concat_captured(
        capture.get_captured(), drafter.target_layer_ids, 1
    )
    assert anchor_row is not None and int(anchor_row.shape[1]) == 1
    rows = _dspine_concat_captured(
        capture.get_captured(), drafter.target_layer_ids, accepted
    )
    assert rows is not None and int(rows.shape[1]) == accepted
    window2 = mx.concatenate([anchor_row, rows], axis=1)
    assert int(window2.shape[1]) == accepted + 1
    assert int(window2.shape[2]) == drafter.config.concat_size

    # Round 2 must run with the grown draft cache and the new window.
    bonus = int(vs[max(0, accepted - 1)])
    drafts2 = drafter.draft_block(window2, bonus, caches)
    assert len(drafts2) == runtime.block_size - 1
    assert all(0 <= t < vocab for t in drafts2)

    capture.uninstall()
    assert not capture.installed


if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__, "-v"])
