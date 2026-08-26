# SPDX-License-Identifier: Apache-2.0
"""Real-model integration tests for /distributed/decode_step (#630).

Bit-exact vs mlx_lm.generate_step is the headline correctness gate: threading
the KVCache through decode_step must reproduce un-split generation. Gated to
the real-model group (@pytest.mark.real_model + FUSION_MLX_REAL_MODEL_TESTS),
matching the test_logprob.py / test_grpo.py convention."""

from __future__ import annotations

import os

import pytest

pytest.importorskip("mlx.core")

_MODEL_CANDIDATES = ["models--mlx-community--Llama-3.2-1B-Instruct-4bit"]


def _find_small_lm() -> str | None:
    base = os.path.expanduser(
        os.environ.get("FUSION_MLX_MODEL_DIR", "~/.fusion-mlx/models")
    )
    for name in _MODEL_CANDIDATES:
        snap_root = os.path.join(base, name, "snapshots")
        if not os.path.isdir(snap_root):
            continue
        for snap in os.listdir(snap_root):
            snap_dir = os.path.join(snap_root, snap)
            if any(f.endswith(".safetensors") for f in os.listdir(snap_dir)):
                return snap_dir
    return None


_LM_PATH = _find_small_lm()


# These tests load real MLX weights and run generate_step, so they follow the
# codebase's real-model convention (test_logprob.py / test_grpo.py): the
# @pytest.mark.real_model marker + an inline FUSION_MLX_REAL_MODEL_TESTS guard
# keep them out of the default `pytest tests/unit` suite (which stays
# weight-free per CLAUDE.md). Running them un-gated in the full suite
# interleaves them with other MLX tests, where stale thread-local Stream
# handles from earlier tests make generate_step's mx.eval raise
# "There is no Stream(gpu, N) in current thread" — a test-ordering artifact,
# not a production bug. Gating to the real-model group avoids it.
def _skip_unless_real_model():
    if not os.environ.get("FUSION_MLX_REAL_MODEL_TESTS"):
        pytest.skip(
            "set FUSION_MLX_REAL_MODEL_TESTS=1 to run real-model decode_step e2e"
        )
    if _LM_PATH is None:
        pytest.skip("no small LM with safetensors found in model dir")


def _ref_tokens(model, tok, prompt, n):
    """Greedy reference tokens from mlx_lm.generate_step."""
    import mlx.core as mx
    from mlx_lm.generate import generate_step
    from mlx_lm.sample_utils import make_sampler

    prompt_ids = tok.encode(prompt)
    gen = generate_step(
        mx.array(prompt_ids, dtype=mx.int32),
        model,
        max_tokens=n,
        sampler=make_sampler(temp=0.0),
    )
    # generate_step yields (token, logprob) tuples; unpack the token.
    return [int(token) for (token, _lp), _ in zip(gen, range(n))]


@pytest.mark.real_model
def test_single_shard_decode_step_matches_generate_step():
    """One shard = whole model [0, total). Prefill via decode_step
    (input_ids, is_last_shard=True) -> token #1; loop single-token decode_step
    for the rest. Bit-exact vs generate_step (greedy)."""
    _skip_unless_real_model()
    import mlx_lm

    from fusion_mlx.distributed.shard import ShardManager

    mgr = ShardManager()
    model, tok = mlx_lm.load(_LM_PATH)
    total = len(model.model.layers)
    info = mgr.load_shard(_LM_PATH, 0, [0, total])

    prompt = "The capital of France is"
    prompt_ids = tok.encode(prompt)
    n = 5

    # prefill (multi-token, last shard) -> token #1
    out = mgr.decode_step(
        info["shard_id"], None, prompt_ids, is_last_shard=True, temperature=0.0
    )
    tok_id = out["token_ids"][0]
    gen = [tok_id]
    assert out["kv_offset"] == len(
        prompt_ids
    ), f"prefill kv_offset {out['kv_offset']} != prompt len {len(prompt_ids)}"
    # decode loop
    for _ in range(n - 1):
        out = mgr.decode_step(
            info["shard_id"], None, [tok_id], is_last_shard=True, temperature=0.0
        )
        tok_id = out["token_ids"][0]
        gen.append(tok_id)
    ref = _ref_tokens(model, tok, prompt, n)
    assert gen == ref, f"single-shard decode_step {gen} != generate_step {ref}"
    assert out["kv_offset"] == len(prompt_ids) + n - 1


@pytest.mark.real_model
def test_two_shard_decode_step_matches_generate_step():
    """Split at the midpoint. Prefill: shard A (input_ids, not last) -> shard
    B (hidden_states, last) -> token #1. Decode loop: A -> B per token.
    Bit-exact vs generate_step (greedy). Pins the boundary activation crossing
    is correct WITH cache."""
    _skip_unless_real_model()
    import mlx_lm

    from fusion_mlx.distributed.shard import ShardManager

    mgr = ShardManager()
    model, tok = mlx_lm.load(_LM_PATH)
    total = len(model.model.layers)
    split = total // 2
    a = mgr.load_shard(_LM_PATH, 0, [0, split])
    b = mgr.load_shard(_LM_PATH, 1, [split, total])

    prompt = "The capital of France is"
    prompt_ids = tok.encode(prompt)
    n = 5

    # prefill: A embeds + [0,split) -> [P,hidden]; B [split,total) + sample
    out_a = mgr.decode_step(a["shard_id"], None, prompt_ids, is_last_shard=False)
    assert out_a["shape"][1] == len(prompt_ids)
    out_b = mgr.decode_step(
        b["shard_id"], out_a["hidden_states"], None, is_last_shard=True, temperature=0.0
    )
    tok_id = out_b["token_ids"][0]
    gen = [tok_id]
    assert out_a["kv_offset"] == len(prompt_ids)
    assert out_b["kv_offset"] == len(prompt_ids)
    # decode loop: single-token A -> B
    for _ in range(n - 1):
        out_a = mgr.decode_step(a["shard_id"], None, [tok_id], is_last_shard=False)
        assert out_a["shape"][1] == 1
        out_b = mgr.decode_step(
            b["shard_id"],
            out_a["hidden_states"],
            None,
            is_last_shard=True,
            temperature=0.0,
        )
        tok_id = out_b["token_ids"][0]
        gen.append(tok_id)
    ref = _ref_tokens(model, tok, prompt, n)
    assert gen == ref, f"two-shard decode_step {gen} != generate_step {ref}"
    assert out_a["kv_offset"] == len(prompt_ids) + n - 1
    assert out_b["kv_offset"] == len(prompt_ids) + n - 1


@pytest.mark.real_model
def test_reset_then_reuse_different_prompt():
    """generate, reset_cache on the shard, generate a DIFFERENT prompt ->
    correct (cache did not bleed across generations)."""
    _skip_unless_real_model()
    import mlx_lm

    from fusion_mlx.distributed.shard import ShardManager

    mgr = ShardManager()
    model, tok = mlx_lm.load(_LM_PATH)
    total = len(model.model.layers)
    info = mgr.load_shard(_LM_PATH, 0, [0, total])

    p1 = "The capital of France is"
    p1_ids = tok.encode(p1)
    out = mgr.decode_step(
        info["shard_id"], None, p1_ids, is_last_shard=True, temperature=0.0
    )
    gen1 = [out["token_ids"][0]]
    for _ in range(3):
        out = mgr.decode_step(
            info["shard_id"], None, gen1[-1:], is_last_shard=True, temperature=0.0
        )
        gen1.append(out["token_ids"][0])

    # reset -> new generation from a different prompt
    reset = mgr.reset_cache(info["shard_id"])
    assert reset["kv_cleared"] is True
    assert reset["prev_offset"] == len(p1_ids) + 3
    p2 = "The largest planet is"
    p2_ids = tok.encode(p2)
    out = mgr.decode_step(
        info["shard_id"], None, p2_ids, is_last_shard=True, temperature=0.0
    )
    gen2 = [out["token_ids"][0]]
    for _ in range(3):
        out = mgr.decode_step(
            info["shard_id"], None, gen2[-1:], is_last_shard=True, temperature=0.0
        )
        gen2.append(out["token_ids"][0])

    ref2 = _ref_tokens(model, tok, p2, 4)
    assert gen2 == ref2, f"after reset, gen2 {gen2} != ref {ref2}"
    # sanity: the two generations differ (different prompts -> different output)
    assert gen1 != gen2


@pytest.mark.real_model
def test_no_auto_reset_appends_wrongly_documented():
    """Pins the contract: the server does NOT auto-reset between generations.
    Two prefills WITHOUT reset on the same shard append into one cache ->
    the second prefill sees the first prompt's KV and produces a DIFFERENT
    output distribution (logits) than a clean run (not a crash). This
    documents the 'reset is the caller's responsibility' contract; it is NOT
    a correctness assertion that the sampled token is right."""
    _skip_unless_real_model()
    import mlx.core as mx
    import mlx_lm

    from fusion_mlx.distributed.shard import ShardManager, deserialize_activation

    mgr = ShardManager()
    model, tok = mlx_lm.load(_LM_PATH)
    total = len(model.model.layers)
    info = mgr.load_shard(_LM_PATH, 0, [0, total])

    p1_ids = tok.encode("The capital of France is")
    mgr.decode_step(info["shard_id"], None, p1_ids, is_last_shard=True, temperature=0.0)
    # Second prefill WITHOUT reset -> appends onto p1's KV. The output is
    # semantically wrong (attention over p1+p2 positions) but must NOT crash.
    p2_ids = tok.encode("The largest planet is")
    out = mgr.decode_step(
        info["shard_id"],
        None,
        p2_ids,
        is_last_shard=True,
        temperature=0.0,
        return_logits=True,
    )
    assert out["token_ids"] is not None  # did not crash
    assert out["kv_offset"] == len(p1_ids) + len(p2_ids)  # appended, not reset
    # Pin the contract on LOGITS, not argmax: causal attention guarantees the
    # appended p1-KV changes the attention context for p2's prefill, so the
    # logits at the last position DIFFER from a clean run. Argmax is a lossy
    # projection that can collapse distinct distributions to the same token
    # (this 1B 4-bit model's greedy argmax is "sticky" — verified to coincide
    # across multiple prompt pairs even though the logits differ), so logits
    # are the robust pre-projection signal. Threshold 1.0 sits between the
    # observed ~11.4 diff (appended vs clean) and 0.0 (identical KV), giving
    # a wide margin on either side.
    mgr.reset_cache(info["shard_id"])
    out_clean = mgr.decode_step(
        info["shard_id"],
        None,
        p2_ids,
        is_last_shard=True,
        temperature=0.0,
        return_logits=True,
    )
    assert out_clean["kv_offset"] == len(p2_ids)  # reset cleared p1's KV
    logits_app = deserialize_activation(out["logits"])
    logits_clean = deserialize_activation(out_clean["logits"])
    diff = float(mx.abs(logits_app - logits_clean).max())
    assert diff > 1.0, (
        f"without reset, appended-cache logits should differ from a clean run "
        f"(max |diff|={diff:.4f}, expected > 1.0)"
    )
