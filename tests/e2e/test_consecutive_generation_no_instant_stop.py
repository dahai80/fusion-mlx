# SPDX-License-Identifier: Apache-2.0
"""#1683 regression: near-tie first-token distribution must not yield an
empty response when the client sets min_tokens.

Root cause: the probe prompt's first-token distribution ties between a
continuation token and <|im_end|> (logit gap ~0.1-0.2 nats), so greedy
argmax legitimately picks EOS on some runs and the engine correctly
emits a 0-token stop. vLLM-parity min_tokens suppresses EOS until N
tokens were generated, guaranteeing a non-empty response. Without
min_tokens the 0-token stop remains valid engine behavior (documented
in the issue, not a bug).
"""

import os

import pytest

pytestmark = pytest.mark.skipif(
    os.environ.get("FUSION_E2E_ENGINE") != "1",
    reason="Set FUSION_E2E_ENGINE=1 to run real-engine tests",
)

SNAP = os.path.expanduser(
    "~/.fusion-mlx/models/models--mlx-community--Qwen3.5-4B-MLX-4bit/"
    "snapshots/32f3e8ecf65426fc3306969496342d504bfa13f3"
)

# The trap prompt from #1683: first-token logits tie between "\n\n"
# (token 271) and <|im_end|> (248046) within ~0.2 nats.
PROMPT = (
    "Explain how continuous batching differs from static batching in LLM "
    "serving, and why prefix cache reuse matters for agent workloads."
)


@pytest.fixture(scope="module")
def engine():
    if not os.path.isdir(SNAP):
        pytest.skip(f"model snapshot not found: {SNAP}")
    from fusion_mlx.engines.vlm import VLMBatchedEngine

    eng = VLMBatchedEngine(model_name=SNAP)
    eng._inject_tool_calling = lambda tok: None
    return eng


@pytest.mark.asyncio
async def test_min_tokens_guarantees_non_empty_response(engine):
    import time

    for run in range(3):
        prompt = f"{PROMPT} nonce-{time.time_ns()}"
        chunks = []
        async for out in engine.stream_generate(
            prompt, max_tokens=64, temperature=0.0, min_tokens=8
        ):
            chunks.append(out)
            if len(chunks) >= 16:
                break
        assert len(chunks) >= 2, (
            f"run {run}: instant-stop despite min_tokens=8 — got "
            f"{len(chunks)} chunk(s), first={chunks[0]!r}"
            if chunks
            else f"run {run}: empty stream despite min_tokens=8"
        )
        assert any(c.new_text for c in chunks), (
            f"run {run}: no text produced despite min_tokens=8"
        )


@pytest.mark.asyncio
async def test_consecutive_runs_stay_healthy_with_min_tokens(engine):
    import time

    for run in range(6):
        prompt = f"{PROMPT} nonce-{time.time_ns()}"
        chunks = []
        async for out in engine.stream_generate(
            prompt, max_tokens=512, temperature=0.0, min_tokens=8
        ):
            chunks.append(out)
            if len(chunks) >= 32:
                break
        assert len(chunks) >= 32, (
            f"run {run}: consecutive generation stopped after "
            f"{len(chunks)} chunk(s) despite min_tokens=8: "
            f"{[repr(c)[:200] for c in chunks]}"
        )
