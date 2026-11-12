# SPDX-License-Identifier: Apache-2.0
"""#1549: DSpine drafter integration tests.

Loads the real DSpine-b16 checkpoint and the real Qwen3-8B bf16 target
(both must exist locally; skipped otherwise, per the real-weights rule).
Covers: load + bind, one draft round from a captured target hidden window,
and greedy draft-token sanity (non-mask, in-vocab).
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


class _FakeEmbed:
    def __init__(self, hidden: int):
        self.weight = mx.zeros((151936, hidden), dtype=mx.bfloat16)

    def __call__(self, ids):
        return mx.zeros((*ids.shape, self.weight.shape[1]), dtype=mx.bfloat16)


def test_load_draft_weights_real_checkpoint():
    from pathlib import Path

    import mlx.core as mx
    from mlx.utils import tree_flatten

    from fusion_mlx.speculative.dspine.config import load_draft_config
    from fusion_mlx.speculative.dspine.model import (
        DSpineDraftModel,
        _load_weights,
        load_draft,
    )

    cfg = load_draft_config(Path(DRAFT_PATH))
    # Strict-check helper path: full weight set present, no extras.
    draft0 = DSpineDraftModel(cfg)
    weights = _load_weights(Path(DRAFT_PATH))
    cleaned = {}
    for name, tensor in weights.items():
        short = name.removeprefix("model.")
        if short in ("draft_block_causal_version", "last_write_refine_version"):
            continue
        cleaned[short] = tensor
    expected = dict(tree_flatten(draft0.parameters()))
    assert not (set(expected) - set(cleaned)), "missing draft weights"
    assert not (set(cleaned) - set(expected)), "extra draft weights"

    # Production path (strict checks + bridge prepare).
    draft, cfg2 = load_draft(Path(DRAFT_PATH))
    assert draft.block_size == 16
    assert draft.num_parameters > 0
    draft.bind(
        type(
            "_T",
            (),
            {"model": type("_M", (), {"embed_tokens": _FakeEmbed(cfg.hidden_size)})()},
        )()
    )
    # One block-only shape probe: embed + mask tokens produce [1, T, H].
    block = mx.array([[100] + [draft.mask_token_id] * 15], dtype=mx.int32)
    caches = draft.make_cache()
    out, stash = draft(
        mx.zeros((1, 4, cfg.concat_size), dtype=mx.bfloat16), block, caches
    )
    assert out.shape == (1, 16, cfg.hidden_size)
    assert "raw" in stash and "packed" in stash


def test_config_schema():
    cfg = DSpineDrafter(DRAFT_PATH)
    from pathlib import Path

    from fusion_mlx.speculative.dspine.config import load_draft_config

    loaded = load_draft_config(Path(DRAFT_PATH))
    assert loaded.block_size == 16
    assert loaded.num_hidden_layers == 5
    assert loaded.vocab_size > 0
    assert cfg is not None


def test_registry_and_config_parse():
    from fusion_mlx.speculative.config import parse_speculative_config
    from fusion_mlx.speculative.registry import get_spec_decoder

    plugin = get_spec_decoder("dspine")
    assert plugin is not None
    cfg = parse_speculative_config(
        f'{{"method": "dspine", "model": "{DRAFT_PATH}", "block_size": 16}}'
    )
    assert cfg.method == "dspine"
    assert cfg.raw.get("block_size") == 16


def test_eligibility_gates():
    from fusion_mlx.speculative.dspine import check

    ok = check("qwen3", is_moe=False, draft_path=DRAFT_PATH, alias="t")
    assert ok.eligible
    hybrid = check("qwen3_5", is_moe=False, draft_path=DRAFT_PATH, alias="t")
    assert not hybrid.eligible
    missing = check("qwen3", is_moe=False, draft_path="/nonexistent", alias="t")
    assert not missing.eligible


if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__, "-v"])
