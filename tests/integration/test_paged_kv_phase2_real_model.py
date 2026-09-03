import os

import pytest

pytestmark = pytest.mark.skipif(
    os.environ.get("FUSION_PAGED_KV_REAL_MODEL") != "on",
    reason="set FUSION_PAGED_KV_REAL_MODEL=on to run real-model paged-KV tests",
)

_MODEL = os.environ.get("FUSION_PAGED_KV_MODEL", "mlx-community/Qwen3-0.6B-4bit")
_PROMPT = os.environ.get("FUSION_PAGED_KV_PROMPT", "The quick brown fox")
_MAX_TOKENS = int(os.environ.get("FUSION_PAGED_KV_MAX_TOKENS", "40"))

import logging

logger = logging.getLogger(__name__)


def _greedy_tokens(model_path, prompt, max_tokens, fused):
    import mlx_lm
    from mlx_lm.generate import stream_generate

    from fusion_mlx.fusion_takeover.config import FusionConfig
    from fusion_mlx.fusion_takeover.patcher import FusionModulePatcher

    if fused:
        os.environ["FUSION_PAGED_FUSED_KERNEL"] = "on"
    else:
        os.environ.pop("FUSION_PAGED_FUSED_KERNEL", None)

    model, tokenizer = mlx_lm.load(model_path)
    cfg = FusionConfig(
        enabled=True,
        paged_kv_enabled=True,
        fused_decode_enabled=fused,
    )
    FusionModulePatcher.patch_model(model, cfg)
    toks = []
    for resp in stream_generate(model, tokenizer, prompt, max_tokens=max_tokens):
        toks.append(int(resp.token))
        if len(toks) >= max_tokens:
            break
    logger.info(
        "greedy decode path=%s model=%s tokens=%s",
        "fused" if fused else "base",
        model_path,
        toks[:10],
    )
    return toks


def test_phase2_fused_matches_concat_tokens():
    base = _greedy_tokens(_MODEL, _PROMPT, _MAX_TOKENS, fused=False)
    fused = _greedy_tokens(_MODEL, _PROMPT, _MAX_TOKENS, fused=True)
    assert base == fused, f"token streams differ: base={base[:10]} fused={fused[:10]}"
