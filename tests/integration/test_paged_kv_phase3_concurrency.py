import logging
import os

import mlx.core as mx
import pytest

from fusion_mlx.custom_kernels.paged_kv_pool import (
    FusionPagedKVPool,
    FusionPagedRequestCache,
)

logger = logging.getLogger(__name__)


def test_interleaved_writes_no_cross_contamination():
    n_kv_heads, head_dim, block_size = 2, 8, 4
    pool = FusionPagedKVPool(
        block_size=block_size,
        num_blocks=32,
        n_kv_heads=n_kv_heads,
        head_dim=head_dim,
        dtype=mx.float32,
    )
    a = FusionPagedRequestCache(pool, request_id="a")
    b = FusionPagedRequestCache(pool, request_id="b")
    mx.random.seed(11)
    a_views, b_views = [], []
    for i in range(10):
        k = mx.random.normal(shape=(1, n_kv_heads, 1, head_dim)) * 0.1
        v = mx.random.normal(shape=(1, n_kv_heads, 1, head_dim)) * 0.1
        if i % 2 == 0:
            a_views.append(a.update_and_fetch(k, v))
        else:
            b_views.append(b.update_and_fetch(k, v))
    assert a.offset == 5
    assert b.offset == 5
    mx.random.seed(11)
    expected_a = []
    expected_b = []
    for i in range(10):
        k = mx.random.normal(shape=(1, n_kv_heads, 1, head_dim)) * 0.1
        v = mx.random.normal(shape=(1, n_kv_heads, 1, head_dim)) * 0.1
        if i % 2 == 0:
            expected_a.append((k, v))
        else:
            expected_b.append((k, v))
    last_k, last_v = a_views[-1]
    for j, (ek, ev) in enumerate(expected_a):
        assert mx.allclose(
            last_k[:, :, j, :], ek[:, :, 0, :]
        ).item(), f"a token {j} contaminated"
        assert mx.allclose(
            last_v[:, :, j, :], ev[:, :, 0, :]
        ).item(), f"a v token {j} contaminated"
    last_k_b, last_v_b = b_views[-1]
    for j, (ek, ev) in enumerate(expected_b):
        assert mx.allclose(
            last_k_b[:, :, j, :], ek[:, :, 0, :]
        ).item(), f"b token {j} contaminated"
        assert mx.allclose(
            last_v_b[:, :, j, :], ev[:, :, 0, :]
        ).item(), f"b v token {j} contaminated"


def test_pool_evict_then_reuse():
    pool = FusionPagedKVPool(block_size=4, num_blocks=4, n_kv_heads=2, head_dim=8)
    a = FusionPagedRequestCache(pool, request_id="a")
    for _ in range(8):
        k = mx.random.normal(shape=(1, 2, 1, 8)) * 0.1
        v = mx.random.normal(shape=(1, 2, 1, 8)) * 0.1
        a.update_and_fetch(k, v)
    assert pool.available() == 2
    pool.free_request("a")
    assert pool.available() == 4
    b = FusionPagedRequestCache(pool, request_id="b")
    for _ in range(4):
        k = mx.random.normal(shape=(1, 2, 1, 8)) * 0.1
        v = mx.random.normal(shape=(1, 2, 1, 8)) * 0.1
        b.update_and_fetch(k, v)
    assert b.offset == 4


_real_mark = pytest.mark.skipif(
    os.environ.get("FUSION_PAGED_KV_REAL_MODEL") != "on",
    reason="set FUSION_PAGED_KV_REAL_MODEL=on for real-model tests",
)

_MODEL = os.environ.get("FUSION_PAGED_KV_MODEL", "mlx-community/Qwen3-0.6B-4bit")
_MAX_TOKENS = int(os.environ.get("FUSION_PAGED_KV_MAX_TOKENS", "30"))


def _to_tokens(res):
    if res is None:
        return []
    if isinstance(res, list):
        return [int(t) for t in res]
    nti = getattr(res, "new_token_ids", None)
    if nti:
        return [int(t) for t in nti]
    tok = getattr(res, "tokens", None)
    if tok:
        return [int(t) for t in tok]
    return []


def _single_stream_ref(model_path, prompt, max_tokens):
    import mlx_lm
    from mlx_lm.generate import stream_generate

    model, tokenizer = mlx_lm.load(model_path)
    toks = []
    for resp in stream_generate(model, tokenizer, prompt, max_tokens=max_tokens):
        toks.append(int(resp.token))
        if len(toks) >= max_tokens:
            break
    logger.info(
        "single-stream ref model=%s prompt=%r tokens=%s",
        model_path,
        prompt,
        toks[:10],
    )
    del model
    del tokenizer
    mx.clear_cache()
    return toks


def _pool_stream(model_path, prompt, max_tokens):
    import mlx_lm
    from mlx_lm.generate import stream_generate

    from fusion_mlx.custom_kernels.fusion_paged_kv import install_paged_kv
    from fusion_mlx.fusion_takeover.config import FusionConfig
    from fusion_mlx.fusion_takeover.patcher import FusionModulePatcher

    model, tokenizer = mlx_lm.load(model_path)
    cfg = FusionConfig(
        enabled=True,
        paged_kv_enabled=True,
        pool_enabled=True,
        pool_num_blocks=256,
    )
    FusionModulePatcher.patch_model(model, cfg)
    install_paged_kv(model, cfg)
    cache = model.make_cache()
    assert isinstance(cache[0], FusionPagedRequestCache), (
        f"make_cache did not return FusionPagedRequestCache; got {type(cache[0]).__name__} "
        f"(install_paged_kv pool override may have silently failed)"
    )
    toks = []
    for resp in stream_generate(
        model, tokenizer, prompt, max_tokens=max_tokens, prompt_cache=cache
    ):
        toks.append(int(resp.token))
        if len(toks) >= max_tokens:
            break
    pool = getattr(model, "_fusion_paged_pool", None)
    assert pool is not None, "pool not installed on model (_fusion_paged_pool missing)"
    pool_stats = pool.stats()
    assert pool_stats["in_use"] >= 1, (
        f"pool has no blocks in_use after generate; pool_stats={pool_stats} "
        f"(pool was never exercised — test is a no-op)"
    )
    logger.info(
        "pool-stream model=%s prompt=%r tokens=%s pool_stats=%s",
        model_path,
        prompt,
        toks[:10],
        getattr(getattr(model, "_fusion_paged_pool", None), "stats", lambda: {})(),
    )
    from fusion_mlx.custom_kernels.fusion_paged_kv import evict_request_by_id

    evict_request_by_id("pool_0")
    del model
    del tokenizer
    mx.clear_cache()
    return toks


@_real_mark
def test_concurrent_pool_matches_single_stream():
    prompts = ["The quick brown fox", "In a galaxy far away"]
    refs = {p: _single_stream_ref(_MODEL, p, _MAX_TOKENS) for p in prompts}
    pool_a = _pool_stream(_MODEL, prompts[0], _MAX_TOKENS)
    pool_b = _pool_stream(_MODEL, prompts[1], _MAX_TOKENS)
    conc = {prompts[0]: pool_a, prompts[1]: pool_b}
    for p in prompts:
        assert refs[p] == conc[p], (
            f"pool stream mismatch for {p!r}: "
            f"ref={refs[p][:10]} pool={conc[p][:10]}"
        )
