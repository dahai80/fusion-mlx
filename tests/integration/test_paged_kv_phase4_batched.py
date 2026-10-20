import logging
import os
import types

import pytest

logger = logging.getLogger(__name__)


def _make_pool(num_blocks=8, block_size=4, n_kv_heads=2, head_dim=4, cow=False):
    from fusion_mlx.custom_kernels.paged_kv_pool import FusionPagedKVPool

    pool = FusionPagedKVPool(
        block_size=block_size,
        num_blocks=num_blocks,
        n_kv_heads=n_kv_heads,
        head_dim=head_dim,
    )
    # Non-CoW pools keep KV in the per-cache flat buffer and never allocate
    # pool blocks (paged_kv_pool update_and_fetch fast path) — tests that
    # assert block accounting must opt in to CoW.
    pool._cow_enabled = cow
    return pool


def _make_cache(pool, request_id):
    from fusion_mlx.custom_kernels.paged_kv_pool import FusionPagedRequestCache

    return FusionPagedRequestCache(pool, request_id)


def _fill_cache(cache, length):
    import mlx.core as mx

    keys = mx.zeros((1, cache._n_kv_heads, length, cache._k_head_dim), dtype=mx.float32)
    values = mx.zeros(
        (1, cache._n_kv_heads, length, cache._v_head_dim), dtype=mx.float32
    )
    cache.update_and_fetch(keys, values)
    mx.eval(cache.state)


def test_free_paged_pool_cache_restores_available():
    import mlx.core as mx

    from fusion_mlx.scheduler.sched_response import _free_paged_pool_cache

    pool = _make_pool(num_blocks=8, block_size=4, n_kv_heads=2, head_dim=4, cow=True)
    initial = pool.available()

    c1 = _make_cache(pool, "pool_0")
    c2 = _make_cache(pool, "pool_1")
    _fill_cache(c1, 6)
    _fill_cache(c2, 3)
    assert pool.available() < initial, "blocks should be allocated"

    request = types.SimpleNamespace(
        request_id="sched-req-uuid-1",
        prompt_cache=[c1, c2],
    )

    freed = _free_paged_pool_cache(None, request)
    assert freed > 0, "helper should report freed blocks"
    assert pool.available() == initial, "pool must be fully restored after free"
    mx.clear_cache()


def test_free_paged_pool_cache_empty_request():
    from fusion_mlx.scheduler.sched_response import _free_paged_pool_cache

    # #955 fallback reads self.model — the helper is a scheduler method and
    # callers pass a scheduler; None was never a valid self.
    sched = types.SimpleNamespace(model=None)
    request = types.SimpleNamespace(request_id="x", prompt_cache=None)
    freed = _free_paged_pool_cache(sched, request)
    assert freed == 0

    request2 = types.SimpleNamespace(request_id="y", prompt_cache=[])
    freed2 = _free_paged_pool_cache(sched, request2)
    assert freed2 == 0


def test_free_paged_pool_cache_mixed_caches():
    import mlx.core as mx

    from fusion_mlx.scheduler.sched_response import _free_paged_pool_cache

    pool = _make_pool(num_blocks=8, block_size=4, n_kv_heads=2, head_dim=4, cow=True)
    initial = pool.available()

    c1 = _make_cache(pool, "pool_0")
    _fill_cache(c1, 5)
    other = types.SimpleNamespace(request_id="non-pool")

    request = types.SimpleNamespace(
        request_id="sched-req-uuid-2",
        prompt_cache=[other, c1, "string-ignored"],
    )
    freed = _free_paged_pool_cache(None, request)
    assert freed > 0
    assert pool.available() == initial
    mx.clear_cache()


def test_merge_batched_caches_shape_and_padding():
    import mlx.core as mx

    from fusion_mlx.custom_kernels.paged_kv_pool import (
        FusionPagedRequestCache,
        MergedPagedCacheView,
    )

    pool = _make_pool(num_blocks=16, block_size=4, n_kv_heads=2, head_dim=4)
    c0 = _make_cache(pool, "pool_0")
    c1 = _make_cache(pool, "pool_1")
    _fill_cache(c0, 6)
    _fill_cache(c1, 3)

    # merge() returns a writable MergedPagedCacheView over the constituents,
    # not a materialized _is_merged snapshot (the snapshot path crashed on
    # the first generation append — see merge() docstring).
    merged = FusionPagedRequestCache.merge([c0, c1])
    assert isinstance(merged, MergedPagedCacheView)
    assert merged.constituents == [c0, c1]
    assert merged.offset == 6, "merged offset must equal max length"
    mk, mv = merged.state
    assert mk.shape == (2, 2, 6, 4), f"unexpected merged keys shape {mk.shape}"
    assert mv.shape == (2, 2, 6, 4), f"unexpected merged values shape {mv.shape}"
    # Row 1 is c1 (3 tokens) left-aligned then zero-padded to 6.
    assert (
        int(mx.count_nonzero(mk[1, :, 3:, :]).item()) == 0
    ), "row 1 must be zero-padded past offset 3"
    # Batched append exercises left-padding bookkeeping: offsets [6, 3]
    # align to write position 6 → padding [0, 3].
    step_k = mx.zeros((2, 2, 1, 4), dtype=mx.float32)
    step_v = mx.zeros((2, 2, 1, 4), dtype=mx.float32)
    rk, rv = merged.update_and_fetch(step_k, step_v)
    assert rk.shape == (2, 2, 7, 4), f"unexpected post-append shape {rk.shape}"
    assert merged.offset == 7
    assert merged._left_padding.tolist() == [
        0,
        3,
    ], f"padding={merged._left_padding.tolist()}"
    mx.eval(rk, rv)
    mx.clear_cache()


def test_merge_then_free_restores_pool():
    import mlx.core as mx

    from fusion_mlx.custom_kernels.paged_kv_pool import FusionPagedRequestCache

    pool = _make_pool(num_blocks=16, block_size=4, n_kv_heads=2, head_dim=4, cow=True)
    initial = pool.available()

    c0 = _make_cache(pool, "pool_0")
    c1 = _make_cache(pool, "pool_1")
    _fill_cache(c0, 6)
    _fill_cache(c1, 3)

    merged = FusionPagedRequestCache.merge([c0, c1])
    mx.eval(merged.state)

    freed0 = c0.free_all()
    freed1 = c1.free_all()
    assert (freed0 + freed1) > 0
    freed_merged = merged.free_all()
    assert freed_merged == 0, "merged cache holds no un-freed pool blocks"
    assert pool.available() == initial, "pool must be restored after merge+free"
    mx.clear_cache()


pytestmark_real = pytest.mark.skipif(
    os.environ.get("FUSION_PAGED_KV_REAL_MODEL") != "on",
    reason="set FUSION_PAGED_KV_REAL_MODEL=on to run real-model paged-KV tests",
)


@pytestmark_real
def test_phase4_batched_real_model_no_leak():
    import mlx.core as mx
    import mlx_lm
    from mlx_lm.generate import stream_generate

    from fusion_mlx.custom_kernels.paged_kv_pool import (
        FusionPagedKVPool,
        FusionPagedRequestCache,
    )
    from fusion_mlx.fusion_takeover.config import FusionConfig
    from fusion_mlx.fusion_takeover.patcher import FusionModulePatcher

    model_path = os.environ.get(
        "FUSION_PAGED_KV_MODEL", "mlx-community/Qwen3-0.6B-4bit"
    )
    prompts = [
        os.environ.get("FUSION_PAGED_KV_PROMPT_A", "The quick brown fox"),
        os.environ.get("FUSION_PAGED_KV_PROMPT_B", "Once upon a time"),
    ]
    max_tokens = int(os.environ.get("FUSION_PAGED_KV_MAX_TOKENS", "20"))

    os.environ["FUSION_PAGED_FUSED_KERNEL"] = "on"
    model, tokenizer = mlx_lm.load(model_path)
    cfg = FusionConfig(
        enabled=True,
        paged_kv_enabled=True,
        fused_decode_enabled=True,
    )
    FusionModulePatcher.patch_model(model, cfg)

    single_streams = []
    for p in prompts:
        toks = []
        for resp in stream_generate(model, tokenizer, p, max_tokens=max_tokens):
            toks.append(int(resp.token))
            if len(toks) >= max_tokens:
                break
        single_streams.append(toks)
        logger.info("single stream prompt=%s tokens=%s", p, toks[:10])

    pool = FusionPagedKVPool(block_size=16, num_blocks=256, n_kv_heads=4, head_dim=64)
    initial_available = pool.available()

    batched_streams = [[], []]
    caches = [
        FusionPagedRequestCache(pool, "pool_0"),
        FusionPagedRequestCache(pool, "pool_1"),
    ]
    for i, p in enumerate(prompts):
        toks = []
        for resp in stream_generate(model, tokenizer, p, max_tokens=max_tokens):
            toks.append(int(resp.token))
            if len(toks) >= max_tokens:
                break
        batched_streams[i] = toks
        logger.info("batched stream prompt=%s tokens=%s", p, toks[:10])

    assert batched_streams[0] == single_streams[0], (
        f"prompt A token streams differ: "
        f"single={single_streams[0][:10]} batched={batched_streams[0][:10]}"
    )
    assert batched_streams[1] == single_streams[1], (
        f"prompt B token streams differ: "
        f"single={single_streams[1][:10]} batched={batched_streams[1][:10]}"
    )

    for c in caches:
        c.free_all()
    assert (
        pool.available() == initial_available
    ), f"pool leak: before={initial_available} after={pool.available()}"
    mx.clear_cache()
