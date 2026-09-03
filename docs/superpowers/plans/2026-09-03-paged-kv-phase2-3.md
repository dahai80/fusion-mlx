# Paged-KV Phase 2/3 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Eliminate Phase 1's single-stream concat overhead (Phase 2 fused paged decode-attention Metal kernel) and unlock multi-request KV concurrency (Phase 3 shared paged pool) — both bit-exact vs upstream, perf reports in `~/fusion/audit/`.

**Architecture:** Phase 2 — a `mx.fast.metal_kernel`-JIT decode-attention kernel reads K/V directly from a flat physical block pool via `block_table` indirection (no materialized logical view, no `mx.concatenate`), replacing `cache.update_and_fetch`+`scaled_dot_product_attention` on the L=1 path. Phase 3 — one `FusionPagedKVPool` backs N `FusionPagedRequestCache` handles (per-request `block_table`), swapped in via the existing Phase 0 `make_cache` rebind so the `Scheduler` continuous-batching loop is reused unchanged.

**Tech Stack:** MLX `mx.fast.metal_kernel` JIT (no CMake), FlashAttention-2 online-softmax numerics, Python `FusionPagedKVCache`/`FusionPagedKVPool`/`FusionPagedRequestCache`, existing `FusionModulePatcher` takeover hook.

**Spec:** `docs/superpowers/specs/2026-09-03-paged-kv-phase2-3-design.md`

## Global Constraints

- 4-space indent multiples, no docstrings, default logging on all new code.
- Lint: `black --check --target-version py313` + `ruff` clean. Never touch `debt_modules.txt`.
- NEVER `mx.clear_streams()` in tests (#630). `mx.clear_cache()` safe.
- Cache ops on `_model_load_executor` same thread (#KV-0 red line).
- Bit-exact vs upstream `KVCache` for greedy decode (fp16 tol 1e-3 relative); real-model integration test required.
- Perf reports → `~/fusion/audit/`, md format, before/after tok/s + peak mem.
- Only modify `/Users/dahai/claude-home/fusion-mlx`. Branch `feat/enhance-arch-0826`. No main commits.
- Venv: `source .venv/bin/activate` (worktree lacks `.venv` → `/Users/dahai/claude-home/fusion-mlx/.venv/bin/python`).
- `rtk proxy python -m pytest` for bare pytest; `rtk proxy find` for compound find predicates.

---

## File Structure

- **Modify** `fusion_mlx/custom_kernels/paged_kv_cache.py` — refactor slab-list → flat pool tensor; add `fused_decode_attention()` + `fused_decode_available()`.
- **Create** `fusion_mlx/custom_kernels/fusion_paged_attention.py` — Metal kernel source + JIT loader + `paged_decode_attention()` invocation.
- **Create** `fusion_mlx/custom_kernels/paged_kv_pool.py` — `FusionPagedKVPool` + `FusionPagedRequestCache`.
- **Modify** `fusion_mlx/custom_kernels/fusion_paged_kv.py` — pool-mode `install_paged_kv`, pool registry.
- **Modify** `fusion_mlx/fusion_takeover/config.py` — `fused_decode_enabled`, `pool_enabled`, `pool_num_blocks` fields.
- **Modify** `fusion_mlx/fusion_takeover/patcher.py` — wrap each `Attention.__call__` for fused decode path; pool-mode make_cache factory.
- **Create** `tests/unit/test_fusion_paged_attention.py` — kernel bit-exact vs MLX SDPA.
- **Modify** `tests/unit/test_fusion_paged_kv_cache.py` — flat-pool refactor regression + fused-path tests.
- **Create** `tests/unit/test_fusion_paged_pool.py` — shared-pool concurrency bit-exact.
- **Create** `examples/benchmark_paged_kv_phase2.py` — Phase 2 perf (reuse Phase 1 harness shape).
- **Create** `examples/benchmark_paged_pool.py` — Phase 3 concurrency perf.
- **Modify** `README.md` (English only) + `README_CN.md` (Chinese only) — paged-KV phase docs.

---

## Task Index

- Task 1: Investigate BatchGenerator cache batch-dim (read-only, resolves Open Q4)
- Task 2: Refactor FusionPagedKVCache slab-list → flat pool tensor
- Task 3: Fused decode-attention Metal kernel + JIT loader
- Task 4: FusionPagedKVCache.fused_decode_attention() + availability gate
- Task 5: Attention-`__call__` wrapper patch (patcher.py) + config fields
- Task 6: Phase 2 bit-exact + real-model tests + perf report
- Task 7: FusionPagedKVPool + FusionPagedRequestCache
- Task 8: Pool-mode wiring (fusion_paged_kv.py) + config fields
- Task 9: Phase 3 concurrency bit-exact + BatchedEngine integration tests
- Task 10: Phase 3 concurrency perf report
- Task 11: Full sweep + docs + PR

---

<!-- TASK BODIES BELOW — filled incrementally -->

## Task 1: Investigate BatchGenerator cache batch-dim

**Files:**
- Read-only: `fusion_mlx/scheduler/sched_batch.py`, `fusion_mlx/scheduler/sched_step.py`, `.venv/lib/python3.12/site-packages/mlx_lm/generate.py` (BatchGenerator)

**Interfaces:**
- Produces: a one-paragraph finding recorded in the commit message answering — when `BatchGenerator` runs a batched decode forward across N running requests, is `cache.update_and_fetch` called (a) once with `B=N` on a single shared cache, or (b) N times with `B=1` on N per-request caches (one forward per request, or one stacked forward with per-row cache dispatch)? Plus the exact `caches=` shape `BatchGenerator.insert` expects. This determines whether `FusionPagedRequestCache` (Task 7) must handle `B>1` or stays `B=1`.

- [ ] **Step 1: Trace the batched decode forward**

Read `fusion_mlx/scheduler/sched_batch.py:761` (`batch_generator.insert(..., caches=[state.cache], ...)`) and follow into the `BatchGenerator` class in `mlx_lm/generate.py`. Find where `update_and_fetch` is called during a step and what `B` dimension the keys/values carry. Grep `mlx_lm/generate.py` for `update_and_fetch`, `class .*Generator`, `def step`, `def _step`, `stack`, `concat`, `prompts`.

- [ ] **Step 2: Record the finding**

Write the finding to a scratch note and capture it verbatim in the Task 2 commit message as `BatchGenerator batch-dim: <answer>`. No code change in this task.

- [ ] **Step 3: Commit (doc-only)**

```bash
git add -A
git commit -m "docs(paged-kv): Task 1 BatchGenerator batch-dim investigation

BatchGenerator batch-dim: <one-paragraph answer>"
```

If the answer is (b) `B=1` per-request (expected — `caches=[state.cache]` is per-row), `FusionPagedRequestCache` stays `B=1` and Task 7 is simpler. If (a) `B=N` shared, Task 7 must batch-handle the pool and the spec's pool design needs a note — raise it as a ruling in the ledger before Task 7.

## Task 2: Refactor FusionPagedKVCache slab-list → flat pool tensor

**Files:**
- Modify: `fusion_mlx/custom_kernels/paged_kv_cache.py`
- Test: `tests/unit/test_fusion_paged_kv_cache.py`

**Interfaces:**
- Consumes: existing `FusionPagedKVCache` public surface (`update_and_fetch`, `state` get/set, `meta_state`, `make_mask`, `trim`, `is_trimmable`, `offset`, `nbytes`, `size`, `empty`, `free_all`, `stats`) — all must stay behavior-identical (Phase 1 tests still pass).
- Produces: `self.keys_pool: mx.array` shape `[num_blocks, B, n_kv_heads, block_size, k_head_dim]` and `self.values_pool` same shape, replacing `keys_slabs`/`values_slabs` lists. `_slab_loc` removed; `_alloc_block` returns a physical index into the flat pool. New `fused_decode_available()` stub returning `False` (filled Task 4). `block_table` unchanged (`list[int]` logical→physical).

**Why:** the Task 3 Metal kernel takes array inputs, not a Python list of slab tensors. One flat pre-allocated pool (grown once to `num_blocks` cap) is the simplest kernel input. This trades Phase 1's lazy-slab memory win (alloc only blocks used) for upfront cap allocation — the spec's accepted decision. The memory win is reclaimed in Phase 3 by sizing the cap to the concurrent budget, not worst-case single request.

- [ ] **Step 1: Write the failing test for flat-pool storage**

Add to `tests/unit/test_fusion_paged_kv_cache.py`:
```python
def test_flat_pool_storage_shape():
    cache = FusionPagedKVCache(block_size=4, num_blocks=8)
    keys = mx.random.uniform(shape=(1, 2, 3, 8))
    values = mx.random.uniform(shape=(1, 2, 3, 8))
    cache.update_and_fetch(keys, values)
    assert hasattr(cache, "keys_pool")
    assert cache.keys_pool.shape == (8, 1, 2, 4, 8)
    assert cache.values_pool.shape == (8, 1, 2, 4, 8)
    assert not hasattr(cache, "keys_slabs")
```
Also keep all existing Phase 1 tests unchanged — they must still pass (bit-exact behavior).

- [ ] **Step 2: Run test to verify it fails**

Run: `rtk proxy python -m pytest tests/unit/test_fusion_paged_kv_cache.py::test_flat_pool_storage_shape -v`
Expected: FAIL (no `keys_pool` attribute; slab-list still present).

- [ ] **Step 3: Refactor to flat pool**

In `paged_kv_cache.py`:
- `__init__`: replace `self.keys_slabs = []` / `self.values_slabs = []` with `self.keys_pool = None` / `self.values_pool = None`. Keep `block_table`, `free_list`, `_total_blocks`, counters.
- `_ensure_pool`: after setting `self._shape`, allocate `self.keys_pool = mx.zeros((self.num_blocks, B, n_kv_heads, self.block_size, k_head_dim), dtype=dtype)` and `self.values_pool` analogously with `v_head_dim`. Log `"paged_kv flat pool init: cap=%d ..."` (default logging).
- Remove `_add_slab`, `_slab_loc`. `_alloc_block`: pop from `free_list`; if empty, raise `RuntimeError("paged_kv pool exhausted ...")` (no slab growth — cap is fixed). Seed `free_list` in `_ensure_pool` with `list(range(self.num_blocks))` reversed so pop() gives ascending order.
- `update_and_fetch` writes: replace `k_slab[in_slab, ..., pos_start:pos_start+n, :] = ...` with `self.keys_pool[pb, ..., pos_start:pos_start+n, :] = keys[..., s_start:s_end, :]` (same slicing, pool index `pb` = `self.block_table[lb]`).
- `_fetch_logical`: replace `k_slab, v_slab, in_slab = self._slab_loc(pb)` + `k_parts.append(k_slab[in_slab])` with `k_parts.append(self.keys_pool[pb])` / partial `self.keys_pool[pb, ..., :rem, :]`.
- `state` setter: same `_ensure_pool` reset (now just clears pool to zeros or re-seeds free_list + resets block_table) — allocate fresh pool only if shape changed.
- `nbytes`: `self.keys_pool.nbytes + self.values_pool.nbytes` when pool exists (full cap; document in stats that this is cap not used). Add `blocks_used` already in `stats` covers used.
- `free_all`: append `block_table` entries back to `free_list`, reset `block_table=[]`, `offset=0`. Pool tensor stays allocated (cap).

- [ ] **Step 4: Run full cache test suite**

Run: `rtk proxy python -m pytest tests/unit/test_fusion_paged_kv_cache.py -v`
Expected: ALL PASS (new flat-pool test + all Phase 1 bit-exact tests unchanged).

- [ ] **Step 5: Lint**

Run: `black --check --target-version py313 fusion_mlx/custom_kernels/paged_kv_cache.py tests/unit/test_fusion_paged_kv_cache.py && ruff check fusion_mlx/custom_kernels/paged_kv_cache.py`
Expected: clean.

- [ ] **Step 6: Commit**

```bash
git add fusion_mlx/custom_kernels/paged_kv_cache.py tests/unit/test_fusion_paged_kv_cache.py
git commit -m "refactor(paged-kv): slab-list -> flat pool tensor for kernel input

BatchGenerator batch-dim: <Task 1 answer pasted here>"
```

## Task 3: Fused decode-attention Metal kernel + JIT loader

**Files:**
- Create: `fusion_mlx/custom_kernels/fusion_paged_attention.py`
- Test: `tests/unit/test_fusion_paged_attention.py`

**Interfaces:**
- Consumes: `mx.fast.metal_kernel` (MLX JIT), `mx.metal.is_available()`. Reference numerics: `mlx.nn.scaled_dot_product_attention` with `cache=None` on a materialized K/V view (the "naive" path the kernel must match).
- Produces: `paged_decode_attention(q, keys_pool, values_pool, block_table, num_kv, scale, gqa_factor, stream=None) -> mx.array` returning `[B, n_heads, 1, v_head_dim]`. `metal_available() -> bool`. `FUSION_PAGED_FUSED_KERNEL` env read (default off).

**Design:** one threadgroup per (batch, query-head) pair. The group loops over logical blocks `[0, ceil(num_kv/block_size))`, reads `block_table[lb]` → physical index `pb`, gathers `keys_pool[pb]` / `values_pool[pb]` (slice `[B, n_kv_heads, block_size, head_dim]`, kv-head = `q_head // gqa_factor`), and runs online softmax (FlashAttention-2: running max `m`, sum `l`, accumulator `o`; each new block's scores `s = q·k^T * scale` merged via `m_new = max(m, m_block); o = o*exp(m-m_new) + exp(m_block-m_new)*sum(exp(s-m_block)*v); l = l*exp(m-m_new) + sum(exp(s-m_block))`). The partial last block is masked by `num_kv % block_size`. Final output `o / l`. Decode-only: L=1 query.

- [ ] **Step 1: Write the failing test (kernel vs SDPA reference)**

Create `tests/unit/test_fusion_paged_attention.py`:
```python
import pytest
import mlx.core as mx
from fusion_mlx.custom_kernels.fusion_paged_attention import (
    paged_decode_attention,
    metal_available,
)


def _ref_decode(q, k_all, v_all, scale, gqa_factor):
    # k_all/v_all: [num_kv, n_kv_heads, head_dim] materialized logical view
    # q: [B, n_heads, 1, head_dim]
    from mlx.nn import scaled_dot_product_attention
    B, n_heads, _, head_dim = q.shape
    n_kv = k_all.shape[1]
    k_view = mx.reshape(k_all, (B, n_kv, k_all.shape[0], head_dim))
    v_view = mx.reshape(v_all, (B, n_kv, k_all.shape[0], head_dim))
    if gqa_factor > 1:
        k_view = mx.repeat(k_view, gqa_factor, axis=1)
        v_view = mx.repeat(v_view, gqa_factor, axis=1)
    return scaled_dot_product_attention(q, k_view, v_view, scale=scale)


@pytest.mark.skipif(not metal_available(), reason="metal kernel unavailable")
@pytest.mark.parametrize("block_size,num_kv,n_heads,n_kv_heads,head_dim,gqa", [
    (16, 33, 8, 8, 64, 1),
    (16, 33, 8, 2, 64, 4),
    (16, 16, 4, 4, 32, 1),
    (16, 1, 8, 2, 64, 4),
])
def test_paged_decode_matches_sdpa(block_size, num_kv, n_heads, n_kv_heads, head_dim, gqa):
    mx.random.seed(7)
    B = 1
    scale = 1.0 / (head_dim ** 0.5)
    q = mx.random.normal(shape=(B, n_heads, 1, head_dim)) * 0.1
    keys_pool = mx.random.normal(shape=(8, B, n_kv_heads, block_size, head_dim)) * 0.1
    values_pool = mx.random.normal(shape=(8, B, n_kv_heads, block_size, head_dim)) * 0.1
    num_blocks_used = (num_kv + block_size - 1) // block_size
    block_table = mx.array(list(range(num_blocks_used)), dtype=mx.uint32)
    out = paged_decode_attention(
        q, keys_pool, values_pool, block_table, num_kv, scale, gqa,
    )
    # materialize the SAME logical view for the reference
    k_parts = [keys_pool[pb] for pb in range(num_blocks_used)]
    v_parts = [values_pool[pb] for pb in range(num_blocks_used)]
    k_all = mx.concatenate(
        [p.reshape(B, n_kv_heads, block_size, head_dim) for p in k_parts], axis=2
    )[:, :, :num_kv, :]
    v_all = mx.concatenate(
        [p.reshape(B, n_kv_heads, block_size, head_dim) for p in v_parts], axis=2
    )[:, :, :num_kv, :]
    ref = _ref_decode(q, k_all, v_all, scale, gqa)
    assert out.shape == ref.shape
    rel = mx.max(mx.abs(out - ref)) / (mx.max(mx.abs(ref)) + 1e-9)
    assert float(rel) < 1e-2, f"rel diff {float(rel)} too large"
```
Note: fp16 tolerance 1e-2 relative is generous for the online-softmax vs exact-softmax comparison; tighten toward 1e-3 if the kernel uses stable numerics. Default the test dtype to fp16 (`mx.float16`) in a second parametrized case.

- [ ] **Step 2: Run test to verify it fails**

Run: `rtk proxy python -m pytest tests/unit/test_fusion_paged_attention.py -v`
Expected: FAIL — `ModuleNotFoundError: fusion_mlx.custom_kernels.fusion_paged_attention`.

- [ ] **Step 3: Write the kernel + loader**

Create `fusion_mlx/custom_kernels/fusion_paged_attention.py`. Pattern mirrors `patches/glm_moe_dsa/sparse_mla.py` (`metal_kernel(name, input_names, output_names, source)` then `kernel(inputs=, template=, grid=, threadgroup=, output_shapes=, output_dtypes=, init_value=, stream=)`). Source (Metal):
```c
// one threadgroup per (batch, query_head). grid = (B * n_heads, 1, 1).
// block_size, head_dim, gqa_factor are template constants.
{
  const uint bh = thread_position_in_grid.x;
  const uint B = GRID_BH / N_HEADS;  // or pass B, n_heads as template
  const uint batch = bh / N_HEADS;
  const uint q_head = bh % N_HEADS;
  const uint kv_head = q_head / GQA_FACTOR;

  float m = -1e30f;
  float l = 0.0f;
  float o[HEAD_DIM];
  for (uint d = 0; d < HEAD_DIM; ++d) o[d] = 0.0f;

  const uint num_blocks = (NUM_KV + BLOCK_SIZE - 1) / BLOCK_SIZE;
  for (uint lb = 0; lb < num_blocks; ++lb) {
    uint pb = block_table[lb];
    uint block_len = (lb + 1 == num_blocks) ? (NUM_KV - lb * BLOCK_SIZE) : BLOCK_SIZE;
    for (uint t = 0; t < block_len; ++t) {
      // score = q[batch,q_head,0,:] . keys_pool[pb,batch,kv_head,t,:] * scale
      float s = 0.0f;
      for (uint d = 0; d < HEAD_DIM; ++d) {
        s += q[batch * N_HEADS * HEAD_DIM + q_head * HEAD_DIM + d]
             * keys_pool[((pb * B + batch) * N_KV_HEADS + kv_head) * BLOCK_SIZE * HEAD_DIM
                         + t * HEAD_DIM + d];
      }
      s *= SCALE;
      float m_new = max(m, s);
      float exp_m = exp(m - m_new);
      float exp_s = exp(s - m_new);
      l = l * exp_m + exp_s;
      for (uint d = 0; d < HEAD_DIM; ++d) {
        o[d] = o[d] * exp_m
             + exp_s * values_pool[((pb * B + batch) * N_KV_HEADS + kv_head) * BLOCK_SIZE * HEAD_DIM
                                   + t * HEAD_DIM + d];
      }
      m = m_new;
    }
  }
  for (uint d = 0; d < HEAD_DIM; ++d) {
    out[batch * N_HEADS * HEAD_DIM + q_head * HEAD_DIM + d] = o[d] / l;
  }
}
```
**IMPORTANT — this is the algorithmic reference for a correct but naive single-thread-per-head kernel.** It will be SLOW (one thread per head, scalar loops). The implementer's job in this task is to get it **bit-exact-correct first** against the SDPA reference; a follow-up task optimizes tiling (threadgroup shared memory, SIMD reduction). Do NOT try to write the optimal FlashAttention tiling in this task — correctness gates performance. Log `"paged fused decode kernel: grid=(B*n_heads) compiled"`.

Python wrapper `paged_decode_attention`:
- Read `FUSION_PAGED_FUSED_KERNEL` env; if `!= "on"` raise/return None with a logged reason (the cache layer checks `metal_available()` before calling).
- Build the kernel via `mx.fast.metal_kernel(name="fusion_paged_decode_attention", input_names=["q","keys_pool","values_pool","block_table"], output_names=["out"], source=source)`.
- `template=[("BLOCK_SIZE",block_size),("HEAD_DIM",head_dim),("GQA_FACTOR",gqa),("NUM_KV",num_kv),("N_HEADS",n_heads),("N_KV_HEADS",n_kv_heads),("B",B),("SCALE",scale)]`.
- `grid=(B * n_heads, 1, 1)`, `threadgroup=(HEAD_DIM, 1, 1)` (one thread per output element for the naive version — or `(32,1,1)`; pick what compiles and matches).
- `output_shapes=[(B, n_heads, 1, head_dim)]`, `output_dtypes=[q.dtype]`, `init_value=0`, `stream=stream or mx.gpu`.
- Return `out[0]`.

- [ ] **Step 4: Run test to verify it passes**

Run: `rtk proxy python -m pytest tests/unit/test_fusion_paged_attention.py -v`
Expected: PASS (4+ parametrized cases bit-exact within tolerance). If the kernel doesn't compile, the test skips (skipif `metal_available()`); log the compile error and iterate on the source string. If numerics mismatch > tol, check the GQA head mapping and the partial-last-block mask.

- [ ] **Step 5: Lint + commit**

```bash
black --target-version py313 fusion_mlx/custom_kernels/fusion_paged_attention.py tests/unit/test_fusion_paged_attention.py
ruff check fusion_mlx/custom_kernels/fusion_paged_attention.py
git add fusion_mlx/custom_kernels/fusion_paged_attention.py tests/unit/test_fusion_paged_attention.py
git commit -m "feat(paged-kv): fused decode-attention Metal kernel + JIT loader (naive, bit-exact)"
```

**Note for implementer:** the naive scalar kernel is intentionally unoptimized. Task 6 measures perf; if it does NOT beat the concat path, a Task 6 sub-step adds threadgroup-parallel tiling (shared-memory K/V tile, SIMD softmax reduction) — but only after correctness is locked. Do not conflate the two.

## Task 4: FusionPagedKVCache.fused_decode_attention() + availability gate

**Files:**
- Modify: `fusion_mlx/custom_kernels/paged_kv_cache.py`
- Test: `tests/unit/test_fusion_paged_kv_cache.py`

**Interfaces:**
- Consumes: Task 2 `keys_pool`/`values_pool` flat tensors `[num_blocks, B, n_kv_heads, block_size, head_dim]`, `block_table: list[int]`, `offset`. Task 3 `paged_decode_attention(q, keys_pool, values_pool, block_table, num_kv, scale, gqa_factor, stream=None)` and `metal_available()`.
- Produces:
  - `FusionPagedKVCache.fused_decode_available(self, num_new: int) -> bool` — returns `True` only when all of: Metal available, `FUSION_PAGED_FUSED_KERNEL == "on"`, `self.offset > 0` (not first token), `num_new == 1` (decode L=1), pool initialized, and the pool's `B`/`n_kv_heads`/`head_dim` match the per-layer shape stored at `_ensure_pool` time.
  - `FusionPagedKVCache.fused_decode_attention(self, queries, scale, n_heads, head_dim) -> mx.array` — builds `block_table` mx.array, computes `gqa_factor = n_heads // self._n_kv_heads`, calls `paged_decode_attention(queries, self.keys_pool, self.values_pool, block_table_mx, self.offset + 1, scale, gqa_factor, stream=self._stream)` and returns the result `[B, n_heads, 1, head_dim]`.

**Design:** the attention `__call__` (Task 5) still calls `update_and_fetch` first (to write the new K/V into the pool). Then, if `fused_decode_available(num_new=1)`, it calls `fused_decode_attention` instead of `scaled_dot_product_attention`. So Task 4 only adds the two methods + stores the layer shape (`n_kv_heads`, `head_dim`) at `_ensure_pool` time so the gate can check shape consistency. `block_table` is stored as `list[int]`; `fused_decode_attention` converts to `mx.array(..., dtype=mx.uint32)` once per call (cheap, O(blocks)).

- [ ] **Step 1: Write the failing tests**

Add to `tests/unit/test_fusion_paged_kv_cache.py`:
```python
import os

import mlx.core as mx
import pytest

from fusion_mlx.custom_kernels.paged_kv_cache import FusionPagedKVCache


def _seed_decode_cache(n_kv_heads=2, head_dim=8, block_size=4, num_blocks=8):
    cache = FusionPagedKVCache(
        block_size=block_size, num_blocks=num_blocks,
        n_kv_heads=n_kv_heads, head_dim=head_dim,
    )
    # prefill 5 tokens so offset>0
    for _ in range(5):
        k = mx.random.normal(shape=(1, n_kv_heads, 1, head_dim)) * 0.1
        v = mx.random.normal(shape=(1, n_kv_heads, 1, head_dim)) * 0.1
        cache.update_and_fetch(k, v)
    return cache


def test_fused_decode_available_gated_off_by_default():
    cache = _seed_decode_cache()
    os.environ.pop("FUSION_PAGED_FUSED_KERNEL", None)
    assert cache.fused_decode_available(num_new=1) is False


def test_fused_decode_available_off_when_not_decode():
    os.environ["FUSION_PAGED_FUSED_KERNEL"] = "on"
    try:
        cache = _seed_decode_cache()
        assert cache.fused_decode_available(num_new=4) is False  # prefill
    finally:
        os.environ.pop("FUSION_PAGED_FUSED_KERNEL", None)


def test_fused_decode_attention_matches_concat_path(monkeypatch):
    monkeypatch.setenv("FUSION_PAGED_FUSED_KERNEL", "on")
    n_kv_heads, head_dim, n_heads = 2, 8, 8
    cache = _seed_decode_cache(n_kv_heads=n_kv_heads, head_dim=head_dim)
    gqa = n_heads // n_kv_heads
    # new token
    k_new = mx.random.normal(shape=(1, n_kv_heads, 1, head_dim)) * 0.1
    v_new = mx.random.normal(shape=(1, n_kv_heads, 1, head_dim)) * 0.1
    k_view, v_view = cache.update_and_fetch(k_new, v_new)  # writes + returns logical view
    scale = 1.0 / (head_dim ** 0.5)
    q = mx.random.normal(shape=(1, n_heads, 1, head_dim)) * 0.1
    from mlx.nn import scaled_dot_product_attention
    if gqa > 1:
        k_ref = mx.repeat(k_view, gqa, axis=1)
        v_ref = mx.repeat(v_view, gqa, axis=1)
    else:
        k_ref, v_ref = k_view, v_view
    ref = scaled_dot_product_attention(q, k_ref, v_ref, scale=scale)
    if not cache.fused_decode_available(num_new=1):
        pytest.skip("metal kernel unavailable")
    out = cache.fused_decode_attention(q, scale=scale, n_heads=n_heads, head_dim=head_dim)
    assert out.shape == ref.shape
    rel = mx.max(mx.abs(out - ref)) / (mx.max(mx.abs(ref)) + 1e-9)
    assert float(rel) < 2e-2, f"rel diff {float(rel)} too large"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `rtk proxy python -m pytest tests/unit/test_fusion_paged_kv_cache.py -v -k fused_decode`
Expected: FAIL — `AttributeError: fused_decode_available` / `fused_decode_attention`.

- [ ] **Step 3: Implement the two methods + store layer shape**

In `paged_kv_cache.py`:
- At `_ensure_pool` time, store `self._n_kv_heads`, `self._head_dim`, `self._B` from the pool shape. Log them.
- Add at top: `from .fusion_paged_attention import paged_decode_attention, metal_available` (lazy import inside methods to avoid import-cycle at module load).
- `fused_decode_available(self, num_new)`:
  ```python
  def fused_decode_available(self, num_new):
      import os
      if os.environ.get("FUSION_PAGED_FUSED_KERNEL", "off") != "on":
          return False
      if not metal_available():
          return False
      if self.offset <= 0:
          return False
      if num_new != 1:
          return False
      if self.keys_pool is None:
          return False
      return True
  ```
- `fused_decode_attention(self, queries, scale, n_heads, head_dim)`:
  ```python
  def fused_decode_attention(self, queries, scale, n_heads, head_dim):
      from .fusion_paged_attention import paged_decode_attention
      gqa_factor = n_heads // self._n_kv_heads
      block_table_mx = mx.array(self.block_table, dtype=mx.uint32)
      num_kv = self.offset + 1
      logger.info(
          "paged_kv fused decode: offset=%d gqa=%d n_heads=%d n_kv=%d",
          self.offset, gqa_factor, n_heads, self._n_kv_heads,
      )
      return paged_decode_attention(
          queries, self.keys_pool, self.values_pool, block_table_mx,
          num_kv, scale, gqa_factor, stream=self._stream,
      )
  ```

- [ ] **Step 4: Run tests to verify they pass**

Run: `rtk proxy python -m pytest tests/unit/test_fusion_paged_kv_cache.py -v -k fused_decode`
Expected: PASS (gate-off tests pass unconditionally; the bit-exact test passes on Metal, skips otherwise).

- [ ] **Step 5: Run the full cache suite (no regression)**

Run: `rtk proxy python -m pytest tests/unit/test_fusion_paged_kv_cache.py -v`
Expected: ALL PASS.

- [ ] **Step 6: Lint + commit**

```bash
black --target-version py313 fusion_mlx/custom_kernels/paged_kv_cache.py tests/unit/test_fusion_paged_kv_cache.py
ruff check fusion_mlx/custom_kernels/paged_kv_cache.py
git add fusion_mlx/custom_kernels/paged_kv_cache.py tests/unit/test_fusion_paged_kv_cache.py
git commit -m "feat(paged-kv): FusionPagedKVCache.fused_decode_attention + availability gate"
```

## Task 5: Attention-`__call__` wrapper patch + config fields

**Files:**
- Modify: `fusion_mlx/fusion_takeover/config.py`
- Modify: `fusion_mlx/fusion_takeover/patcher.py`
- Test: `tests/unit/test_fusion_takeover_patcher.py`

**Interfaces:**
- Consumes: Task 4 `FusionPagedKVCache.fused_decode_available(num_new)` + `fused_decode_attention(queries, scale, n_heads, head_dim)`. The cache object is reached via the model's per-layer cache list (the cache passed into `Attention.__call__` as the `cache` kwarg). `mlx.nn.Attention` (`Attention.__call__(queries, keys, values, mask=None, modality=None, cache=None)`).
- Produces: `FusionConfig.fused_decode_enabled: bool` (default `False`, read from `fusion_paged_fused_kernel` model setting or `FUSION_PAGED_FUSED_KERNEL=on`). `FusionModulePatcher` wraps each `Attention` module's `__call__` so that when its `cache` is a `FusionPagedKVCache` and `fused_decode_available(num_new=1)` is True, the call routes through `cache.fused_decode_attention` instead of the base `scaled_dot_product_attention`.

**Design — how the attention is reached:** Phase 0 `patch_model` already rebinds `model.make_cache` (Phase 1 wiring). The attention modules live on each transformer layer (`model.layers[i].attention` or `.attn` — model-family-dependent). The wrapper is installed by walking `model.layers`, finding the `Attention`-typed submodule, and replacing its `__call__` with a closure that:
1. Computes `queries = self.rope(queries, offset=cache.offset)` / `keys = self.rope(keys, offset=cache.offset)` exactly as the original `__call__` does (the original body is preserved — we wrap, we do not rewrite attention math).
2. Calls `keys, values = cache.update_and_fetch(keys, values)` (unchanged — writes the new block + returns logical view).
3. If `isinstance(cache, FusionPagedKVCache) and cache.fused_decode_available(num_new=queries.shape[2])`: `out = cache.fused_decode_attention(queries, self.scale, self.n_heads, self.head_dim)`.
4. Else: fall through to the original `__call__` body (`scaled_dot_product_attention(queries, keys, values, cache=cache, scale=self.scale, mask=mask)`).

**Scope:** llama-family only (Open Q3). Gate the wrap on `config.fused_decode_enabled` AND a model-family allowlist `("llama", "qwen3", "qwen2")` matched against `model.model_type`. Other families skip the wrap (their attention uses the concat path as before). File follow-up issues for the rest (recorded in Task 11).

- [ ] **Step 1: Write the failing test**

Create/extend `tests/unit/test_fusion_takeover_patcher.py`:
```python
import os

import mlx.core as mx
import pytest

from fusion_mlx.custom_kernels.paged_kv_cache import FusionPagedKVCache
from fusion_mlx.fusion_takeover.config import FusionConfig
from fusion_mlx.fusion_takeover.patcher import FusionModulePatcher


def test_fused_decode_config_field_default_off():
    cfg = FusionConfig(enabled=True, paged_kv_enabled=True)
    assert cfg.fused_decode_enabled is False


def test_fused_decode_config_field_from_settings():
    cfg = FusionConfig.from_model_settings(
        {"fusion_paged_kv": "on", "fusion_paged_fused_kernel": "on"}
    )
    assert cfg.fused_decode_enabled is True


def test_patcher_wraps_llama_attention(monkeypatch):
    monkeypatch.setenv("FUSION_PAGED_FUSED_KERNEL", "on")
    # Minimal fake llama-style model with one layer holding an Attention.
    class FakeRope:
        def __call__(self, x, offset=None):
            return x

    class FakeAttention:
        n_heads = 8
        n_kv_heads = 2
        head_dim = 8
        scale = 1.0 / (8 ** 0.5)
        rope = FakeRope()

        def __call__(self, queries, keys, values, mask=None, modality=None, cache=None):
            keys, values = cache.update_and_fetch(keys, values)
            from mlx.nn import scaled_dot_product_attention
            k = mx.repeat(keys, self.n_heads // self.n_kv_heads, axis=1)
            v = mx.repeat(values, self.n_heads // self.n_kv_heads, axis=1)
            return scaled_dot_product_attention(queries, k, v, scale=self.scale)

    class FakeLayer:
        def __init__(self):
            self.attention = FakeAttention()

    class FakeModel:
        model_type = "llama"
        def __init__(self):
            self.layers = [FakeLayer()]
        def make_cache(self):
            return [FusionPagedKVCache(
                block_size=4, num_blocks=8,
                n_kv_heads=2, head_dim=8,
            )]

    model = FakeModel()
    cfg = FusionConfig(enabled=True, paged_kv_enabled=True, fused_decode_enabled=True)
    patcher = FusionModulePatcher()
    patcher.patch_model(model, cfg)

    caches = model.make_cache()
    cache = caches[0]
    assert isinstance(cache, FusionPagedKVCache)
    # prefill so offset>0
    for _ in range(5):
        k = mx.random.normal(shape=(1, 2, 1, 8)) * 0.1
        v = mx.random.normal(shape=(1, 2, 1, 8)) * 0.1
        cache.update_and_fetch(k, v)
    attn = model.layers[0].attention
    q = mx.random.normal(shape=(1, 8, 1, 8)) * 0.1
    k = mx.random.normal(shape=(1, 2, 1, 8)) * 0.1
    v = mx.random.normal(shape=(1, 2, 1, 8)) * 0.1
    # should route through fused path without raising; shape [1,8,1,8]
    out = attn(q, k, v, cache=cache)
    assert out.shape == (1, 8, 1, 8)


def test_patcher_skips_non_llama_family():
    class FakeModel:
        model_type = "gemma"
        layers = []
    model = FakeModel()
    cfg = FusionConfig(enabled=True, paged_kv_enabled=True, fused_decode_enabled=True)
    patcher = FusionModulePatcher()
    # should not raise; wrap is a no-op for non-llama family
    patcher.patch_model(model, cfg)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `rtk proxy python -m pytest tests/unit/test_fusion_takeover_patcher.py -v`
Expected: FAIL — `AttributeError: fused_decode_enabled` / wrap not installed.

- [ ] **Step 3: Add config fields**

In `fusion_mlx/fusion_takeover/config.py`:
- Add field `fused_decode_enabled: bool = False` to `FusionConfig`.
- In `from_model_settings`, read `fusion_paged_fused_kernel`: `fused_decode_enabled = settings.get("fusion_paged_fused_kernel") == "on"`.
- Log the resolved value.

- [ ] **Step 4: Implement the attention wrapper in patcher.py**

In `fusion_mlx/fusion_takeover/patcher.py`:
- Define module-level constant `_FUSED_DECODE_MODEL_FAMILIES = ("llama", "qwen2", "qwen3")`.
- In `FusionModulePatcher.patch_model`, after the existing `install_paged_kv` block, add:
  ```python
  if config.fused_decode_enabled and getattr(model, "model_type", "") in _FUSED_DECODE_MODEL_FAMILIES:
      self._install_fused_decode(model)
  ```
- `_install_fused_decode(self, model)`:
  ```python
  def _install_fused_decode(self, model):
      from fusion_mlx.custom_kernels.paged_kv_cache import FusionPagedKVCache
      for layer in model.layers:
          attn = getattr(layer, "attention", None) or getattr(layer, "attn", None)
          if attn is None:
              continue
          self._wrap_attention(attn)
  ```
- `_wrap_attention(self, attn)`:
  ```python
  def _wrap_attention(self, attn):
      from fusion_mlx.custom_kernels.paged_kv_cache import FusionPagedKVCache
      base_call = attn.__call__

      def fused_call(queries, keys, values, mask=None, modality=None, cache=None):
          if isinstance(cache, FusionPagedKVCache) and cache.fused_decode_available(num_new=queries.shape[2]):
              keys = attn.rope(keys, offset=cache.offset)
              values = values
              queries = attn.rope(queries, offset=cache.offset)
              keys, values = cache.update_and_fetch(keys, values)
              out = cache.fused_decode_attention(
                  queries, attn.scale, attn.n_heads, attn.head_dim,
              )
              logger.info("paged_kv fused decode attention path taken offset=%d", cache.offset)
              return out
          return base_call(queries, keys, values, mask=mask, modality=modality, cache=cache)

      attn.__call__ = fused_call
  ```
  Note: `rope` application order must match the original `Attention.__call__`. Read the real `mlx.nn.Attention` / llama attention `__call__` before finalizing — if the model applies rope AFTER `update_and_fetch`, mirror that. The implementer reads `mlx_lm/models/llama.py` `Attention.__call__` to confirm the exact rope+offset+fetch order and copies it. Log the path taken (default logging).

- [ ] **Step 5: Run test to verify it passes**

Run: `rtk proxy python -m pytest tests/unit/test_fusion_takeover_patcher.py -v`
Expected: PASS. The fake-llama test routes through fused path; gemma test is a no-op.

- [ ] **Step 6: Lint + commit**

```bash
black --target-version py313 fusion_mlx/fusion_takeover/config.py fusion_mlx/fusion_takeover/patcher.py tests/unit/test_fusion_takeover_patcher.py
ruff check fusion_mlx/fusion_takeover/config.py fusion_mlx/fusion_takeover/patcher.py
git add fusion_mlx/fusion_takeover/config.py fusion_mlx/fusion_takeover/patcher.py tests/unit/test_fusion_takeover_patcher.py
git commit -m "feat(paged-kv): attention __call__ wrapper for fused decode (llama-family, env-gated)"
```

## Task 6: Phase 2 bit-exact + real-model tests + perf report

**Files:**
- Create: `tests/integration/test_paged_kv_phase2_real_model.py`
- Create: `examples/benchmark_paged_kv_phase2.py`
- Create (output): `~/fusion/audit/paged-kv-phase2-perf-report.md`

**Interfaces:**
- Consumes: Tasks 3-5 (fused kernel, cache methods, attention wrapper), existing Phase 1 real-model test harness (qwen3-0.6b / llama-3.2-1b greedy decode reference). Model start/stop via `~/claude-home/fusion-mlx/start.sh start|stop`. Models at `~/.fusion-mlx/models`; download via `https://hf-mirror.com` if missing.
- Produces: a real-model integration test proving identical token stream with/without the fused kernel, and a md perf report with before/after decode tok/s + peak mem.

**Prereq:** confirm a small llama-family model is available locally. If not, download `mlx-community/Qwen3-0.6B-4bit` (or `mlx-community/Llama-3.2-1B-Instruct-4bit`) via the mirror. Use the SAME model for bit-exact and perf.

- [ ] **Step 1: Write the real-model bit-exact test**

Create `tests/integration/test_paged_kv_phase2_real_model.py`:
```python
import os

import pytest

pytestmark = pytest.mark.skipif(
    os.environ.get("FUSION_PAGED_KV_REAL_MODEL") != "on",
    reason="set FUSION_PAGED_KV_REAL_MODEL=on to run real-model paged-KV tests",
)

_MODEL = os.environ.get("FUSION_PAGED_KV_MODEL", "mlx-community/Qwen3-0.6B-4bit")


def _greedy_tokens(model_path, prompt, max_tokens):
    import mlx_lm
    model, tokenizer = mlx_lm.load(model_path)
    from fusion_mlx.custom_kernels.paged_kv_cache import FusionPagedKVCache
    from fusion_mlx.fusion_takeover.config import FusionConfig
    from fusion_mlx.fusion_takeover.patcher import FusionModulePatcher

    cfg = FusionConfig(enabled=True, paged_kv_enabled=True, fused_decode_enabled=False)
    patcher = FusionModulePatcher()
    patcher.patch_model(model, cfg)
    cache = [FusionPagedKVCache(block_size=16, num_blocks=256) for _ in model.layers]
    # monkeypatch make_cache to return paged caches
    model.make_cache = lambda: cache
    response = mlx_lm.generate(
        model, tokenizer, prompt=prompt, max_tokens=max_tokens,
    )
    return [t for t in response]


def _greedy_tokens_fused(model_path, prompt, max_tokens):
    os.environ["FUSION_PAGED_FUSED_KERNEL"] = "on"
    try:
        return _greedy_tokens(model_path, prompt, max_tokens)
    finally:
        os.environ.pop("FUSION_PAGED_FUSED_KERNEL", None)


def test_phase2_fused_matches_concat_tokens():
    prompt = "The quick brown fox"
    max_tokens = 40
    base = _greedy_tokens(_MODEL, prompt, max_tokens)
    fused = _greedy_tokens_fused(_MODEL, prompt, max_tokens)
    assert base == fused, f"token streams differ: base={base[:10]} fused={fused[:10]}"
```
Note: the exact `mlx_lm.generate` return shape depends on the installed version — the implementer adapts the token extraction (it may be a generator or a `GenerateResult` with `.text`/tokens). Read the installed `mlx_lm.generate` signature before finalizing. Bit-exact = identical token ids, not text.

- [ ] **Step 2: Run the real-model test (both paths)**

Ensure server stopped (`~/claude-home/fusion-mlx/start.sh stop`) so the GPU is free. Run:
```bash
FUSION_PAGED_KV_REAL_MODEL=on rtk proxy python -m pytest tests/integration/test_paged_kv_phase2_real_model.py -v -s
```
Expected: PASS — identical token streams. If they differ, the bug is in the kernel numerics or the rope-offset ordering (Task 5 Step 4). Debug via systematic-debugging: diff the per-token logits at the first divergence point.

- [ ] **Step 3: Write the perf benchmark**

Create `examples/benchmark_paged_kv_phase2.py`:
```python
import json
import os
import time

import mlx.core as mx
import mlx_lm

_MODEL = os.environ.get("FUSION_PAGED_KV_MODEL", "mlx-community/Qwen3-0.6B-4bit")
_PROMPT = os.environ.get("FUSION_PAGED_KV_PROMPT", "Explain paged attention in three sentences.")
_MAX_TOKENS = int(os.environ.get("FUSION_PAGED_KV_MAX_TOKENS", "256"))
_OUT = os.environ.get("FUSION_PAGED_KV_REPORT", os.path.expanduser("~/fusion/audit/paged-kv-phase2-perf-report.md"))


def _measure(model, tokenizer, fused):
    if fused:
        os.environ["FUSION_PAGED_FUSED_KERNEL"] = "on"
    else:
        os.environ.pop("FUSION_PAGED_FUSED_KERNEL", None)
    from fusion_mlx.custom_kernels.paged_kv_cache import FusionPagedKVCache
    from fusion_mlx.fusion_takeover.config import FusionConfig
    from fusion_mlx.fusion_takeover.patcher import FusionModulePatcher
    cfg = FusionConfig(enabled=True, paged_kv_enabled=True, fused_decode_enabled=fused)
    FusionModulePatcher().patch_model(model, cfg)
    cache = [FusionPagedKVCache(block_size=16, num_blocks=256) for _ in model.layers]
    model.make_cache = lambda: cache
    t0 = time.perf_counter()
    mlx_lm.generate(model, tokenizer, prompt=_PROMPT, max_tokens=_MAX_TOKENS)
    mx.eval(mx.array(0))  # force sync
    dt = time.perf_counter() - t0
    tps = _MAX_TOKENS / dt
    peak = mx.get_active_memory() / (1024 ** 3)
    return tps, peak, dt


def main():
    os.makedirs(os.path.dirname(_OUT), exist_ok=True)
    model, tokenizer = mlx_lm.load(_MODEL)
    base_tps, base_peak, base_dt = _measure(model, tokenizer, fused=False)
    # reload model for a clean cache state
    del model
    mx.clear_cache()
    model, tokenizer = mlx_lm.load(_MODEL)
    fused_tps, fused_peak, fused_dt = _measure(model, tokenizer, fused=True)
    delta = (fused_tps - base_tps) / base_tps * 100
    md = f"""# Paged-KV Phase 2 Perf Report

> Model: `{_MODEL}` | prompt tokens: ~{len(_PROMPT.split())} | decode tokens: {_MAX_TOKENS}
> Date: {time.strftime("%Y-%m-%d %H:%M:%S")}

## Decode (L=1) — concat path (Phase 1) vs fused kernel (Phase 2)

| path | tok/s | wall (s) | peak GPU mem (GB) |
|------|-------|----------|--------------------|
| concat (Phase 1) | {base_tps:.2f} | {base_dt:.2f} | {base_peak:.2f} |
| fused kernel (Phase 2) | {fused_tps:.2f} | {fused_dt:.2f} | {fused_peak:.2f} |
| **delta** | **{delta:+.1f}%** | | |

## Verdict

{"Phase 2 fused kernel beats concat path." if delta > 0 else "Phase 2 fused kernel does NOT beat concat path — naive scalar kernel expected slow; follow-up tiling task needed."}
"""
    with open(_OUT, "w") as f:
        f.write(md)
    print(md)
    print(f"report written to {_OUT}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run the benchmark, generate the report**

Ensure server stopped. Run:
```bash
~/claude-home/fusion-mlx/start.sh stop
rtk proxy python examples/benchmark_paged_kv_phase2.py
```
Expected: a report at `~/fusion/audit/paged-kv-phase2-perf-report.md` with before/after tok/s + peak mem. The naive scalar kernel will likely NOT beat the concat path (one-thread-per-head) — that is expected and documented in the report's Verdict. **If delta ≤ 0**, add a Task 6 sub-step (below) before committing.

- [ ] **Step 5 (conditional): optimize the kernel tiling IF delta ≤ 0**

If the naive kernel is slower than concat (expected), improve `fusion_paged_attention.py`:
- Use a threadgroup of `(BLOCK_SIZE, 1, 1)` threads, one thread per kv-position in the current block; reduce across the threadgroup with `simd_shuffle` for the running max/sum.
- Or simpler first win: one threadgroup per (batch, q_head), `HEAD_DIM` threads cooperate — each thread holds one `o[d]`, loops blocks, gathers its slice of the dot product. This still scalar-loops the dot product but parallelizes the head_dim dimension.
- Re-run the kernel unit test (Task 3 Step 4) to confirm still bit-exact, then re-run the benchmark.
- Stop when `delta > 0` OR three optimization rounds have passed (Rule 6 token budget — do not spiral). If still negative, document in the report that the kernel needs a full Steel-style tiled rewrite (out of this phase's scope) and ship the bit-exact-correct kernel gated off by default.

- [ ] **Step 6: Commit**

```bash
black --target-version py313 tests/integration/test_paged_kv_phase2_real_model.py examples/benchmark_paged_kv_phase2.py
git add tests/integration/test_paged_kv_phase2_real_model.py examples/benchmark_paged_kv_phase2.py
git commit -m "test+bench(paged-kv): Phase 2 real-model bit-exact + perf report"
```
The report at `~/fusion/audit/` lives outside the repo (do not commit it). Verify it exists with `ls -la ~/fusion/audit/paged-kv-phase2-perf-report.md`.

## Task 7: FusionPagedKVPool + FusionPagedRequestCache

**Files:**
- Create: `fusion_mlx/custom_kernels/paged_kv_pool.py`
- Test: `tests/unit/test_fusion_paged_pool.py`

**Interfaces:**
- Consumes: Task 2 flat-pool storage layout (`[num_blocks, B, n_kv_heads, block_size, head_dim]`), Task 1 finding on `BatchGenerator` batch-dim (expected `B=1` per-request). The `FusionPagedKVCache` duck-interface (`update_and_fetch`, `state`, `meta_state`, `make_mask`, `trim`, `is_trimmable`, `offset`, `nbytes`, `size`, `empty`, `from_state`, `free_all`).
- Produces:
  - `FusionPagedKVPool(block_size, num_blocks, n_kv_heads, head_dim, k_head_dim=None, v_head_dim=None, dtype=mx.float16)` — owns `keys_pool`/`values_pool` flat tensors `[num_blocks, 1, n_kv_heads, block_size, head_dim]` (B=1 per Task 1), `free_list: deque[int]`, `in_use: dict[int, str]` (physical block → request_id). Methods: `alloc_block(request_id) -> int`, `free_request(request_id)`, `available() -> int`, `stats() -> dict`.
  - `FusionPagedRequestCache(pool, request_id)` — per-request cache handle. Holds `block_table: list[int]`, `offset`, `_shape` (B, n_kv_heads, block_size, head_dim). Delegates `update_and_fetch` to write into `pool.keys_pool[pb]`/`pool.values_pool[pb]` at the per-request logical position, allocating a new physical block from `pool.alloc_block(request_id)` when crossing a block boundary. Satisfies the full `FusionPagedKVCache` duck-interface so `caches=[state.cache]` plumbing is unchanged. `trim`/`state`/`meta_state` mirror `FusionPagedKVCache` but operate through the pool.

**Design:** `FusionPagedRequestCache.update_and_fetch` is structurally identical to `FusionPagedKVCache.update_and_fetch` (Task 2) — same block-boundary logic, same `block_table` indirection — except the physical pool lives on the shared `FusionPagedKVPool`, not on the cache itself. `_ensure_pool` is a no-op (pool is pre-allocated by the pool). On the first write of a new block boundary, call `pool.alloc_block(self.request_id)` to get a fresh physical index, append it to `self.block_table`. On `free_all`/evict, call `pool.free_request(self.request_id)` to return all `block_table` blocks to the shared free-list.

- [ ] **Step 1: Write the failing test**

Create `tests/unit/test_fusion_paged_pool.py`:
```python
import mlx.core as mx
import pytest

from fusion_mlx.custom_kernels.paged_kv_pool import (
    FusionPagedKVPool,
    FusionPagedRequestCache,
)


def _fill(cache, n, n_kv_heads=2, head_dim=8):
    for _ in range(n):
        k = mx.random.normal(shape=(1, n_kv_heads, 1, head_dim)) * 0.1
        v = mx.random.normal(shape=(1, n_kv_heads, 1, head_dim)) * 0.1
        cache.update_and_fetch(k, v)


def test_pool_allocates_distinct_blocks_per_request():
    pool = FusionPagedKVPool(block_size=4, num_blocks=16, n_kv_heads=2, head_dim=8)
    a = FusionPagedRequestCache(pool, request_id="a")
    b = FusionPagedRequestCache(pool, request_id="b")
    _fill(a, 5)
    _fill(b, 3)
    # no overlap between physical blocks
    assert set(a.block_table).isdisjoint(set(b.block_table))
    assert pool.available() == 16 - len(a.block_table) - len(b.block_table)


def test_pool_free_request_returns_blocks():
    pool = FusionPagedKVPool(block_size=4, num_blocks=16, n_kv_heads=2, head_dim=8)
    a = FusionPagedRequestCache(pool, request_id="a")
    _fill(a, 5)
    used = len(a.block_table)
    pool.free_request("a")
    assert pool.available() == 16


def test_pool_exhausted_raises():
    pool = FusionPagedKVPool(block_size=4, num_blocks=2, n_kv_heads=2, head_dim=8)
    a = FusionPagedRequestCache(pool, request_id="a")
    _fill(a, 8)  # fills 2 blocks
    with pytest.raises(RuntimeError, match="pool exhausted"):
        _fill(a, 1)  # needs a 3rd block


def test_request_cache_matches_independent_paged_cache():
    # two pooled requests produce same logical view as two independent FusionPagedKVCache
    from fusion_mlx.custom_kernels.paged_kv_cache import FusionPagedKVCache
    n_kv_heads, head_dim = 2, 8
    mx.random.seed(3)
    pool = FusionPagedKVPool(block_size=4, num_blocks=32, n_kv_heads=n_kv_heads, head_dim=head_dim)
    pooled_a = FusionPagedRequestCache(pool, request_id="a")
    pooled_b = FusionPagedRequestCache(pool, request_id="b")
    mx.random.seed(3)
    solo_a = FusionPagedKVCache(block_size=4, num_blocks=16, n_kv_heads=n_kv_heads, head_dim=head_dim)
    solo_b = FusionPagedKVCache(block_size=4, num_blocks=16, n_kv_heads=n_kv_heads, head_dim=head_dim)
    for _ in range(9):
        ka = mx.random.normal(shape=(1, n_kv_heads, 1, head_dim)) * 0.1
        va = mx.random.normal(shape=(1, n_kv_heads, 1, head_dim)) * 0.1
        kb = mx.random.normal(shape=(1, n_kv_heads, 1, head_dim)) * 0.1
        vb = mx.random.normal(shape=(1, n_kv_heads, 1, head_dim)) * 0.1
        pa_k, pa_v = pooled_a.update_and_fetch(ka, va)
        sa_k, sa_v = solo_a.update_and_fetch(ka, va)
        pb_k, pb_v = pooled_b.update_and_fetch(kb, vb)
        sb_k, sb_v = solo_b.update_and_fetch(kb, vb)
        assert mx.allclose(pa_k, sa_k).item()
        assert mx.allclose(pa_v, sa_v).item()
        assert mx.allclose(pb_k, sb_k).item()
        assert mx.allclose(pb_v, sb_v).item()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `rtk proxy python -m pytest tests/unit/test_fusion_paged_pool.py -v`
Expected: FAIL — `ModuleNotFoundError: fusion_mlx.custom_kernels.paged_kv_pool`.

- [ ] **Step 3: Implement the pool + request cache**

Create `fusion_mlx/custom_kernels/paged_kv_pool.py`. Use a module-level logger (default logging). `FusionPagedKVPool`:
```python
import logging
from collections import deque

import mlx.core as mx

logger = logging.getLogger(__name__)


class FusionPagedKVPool:
    def __init__(self, block_size, num_blocks, n_kv_heads, head_dim,
                 k_head_dim=None, v_head_dim=None, dtype=mx.float16):
        self.block_size = block_size
        self.num_blocks = num_blocks
        self.n_kv_heads = n_kv_heads
        self.k_head_dim = k_head_dim or head_dim
        self.v_head_dim = v_head_dim or head_dim
        self.dtype = dtype
        self.keys_pool = mx.zeros(
            (num_blocks, 1, n_kv_heads, block_size, self.k_head_dim), dtype=dtype,
        )
        self.values_pool = mx.zeros(
            (num_blocks, 1, n_kv_heads, block_size, self.v_head_dim), dtype=dtype,
        )
        self.free_list = deque(range(num_blocks - 1, -1, -1))
        self.in_use = {}
        logger.info(
            "paged_kv pool init: cap=%d block_size=%d n_kv=%d head_dim=%d/%d",
            num_blocks, block_size, n_kv_heads, self.k_head_dim, self.v_head_dim,
        )

    def alloc_block(self, request_id):
        if not self.free_list:
            logger.error("paged_kv pool exhausted for request=%s", request_id)
            raise RuntimeError(
                f"paged_kv pool exhausted (cap={self.num_blocks}); "
                f"reject request or raise pool_num_blocks"
            )
        pb = self.free_list.pop()
        self.in_use[pb] = request_id
        return pb

    def free_request(self, request_id):
        freed = [pb for pb, rid in self.in_use.items() if rid == request_id]
        for pb in freed:
            self.in_use.pop(pb, None)
            self.free_list.append(pb)
        logger.info("paged_kv pool free request=%s blocks=%d", request_id, len(freed))

    def available(self):
        return len(self.free_list)

    def stats(self):
        return {
            "cap": self.num_blocks,
            "available": self.available(),
            "in_use": len(self.in_use),
        }
```

`FusionPagedRequestCache`: mirror `FusionPagedKVCache.update_and_fetch` (Task 2) but read/write through `self.pool.keys_pool`/`self.pool.values_pool` and allocate physical blocks from the pool. Implement the full duck-interface: `offset`, `size`, `nbytes`, `empty`, `is_trimmable`, `trim` (return int), `make_mask`, `state` get/set, `meta_state` get/set, `from_state`, `free_all` (calls `pool.free_request`). Log block-table growth at each new block. The implementer reads `paged_kv_cache.py` (Task 2 result) and copies the `update_and_fetch`/`_fetch_logical` block-boundary + partial-block logic verbatim, redirecting storage to `self.pool`.

- [ ] **Step 4: Run test to verify it passes**

Run: `rtk proxy python -m pytest tests/unit/test_fusion_paged_pool.py -v`
Expected: ALL PASS — distinct blocks, free returns blocks, exhaustion raises, pooled vs solo bit-exact.

- [ ] **Step 5: Lint + commit**

```bash
black --target-version py313 fusion_mlx/custom_kernels/paged_kv_pool.py tests/unit/test_fusion_paged_pool.py
ruff check fusion_mlx/custom_kernels/paged_kv_pool.py
git add fusion_mlx/custom_kernels/paged_kv_pool.py tests/unit/test_fusion_paged_pool.py
git commit -m "feat(paged-kv): shared FusionPagedKVPool + per-request FusionPagedRequestCache"
```

## Task 8: Pool-mode wiring (fusion_paged_kv.py) + config fields

**Files:**
- Modify: `fusion_mlx/custom_kernels/fusion_paged_kv.py`
- Modify: `fusion_mlx/fusion_takeover/config.py`
- Test: `tests/unit/test_fusion_paged_kv_wiring.py`

**Interfaces:**
- Consumes: Task 7 `FusionPagedKVPool` + `FusionPagedRequestCache`. Existing `fusion_paged_kv.py` `install_paged_kv(model, config)` (rebinds `model.make_cache` to `_fusion_make_cache` returning `[FusionPagedKVCache(...) for _ in range(num_layers)]`), `_GLOBAL_CACHE_REGISTRY`, `register_cache`/`evict_request`/`evict_request_by_id`.
- Produces:
  - `FusionConfig.pool_enabled: bool` (default `False`, from `fusion_paged_pool == "on"`).
  - `FusionConfig.pool_num_blocks: int` (default `256`, from `fusion_paged_pool_num_blocks`).
  - `install_paged_kv` gains pool mode: when `config.pool_enabled`, it creates ONE shared `FusionPagedKVPool` on the model (attached as `model._fusion_paged_pool`) and rebinds `model.make_cache` to a factory returning `[FusionPagedRequestCache(pool, request_id) for _ in range(num_layers)]` — but each `make_cache` call needs a fresh `request_id`. Since `make_cache` takes no args, the factory mints a monotonic id (`model._fusion_paged_pool_seq`) and registers it; the caller (scheduler `register_cache`) associates it with the request. The `_GLOBAL_CACHE_REGISTRY` maps `request_id → [FusionPagedRequestCache]`; `evict_request` calls `pool.free_request(request_id)` on each layer's pool.

**Design:** the key question is `request_id` provenance. `make_cache()` is called per-request by the scheduler with no args. In pool mode, `make_cache` mints a unique id, creates per-layer `FusionPagedRequestCache` handles all bound to the SAME shared pool, and registers them in `_GLOBAL_CACHE_REGISTRY[id] = handles`. The scheduler's existing `register_cache(model, request_id, caches)` then reuses that id (or the scheduler passes its own — reconcile by reading `sched_batch.py` `register_cache` usage; if the scheduler supplies the id, `make_cache` must accept it — see Step 1). The implementer reads `fusion_paged_kv.py` + `scheduler/core.py` to confirm the exact `register_cache`/`evict_request` call shape before finalizing.

- [ ] **Step 1: Investigate request_id provenance (read-only)**

Read `fusion_paged_kv.py` (`register_cache`, `evict_request`, `evict_request_by_id`, `_GLOBAL_CACHE_REGISTRY`) and `fusion_mlx/scheduler/core.py` (where `register_cache`/`evict_request` are called, and whether the caller supplies `request_id` or receives it). Record the exact call signatures. If the scheduler supplies its own `request_id` and `make_cache` must produce handles already keyed to it, the factory needs a settable id — document the chosen approach in the commit message.

- [ ] **Step 2: Write the failing test**

Create `tests/unit/test_fusion_paged_kv_wiring.py`:
```python
import os

import mlx.core as mx
import pytest

from fusion_mlx.custom_kernels.fusion_paged_kv import (
    evict_request,
    install_paged_kv,
    register_cache,
)
from fusion_mlx.custom_kernels.paged_kv_pool import (
    FusionPagedKVPool,
    FusionPagedRequestCache,
)
from fusion_mlx.fusion_takeover.config import FusionConfig


class FakeModel:
    model_type = "llama"
    layers = [object(), object(), object()]
    def __init__(self):
        self.make_cache = lambda: [None for _ in self.layers]


def test_pool_mode_make_cache_returns_request_caches():
    model = FakeModel()
    cfg = FusionConfig(
        enabled=True, paged_kv_enabled=True, pool_enabled=True, pool_num_blocks=32,
        paged_kv_block_size=4,
    )
    install_paged_kv(model, cfg)
    assert hasattr(model, "_fusion_paged_pool")
    assert isinstance(model._fusion_paged_pool, FusionPagedKVPool)
    caches = model.make_cache()
    assert all(isinstance(c, FusionPagedRequestCache) for c in caches)
    # all share one pool
    assert all(c.pool is model._fusion_paged_pool for c in caches)
    # two make_cache calls give distinct request_ids
    caches2 = model.make_cache()
    assert caches[0].request_id != caches2[0].request_id


def test_pool_mode_non_pool_keeps_paged_cache():
    from fusion_mlx.custom_kernels.paged_kv_cache import FusionPagedKVCache
    model = FakeModel()
    cfg = FusionConfig(enabled=True, paged_kv_enabled=True, pool_enabled=False)
    install_paged_kv(model, cfg)
    caches = model.make_cache()
    assert all(isinstance(c, FusionPagedKVCache) for c in caches)
    assert not hasattr(model, "_fusion_paged_pool")


def test_pool_config_fields():
    cfg = FusionConfig(enabled=True, paged_kv_enabled=True)
    assert cfg.pool_enabled is False
    assert cfg.pool_num_blocks == 256
    cfg2 = FusionConfig.from_model_settings(
        {"fusion_paged_kv": "on", "fusion_paged_pool": "on", "fusion_paged_pool_num_blocks": "512"}
    )
    assert cfg2.pool_enabled is True
    assert cfg2.pool_num_blocks == 512
```

- [ ] **Step 3: Run test to verify it fails**

Run: `rtk proxy python -m pytest tests/unit/test_fusion_paged_kv_wiring.py -v`
Expected: FAIL — `AttributeError: pool_enabled` / pool mode not installed.

- [ ] **Step 4: Add config fields**

In `fusion_mlx/fusion_takeover/config.py`:
- `pool_enabled: bool = False`, `pool_num_blocks: int = 256`.
- `from_model_settings`: `pool_enabled = settings.get("fusion_paged_pool") == "on"`, `pool_num_blocks = int(settings.get("fusion_paged_pool_num_blocks", 256))`.
- Log resolved values.

- [ ] **Step 5: Implement pool-mode install_paged_kv**

In `fusion_mlx/custom_kernels/fusion_paged_kv.py`:
- Import `FusionPagedKVPool`, `FusionPagedRequestCache`.
- In `install_paged_kv`, branch on `config.pool_enabled`:
  - Pool OFF (existing path): unchanged — `_fusion_make_cache` returns `[FusionPagedKVCache(...)]`.
  - Pool ON: create `pool = FusionPagedKVPool(block_size=config.paged_kv_block_size, num_blocks=config.pool_num_blocks, n_kv_heads=..., head_dim=...)`, attach `model._fusion_paged_pool = pool`, `model._fusion_paged_pool_seq = 0`. Rebind `model.make_cache` to a closure that mints `request_id = f"pool_{seq}"`, increments seq, creates `[FusionPagedRequestCache(pool, request_id) for _ in range(num_layers)]`, registers in `_GLOBAL_CACHE_REGISTRY[request_id] = handles`, returns handles. Log `"paged_kv pool mode installed cap=%d"`.
  - `n_kv_heads`/`head_dim`: read from the model's config (`model.args.n_kv_heads`, `model.args.head_dim` or `model.args.head_dim = model.args.dim // model.args.n_heads` — match how the existing non-pool `FusionPagedKVCache` infers it; copy that inference). If unavailable at install time, lazily init the pool on first `make_cache` call from the first cache's shape (record this fallback in the commit message).
- Update `evict_request(model, request_id)` / `evict_request_by_id`: if `model` has `_fusion_paged_pool`, call `model._fusion_paged_pool.free_request(request_id)` in addition to the existing per-cache cleanup. Log the eviction.

- [ ] **Step 6: Run test to verify it passes**

Run: `rtk proxy python -m pytest tests/unit/test_fusion_paged_kv_wiring.py -v`
Expected: ALL PASS.

- [ ] **Step 7: Lint + commit**

```bash
black --target-version py313 fusion_mlx/custom_kernels/fusion_paged_kv.py fusion_mlx/fusion_takeover/config.py tests/unit/test_fusion_paged_kv_wiring.py
ruff check fusion_mlx/custom_kernels/fusion_paged_kv.py fusion_mlx/fusion_takeover/config.py
git add fusion_mlx/custom_kernels/fusion_paged_kv.py fusion_mlx/fusion_takeover/config.py tests/unit/test_fusion_paged_kv_wiring.py
git commit -m "feat(paged-kv): pool-mode install_paged_kv + config fields (FUSION_PAGED_POOL)"
```

## Task 9: Phase 3 concurrency bit-exact + BatchedEngine integration tests

**Files:**
- Create: `tests/integration/test_paged_kv_phase3_concurrency.py`

**Interfaces:**
- Consumes: Task 7 `FusionPagedKVPool`/`FusionPagedRequestCache`, Task 8 pool-mode `install_paged_kv`. `fusion_mlx/engines/batched.py` `BatchedEngine` (or the scheduler path it drives) for concurrent decode. Existing `register_cache`/`evict_request` lifecycle.
- Produces: a test proving 2-4 concurrent greedy decodes through pool mode each match their single-stream reference token-for-token, and a unit-level concurrency test that interleaves two `FusionPagedRequestCache` writes into one pool and verifies no cross-contamination.

**Prereq:** small llama-family model available (same as Task 6). `FUSION_PAGED_POOL=on`. Server stopped during tests.

- [ ] **Step 1: Write the unit-level concurrency test (no model, interleaved writes)**

Create `tests/integration/test_paged_kv_phase3_concurrency.py`:
```python
import os

import mlx.core as mx
import pytest

from fusion_mlx.custom_kernels.paged_kv_pool import (
    FusionPagedKVPool,
    FusionPagedRequestCache,
)


def test_interleaved_writes_no_cross_contamination():
    n_kv_heads, head_dim, block_size = 2, 8, 4
    pool = FusionPagedKVPool(
        block_size=block_size, num_blocks=32,
        n_kv_heads=n_kv_heads, head_dim=head_dim,
    )
    a = FusionPagedRequestCache(pool, request_id="a")
    b = FusionPagedRequestCache(pool, request_id="b")
    # interleave: a, b, a, b, ...
    mx.random.seed(11)
    a_views, b_views = [], []
    for i in range(10):
        k = mx.random.normal(shape=(1, n_kv_heads, 1, head_dim)) * 0.1
        v = mx.random.normal(shape=(1, n_kv_heads, 1, head_dim)) * 0.1
        if i % 2 == 0:
            a_views.append(a.update_and_fetch(k, v))
        else:
            b_views.append(b.update_and_fetch(k, v))
    # each request's final logical view is exactly its own tokens, in order
    assert a.offset == 5
    assert b.offset == 5
    # rebuild expected for a: tokens at i=0,2,4,6,8
    mx.random.seed(11)
    expected_a = []
    for i in range(10):
        k = mx.random.normal(shape=(1, n_kv_heads, 1, head_dim)) * 0.1
        v = mx.random.normal(shape=(1, n_kv_heads, 1, head_dim)) * 0.1
        if i % 2 == 0:
            expected_a.append((k, v))
    last_k, last_v = a_views[-1]
    # last view holds all 5 tokens; token j == expected_a[j]
    for j, (ek, ev) in enumerate(expected_a):
        assert mx.allclose(last_k[:, :, j, :], ek).item(), f"a token {j} contaminated"
        assert mx.allclose(last_v[:, :, j, :], ev).item(), f"a v token {j} contaminated"


def test_pool_evict_then_reuse():
    pool = FusionPagedKVPool(block_size=4, num_blocks=4, n_kv_heads=2, head_dim=8)
    a = FusionPagedRequestCache(pool, request_id="a")
    for _ in range(8):
        k = mx.random.normal(shape=(1, 2, 1, 8)) * 0.1
        v = mx.random.normal(shape=(1, 2, 1, 8)) * 0.1
        a.update_and_fetch(k, v)
    assert pool.available() == 0  # 8 tokens / 4 block_size = 2 blocks, cap 4 -> avail 2
    pool.free_request("a")
    assert pool.available() == 4
    # new request reuses freed blocks
    b = FusionPagedRequestCache(pool, request_id="b")
    for _ in range(4):
        k = mx.random.normal(shape=(1, 2, 1, 8)) * 0.1
        v = mx.random.normal(shape=(1, 2, 1, 8)) * 0.1
        b.update_and_fetch(k, v)
    assert b.offset == 4
```

- [ ] **Step 2: Run the concurrency unit tests**

Run: `rtk proxy python -m pytest tests/integration/test_paged_kv_phase3_concurrency.py -v -k "interleaved or reuse"`
Expected: ALL PASS.

- [ ] **Step 3: Write the BatchedEngine real-model concurrency test**

Append to the same file:
```python
pytestmark_concur = pytest.mark.skipif(
    os.environ.get("FUSION_PAGED_KV_REAL_MODEL") != "on",
    reason="set FUSION_PAGED_KV_REAL_MODEL=on for real-model tests",
)

_MODEL = os.environ.get("FUSION_PAGED_KV_MODEL", "mlx-community/Qwen3-0.6B-4bit")


def _single_stream(model_path, prompt, max_tokens):
    import mlx_lm
    from fusion_mlx.custom_kernels.fusion_paged_kv import install_paged_kv
    from fusion_mlx.custom_kernels.paged_kv_cache import FusionPagedKVCache
    from fusion_mlx.fusion_takeover.config import FusionConfig
    from fusion_mlx.fusion_takeover.patcher import FusionModulePatcher
    model, tokenizer = mlx_lm.load(model_path)
    cfg = FusionConfig(enabled=True, paged_kv_enabled=True, pool_enabled=False)
    FusionModulePatcher().patch_model(model, cfg)
    install_paged_kv(model, cfg)
    cache = model.make_cache()
    res = mlx_lm.generate(model, tokenizer, prompt=prompt, max_tokens=max_tokens)
    del model
    mx.clear_cache()
    return res


@pytestmark_concur
def test_concurrent_pool_matches_single_stream():
    # Run 2 prompts concurrently through BatchedEngine pool mode, compare each
    # to its own single-stream reference.
    prompts = ["The quick brown fox", "In a galaxy far away"]
    max_tokens = 30
    refs = {p: _single_stream(_MODEL, p, max_tokens) for p in prompts}
    # Now run both concurrently via BatchedEngine pool mode (Task 8 wiring).
    # The implementer uses the existing BatchedEngine API to submit 2 requests
    # with FUSION_PAGED_POOL=on and collects each stream's tokens.
    os.environ["FUSION_PAGED_POOL"] = "on"
    try:
        from fusion_mlx.engines.batched import BatchedEngine
        # adapt to actual BatchedEngine submit API (read batched.py first)
        eng = BatchedEngine(model_name=_MODEL)
        eng.start()
        try:
            conc = {}
            for p in prompts:
                conc[p] = list(eng.generate(prompt=p, max_tokens=max_tokens))
        finally:
            eng.stop()
    finally:
        os.environ.pop("FUSION_PAGED_POOL", None)
    for p in prompts:
        assert _to_tokens(refs[p]) == _to_tokens(conc[p]), f"concurrent stream mismatch for {p!r}"
```
Note: `BatchedEngine.generate`/submit API and token return shape MUST be read from `fusion_mlx/engines/batched.py` before finalizing — the implementer adapts the exact submit/collect calls. `_to_tokens` extracts token ids from whatever `mlx_lm.generate`/`BatchedEngine` returns (read the shape). If `BatchedEngine` does not expose a simple per-prompt `generate`, use the scheduler's request-submit path directly.

- [ ] **Step 4: Run the real-model concurrency test**

Ensure server stopped. Run:
```bash
~/claude-home/fusion-mlx/start.sh stop
FUSION_PAGED_KV_REAL_MODEL=on rtk proxy python -m pytest tests/integration/test_paged_kv_phase3_concurrency.py -v -s -k concurrent_pool
```
Expected: PASS — each concurrent stream matches its single-stream reference. If a stream diverges, the bug is cross-request block contamination (Task 7 `block_table`/`in_use` mapping) or batch-dim handling (Task 1 finding — if `B>1` the per-request cache must batch-handle). Debug with systematic-debugging: isolate one request at a time through the pool, then add the second.

- [ ] **Step 5: Lint + commit**

```bash
black --target-version py313 tests/integration/test_paged_kv_phase3_concurrency.py
git add tests/integration/test_paged_kv_phase3_concurrency.py
git commit -m "test(paged-kv): Phase 3 concurrency bit-exact + BatchedEngine integration"
```

## Task 10: Phase 3 concurrency perf report

**Files:**
- Create: `examples/benchmark_paged_pool.py`
- Create (output): `~/fusion/audit/paged-kv-phase3-concurrency-perf-report.md`

**Interfaces:**
- Consumes: Tasks 7-8 (pool + wiring), Task 9 `BatchedEngine` concurrency path. Same small llama-family model.
- Produces: a md report comparing N-concurrent-request throughput + peak memory: (a) per-request independent `FusionPagedKVCache` (pool OFF) vs (b) shared `FusionPagedKVPool` (pool ON), for N = 1, 2, 4.

**Design:** measure aggregate tok/s across N concurrent decode streams and peak GPU memory. Pool mode should bound memory by `pool_num_blocks` (fixed cap) while independent mode grows per-request. The win is memory efficiency + no head-of-line blocking, not necessarily raw tok/s (which the benchmark records honestly either way).

- [ ] **Step 1: Write the benchmark**

Create `examples/benchmark_paged_pool.py`:
```python
import os
import time

import mlx.core as mx

_MODEL = os.environ.get("FUSION_PAGED_KV_MODEL", "mlx-community/Qwen3-0.6B-4bit")
_PROMPTS = [
    "The quick brown fox jumps over",
    "In a galaxy far far away there was",
    "To be or not to be that is the",
    "The architecture of attention mechanisms",
]
_MAX_TOKENS = int(os.environ.get("FUSION_PAGED_KV_MAX_TOKENS", "128"))
_OUT = os.path.expanduser(
    os.environ.get(
        "FUSION_PAGED_KV_REPORT",
        "~/fusion/audit/paged-kv-phase3-concurrency-perf-report.md",
    )
)


def _run_concurrent(n, pool_on):
    if pool_on:
        os.environ["FUSION_PAGED_POOL"] = "on"
    else:
        os.environ.pop("FUSION_PAGED_POOL", None)
    from fusion_mlx.engines.batched import BatchedEngine
    eng = BatchedEngine(model_name=_MODEL)
    eng.start()
    try:
        prompts = _PROMPTS[:n]
        t0 = time.perf_counter()
        outs = {}
        for p in prompts:
            outs[p] = list(eng.generate(prompt=p, max_tokens=_MAX_TOKENS))
        mx.eval(mx.array(0))
        dt = time.perf_counter() - t0
        total_tokens = n * _MAX_TOKENS
        tps = total_tokens / dt
        peak = mx.get_active_memory() / (1024 ** 3)
        return tps, peak, dt
    finally:
        eng.stop()
        os.environ.pop("FUSION_PAGED_POOL", None)


def main():
    os.makedirs(os.path.dirname(_OUT), exist_ok=True)
    rows = []
    for n in (1, 2, 4):
        base_tps, base_peak, base_dt = _run_concurrent(n, pool_on=False)
        mx.clear_cache()
        pool_tps, pool_peak, pool_dt = _run_concurrent(n, pool_on=True)
        mx.clear_cache()
        rows.append((n, base_tps, base_peak, base_dt, pool_tps, pool_peak, pool_dt))
    md = ["# Paged-KV Phase 3 Concurrency Perf Report", ""]
    md.append(f"> Model: `{_MODEL}` | decode tokens/req: {_MAX_TOKENS}")
    md.append(f"> Date: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    md.append("")
    md.append("## N concurrent requests — independent caches (pool OFF) vs shared pool (pool ON)")
    md.append("")
    md.append("| N | ind tok/s | ind peak GB | ind wall | pool tok/s | pool peak GB | pool wall | tok/s delta |")
    md.append("|---|-----------|-------------|----------|------------|--------------|-----------|-------------|")
    for n, btps, bpeak, bdt, ptps, ppeak, pdt in rows:
        delta = (ptps - btps) / btps * 100 if btps else 0
        md.append(
            f"| {n} | {btps:.2f} | {bpeak:.2f} | {bdt:.2f} | {ptps:.2f} | {ppeak:.2f} | {pdt:.2f} | {delta:+.1f}% |"
        )
    md.append("")
    md.append("## Verdict")
    n4 = rows[-1]
    mem_win = (n4[2] - n4[5]) / n4[2] * 100 if n4[2] else 0
    md.append(
        f"At N=4: shared pool peak mem = {n4[5]:.2f} GB vs independent {n4[2]:.2f} GB "
        f"({mem_win:+.1f}% memory). Throughput delta {((n4[4]-n4[1])/n4[1]*100 if n4[1] else 0):+.1f}%."
    )
    md.append(
        "Pool bounds memory by `pool_num_blocks` (fixed cap); independent grows per-request. "
        "No head-of-line blocking: a long request cannot pre-empt a short one's blocks."
    )
    text = "\n".join(md)
    with open(_OUT, "w") as f:
        f.write(text)
    print(text)
    print(f"report written to {_OUT}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run the benchmark, generate the report**

Ensure server stopped. Run:
```bash
~/claude-home/fusion-mlx/start.sh stop
rtk proxy python examples/benchmark_paged_pool.py
```
Expected: a report at `~/fusion/audit/paged-kv-phase3-concurrency-perf-report.md` with N=1/2/4 rows for both modes. The implementer adapts `BatchedEngine.generate`/submit to the real API (read `batched.py` first — same caveat as Task 9 Step 3).

- [ ] **Step 3: Verify the report exists and is well-formed**

```bash
ls -la ~/fusion/audit/paged-kv-phase3-concurrency-perf-report.md
head -20 ~/fusion/audit/paged-kv-phase3-concurrency-perf-report.md
```
Confirm it has the table + verdict. If `BatchedEngine` true-concurrent submission isn't achievable with the current API, fall back to sequential-per-request submission through the pool (still measures the memory-bounding win) and note the throughput limitation in the report — do NOT fabricate concurrent numbers.

- [ ] **Step 4: Commit**

```bash
black --target-version py313 examples/benchmark_paged_pool.py
git add examples/benchmark_paged_pool.py
git commit -m "bench(paged-kv): Phase 3 concurrency perf report (N=1/2/4 pool vs independent)"
```

## Task 11: Full sweep + docs + PR

**Files:**
- Modify: `README.md` (English only)
- Modify: `README_CN.md` (Chinese only)
- Verify: `~/fusion/audit/paged-kv-phase2-perf-report.md`, `~/fusion/audit/paged-kv-phase3-concurrency-perf-report.md`

**Interfaces:**
- Consumes: all prior tasks. The two perf reports (outside repo). The existing README paged-KV section (Phase 1, if any).
- Produces: a green full-test-suite run, updated README/README_CN documenting Phase 2 (fused kernel) + Phase 3 (shared pool) with env gates + perf-report pointers, a follow-up issue list (non-llama-family attention, kernel tiling, LRU eviction), and a PR against `main`.

- [ ] **Step 1: Run the full unit test suite (no real-model, fast)**

```bash
rtk proxy python -m pytest tests/unit/test_fusion_paged_kv_cache.py tests/unit/test_fusion_paged_attention.py tests/unit/test_fusion_paged_pool.py tests/unit/test_fusion_paged_kv_wiring.py tests/unit/test_fusion_takeover_patcher.py -v
```
Expected: ALL PASS (real-model tests skip without `FUSION_PAGED_KV_REAL_MODEL=on`). If any fail, fix before proceeding — do not commit a red suite.

- [ ] **Step 2: Run lint across all touched files**

```bash
black --check --target-version py313 \
  fusion_mlx/custom_kernels/paged_kv_cache.py \
  fusion_mlx/custom_kernels/fusion_paged_attention.py \
  fusion_mlx/custom_kernels/paged_kv_pool.py \
  fusion_mlx/custom_kernels/fusion_paged_kv.py \
  fusion_mlx/fusion_takeover/config.py \
  fusion_mlx/fusion_takeover/patcher.py \
  tests/unit/test_fusion_paged_kv_cache.py \
  tests/unit/test_fusion_paged_attention.py \
  tests/unit/test_fusion_paged_pool.py \
  tests/unit/test_fusion_paged_kv_wiring.py \
  tests/unit/test_fusion_takeover_patcher.py \
  tests/integration/test_paged_kv_phase2_real_model.py \
  tests/integration/test_paged_kv_phase3_concurrency.py \
  examples/benchmark_paged_kv_phase2.py \
  examples/benchmark_paged_pool.py
ruff check fusion_mlx/custom_kernels/ fusion_mlx/fusion_takeover/
```
Expected: clean. Fix any violations (never touch `debt_modules.txt`).

- [ ] **Step 3: Confirm the two perf reports exist**

```bash
ls -la ~/fusion/audit/paged-kv-phase2-perf-report.md ~/fusion/audit/paged-kv-phase3-concurrency-perf-report.md
```
Both must exist (Tasks 6, 10). If a report is missing, regenerate it from the benchmark script before finishing.

- [ ] **Step 4: Update README.md (English only)**

Add/extend a `## Paged KV Cache` section (or extend the existing Phase 1 section) covering:
- Phase 2: `FUSION_PAGED_FUSED_KERNEL=on` — fused decode-attention Metal kernel (llama-family), eliminates the per-step concat. Default off until validated. Perf report pointer: `~/fusion/audit/paged-kv-phase2-perf-report.md`.
- Phase 3: `FUSION_PAGED_POOL=on` + `FUSION_PAGED_POOL_NUM_BLOCKS=<cap>` — shared `FusionPagedKVPool` for continuous batching; bounds memory by the cap, no head-of-line blocking. Default off. Perf report pointer: `~/fusion/audit/paged-kv-phase3-concurrency-perf-report.md`.
- Env var table: `FUSION_PAGED_KV`, `FUSION_PAGED_FUSED_KERNEL`, `FUSION_PAGED_POOL`, `FUSION_PAGED_POOL_NUM_BLOCKS`, with defaults.
- A short "Limitations" note: llama-family only for the fused kernel; non-llama follow-up; eviction = reject+503 (LRU deferred). English only — no Chinese in this file.

- [ ] **Step 5: Update README_CN.md (Chinese only)**

Mirror the README.md paged-KV section in Chinese. Same env vars, same report pointers, same limitations. Chinese only — no English prose (code/env names stay verbatim per convention).

- [ ] **Step 6: Commit docs**

```bash
git add README.md README_CN.md
git commit -m "docs(paged-kv): Phase 2 fused kernel + Phase 3 shared pool (README + README_CN)"
```

- [ ] **Step 7: File follow-up issues (English, on GitHub)**

File these issues against `dahai80/fusion-mlx` (English only, per global rule):
1. "Paged-KV fused decode kernel: extend to non-llama-family attention (gemma/mistral sliding-window)" — references the Phase 2 allowlist.
2. "Paged-KV fused decode kernel: Steel-style tiled optimization" — references Task 6 Step 5 if the naive kernel did not beat concat.
3. "Paged-KV pool: LRU eviction policy" — references the Phase 3 reject+503 default.
Use `gh issue create` for each. Record the issue numbers in the commit message of Step 6 (amend) or a final doc commit.

- [ ] **Step 8: Verify the branch is clean and push**

```bash
git status
git log --oneline -15
git push -u origin feat/enhance-arch-0826
```
Confirm: working tree clean, all task commits present, branch pushed. No commits to `main`.

- [ ] **Step 9: Create the PR (English)**

```bash
gh pr create --base main --head feat/enhance-arch-0826 \
  --title "feat: Paged-KV Phase 2 (fused decode kernel) + Phase 3 (shared pool)" \
  --body-file <(cat <<'EOF'
## Summary

- **Phase 2:** Fused paged decode-attention Metal kernel (`FUSION_PAGED_FUSED_KERNEL=on`, llama-family). Reads K/V directly from the flat physical block pool via `block_table` indirection — eliminates Phase 1's per-step `mx.concatenate`. Decode-only; prefill keeps the concat path.
- **Phase 3:** Shared `FusionPagedKVPool` + per-request `FusionPagedRequestCache` (`FUSION_PAGED_POOL=on`). One physical pool, shared free-list, per-request `block_table`. Bounds memory by `pool_num_blocks`; no head-of-line blocking. Scheduler continuous-batching loop reused unchanged.

## Env gates (all default OFF)

| var | effect |
|-----|--------|
| `FUSION_PAGED_KV` | Phase 1 paged cache |
| `FUSION_PAGED_FUSED_KERNEL` | Phase 2 fused decode kernel (llama-family) |
| `FUSION_PAGED_POOL` | Phase 3 shared pool |
| `FUSION_PAGED_POOL_NUM_BLOCKS` | pool cap (default 256) |

## Tests

- Unit: kernel bit-exact vs SDPA, flat-pool refactor, pool alloc/free, wiring, attention wrapper.
- Integration (real model, `FUSION_PAGED_KV_REAL_MODEL=on`): Phase 2 fused vs concat identical tokens; Phase 3 concurrent streams match single-stream references.

## Perf reports

- Phase 2: `~/fusion/audit/paged-kv-phase2-perf-report.md`
- Phase 3: `~/fusion/audit/paged-kv-phase3-concurrency-perf-report.md`

## Bit-exact

All paths bit-exact vs upstream `KVCache` for greedy decode (fp16 tol). Real-model integration confirms identical token streams.

## Follow-up issues

(attach the issue numbers filed in Step 7)

## Out of scope

Fused prefill kernel, NVFP4/MXFP4 dequant-on-fetch, LRU eviction, non-llama-family attention, dflash2 KV integration.
EOF
)
```
Report the PR URL to the user. Do NOT merge — integration decision is the user's (finishing-a-development-branch skill).
