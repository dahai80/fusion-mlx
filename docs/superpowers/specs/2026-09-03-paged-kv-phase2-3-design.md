# Paged-KV Phase 2/3 Design — Fused Paged Attention + Continuous Batching Pool

> **Status:** DRAFT for user review. Not yet implemented.
> **Branch:** `feat/enhance-arch-0826` (Phase 0 shim + Phase 1 `FusionPagedKVCache` already landed).
> **Predecessor:** Phase 1 `FusionPagedKVCache` (commit `a84ebe9`) — slab-based lazy-growth, block_table indirection, raw unquantized KV, bit-exact. Single-stream decode overhead -15%/-37% (concat-per-step).

## Goal

Eliminate Phase 1's single-stream concat overhead and unlock multi-request concurrency by (Phase 2) fusing the paged-KV fetch into a single Metal decode-attention kernel that reads blocks directly via `block_table`, and (Phase 3) unifying per-request KV into a single physical block pool shared across concurrently-decoding requests.

## Architecture (summary)

- **Phase 2:** A `mx.fast.metal_kernel`-JIT paged decode-attention kernel. Replaces `cache.update_and_fetch` + `scaled_dot_product_attention` for the decode (L=1) path with one kernel that gathers K/V from physical blocks via `block_table` and computes GQA attention in-register — no materialized logical K/V view, no `mx.concatenate`. Prefill (L>1) keeps the existing concat path (concat cost amortized over many tokens).
- **Phase 3:** Lift `FusionPagedKVCache` from per-request to a shared `FusionPagedKVPool` — one slab pool, one free-list, per-request `block_table` arrays. The existing `Scheduler` continuous-batching loop already batches decode forwards; Phase 3 swaps per-request `make_prompt_cache` for pooled caches so N concurrent requests share one physical memory pool without head-of-line blocking.

## Tech Stack

- MLX `mx.fast.metal_kernel` JIT (no CMake/nanobind — path (b) from the scaffolding map).
- Steel attention tiling idioms from `steel_attention_block_token.h` (reference, not link dependency).
- Python: `FusionPagedKVCache` (extend), `FusionPagedKVPool` (new), `fusion_paged_kv.py` (wiring).

## Constraints (global)

- 4-space indent multiples, no docstrings, default logging on all new code.
- Lint: `black --check --target-version py313` + `ruff` clean. Never touch `debt_modules.txt`.
- NEVER `mx.clear_streams()` in tests (#630 stream pollution). `mx.clear_cache()` safe.
- Cache ops on `_model_load_executor` same thread (#KV-0 red line).
- Tests bit-exact vs upstream `KVCache` for greedy decode; real-model integration test required.
- Perf report → `~/fusion/audit/`, md format, before/after tok/s + peak mem.
- Only modify `/Users/dahai/claude-home/fusion-mlx`. Branch `feat/enhance-arch-0826` from `main`. No main commits.
- `source .venv/bin/activate` (worktree has no `.venv` → use `/Users/dahai/claude-home/fusion-mlx/.venv/bin/python`).

---

## Background — what exists (verified)

_(scaffolding facts from explore agent — grounds every design choice)_

1. **No native paged kernel.** `glm_moe_dsa` `exact_block_attention.metal` / `sparse_mla.metal` read KV **contiguously** (no `block_table`). `is_paged_kv_available()` only checks the GLM extension loads — no paged symbol. Reuse = idiom reference only, not code.

2. **Cache entry point** (mlx_lm `llama.py:89-92`):
   ```python
   queries = self.rope(queries, offset=cache.offset)
   keys = self.rope(keys, offset=cache.offset)
   keys, values = cache.update_and_fetch(keys, values)   # ← Phase 2 replaces fetch
   output = scaled_dot_product_attention(queries, keys, values, cache=cache, scale=self.scale, mask=mask)
   ```
   Decode (L=1): `keys`/`values` returned are the entire cached sequence `[B, n_kv_heads, offset+1, head_dim]`. The concat-per-step in Phase 1's `_fetch_logical` rebuilds this view every token. Fused kernel computes attention directly from physical blocks — the view is never materialized.

3. **KVCache interface** (duck-typed, `mlx_lm/models/cache.py`): `update_and_fetch`, `state` get/set, `meta_state` get/set, `make_mask`, `trim`→int, `is_trimmable`, `offset` (plain int), `nbytes`, `size`, `empty`, `from_state`. Generate path calls **only** `update_and_fetch` + `offset` + `make_mask`. No `fetch`/`concat` method on the cache.

4. **MLX JIT** (`mx.fast.metal_kernel`): flash_kda loads `.metal` source text, JITs on first use, invokes with `kernel(inputs=, template=, grid=, threadgroup=, output_shapes=, output_dtypes=, stream=)`. No build step. This is the Phase 2 path.

5. **Continuous batching exists** (`fusion_mlx/engines/batched.py` `BatchedEngine` → `fusion_mlx/scheduler/core.py` `Scheduler` → `sched_step.step()`). Batches **requests** (decode forwards across running requests). KV is **per-request** (`make_prompt_cache(self.model)` per request, `caches=[state.cache]`). `block_aware_cache` = paged SSD prefix cache (disk snapshots), NOT a unified in-memory pool. Phase 3 = swap per-request caches for pooled ones; the scheduler loop is reused unchanged.

6. **dflash2** = single-stream spec-decode, KV internal to `_dflash.stream_generate`. Out of scope for Phase 3 (orthogonal; own KV).

---

## Phase 2 — Fused Paged Decode-Attention Kernel

### Problem

Phase 1 `_fetch_logical` does `mx.concatenate(k_parts, axis=-2)` every `update_and_fetch` call to rebuild the logical K/V view from non-contiguous physical blocks. At decode (1 token/step) this concat is O(blocks) work per token with no useful output — it only exists to feed `scaled_dot_product_attention` a contiguous array. Measured -15% to -37% decode tok/s.

### Design

A single Metal kernel computes decode attention (L=1 query) directly from the paged physical layout:

```
inputs:
  q            [B, n_heads, 1, head_dim]      # RoPE already applied in Python (offset known)
  keys_slabs   [num_slabs, B, n_kv_heads, block_size, k_head_dim]   # or flattened slab list
  values_slabs [num_slabs, B, n_kv_heads, block_size, v_head_dim]
  block_table  [num_logical_blocks]   int32   # logical→physical (slab_idx*slab_size + in_slab, or split)
  num_kv       int32                            # total cached length (offset+1)
  scale        float
outputs:
  out          [B, n_heads, 1, v_head_dim]
```

Kernel per query head: loop over logical blocks in `block_table[0:ceil(num_kv/block_size)]`, gather each block's K/V slice (respecting the partial last block), accumulate `exp(q·k / scale)·v` running softmax (online softmax / FlashAttention-2 numerics) in registers, write `out`. GQA handled by query-head→kv-head mapping `kv_head = q_head // gqa_factor`.

**Why in-register, not materialized:** the whole point is the K/V view is never built. Each threadgroup handles one query head; iterates blocks; the partial last block is masked by `num_kv % block_size`.

### Python integration

`FusionPagedKVCache` gains a `fused_decode_attention(self, q, rope_keys, rope_values, scale, n_heads, n_kv_heads, head_dim)` method. The model attention path calls it instead of `update_and_fetch`+SDPA when:
- `self.offset > 0` (not the first token — prefill is L>1),
- the new token count is 1 (decode),
- Metal JIT is available,
- `FUSION_PAGED_FUSED_KERNEL != "off"` (env gate, default on when available).

`update_and_fetch` still does the block writes (append new K/V into the right physical block) AND returns the logical view for the **non-fused fallback** + prefill. The fused path reads from the already-written physical slabs. So the write stays in `update_and_fetch`; the **attention** moves to the kernel. Concretely the attention `__call__` becomes:

```python
keys, values = cache.update_and_fetch(keys, values)   # writes new block, returns view
if cache.fused_decode_available(num_new=1):
    out = cache.fused_decode_attention(queries, keys[..., -1:, :], values[..., -1:, :], scale, n_heads, n_kv_heads, head_dim)
else:
    out = scaled_dot_product_attention(queries, keys, values, cache=cache, scale=scale, mask=mask)
```

Hooking the model attention: Phase 0 shim's settings-driven takeover already rebinds `model.make_cache`. Phase 2 adds a **model-class patch** that wraps each layer's `Attention.__call__` to use the fused path when the cache is a `FusionPagedKVCache`. Patch is opt-in via `FUSION_PAGED_FUSED_KERNEL=on` (default off until validated, then on) — same gating convention as Phase 0/1.

### Correctness

- Bit-exact vs `scaled_dot_product_attention` for greedy decode — unit test `_NaiveRef` extended to compare fused-kernel output vs MLX SDPA on random K/V within tolerance (fp16: 1e-3 relative; the kernel uses the same online-softmax numerics MLX Steel uses).
- Real-model integration: qwen3-0.6b / llama-3.2-1b greedy decode produces identical token stream with and without the fused kernel (reuse the Phase 1 test harness).

### Failure modes

- Metal JIT unavailable / kernel compile fails → fall back to concat path (env gate already off by default; runtime try/except logs + disables).
- Slab list as separate tensors → `mx.fast.metal_kernel` inputs must be arrays, not a Python list. Resolution: store slabs as **one concatenated physical pool tensor** `[max_blocks, B, n_kv_heads, block_size, head_dim]` (pre-allocated, grown lazily by appending slabs into a growable array — or accept the cap cost). Decision deferred to implementation; spec picks **one flat pool tensor** for kernel simplicity, trading the lazy-growth memory win for kernel-input simplicity. (See Open Questions.)

---

## Phase 3 — Shared Paged KV Pool for Continuous Batching

### Problem

`Scheduler` already runs N concurrent decode requests in one batched forward, but each gets its own `make_prompt_cache` — N independent KV arrays, no sharing, head-of-line blocking on memory, no cross-request defragmentation. The whole point of paging (vLLM-style) is one physical pool, per-request `block_table`, on-demand allocation from a shared free-list.

### Design

`FusionPagedKVPool` — one physical block pool, shared free-list, per-request `block_table`:

```
pool keys   [num_blocks_cap, B_pool, n_kv_heads, block_size, k_head_dim]
pool values [num_blocks_cap, B_pool, n_kv_heads, block_size, v_head_dim]
free_list   deque[int]      # physical block indices
requests    {request_id: FusionPagedRequestCache}
```

`FusionPagedRequestCache` wraps a per-request `block_table: list[int]` + `offset` and delegates `update_and_fetch` to write into the shared pool at its allocated physical blocks (allocating from `free_list` as it crosses block boundaries). Satisfies the same KVCache duck-interface so the scheduler's `caches=[state.cache]` plumbing is unchanged.

Per-request lifecycle (already plumbed in `fusion_paged_kv.py`): `register_cache(model, request_id, caches)` on insert → `evict_request(model, request_id)` on finish/abort → blocks returned to `free_list`. The existing `_GLOBAL_CACHE_REGISTRY` becomes the pool registry.

### Scheduler integration

The scheduler already passes `caches=[state.cache]` per request into `BatchGenerator.insert`. Phase 3 changes **only** what `make_cache` returns: instead of `FusionPagedKVCache` (own pool), it returns `FusionPagedRequestCache(handle into the shared pool)`. The `install_paged_kv` wiring gains a pool mode: `FUSION_PAGED_POOL=on` (default off) → model gets one `FusionPagedKVPool` + a factory producing `FusionPagedRequestCache` handles bound to it.

**No scheduler code changes.** The batching loop, abort handling, prefill eviction are reused verbatim. This keeps the blast radius to the cache + wiring layer.

### Concurrency win

- N concurrent requests share `num_blocks_cap` physical blocks; a request only allocates blocks as it grows (lazy), returns them on finish. No head-of-line blocking — a long request can't pre-empt a short one's memory.
- Memory: `num_blocks_cap * block_size * head_dim * 2 * n_kv_heads * dtype` bounded once, vs N unbounded per-request caches. Quantifiable in the perf report.

### Correctness

- Bit-exact vs per-request `FusionPagedKVCache` (Phase 1) — same block writes, same logical view, just a shared physical backing. Unit test: two `FusionPagedRequestCache` in one pool produce identical decode tokens to two independent `FusionPagedKVCache`.
- Concurrency integration test: run 2-4 concurrent greedy decodes through `BatchedEngine` pool mode, verify each stream matches its single-stream reference.

### Failure modes

- Pool exhausted → `RuntimeError` with evict hint (already in Phase 1 `_alloc_block`). Phase 3 adds an LRU eviction policy candidate (lowest-`offset` idle request) — but **eviction policy is a product decision**, default = reject + 503 (fail-visible, consistent with server HA conventions). Eviction deferred to a follow-up.
- Batch dimension: the existing per-request cache uses `B=1`. Pooled decode in the scheduler batches requests into one forward — the batch dim handling in `FusionPagedRequestCache` must match `BatchGenerator`'s expected cache shape. (See Open Questions.)

---

## Open Questions (resolve before plan)

1. **Slab storage: flat pool tensor vs slab list.** `mx.fast.metal_kernel` takes array inputs, not Python lists. Phase 1 uses a list of slab tensors. Phase 2 kernel needs array input → either (a) one flat pre-allocated pool `[cap, ...]` (simple, loses lazy-growth memory win, pays cap cost upfront), or (b) pass the slab list as separate kernel inputs with a slab-index lookup (complex, many inputs). **Recommendation: flat pool tensor** — it also serves Phase 3's shared pool directly. The lazy-growth memory win from Phase 1 is reclaimed differently: the cap is sized to concurrent-budget, not worst-case-single-request. Accept the upfront allocation; document the tradeoff in the perf report.

2. **Prefill path.** Phase 2 fuses **decode only** (L=1). Prefill (L>1) keeps `update_and_fetch` concat path — concat cost amortized over many tokens, not the bottleneck. Confirm in perf report (prefill tok/s unchanged). No fused prefill kernel in Phase 2.

3. **Model attention patch scope.** Wrapping every `Attention.__call__` across all mlx_lm model families is broad. Phase 2 lands for **llama-family** first (llama, qwen3 — same attention layout), gated by `FUSION_PAGED_FUSED_KERNEL`, validated bit-exact, then extends. Other families (gemma, mistral sliding-window) follow once llama path green. **Recommendation: llama-family-only in this phase**, file follow-up issues for the rest.

4. **Batch dim in pooled decode.** Does `BatchGenerator` call `update_and_fetch` with `B>1` (request-batched) or `B=1` per-request-cached then stacked? This determines `FusionPagedRequestCache` shape handling. **Must verify** in implementation by reading `BatchGenerator` — flagged as a Task-1 investigation in the plan, not a spec blocker.

5. **Env gating defaults.** Phase 2 fused kernel: default **off** until bit-exact validated across the test matrix, then flipped on (matches Phase 0/1 convention — ship off, validate, enable). Phase 3 pool: default **off** until concurrency integration green, then on. Both behind `FUSION_PAGED_*` env vars, 503/fallback when off.

---

## Scope Check

Single subsystem (the KV cache + attention path). One spec, one plan. Phases 2 and 3 are sequential (3 depends on 2's flat-pool decision) but one plan with phase boundaries. No multi-subsystem decomposition needed.

## Out of Scope

- Fused **prefill** kernel (Phase 2 decode-only).
- NVFP4/MXFP4 GEMV dequantize-on-fetch (mentioned in Phase 1 memory as "Phase 2 concern" — deferred; raw KV stays).
- LRU/swap-to-disk eviction policy (default reject; follow-up).
- dflash2 / spec-decode KV integration (orthogonal).
- Non-llama-family attention patches (follow-up issues).

## Success Criteria

- Phase 2: decode tok/s ≥ upstream `KVCache` (reclaims the -15/-37% overhead), bit-exact tokens, perf report in `~/fusion/audit/`.
- Phase 3: N concurrent requests share one pool, memory bounded by `num_blocks_cap`, no head-of-line blocking, bit-exact per-stream, concurrency perf report in `~/fusion/audit/`.
- All new code: 4-space indent, no docstrings, default logging, black+lint clean, real-model tests bit-exact.
