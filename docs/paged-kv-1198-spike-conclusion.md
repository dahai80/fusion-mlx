# #1198 Spike Conclusion — Paged KV dynamic block management (Exit B)

Time-boxed 2-day spike, per issue DoD. Verdict: **Exit B (re-plan)** — the
requested capability already exists in-tree, is default-OFF for measured
reasons, and the fragmentation motivation does not hold on this codebase.

## Hookability (answered: yes, and already exercised)

The decode-path KV lifecycle is scheduler-owned up to one seam
(`batch_generator.insert(caches=...)`, sched_batch.py:857,
sched_schedule.py:1065). `install_paged_kv(model, config)` rebinds
`model.make_cache` to return per-request handles over a shared
`FusionPagedKVPool` (physical block pool + per-request block table + LRU
eviction) — fusion_mlx/custom_kernels/fusion_paged_kv.py:81-189. Scheduler
hooks already exist: active-set sync (sched_step.py:40-88), pool-id
bookkeeping (sched_batch.py:883-886), CoW donor registration
(sched_batch.py:892-906), free-on-finish (sched_response.py:371-405).

No fork of cache storage is required — but this IS the feature the issue
asks to build, and it is shipped and gated OFF:
`fusion_paged_kv_enabled=False`, `fusion_paged_pool="off"`
(model_settings/__init__.py:121-125).

## Why default-OFF is correct (prior measurements)

| Attempt | Result |
|---|---|
| Phase 1 paged cache (concat-per-step logical view) | decode -14.7..-36.6%, prefill -19.9..-23.7%, memory neutral |
| Phase 2 fused Metal paged decode kernel | -77.3% (57.67 vs 253.84 tok/s), bit-exact, gated `FUSION_PAGED_FUSED_KERNEL=off` |
| Phase 3 shared pool concurrency | -29.0% N=4 throughput, memory neutral |
| CoW paged-KV | -5~7x, "incompatible with mlx-lm batched KV structure" (docs/innovation-weight-amortized-spec-decode.md:99), DELETED |

Reports: ~/fusion/audit/perf-paged-kv-2026-09-03T00-25-41.md,
paged-kv-phase2-perf-report.md, paged-kv-phase3-concurrency-perf-report.md.

## Fragmentation premise does not hold

- On the default path each request's KV is one contiguously-grown array pair
  (mlx_lm KVCache.update_and_fetch = concatenate, cache.py:191-200). There is
  no shared pool to fragment; per-request arrays die by refcount on finish
  (deferred `batch_generator.remove`, sched_response.py:456-458) and MLX's
  allocator recycles the buffers.
- The only fragmentation metric measures non-MLX overhead, not KV layout
  (routes_internal/metrics.py:450-497). The measured memory pain is prefill
  transients (113GB+ footprint on 40K-token prefill, sched_batch.py:773-783)
  — already attacked by the adaptive prefill throttle, the heavy-admission
  gate (#1187) and tier budgets (#1194), not by block layout.
- The production block machinery that IS default-on
  (`BlockAwarePrefixCache`, 64-token blocks, refcount, chain-hash, LRU)
  operates on boundary snapshots at admission/finish — it gives prefix reuse
  without touching live decode storage.

## Re-plan proposal

1. Do NOT revive the pool path for capacity: the >=2x capacity target has no
   supporting measurement, and enabling the pool re-introduces the measured
   -15..-77% decode losses. Keep `fusion_paged_pool=off`.
2. Capacity levers that are real on this codebase: prefix-cache reuse
   (snapshot path — see #1236 store_cache defect fix), admission control
   (#1187/#1194), and transient bounding during prefill (#1204 throttle
   floor).
3. If mlx-lm upstream ever exposes native block-structured KV in BatchGenerator,
   re-evaluate then; until that exists the concat-per-step tax is structural.

DoD satisfied: spike conclusion documented; no half-done fork committed.
