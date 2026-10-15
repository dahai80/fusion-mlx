# Lock inventory + global ordering (issue #1190)

PRD opti-1014 F5/E7, deep-bug H2 (deadlock prevention). Status: landed with
debug-gated runtime verification.

## Scope

140 lock sites inventoried across `fusion_mlx/` (non-test, `patches/`
excluded). Detailed below: the `pool` package (the H2 core — engine pool +
memory enforcer share nested acquisitions) plus every verified
nested-acquisition edge. The remaining modules follow the single-lock-per-
site pattern with no verified nested acquisition.

## Pool package locks

| Lock | File:line | Kind | Guards |
|---|---|---|---|
| `pool.main` | `engine_pool.py:206` | `TrackedAsyncLock` (asyncio.Lock) | `_entries`, memory accounting, load markers, eviction selection. Enforcer re-acquires with a 2.0s `wait_for` timeout. |
| `pool.pinned` | `engine_pool.py:209` | `TrackedThreadLock` (threading.Lock) | `is_pinned` writes from sync/threadpool callers (P3 #811). |
| `pool.discover_sync` | `engine_pool.py:216` | `TrackedThreadLock` (threading.Lock) | `_merge_discovered` on the sync discover path (AS-12). |
| `pool.adapter_swap` | `engine_pool.py:245` (dict), created in `_swap_lock` :1244 | per-base `TrackedAsyncLock` | in-place LoRA swap apply→infer→restore window (issue #389). |
| `_load_semaphore` | `engine_pool.py:257` | `asyncio.Semaphore` | caps concurrent model loads (load-storm OOM guard, #819). Not part of the lock order. |
| `enforcer.state` | `memory_enforcer.py:430` | `TrackedThreadLock` (threading.Lock) | `_loaded_model_bytes`, `_last_evicted_at`, `_eviction_marked`, `_ane_resident`, pressure bookkeeping. |
| RLock | `unified_memory_pool.py:67` | `threading.RLock` | dead-code pool (P3 #811), not wired into serving. |
| BackendQuota `_lock` | `unified_memory_pool.py:239` | `threading.Lock` | per-backend request quota counters. |
| KVCacheBridge `_lock` | `unified_memory_pool.py:318` | `threading.Lock` | KV cache bridge registration/lookup. |

## Verified nested-acquisition edges (no reverse-order pairs found)

- `pool.main` → `pool.discover_sync` — `discover_models_async` →
  `_merge_discovered`.
- `pool.main` → `enforcer.state` — `update_loaded_model_bytes` /
  eviction cooldown write inside the enforcement loop (both directions of
  the enforcer's outer/inner split are audited; enforcer enters
  `pool.main` WITHOUT holding `enforcer.state`, and releases `pool.main`
  around `unload_engine_async`).
- `KVCacheBridge._lock` → `MetalBufferRegistry._lock`
  (`unified_memory_pool.py:385` → `:331`).
- Swap locks (`pool.adapter_swap`) are always acquired WITHOUT `pool.main`
  held — bottom tier of the order.
- No edge acquires a lock higher in the declared order while holding one
  lower; no reentrant acquisition of `pool.main` (documented
  non-reentrancy at `engine_pool.py:2622`).

## Declared global order

`fusion_mlx/pool/lock_order.py`, outermost first:

```
server.app_init
enforcer.state
pool.main
pool.discover_sync
pool.pinned
pool.adapter_swap
```

Rule: a thread/task must never acquire a lock that appears EARLIER in this
list while already holding one that appears LATER. All pool-package locks
are wired through `TrackedAsyncLock` / `TrackedThreadLock`, which check the
order on acquire when debugging is enabled.

## TOCTOU register (11 sites)

Documented and accepted unless noted:

1. `engine_pool.py:1244` `_swap_lock` — get-or-create of per-base swap
   locks is not atomic; two tasks can transiently create two locks. Single
   event loop makes the window practically closed; worst case one request
   serializes on a stale lock object.
2. `engine_pool.py:1288` `_acquire_inplace_adapter` — reads `_entries`
   before taking the swap lock; the base can be unloaded between the check
   and the swap. Guarded upstream by the `_inplace_swap` gate plus the
   engine-strong-ref read.
3. `engine_pool.py:1551` `get_engine` inplace gate — `_entries.get` outside
   the pool lock; a concurrent unload makes the swap no-op fall through to
   the derived-engine path (safe, cold load).
4. `engine_pool.py:1559` active-swap check — `model_id in self._active_swap`
   outside any lock; a restore completing between check and wait means one
   request waits unnecessarily (safe direction).
5. `engine_pool.py:1362` `_release_inplace_adapter` — `_entries.get` +
   `_active_swap.pop` without the pool lock; restore-failure marks the
   engine dead so a raced eviction cannot serve wrong weights.
6. `engine_pool.py:1564-1700` `get_engine` Phase 1 → Phase 2 marker —
   `is_loading` reservation is set under the lock and cleared in phase 2;
   failure marker (`last_load_failed`) documented, guarded.
7. `engine_pool.py:2108` `_find_lru_victim` — iterates a shallow snapshot
   of `_entries` outside the pool lock (AS-1); selection is advisory,
   re-checked by the next admission iteration.
8. `memory_enforcer.py:1742` lock-timeout fallback — if the 2.0s
   `wait_for` on `pool.main` times out, the enforcer proceeds on a stale
   usage snapshot and marks victims for eviction (documented AS-2,
   `record_enforcer_timeout` counter).
9. `memory_enforcer.py:~1898` `_last_evicted_at[victim]` cooldown write —
   was written under `pool.main` only while `prune_model_state` pops the
   same key under `enforcer.state`. **FIXED in this PR**: the write is now
   under `enforcer.state` (nested `pool.main` → `enforcer.state`, the
   declared order).
10. `memory_enforcer.py:481` `_eviction_marked.discard` — was immediately
    after the `enforcer.state` block that pops `_last_evicted_at`;
    **FIXED in this PR**: the discard is inside the lock block.
11. `engine_pool.py:2021/:2088` vs `:3426` `update_loaded_model_bytes` /
    `_current_model_memory` — mixed protection (some +- under the pool
    lock, phase-2 increments lockless by design). Left as-is: pool-lock-
    adjacent refactor, out of scope for #1190.

## Debug assertions

```bash
# Log-only (default): violations log ERROR, server keeps running.
FUSION_MLX_LOCK_DEBUG=1 ./start.sh start

# Strict: first violation raises LockOrderViolation (fail fast).
FUSION_MLX_LOCK_DEBUG=1 FUSION_MLX_LOCK_DEBUG_STRICT=1 ./start.sh start
```

When `FUSION_MLX_LOCK_DEBUG` is unset the tracker is inert: every check is
one boolean test, wrapper classes delegate straight to
`asyncio.Lock` / `threading.Lock`, and `snapshot_held()` stays empty.
Tests: `tests/unit/test_lock_order_1190.py`.

## Hardening already in place (not changed here)

- `faulthandler` SIGUSR1 dump wired in `serve_command.py` (dumps all
  thread stacks on demand).
- `record_enforcer_timeout` Prometheus counter on the enforcer's bounded
  lock acquisition.
- Bounded acquisitions: enforcer's `wait_for(pool._lock.acquire(), 2.0)`,
  `boundary_snapshot_store` timeout acquire, `_bounded_mlx_sync_clear`.
