# Chaos fault-injection suite (#1195)

`tests/chaos/` validates detection + recovery for every failure mode the
optimization PRD (`architecture/fusion-mlx-opti-prd-1014.md` §6.2) addresses.

## Per-scenario contract

Every scenario asserts all three DoD targets:

| Target | Bound | How asserted |
|---|---|---|
| Detection | < 60 s | fault injected, watchdog/metric fires, wall-clock elapsed measured against `DETECTION_BOUND_S` |
| Recovery | < 120 s | service returns to admitting/serving work, wall-clock measured against `RECOVERY_BOUND_S` |
| Silent loss | zero | every request/job either served or carries an explicit error/decision |

Windows are scaled down via the same env knobs/knobs-as-params production
uses (`FUSION_LLM_WATCHDOG_TIMEOUT`, `ExecutorWatchdog(window_s=...)`,
`FUSION_MLX_FD_*_RATIO`), so a green test maps 1:1 to the production
configuration bound.

## Scenario map

| File | PRD shape | Injection point |
|---|---|---|
| `test_h1_gpu_stall.py` | H1 GPU/command-queue stall | watchdog INPUT level: pending step deadline + stale heartbeat + frozen step counter |
| `test_h2_lock_inversion.py` | H2 lock-order inversion | wrong nested order on `TrackedAsyncLock`/`TrackedThreadLock` under `FUSION_MLX_LOCK_DEBUG=1` + `_STRICT=1` (#1190) |
| `test_h3_executor_block.py` | H3 blocking op on executor | single-worker executor job with no heartbeat (#1189 `ExecutorWatchdog`) |
| `test_h4_fd_leak.py` | H4 FD leak | real unclosed sockets + tight synthetic limit (#1192 `FDWatchdog`) |
| `test_training_contention.py` | #1184 training vs serving | admission gate with monkeypatched physical/usage (#1187) + laya defer_check |
| `test_long_request_spike.py` | long request + spike | slow usage probe concurrent with 20–100 admission decisions |

## H1 rationale (documented skip of a REAL GPU stall)

A genuine GPU/Metal command-queue stall **cannot be injected from pytest
without risking the CI runner itself** — the stall is the machine hanging.
The suite instead injects the exact observable the G2 LLM watchdog reads in
production: `_llm_step_deadline` pending, `_llm_heartbeat` stale beyond the
window, engine step counter frozen. That is byte-for-byte what a wedged
Metal submission produces; the watchdog cannot distinguish it from the real
thing (which is the point — detection is input-driven).

The media-starvation variant (GPU shared with a media job) is separately
covered: the watchdog must DEFER, not poison, while a media job is active
within the defer budget (`_LLM_WATCHDOG_MEDIA_DEFER_S`).

## CI

Dedicated `chaos` job (macos-14, timeout 15 min) runs
`pytest tests/chaos -v`. Kept out of the unit shards: the shard emitter
(`scripts/shard_tests.py`) globs `tests/unit` only, and the suite's
wall-clock assertions deserve their own time-boxed lane.
