# Submission pacing gate (#1197)

Per PRD `architecture/fusion-mlx-opti-prd-1014.md` F4/§6.1 (redefined): a
classic TDM bus scheduler is not measurable/controllable from Python — the
isolation mechanism is a **submission pacing gate**: every heavy-workload
submission to a Metal-bound executor passes a per-workload-class token
bucket before the executor submit.

## Stage 1 — static pacing gate

`fusion_mlx/pacing.py` — `PacingGate` singleton, per-class `TokenBucket`:

| Class | Default rate | Burst | Gate site |
|---|---|---|---|
| llm | 200/s | 40 | engine loop step-burst submit (`engine_core.py`, BEFORE the G2 deadline is armed) |
| image | 8/s | 4 | `image_gen.py` `_generate` executor submit |
| video | 4/s | 2 | `VideoGenEngine.generate` per-job |
| audio | 10/s | 5 | (reserved — no gate wired yet) |
| laya | 50/s | 20 | `/v1/laya/decide` router dispatch |
| training | 1/s | 1 | (reserved — cross-process channel, see below) |

Knobs: `FUSION_MLX_PACE_<CLASS>_RATE` / `_BURST`, master switch
`FUSION_MLX_PACING=off` (every acquire succeeds — the gate can never become
a new hang path).

**Safety rules** (see `scheduler/sched_batch.py` #1171 note):

- `acquire` runs ONLY on the async/route side — never inside an executor
  worker. A bounded wait inside a max_workers=1 worker is a self-deadlock.
- The llm gate sits BEFORE the G2 step-deadline arming: a deferred
  submission must never leave a stale armed deadline (watchdog
  false-poison hazard).
- A throttled submission surfaces an explicit error (503 via the route for
  laya, `RuntimeError` for image/video) — never a silent drop. The llm
  loop defers to the next tick instead of erroring requests.

## Stage 2 — dynamic TDM throttle (AIMD)

Victim quality signals shrink AGGRESSOR class budgets (never the victim's
own — it is already the starved party):

- **LLM TPS**: the engine loop reports windowed steps/s vs its own learned
  solo baseline (running max — self-calibrating, no config).
- **Diffusion FPS**: the video engine's wrapped `on_step` callback reports
  inter-step FPS vs the engine's learned baseline. Only fires for
  engine-layer callers that supply `on_step` (ComfyUI/SDK paths) — the
  plain HTTP video route passes none (the #171 contract forbids the
  engine injecting a callback the caller did not ask for), so API-served
  video relies on the stage-1 static gate alone.

Controller: multiplicative decrease (x0.8) after a sustained bad streak
(3 consecutive samples < 0.85 ratio — single-sample noise never
throttles), additive increase (+1/s) on recovery, 10s adjustment cooldown,
per-class floor at 25% of the static budget (decode-critical classes stay
alive). Square-wave convergence is proven in
`tests/unit/test_pacing_gate_1197.py::test_aimd_convergence_no_oscillation`.

## Metrics

`pacing_rate_budget_per_second`, `pacing_rate_current_per_second`,
`pacing_submissions_total`, `pacing_throttled_total`,
`pacing_bucket_tokens` — all labeled `class="..."` on `/metrics`.

## Known limitation — training cross-process channel

The training class runs in a spawned child (#1186); the parent's token
bucket cannot see the child's Metal submissions. The class is reserved
with a default budget; a shared-memory token counter is the follow-up
(filed separately — needs shm accounting across the pipe boundary).
