# Hang diagnosis playbook (issue #1188)

Concurrent LLM + Diffusion workloads intermittently wedge the process
(memory not high, restart recovers). This playbook maps the four
hypotheses from `architecture/fusion-mlx-deep-bug-1014.md` (H1-H4) to the
evidence each capture command produces, so the NEXT hang can be
attributed before restart.

**Rule: capture BEFORE restart.** One command gathers everything:

```bash
scripts/hang-capture.sh <PID> [PORT] [OUTDIR]
```

It writes `ps.txt`, top profile, py-spy/python stacks (SIGUSR1
faulthandler), native `sample` profile, fd usage vs limit, kernel Metal
log (10m), live `/metrics` (watchdog/fd/admission/eviction series), and
the workload state (`/v1/admission/status`, fine-tune jobs) to the output
dir, plus a `summary.txt` with classification hints.

## Fingerprint table

| Hypothesis | Fingerprint | Decisive evidence |
|---|---|---|
| **H1** Metal command-queue deadlock (shared queue, missing MTLEvent fence) | CPU ~0%, threads Sleeping, memory static; GPU submissions never complete | `sample.txt`: main thread blocked in `IOAccelCommandBuffer`/`MTLCommandQueue` wait; `kernel-metal.log` shows stalled command buffers; watchdog `fusion_mlx_watchdog_heartbeat_age_seconds` climbs while CPU stays 0 |
| **H2** lock deadlock / inversion (all workers lock-waiting) | CPU 0% or one thread spinning; every worker parked on a mutex | `pyspy-dump.txt`: 2+ threads parked in `lock.acquire` on the same pair of locks in reverse order; with `FUSION_MLX_LOCK_DEBUG=1` the pool logs a violation naming both locks (see `docs/LOCK_INVENTORY.md`); `fusion_mlx_enforcer_lock_timeouts_total` climbing |
| **H3** event loop blocked by C++ holding GIL / un-guarded eval | single Python thread ~100%, all coroutines stopped, heartbeats stall | `sample.txt`/`pyspy`: one thread inside a C++ frame with GIL; shim offenders closed by #1193 (`fusion_mlx/shim/GIL_AUDIT.md`); if the hot frame is `mx.eval`, that is the eval-wedge case -> watchdog is the designed backstop (#1189) |
| **H4** FD/IPC exhaustion (SSE keep-alive leak) | `lsof` count near `ulimit -n`, EMFILE in logs | `fd.txt` >= 90%; `fusion_mlx_fd_by_type{type="socket"}` grew steadily; `fusion_mlx_fd_leak_delta{type="socket"}` positive -> #1192 gate already rejects NEW streams at 90% |
| **New hypothesis H5+** | none of the above | attach capture dir + assign next number in the deep-bug doc |

## Workload attribution (#1184 overlap)

`fine-tune-jobs.json` + `admission-status.json` in the capture dir record
what was running at hang time. If a training job was active during the
hang, note it explicitly in the diagnosis report — #1186 (process
isolation + Metal memory limit) and #1187 (heavy admission gate) already
contain that scenario; the report should say whether the observed hang
predates or survived those fixes.

## Tooling that already ships (no capture needed)

- `kill -USR1 <pid>` — all-thread Python stack dump to the server log
  (faulthandler, installed at boot, `cli_serve/serve_command.py`).
- `/metrics` — `fusion_mlx_watchdog_heartbeat_age_seconds{executor}`,
  `fusion_mlx_watchdog_poison_total{executor}` (#1189),
  `fusion_mlx_fd_*` (#1192), `fusion_mlx_admission_*` (#1187),
  `fusion_mlx_tier_budget_rejections_*` (#1194),
  `fusion_mlx_enforcer_lock_timeouts_total`.
- `FUSION_MLX_LOCK_DEBUG=1` — lock-order violation log (#1190).

## Diagnosis report

When a hang is attributed, add a section to
`architecture/fusion-mlx-hang-diagnosis-1014.md` with the capture dir
path (or its summary.txt), the hypothesis verdict, and the command-output
evidence. Until the next hang occurs, attribution is intentionally
pending — this PR ships the tooling + playbook only (no speculative
fixes, per the issue).
