---
doc_id: fusion-mlx/guide/benchmark-thermal-protocol
product: fusion-mlx
class: core
entry: service
doc_type: guide
title: { zh: "热感知基准协议 (#1588)", en: "Thermal-aware benchmark protocol (#1588)" }
audience: engineer
status: draft
version: 0.10.63
updated: 2026-11-13
owner: @dahai
source_of_truth:
  - docs/benchmark-thermal-protocol.md
---

# Thermal-Aware Benchmark Protocol (#1588)

Apple Silicon throttles **silently** — `pmset` records no warning. Measured
envelope on the #1582 investigation chassis (constant-workload decode loop,
KV trimmed every 256 steps):

- Drift: **7.2 → 14.5 ms/step** over 10 sustained minutes (step-function
  clock drop around minute 3).
- Recovery: full recovery to **7.0 ms/step after 10 min idle**.

Consequence: any comparative benchmark that mixes a cool run with a
heat-soaked run is invalid. Several historical results were artifacts of
measurement order (offline replicates ran cool, server benches ran
heat-soaked after hours of benching).

## Rules

1. **Comparative runs require a pre-run GPU idle window** — 10 minutes
   (`THERMAL_IDLE_RECOMMENDED_S = 600`). The idle duration must appear in
   the bench log / artifact.
2. **Report per-window decode ms/step** so within-run throttle drift is
   visible. Never report a single average across a throttling run.
3. **Never compare cold vs heat-soaked numbers in one table** without
   explicit thermal-state labels.
4. **Sustained production decode is thermally capped.** Exit criteria
   defined on cool-GPU numbers (e.g. `decode_b1 >= 150 tok/s`) are not
   valid for heat-soaked production (~135 cool → ~90-110 hot for
   Qwen3.5-4B-4bit b1).
5. **Perf regression gates must pin the cooldown protocol**, or they flake
   with machine temperature.
6. `gb_step_profile` numbers are only comparable at similar thermal state
   and similar step-count-within-boot.

## Tooling in this repo

`fusion_mlx.bench.thermal` provides the protocol primitives:

- `record_idle(seconds)` — pre-run idle with a log line stating duration.
- `ThermalTracker` — samples the scheduler's rolling decode ms/step
  during a run and summarizes:
  `idle_before_s`, `first_window_ms_per_step`, `last_window_ms_per_step`,
  `drift_ratio`.

`fusion-mlx bench` wires both: pass `--thermal-idle-s 600` for a
comparative run; the Results block always prints the thermal-state line.

```bash
fusion-mlx bench <model> --num-prompts 10 --max-tokens 256 --thermal-idle-s 600
```

## Cross-project note

`fusion-bench` (separate repository) runner wiring should adopt the same
protocol: pre-run idle metadata, per-window ms/step in the output schema
(`thermal_state` block), and thermal-state labels wherever cold and hot
runs could appear together.

## Reproducing the envelope

A re-run of the P2-1 b1 bench under this protocol should reproduce, as
labeled artifacts: cool ~7.2 ms/step first window, heat-soaked ~14.5
ms/step last window after 10 sustained minutes, ~7.0 after the 10-minute
recovery idle. See issue #1582 for the raw measurement series.
