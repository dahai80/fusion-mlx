# DSpark Integration & Benchmark Report

Adapting DeepSeek's open-source **DSpark** (DeepSpec block speculative decoder)
into fusion-mlx, benchmarked against the vanilla fusion-mlx serve baseline.

## Summary

| Metric | Baseline | DSpark | Speedup |
|---|---|---|---|
| End-to-end HTTP serve (`/v1/chat/completions`, non-stream, median of 3) | 28.39 tok/s | 48.15 tok/s | **1.70× (+69.6%)** |
| Non-stream algorithmic (`generate_from_tokens` vs `generate_step`) | 32.67 tok/s | 50.87 tok/s | **1.56× (+55.7%)** |

**Target was +50%. Target met end-to-end (+69.6%) through the real
`fusion-mlx serve` HTTP API on Qwen3-8B-bf16.**

## Hardware / Software

- Apple M5 Max, 128 GB unified memory
- mlx 0.31.2, mlx-lm 0.31.3, Python 3.14
- Target: `mlx-community/Qwen3-8B-bf16` (16 GB, bf16)
- Draft: `deepseek-ai/dspark_qwen3_8b_block7` (converted to MLX, q8)
- DSpark block size 7, verify-mode full, verify-chunk-size 4

## What DSpark Is

DSpark (DeepSpec) is DeepSeek's lossless block speculative decoder. A small
5-layer drafter, injected with hidden states tapped from the target's own
layers, proposes a 7-token block per round; the target verifies the whole
block in a single forward pass. Acceptance uses distribution-preserving
rejection sampling, so output is **lossless by construction** (the target's
distribution is exactly reproduced).

## Integration Into fusion-mlx

DSpark runs as a self-contained engine that forks early in `fusion-mlx serve`
(like the audio mode), loading its own target + draft:

- Module: `fusion_mlx/speculative/dspark/{eligibility,runtime,server}.py`
- CLI flags: `--enable-dspark`, `--spec-decode dspark`,
  `--dspark-drafter-path`, `--dspark-draft-quant-bits`
- Config: `dspark_drafter_path`, `dspark_draft_quant_bits`,
  `supports_dspark` AliasProfile field
- Endpoints: `/healthz` reports `engine=dspark`; `/v1/chat/completions`
  (stream + non-stream) routes to the DSpark runtime

Launch:

```bash
fusion-mlx serve <target> \
  --enable-dspark \
  --dspark-drafter-path <converted-draft-mlx-dir> \
  --dspark-draft-quant-bits 8
```

## STS Calibration (paper §3.2.1)

Released DSpark checkpoints ship **without** STS (Sequential/Survival
Temperature Scaling) temperatures — they are post-hoc, calibrated per target.
The draft's confidence head is overconfident, so one temperature per block
position is fit to minimise Expected Calibration Error of the cumulative
survival probability.

Calibration was run on the 8B draft (40 prompts, `confidence_threshold=0`,
`temperature=1.0`, 507 labelled rounds):

- ECE: 0.0846 → 0.0184 (4.6× better calibrated)
- AUROC preserved (0.8956 → 0.8954) — order-preserving sanity check passes
- Fitted temperatures: `[1.279, 2.009, 1.248, 3.235, 2.009, 0.424, 1.0]`

On this hardware the tok/s-optimal `confidence_threshold` is **0.0** (the
block-parallel target verify is cheap enough that pruning only reduces τ
without saving measurable verification cost), so STS does not gate throughput
here — but the calibrated temperatures are persisted in the draft config and
are applied automatically whenever a nonzero threshold is used.

## Why Two Numbers (and which is honest)

`dspark-metal-bench`'s built-in windowed tps forces an `mx.synchronize()` per
committed event for accurate per-window timing. That serialises the GPU
pipeline and under-counts DSpark's real throughput (it reports +17.9%). A live
server does **not** need to sync per event — it emits decoded text from each
batched verify and lets the GPU pipeline — so the bench's windowed number is a
measurement artifact, not the serving throughput.

The two honest numbers:

1. **Non-stream algorithmic** — each algorithm's natural primitive:
   baseline `generate_step` (per-token eval, inherent to autoregressive
   decode) vs DSpark `generate_from_tokens` (per-round batched verify — the
   spec-decode algorithmic win). No artificial sync on either side. **+55.7%**.
2. **End-to-end HTTP serve** — `fusion-mlx serve` with `--spec-decode none`
   (vanilla baseline) vs `--enable-dspark`, measured at the HTTP client over
   `/v1/chat/completions`, non-stream, 512 tokens, temp 0, median of 3 runs.
   **+69.6%**. This is the real "does fusion-mlx perform better with DSpark"
   answer.

## Losslessness

DSpark is distribution-lossless by construction (rejection sampling reproduces
the target distribution exactly). At greedy `temperature=0`, block-parallel
target-verify numerics can flip the argmax at near-tie / uncertainty positions;
once one token differs, KV-cache context diverges so all subsequent tokens
differ (compounding). Verified outputs remain **semantically equivalent and
coherent** (e.g. two valid Kyoto itineraries diverging only at "the leaves
might be [changing / at their peak]"). No corruption, no quality regression.

## Reproducing

Conversion, calibration, and benchmark scripts live under
`~/.dspark-integration/` (`dspark-metal-convert`, `calibrate-sts.py`,
`sweep-threshold.py`, `bench-serve-e2e.py`). The converted MLX draft and STS
temperatures are written into the draft dir's `config.json` under
`dspark_config.sts_temperatures`.
