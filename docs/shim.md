# Shim Enhancement Layer

## Overview

The shim is an enhancement layer on top of the native MLX stack — not a
ggml reimplementation. Every capability borrowed from llama.cpp lands as
an opt-in module under `fusion_mlx/shim/` (plus a few wired modules in
their home packages) with a dedicated `FUSION_SHIM_*` degrade switch that
defaults OFF, so the default inference path stays stock mlx_lm/mlx_vlm.

The layer follows the v2 roadmap doc
(`architecture/fusion-mlx-vs-llamacpp-doubao-gemini-v2-0917.md`): Tier-1
safety gates are hard requirements, Tier-2 capabilities are conditional
on product support, Tier-3 items are deliberately not built.

## Package layout

- `shim/__init__.py` — master switch `is_shim_enabled()` (FUSION_SHIM_ENABLED),
  status snapshot `shim_status()`
- `shim/fast.py` — C++ extension loader with Python fallback
  (`is_native_available()`), C-ABI envelope, `MemorySentinel`
  bridge to `ProcessMemoryEnforcer`, `is_engine_runner_enabled()`
- `shim/fused_ops.py` — fused RMSNorm+residual and YaRN/NTK RoPE
  (FUSION_SHIM_FUSED_RMSNORM / FUSION_SHIM_FUSED_ROPE)
- `shim/quant_kv.py` — Q4_0/Q8_0 quantized-KV online decompression with
  FP32 softmax accumulation (FUSION_SHIM_QUANT_KV)
- `shim/mixed_quant.py` — imatrix metadata + IQ/GGUF mixed-quant
  consumption (FUSION_SHIM_IQ)
- `shim/grammar_ring.py` — grammar bitmask ring + bucket padding
  (FUSION_SHIM_GRAMMAR_RING)
- `shim/bucket_pad.py` — shape-bucket padding so JIT sees a closed shape
  set (used by the grammar ring)
- `shim/moe_dispatch.py` — deterministic MoE token reorder + grouped
  matmul + gather combine (FUSION_SHIM_MOE)
- `shim/ssm_scan.py` — Mamba-2 SSD 3-pass parallel prefix scan
  (FUSION_SHIM_SSM)
- `shim/csrc/` + `scripts/build_shim.sh` — nanobind+MLX C++ extension
  (hardware probe, memory sentinel, C-ABI envelope)
- `migrate/asfw.py`, `migrate/gguf_loader.py` — GGUF loader + ASFW
  layout conversion (FUSION_SHIM_ASFW)
- `custom_kernels/paged_kv_cow.py` — two-level paged KV with CoW
  (FUSION_SHIM_TWO_LEVEL_KV)
- `custom_kernels/fused_quant_gemv.py` — fused INT4 dequant+GEMV Metal
  kernel (FUSION_FUSED_QUANT_GEMV, default OFF). Parity-verified vs
  `mx.quantized_matmul` (max diff 4.8e-7; end-to-end real-model decode
  produces IDENTICAL tokens). Custom kernel WINS in isolated compiled
  MLP chains (-5% to -16% on large-K, e.g. Qwen2.5-7B down_proj K=18944
  -16.1%), but `mx.fast.metal_kernel` is opaque to `mx.compile` — in a
  full model forward (attention+RMSNorm+RoPE+MLP one graph) the opaque
  kernel fragments compiler scheduling, erasing the isolated win. Real
  Qwen3-8B-4bit decode (ON vs OFF, parity IDENTICAL) tok/s within
  run-to-run noise (18-30 tok/s) — no reliable end-to-end win. Kept as
  opt-in base-layer capability + foundation for PRD stage-4 (C++ graph
  pass registering dequant+GEMV as a fused primitive). Production int4
  quant path = native `mx.quantized_matmul` via `nn.QuantizedLinear`
  (default-ON). When gate ON: batch==1 AND bits==4 AND K>=8192 ->
  custom; everything else -> native (zero regression).
- `speculative/tree_mask.py` — tree-mask verify + virtual append offset
  (FUSION_SHIM_TREE_MASK)
- `utils/hardware.py` — chip-generation/MMA probe shared by the C++
  probe and Python fallback

## Building the C++ extension

```bash
./scripts/build_shim.sh     # produces fusion_mlx/shim/_ext*.so (+ metallib when kernels exist)
```

The Python layer degrades gracefully when the extension is missing —
`is_native_available()` reports False and every shim op runs its Python
fallback. Build is optional and macOS-only (MLX headers + nanobind).

## Degrade switches

All switches read live (`os.environ.get(...) == "1"`) so ops can flip at
runtime via SIGHUP settings reload or test monkeypatching. Defaults:

| Env switch | Default | Gates |
|---|---|---|
| FUSION_SHIM_ENABLED | OFF | Master switch for the C++ extension path |
| FUSION_ENGINE_RUNNER | OFF | C++ EngineRunner prototype (exclusive decode thread) |
| FUSION_SHIM_FUSED_RMSNORM | ON | Fused RMSNorm+residual kernel (microbench -47%; see audit — not wired into engine, contract mismatch) |
| FUSION_SHIM_FUSED_ROPE | OFF | Fused YaRN/NTK RoPE kernel |
| FUSION_SHIM_TWO_LEVEL_KV | OFF | Two-level paged KV + CoW |
| FUSION_SHIM_TREE_MASK | OFF | Tree-mask speculative verify |
| FUSION_SHIM_QUANT_KV | OFF | Q4_0/Q8_0 quantized-KV online decompression |
| FUSION_SHIM_IQ | OFF | imatrix + IQ/GGUF mixed quant |
| FUSION_SHIM_ASFW | OFF | ASFW layout conversion at GGUF load |
| FUSION_SHIM_GRAMMAR_RING | ON | Grammar bitmask ring (wired, -54% end-to-end) |
| FUSION_SHIM_MOE | OFF | MoE deterministic route dispatch + gather combine |
| FUSION_SHIM_SSM | OFF | Mamba SSD parallel prefix scan |
| FUSION_FUSED_QUANT_GEMV | OFF | Custom INT4 dequant+GEMV Metal kernel (parity PASS; isolated-chain -5% to -16% win, end-to-end within noise due to mx.compile opacity; native quantized_matmul is prod path) |

`tests/unit/test_shim_switches.py` enforces the contract: default values,
literal "0"/"1" handling, and switch independence (enabling one never
enables another).

## Production status (audited 2026-09-21)

| Module | Wired? | Production path |
|---|---|---|
| grammar_ring | yes (sched_thinking, monkeypatches) | default ON, -54% real |
| fused_quant_gemv (int4) | opt-in (patch wired, default OFF) | native `mx.quantized_matmul` via `nn.QuantizedLinear` (default ON); custom kernel wins isolated chains -5%/-16% but no end-to-end win (mx.compile opacity) |
| fused_rmsnorm_residual | no (contract mismatch) | `mx.fast.rms_norm` native (default) |
| fused_rope | no (slower +6-9%) | `mx.fast.rope` native (default) |
| quant_kv (shim) | no (redundant w/ turboquant_kv) | `turboquant_kv.py` int4 (default ON) |
| moe_dispatch | no | native `mx.quantized_matmul` grouped (default) |

See `audit/fusion-mlx-vs-llamacpp-audit-report-0921.md` for the full audit
and `audit/shim-audit-verified-0921.md` context.

## Verification

- Switch contract: `pytest tests/unit/test_shim_switches.py -q` —
  default-OFF, literal parsing, independence.
- Op correctness: `pytest tests/unit/test_moe_dispatch.py tests/unit/test_ssm_scan.py -q`
  — numpy reference comparisons + `ssm_attn` parity to 1-2 ulp.
- Full-chain composition + memory gates:
  `pytest tests/unit/test_shim_fullchain.py -q` — MoE→SSM chain vs a
  float64 recurrence on the CPU device; 300-iteration active-memory
  return and RSS-slope bound via the PR-F `MemoryGrowthTracker`.
- Golden logits alignment: the PR-F harness
  (`fusion_mlx/eval/golden_reference.py`) provides `assert_logits_aligned`
  (KL tol 1e-6) and `MemoryGrowthTracker`; real-model cases are gated by
  the `real_model` marker and require the live server
  (`./start.sh start`).

### Known numeric platform facts

- MLX GPU fp32 matmul computes in fp16 (~1e-3 relative error); CPU
  matmul is exact. fp64-reference tests for GPU mx code therefore run
  under `mx.set_default_device(mx.cpu)`. GPU GEMM last-ulp also varies
  with buffer context — avoid bit-exact assertions across calls.
- This MLX build has no `mx.scatter_add` / `bincount` / `searchsorted`;
  MoE expert counts use a one-hot equality sum instead.

## Performance baseline

```bash
./scripts/bench_shim_perf.py   # grammar-ring + bucket-pad microbench (PR-M)
```

Baseline JSON is archived at `benchmarks/reports/shim_perf_<date>.json`
(current: `shim_perf_20260918.json`) for across-version comparison. The
grammar-apply stock column is the prod per-bit manual fallback
(`api/grammar.py _apply_bitmask_manual`, the active path when xgrammar is
absent), not a no-op — measured -54% vs stock on M5 Max. The grammar ring
runs one persistent daemon worker (no per-token thread spawn), and
`apply_bitmask` transfers only the int32 bitmask words, expanding bits on
the GPU.

Real-model golden runs: start the server
(`./start.sh start`), set the desired switches, run the golden harness
or `fusion-mlx bench <model>`. Compare ON vs OFF for each switch before
enabling in operations.

## Operations metrics

- `fusion-mlx status` shows loaded models and memory; shim modules log
  at DEBUG under `fusion_mlx.shim.*` loggers.
- Device memory after each generation burst: `mx.get_active_memory()` /
  `mx.get_cache_memory()` (call `mx.synchronize()` + `mx.clear_cache()`
  before measuring — in-flight buffers otherwise distort the reading).
- Memory sentinel: with FUSION_SHIM_ENABLED=1 the C++ sentinel bridges
  kernel memory-pressure events into `ProcessMemoryEnforcer`.

## Upstream notes

- PR-E (GBNF C++ DFA engine) was deferred: llguidance/xgrammar remains
  the guided-decode backend, with the PR-M ring + bucket padding as the
  perf layer. File upstream issues for MLX limitations rather than
  patching vendored code.
- MLX GPU fp32 matmul computes in fp16 — relevant to any future
  golden-KL comparison against non-MLX runtimes.
- No `mx.scatter_add`/`bincount`/`searchsorted` in this build — shim ops
  must not assume them.
