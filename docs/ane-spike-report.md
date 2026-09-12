# PR-D7: ANE Spike Report — Apple Neural Engine Direct-Integration Feasibility

> Date: 2026-09-12
> Source: oMLX codebase survey at `/Users/dahai/claude-home/omlx`
> Plan ref: enhance-0911 §S4.3 / PR-D7

## Verdict: ANE direct integration is NOT viable

oMLX is **Metal-only**, identical to upstream MLX. No ANE execution path exists anywhere in the codebase.

## Evidence

### 1. Strict keyword search — zero matches

Whole-word search across every source file (`.py/.swift/.m/.mm/.h/.hpp/.cpp/.cc/.c/.metal`) for the ANE framework surface:

`CoreML, coreml, MLProgram, NeuralEngine, ANECompiler, ane_runtime, Espresso, MPSGraph, tensor_anne, _ane, ANEDevice, ANEProxy`

→ **zero matches** (only false positives: `anext` async wrapper, a chat-template fixture string mentioning "neural networks").

### 2. oMLX explicitly adds nothing over mlx-lm

`omlx/optimizations.py:9-14`:
```
Note: mlx-lm already includes optimized implementations internally:
- Flash Attention via mx.fast.scaled_dot_product_attention
- Efficient memory management
- Optimized Metal kernels

No additional optimization is needed - mlx-lm is already fast out of the box.
```

### 3. Swift app does zero inference

`apps/omlx-mac/` (106 `.swift` files) is a menubar/status UI that spawns the Python server over HTTP. No `MLX`/`Metal`/`CoreML`/`NeuralEngine` symbol imports — every "MLX" hit is a localization string.

## Fallback: Metal Deepening (recommended path)

Since ANE is unreachable, the performance ceiling lives in custom Metal kernels. oMLX's deepest original Metal work:

### A. GLM MoE DSA custom-kernel subsystem

`omlx/custom_kernels/glm_moe_dsa/` — the only place shipping genuinely custom Metal kernels (everything else calls `mx.fast.*`).

Key kernels:
- `sparse_mla_attention` — fused sparse Multi-head Latent Attention with top-k token selection
- `glm_dsa_exact_block_attention` — block-token attention
- `dsa_indexer_scores` — DSA indexer scoring
- `glm_moe_weighted_sum` — fused MoE weight-reduction

Python dispatcher (`fast.py`) falls back to `mx.fast` when native extension not built.

### B. TurboQuant quantized-KV attention

`omlx/patches/turboquant_attention.py:73-95` — monkey-patches `scaled_dot_product_attention` so KV cache stays 4-bit; decode/prefill run quantized Metal kernels without dequantizing.

**fusion-mlx already has this**: `fusion_mlx/patches/turboquant_attention.py` (6.3K) ports the same pattern — decode via `cache.decode_attention()`, prefill via `cache.prefill_attention()` with dequantize fallback.

## What fusion-mlx already covers

| Metal optimization | oMLX | fusion-mlx | Status |
|---|---|---|---|
| TurboQuant quantized-KV attention | ✅ | ✅ `patches/turboquant_attention.py` | Already ported |
| Flash Attention (mx.fast SDPA) | ✅ (upstream) | ✅ (upstream) | Shared |
| Custom Metal kernel infrastructure | ✅ `custom_kernels/glm_moe_dsa/` | ✅ `custom_kernels/` (10+ files) | Infrastructure exists |
| GLM DSA sparse MLA kernel | ✅ | ❌ | **Gap — model-specific, only GLM MoE** |
| MFA (Metal Flash Attention) bridge | ❌ | ✅ `custom_kernels/mfa_bridge.py` | fusion-mlx ahead |

## Conclusion

- **ANE: no path.** Do not pursue. oMLX confirms MLX ecosystem is GPU-via-Metal only.
- **Metal deepening: partially done.** TurboQuant already ported. GLM DSA kernels are model-specific (only benefit GLM-family MoE) — port only if GLM models become a priority use case.
- **No code changes needed for D7.** This report is the deliverable. The spike gates PR-D7 as **closed — not viable, documented**.

## Recommendation for performance work

Redirect performance investment to:
1. **Prefill optimization** — MFA/w4a8 prefill path (`custom_kernels/mfa_bridge.py` exists, needs benchmarking)
2. **KV cache compression** — TurboQuant already wired, ensure default-on for long-context models
3. **Speculative decode** — DSpark/DFlash2 (already in fusion-mlx, separate from this spike)
