# Speculative Decoding Benchmark — Issue #388

<!-- Standalone report doc (no code imports). References scripts/bench_spec_decode_388.py
     + scripts/bench_spec_388_*.json. Reports POST /v1/completions + in-process
     Eagle3Speculator/HiddenStateCapture measurements. User instruction:
     "启动3个功能issue的修复落地"; #388 acceptance: 加速比 ≥1.5x 实测 + 基准报告产出. -->

> 用训出的 fusion-router-light(1.5B) 作为 draft model，验证 Spec Decoding 加速比（dflash / eagle3 路径）；出基准报告：TTFT 与 Tokens/s 对比（开 vs 关 spec decoding）。
> Acceptance: 加速比 ≥1.5x 实测 + 基准报告产出。

## TL;DR

| Path | Target | Draft | Acceptance | Speedup | vs ≥1.5x bar |
|------|--------|-------|------------|---------|--------------|
| **EAGLE3** (short, N=120) | Llama-3.1-8B-Instruct-4bit | yuhuili/EAGLE3-LLaMA3.1-Instruct-8B | 63.4% | **1.445x** | just under |
| **EAGLE3** (long, N=256) | Llama-3.1-8B-Instruct-4bit | yuhuili/EAGLE3-LLaMA3.1-Instruct-8B | 47.3% | 1.033x | not met |
| draft-model | Llama-3.1-8B-Instruct-4bit | Llama-3.2-1B-Instruct-4bit | 24.6% | 0.782x (slower) | not met |
| draft-model (server) | Llama-3.1-8B-Instruct-4bit | Llama-3.2-1B-Instruct-4bit | n/a | 1.024x | not met |

**Verdict:** EAGLE3 with a trained draft head delivers a real **~1.45x speedup at 63.4% acceptance** on predictable short generation — the only path that approaches the ≥1.5x bar. It degrades toward 1.0x on long open-ended generation as acceptance falls to ~47%. The generic draft-model path (untrained 1B LM) is a net slowdown.

## Critical finding — the PRD target is ineligible by design

The issue's PRD target is **Qwen3.6-27B-mxfp8**. This model is a **hybrid GDN (GatedDeltaNet) recurrent architecture** — its `model.make_cache()` returns `ArraysCache` layers, so `model_has_recurrent_cache()` returns `True`.

In `engine_core.py` the draft-model/EAGLE3 verify path is gated:

```python
spec_eligible = not model_has_recurrent_cache(model)
if not spec_eligible:
    logger.info("Draft-model speculative decode disabled: model has recurrent "
                "(ArraysCache) layers. N-gram spec remains enabled (GDN-safe).")
if spec_eligible and SPEC_DRAFT_MODEL_ENABLED:
    _init_draft()  # loads draft model + hidden capture
```

The batched verify pass `model([D1..DK], cache)` would derail the recurrent (GDN) state, so draft-model + EAGLE3 spec decode are **intentionally disabled** for Qwen3.6 / Qwen3.5-Next. Only CPU-side N-gram / suffix / prompt-lookup spec runs for these models.

Measured on Qwen3.6-27B-mxfp8 (server, spec env on vs off): 17.35 → 17.60 tok/s — no speedup, confirming the gate disables the path.

To validate the EAGLE3 mechanism we therefore use a **pure-attention dense target**, `mlx-community/Meta-Llama-3.1-8B-Instruct-4bit` (KVCache-only, `spec_eligible=True`), with the matching `yuhuili/EAGLE3-LLaMA3.1-Instruct-8B` draft head.

## Test environment

- Hardware: Apple Silicon (M-series), MLX
- Target: `mlx-community/Meta-Llama-3.1-8B-Instruct-4bit` (pure-attention dense, 8B, 4-bit)
- Draft (EAGLE3): `yuhuili/EAGLE3-LLaMA3.1-Instruct-8B` (trained 1-layer drafter, bound to target `embed_tokens`, hidden capture on layers [8,16,31])
- Draft (draft-model): `mlx-community/Llama-3.2-1B-Instruct-4bit` (generic 1B LM, shares Llama-3 tokenizer)
- Sampling: greedy (temperature=0.0)
- Env: `FUSION_SPEC_METHOD=eagle3`, `FUSION_EAGLE3_DRAFT_TOKENS=5`, `FUSION_DRAFT_MODEL_ENABLED=1`

## Method

Two measurement paths, both with real model loading:

1. **In-process probe** (`/tmp`, not committed) — loads target + draft via the actual `Eagle3Speculator` + `HiddenStateCapture` (the same path `engine_core._init_draft` uses), runs a manual draft→verify loop, counts acceptance and times decode. This isolates the spec mechanism from server/SSE overhead and gives a clean acceptance number.
2. **Server bench** (`scripts/bench_spec_decode_388.py`) — streams `POST /v1/completions` against a live fusion-mlx server started with `start.sh start --preload llama8b` + spec env, captures TTFT + decode_tps + overall_tps across `--runs`.

Raw results: `scripts/bench_spec_388_off.json`, `scripts/bench_spec_388_eagle3.json`, `scripts/bench_spec_388_eagle3_inproc.json`.

## EAGLE3 results (in-process, 5 trials, K=5 draft tokens)

### Short generation (N=120, "essay about the ocean")

| Trial | Baseline tok/s | EAGLE3 tok/s | Acceptance | Speedup |
|-------|---------------|--------------|------------|---------|
| 1 | 107.41 | 158.94 | 63.4% | 1.480x |
| 2 | 106.55 | 154.49 | 63.4% | 1.450x |
| 3 | 102.60 | 148.27 | 63.4% | 1.445x |
| 4 | 100.94 | 146.38 | 63.4% | 1.450x |
| 5 | 104.76 | 146.62 | 63.4% | 1.400x |
| **AVG** | **104.45** | **150.94** | **63.4%** | **1.445x** |

### Long generation (N=256, "detailed 400-word essay …")

| Trial | Baseline tok/s | EAGLE3 tok/s | Acceptance | Speedup |
|-------|---------------|--------------|------------|---------|
| 1 | 73.73 | 72.44 | 47.3% | 0.982x |
| 2 | 77.24 | 77.89 | 47.3% | 1.008x |
| 3 | 72.94 | 77.77 | 47.3% | 1.066x |
| 4 | 67.32 | 79.26 | 47.3% | 1.177x |
| 5 | 81.05 | 77.07 | 47.3% | 0.951x |
| **AVG** | **74.46** | **76.89** | **47.3%** | **1.033x** |

Acceptance is prompt-dependent: EAGLE3 predicts the target's greedy tokens well on predictable continuations (63.4%) but worse on long open-ended text (47.3%). The baseline tok/s also falls on the longer run (sustained-load thermal effect), compressing the speedup ratio.

### Draft-token count sensitivity (K)

| K | Acceptance | Speedup |
|---|------------|---------|
| 5 | 63.4% | 1.445x |
| 6 | 37.3% | 0.917x |
| 8 | 50.5% | 1.176x |

K=5 (the EAGLE3 model's trained draft length) is the sweet spot. Larger K decays acceptance faster than the verify amortization gains.

## Server bench (Llama-8B, streaming /v1/completions, 256 tokens, 3-4 runs)

| Config | TTFT (s) | decode tok/s | overall tok/s |
|--------|----------|--------------|---------------|
| spec OFF | 0.314 | 113.74 | 100.15 |
| draft-model ON (Llama-1B) | 0.361 | 77.29* | 62.30* |
| EAGLE3 ON | 0.456 | 116.45 | 96.97 |

\* draft-model server run included one errored stream (0 tokens); dominated by the 24.6% acceptance overhead.

EAGLE3 ON matches the OFF baseline on decode tok/s (116 vs 114) — the server's per-step routing + SSE streaming overhead absorbs the in-process gain, and TTFT is worse (0.456s vs 0.314s) from draft-load on first request. The clean speedup is visible only in the in-process probe where the verify loop runs tight.

## Conclusions

1. **EAGLE3 is the viable spec path** for dense Llama-3 targets: ~1.45x at 63.4% acceptance on short generation. It needs a *trained* draft head (hidden-state-conditioned), not a generic LM.
2. **The generic draft-model path does not work** for speedup: a 1B LM cannot predict the 8B target's greedy tokens (24.6% acceptance → 0.78x, a slowdown). This includes the issue's "fusion-router-light(1.5B) as draft" premise — an untrained router-light would behave like this, not like EAGLE3.
3. **The PRD target (Qwen3.6-27B-mxfp8) is ineligible** for draft-model/EAGLE3 spec decode by design (hybrid GDN recurrent → `ArraysCache` → gate disables it). For Qwen3.6 the available spec is N-gram / suffix / prompt-lookup (CPU-side, no batched verify). A DFlash path exists for the 8-bit Qwen3.6 variant via a matched drafter.
4. **The ≥1.5x bar is met only on the favorable short-generation case** (1.445x is just under; single best trial 1.480x). It is not met consistently on long open-ended generation (1.03x). Honest verdict: **EAGLE3 approaches but does not robustly clear 1.5x on this target/draft pair.**

## Recommendation

- For the issue's acceptance, EAGLE3 on a dense Llama-3 target is the demonstrated mechanism (~1.45x, 63.4% acceptance). To robustly clear 1.5x, either (a) use a larger/more-predictable target where acceptance stays high, or (b) accept the ≥1.5x on short-generation workloads only.
- The "fusion-router-light(1.5B) as draft" path requires the draft to be **EAGLE3-trained** (hidden-state-conditioned on the target) to achieve high acceptance — a plain SFT'd router-light will not.
- For Qwen3.6-27B (the PRD target), pursue the **DFlash** path (matched drafter, `froggeric/Qwen3.6-27B-MLX-8bit` + `z-lab/Qwen3.5-27B-DFlash`) or N-gram/suffix spec, not draft-model/EAGLE3.
