# Speculative Decoding

fusion-mlx ships **five speculative-decoding algorithms** — SuffixDecoding,
DFlash, DSpark, MTP, and VLM MTP — selectable per serve. This is the single
largest differentiator vs. Ollama / vLLM-mac, none of which ship a
spec-decode path. This document is the authoritative reference: which method to
pick, how to activate it, the architectural constraint that governs selection,
and the auto-router that automates the choice.

> **Status legend** — ✅ shipping and verified · 🧪 shipped but PoC / workload-gated · 🗺️ per-request routing is Phase B (boot-time `--spec-decode auto` landed; see [Auto-router](#auto-router-specroute)).

---

## Why speculative decoding

Speculative decoding trades a small amount of extra compute for a large
reduction in decode latency. A cheap **drafter** proposes K candidate tokens;
the target model verifies all K in a single forward pass. If the verifier
accepts `a` of them, you pay one forward pass but emit `a+1` tokens — a
speedup whenever `a` is high and the drafter is cheap.

The speedup is **workload-dependent**, not free:

| Workload | Typical speedup | Why |
|---|---|---|
| Tool calls / JSON / code edit | 3–5× | Highly repetitive token sequences — the drafter guesses whole spans |
| Long-document / RAG | 1.5–3× | Block drafters exploit repetition in the source text |
| Free-form chat / reasoning | ~1× | Low repetition; draft acceptance collapses, overhead dominates |
| Hostile (random / multilingual switches) | <1× possible | A naive drafter *regresses* — fusion-mlx gates these off (see [Pitfalls](#pitfalls)) |

**Pick a method only when the workload rewards it.** The auto-router below
encodes this judgment; manual selection should follow the same logic.

---

## Methods at a glance

| Method | Mechanism | Draft cost | Best workload | Status |
|---|---|---|---|---|
| **SuffixDecoding** (`suffix`) | Drafter-free suffix tree over already-generated tokens | None — no draft model | Tool calls, JSON, code edit | ✅ |
| **DFlash** (`ddtree`) | Block-diffusion drafter (arXiv 2410.04097), bound to a Qwen3.5/3.6 target | One drafter load at boot | Long-document / RAG | 🧪 |
| **DFlash2** (`dflash2`) | Block-diffusion drafter (z-lab `dflash` pkg, GDN conv + CandidateSelector), bound to a Qwen3.8 dense target; reads target hidden states at `target_layer_ids` | One drafter load at boot | Qwen3.8-27B dense, greedy + sampling | ✅ |
| **MTP** (`mtp`) | Model-native multi-token-prediction heads (Qwen3.5/3.6, DeepSeek-V4) via mlx-lm PR #990 | None — uses target's own heads | Any eligible model | ✅ |
| **DSpark** (`dspark`) | DeepSeek DeepSpec lossless block speculative decode, distribution-preserving rejection sampling | One converted draft load at boot | Qwen3 4B/8B/14B bf16 targets | 🧪 |
| **VLM MTP** (`vlm-mtp`) | MTP drafter for vision-language models (`gemma4_assistant` drafter) | Drafter load | VLM generation | ✅ |

Canonical method names (the `method=` field in `speculative/registry.py`) are
shown in parentheses. CLI aliases are documented under [Activation](#activation).

---

## Activation

Speculative decoding is a **serve-time** setting. There are three activation
surfaces; the primary selector and the individual toggles are mutually aware.

### Primary selector

```bash
fusion-mlx serve --model <model> --spec-decode {none,mtp,dflash,dflash2,dspark}
```

`--spec-decode` picks **one** model-side method. `none` (default) disables all
model-side spec decode. The flag fails loud at boot if the model is ineligible
(e.g. `mtp` against a checkpoint without `mtp_num_hidden_layers >= 1`) so
misuse never silently falls back.

### Individual toggles

```bash
--suffix-decoding          # SuffixDecoding (drafter-free)
--enable-mtp               # native MTP (Qwen3.5/3.6 runtime)
--enable-dflash            # DFlash block-diffusion drafter
--enable-dflash2           # DFlash2 block-diffusion drafter (z-lab dflash pkg, Qwen3.8)
--enable-dspark            # DSpark lossless block spec-decode (serial mode)
```

### Method-specific knobs

| Flag | Default | Purpose |
|---|---|---|
| `--suffix-max-draft` | 8 | Max draft tokens per verify step (SuffixDecoding). Verify cost grows linearly. |
| `--dflash-drafter-path` | empty | Override the per-alias DFlash drafter HF path; empty = use the registry binding. |
| `--dflash2-drafter-path` | empty | Path/HF-id to the DFlash2 draft model (e.g. `z-lab/Qwen3.8-27B-DFlash2` or a local dir). Required with `--enable-dflash2` / `--spec-decode dflash2`. |
| `--dflash2-block-size` | 5 | Block size (draft tokens per verify step). **Must be ≤ 5** for MLX quantized targets — larger verify widths are matmul-inefficient on quantized weights. The official draft config uses 8; we cap at 5. |
| `--dspark-drafter-path` | — | Path to a converted MLX DSpark draft (from `dspark-metal-convert`). Required with `--enable-dspark`. |
| `--dspark-draft-quant-bits` | 8 | Draft quantization bits; lower = faster drafter, lower acceptance. |
| `--vlm-dev` | off | [.dev/experimental] Enable multimodal (image) input on the DSpark server's `/v1/chat/completions`. Only takes effect under `--enable-dspark` with a `qwen3_vl` target; images are dropped (with a warning) otherwise. Also set via `DSPARK_VLM_DEV=1`. |
| `--force-spec-decode` | off | Override the eligibility/auto-disable gates. Mutually exclusive with `--no-spec-decode`. |
| `--no-spec-decode` | off | Hard-disable all spec decode for this serve. Mutually exclusive with `--force-spec-decode`. |

### Per-model config

In per-model settings (`docs/configuration.md` → Per-Model Settings):

```python
{
    "specprefill_enabled": false,   # speculative prefill
    "dflash_enabled": false,        # DFlash speculative decoding
    "mtp_enabled": false,           # native MTP (Qwen3.5/3.6, DeepSeek-V4)
    "vlm_mtp_enabled": false        # VLM MTP with gemma4_assistant drafter
}
```

These mirror the CLI toggles and let you enable a method for one model without
passing the flag every serve.

---

## Method selection guide

Use this when picking manually. The [auto-router](#auto-router-specroute)
automates the same logic.

```
Is the model a VLM with a gemma4_assistant drafter available?
  → VLM MTP (vlm-mtp)

Does the model expose native MTP heads (mtp_num_hidden_layers >= 1)?
  → MTP (mtp)            # no draft-model load, good quality

Is the workload long-document / RAG (≥ ~4k prompt tokens)?
  → DFlash (ddtree)      # block drafter exploits source-text repetition

Is the workload tool-call / JSON / code-edit (high token repetition)?
  → SuffixDecoding (suffix)  # drafter-free, ~zero overhead when it misses

Is the target Qwen3 4B/8B/14B bf16 with a converted DSpark draft?
  → DSpark (dspark)      # lossless; serial single-user mode

Is the target Qwen3.8 dense (e.g. Qwen3.8-27B-4bit) with a DFlash2 draft?
  → DFlash2 (dflash2)    # block-diffusion, reads target hidden states;
                         #   2.47× speedup, accept avg 3.56 (real 27B-4bit)

Otherwise / free-form chat
  → none                 # spec decode overhead not worth it
```

The single most important question is **workload, not capability**. Enabling
SuffixDecoding on free-form chat costs ~1× (no regression — its D1-match gate
self-disables), but enabling DFlash/DSpark on free-form chat pays a draft-model
load and a verify overhead for ~no gain. Match the method to the traffic.

---

## The boot-time loading constraint

> This is the architectural fact that governs all spec-decode selection.

**Speculative-decoding methods load their state at boot time.** DFlash and
DSpark load a draft model; MTP requires a checkpoint converted through the PR
#990 `sanitize()` path that preserves `mtp.*` weights; SuffixDecoding builds
its suffix tree as tokens stream. Once a serve is up, the active method's
state is fixed for that serve's lifetime.

Consequences:

- **Selection happens at serve start, not per request.** You cannot, today,
  route request A to DFlash and request B to MTP within one running serve —
  that would require loading (or hot-swapping) draft state mid-flight.
- **Per-request cross-method routing is engine work, not a flag.**
  Boot-time `--spec-decode auto` *is* shipping — it picks `mtp` vs `suffix`
  from the model's shape at startup (see [Auto-router](#auto-router-specroute)).
  What is Phase B is *per-request* routing: switching method per request
  without a draft reload, which needs the engine refactor above.
- **Runtime tuning is already per-request.** Within the *active* method, the
  scheduler's adaptive gating (below) pauses/resumes spec decode per request
  based on observed acceptance. That is gating, not cross-method routing.

If your workload spans categories (e.g. a mix of free-form chat and tool
calls), run the method that matches the *dominant* traffic and let the
adaptive gate disable it on the hostile minority.

---

## Adaptive gating (runtime)

Within the active method, `scheduler/spec_decode.py` does per-request
**hysteresis** so a method that starts failing doesn't drag throughput down:

- Each spec step records its acceptance (`record_accepted(n_accepted, K)`).
- A sliding window (`SPEC_ADAPTIVE_WINDOW`, default 20, env
  `FUSION_SPEC_ADAPTIVE_WINDOW`) tracks the recent acceptance rate.
- If the windowed rate drops below `SPEC_MIN_ACCEPT_RATE` (default 0.05, env
  `FUSION_SPEC_MIN_ACCEPT_RATE`), the method is **paused** (`_spec_paused`)
  — subsequent steps decode without spec.
- **Phase-2 fix: the resume path.** Previously pause was a dead end — once
  paused, `should_speculate()` always returned False, so no drafts ran and
  `record_accepted()` never fired again, so the method could never
  un-pause. The fix adds a periodic **re-probe**: while paused,
  `should_speculate()` lets one spec step through every
  `SPEC_RESUME_CHECK_INTERVAL` steps (default 10, env
  `FUSION_SPEC_RESUME_CHECK_INTERVAL`). The re-probe step produces a fresh
  acceptance sample; after ≥3 probe samples, if the probe rate has recovered
  to ≥ `SPEC_MIN_ACCEPT_RATE` the method **resumes**, otherwise it stays
  paused and clears the probe set to re-sample.
- This is automatic and free; no flag required.

This gating is what makes SuffixDecoding safe to leave on by default for
agent workloads: on hostile inputs its D1-match gate and the scheduler's
pause/re-probe hysteresis combine to fall back to plain decode, so it never
regresses below baseline.

### Real-model verification (2026-08-23)

Verified on Eagle3 + Llama-3.1-8B-Instruct-4bit with forced-pause params
(`FUSION_SPEC_MIN_ACCEPT_RATE=0.5 FUSION_SPEC_ADAPTIVE_WINDOW=10
FUSION_SPEC_RESUME_CHECK_INTERVAL=3`): `PAUSING` fires at ~step 10 when the
windowed rate crosses the threshold; `paused re-probe` fires every 3 paused
steps (the dead-code fix is live); `STAYING PAUSED` repeats correctly while
the probe rate stays below threshold. At the natural 0.05 threshold with
~20% acceptance, `paused` never flips — correct, the threshold is never
crossed.

---

## Auto-router (`--spec-route`)

`fusion_mlx/speculative/auto_router.py` provides `SpecAutoRouter` — a
**deterministic, pure-Python** decision function that picks a spec-decode
method from cheap signals at request-setup time:

- `prompt_token_count` — long prompts route to DFlash.
- `has_mtp` — model-native MTP wins when available (no draft load).
- `recent_accept_rate` — hysteresis against the previous request's acceptance.
- `current_method` / `available` — the registry's config-enabled methods.

The router never invokes a model forward pass; same inputs always yield the
same method, so the whole decision table is unit-tested
(`tests/unit/test_spec_auto_router.py`).

### Decision order (`SpecAutoRouter.decide`)

1. **Abandon** a clearly-failing current method (`acceptance < abandon_accept`,
   default 0.20) — drop it and exclude it from immediate re-selection.
2. **Hysteresis** — keep the current method if it's working
   (`acceptance >= keep_accept`, default 0.40) to avoid thrashing.
3. **Long-context** — prompts ≥ `long_doc_threshold` (default 4096 tokens)
   route to DFlash.
4. **Model-native MTP** when the model exposes MTP heads.
5. **SuffixDecoding** as the cheapest default (its D1-match gate self-disables
   on hostile input, so it never regresses below baseline).
6. Degenerate fallback — anything still available, else the n-gram sentinel.

Thresholds are public dataclass fields, tunable without touching decision
code.

### API

```python
from fusion_mlx.speculative.auto_router import (
    SpecAutoRouter,
    RouteSignals,
    auto_route,
    available_methods,
)

router = SpecAutoRouter(long_doc_threshold=4096, abandon_accept=0.20, keep_accept=0.40)
method = router.decide(RouteSignals(
    prompt_token_count=8192,
    has_mtp=False,
    recent_accept_rate=0.55,
    current_method="suffix",
    available=available_methods(),   # registry methods that are config-enabled
))
# → "ddtree"  (long doc, current method working but long-doc rule overrides)
```

`available_methods()` reads the spec-decode registry and returns only methods
that are both registered and config-enabled, so the router never recommends a
method the serve can't actually provide.

### Status: ✅ boot-time auto landed · 🗺️ per-request routing is Phase B

The router is a **ready-to-call library** with full unit coverage, and
**boot-time CLI wiring has landed** via `--spec-decode auto`: at startup
`resolve_spec_auto()` inspects the model's config and picks a zero-config
method — `mtp` for MTP-eligible Qwen3.5/3.6 checkpoints (`model_type` ∈
{`qwen3_5`, `qwen3_5_moe`, `gemma4_unified`} with `mtp_num_hidden_layers ≥ 1`),
`suffix` (n-gram) otherwise. Drafter-backed methods (dflash/dspark) stay
operator-selected; auto does not duplicate their drafter-binding and
eligibility checks.

```bash
fusion-mlx serve --spec-decode auto --model mlx-community/Qwen3.5-9B-4bit
# → Spec-decode: auto → suffix (n-gram suffix decoding (safe default, ...))
```

What remains Phase B is **per-request routing** — selecting the method per
request based on prompt length and observed acceptance rate, *without* a
draft-model reload. The [boot-time loading constraint](#the-boot-time-loading-constraint)
means a running serve cannot today hot-swap draft state per request; that
engine refactor (lazy draft load where methods don't overlap, or multi-method
resident state) is Phase B. `available_methods()` and `SpecAutoRouter.decide`
remain usable from the admin panel, tests, and per-model settings.

---

## Pitfalls

- **Hybrid recurrent models auto-disable spec decode.** Qwen3.5/3.6
  (GatedDeltaNet), Granite4, Mamba/Jamba/RWKV ship with spec decode
  auto-gated off because verifying speculative tokens against a recurrent
  state corrupts output. `--force-spec-decode` overrides this — only do so if
  you have verified coherence on your specific model.
- **Earlier benchmark figures were spec-corrupted.** README figures showing
  ~29.8 tok/s (single) and ~36 tok/s (concurrent) on Qwen3.6-27B were
  measured with spec decode enabled against a hybrid model — the output was
  incoherent and the speed was not real. Coherent ceiling for that model is
  ~18.5 tok/s. See `docs/configuration.md` and the README Performance note.
- **`mtp` requires a converted checkpoint.** Passing `--spec-decode mtp`
  against an unconverted checkpoint fails at boot (loud, not silent). Convert
  via the PR #990 `sanitize()` path that preserves `mtp.*` weights.
- **DSpark is serial single-user mode.** `--enable-dspark` early-forks the
  serve path into a dedicated single-user-serial server (like audio mode) —
  it does not participate in continuous batching.
- **DFlash/DSpark draft loads cost memory.** Budget for the draft model in
  addition to the target; on memory-constrained machines prefer
  SuffixDecoding (drafter-free) or MTP (uses the target's own heads).

---

## Eagle3 (`eagle3`)

Eagle3 is a draft-model speculative decoder: a small one-layer drafter
reads the target model's multi-layer hidden states (captured at fixed
`capture_layers=[8,16,31]`) projected through `Eagle3Model.fc` (a
`Linear(3 * target_hidden_size, hidden_size)`), and proposes K draft tokens
the target verifies in one forward pass. fusion-mlx ships a custom
MLX-native Eagle3 model (`fusion_mlx/speculative/eagle3/`) instead of
`mlx_lm.load()`, which fails on Eagle3's non-standard weight keys.

### Activation

```bash
FUSION_SPEC_METHOD=eagle3 fusion-mlx serve --model mlx-community/Meta-Llama-3.1-8B-Instruct-4bit
```

Optional env knobs:

| Env | Default | Purpose |
|---|---|---|
| `FUSION_SPEC_METHOD` | `draft_model` | Selects `eagle3` vs the legacy draft-model path. |
| `FUSION_EAGLE3_DRAFT_MODEL` | `llama3.1-8b` | Draft registry key (`llama3.1-8b`, `qwen3-8b`). |
| `FUSION_EAGLE3_DRAFT_TOKENS` | `5` | K — draft tokens per verify step. |
| `FUSION_EAGLE3_DRAFT_TEMP` | `0.1` | Draft sampler temperature. A small temp (Phase-2 item 4) helps the draft distribution match the target's, improving acceptance vs greedy argmax (`0.0`). |

### Phase-2 hardening (PR #609)

Four items, all real-model verified on Eagle3 + Llama-3.1-8B-Instruct-4bit:

1. **Multi-layer hidden capture** — `HiddenStateCapture` grabs target
   hidden states at layers `[8,16,31]`, concatenated and projected via
   `fc`. Already implemented; verified `hidden_capture installed
   layers=[8, 16, 31]` in the real-model log.
2. **Family compatibility guard** — `Eagle3Speculator.is_compatible()` does
   case-insensitive substring matching on `_FAMILY_MATCHERS` (llama3/qwen3),
   including the local-path basename, and **disables spec decode** if the
   loaded target's family doesn't match the draft's (e.g. EAGLE3-LLaMA3
   against a Qwen target would produce silent garbage drafts). Verified
   `family=llama3 incompatible_block=False` for the matched Llama3 target.
3. **Adaptive pause/resume** — see [Adaptive gating](#adaptive-gating-runtime).
   The resume-path dead-code fix is the core change.
4. **Draft temperature** — default `0.0`→`0.1` (`FUSION_EAGLE3_DRAFT_TEMP`);
   surfaced in the "draft model enabled" log line.

### Code map

- `fusion_mlx/speculative/eagle3/speculator.py` — `Eagle3Speculator`,
  `is_compatible`, `Eagle3DraftConfig`
- `fusion_mlx/speculative/eagle3/model.py` — `Eagle3Model`, weight loading
  (`safe_open` keys iteration)
- `fusion_mlx/speculative/hidden_capture.py` — `HiddenStateCapture`
- `fusion_mlx/engine_core.py` — spec-init block (`_init_draft`), compat guard
- `fusion_mlx/scheduler/spec_decode.py` — `SpecDecodeState`, pause/re-probe
- `tests/unit/test_eagle3_compatibility.py` — 7 family-match tests
- `tests/unit/test_spec_decode_adaptive.py` — 7 pause/resume tests

---

## DFlash2 (block-diffusion, z-lab `dflash` pkg)

DFlash2 is the second-generation block-diffusion speculative decoder from
z-lab (PyPI `dflash==0.1.0`). A single draft forward predicts a whole
**block** of candidate tokens; a lightweight `CandidateSelector` traces one
coherent path through them, and two-tap grouped-dynamic convolutions
(`GroupedDynamicCausalConv` / GDN) keep the draft from decaying across the
block. The draft reads the target model's hidden states at fixed
`target_layer_ids` (e.g. `[5, 19, 33, 47, 61]` for the 27B draft), so it is
quality-matched to the target — not a separate small model.

Unlike DFlash v1 and DSpark, DFlash2 does **not** fork a dedicated server. It
loads in-place via `BatchedEngine`'s load branch and participates in normal
continuous batching as a self-contained generator.

### Activation

```bash
fusion-mlx serve --model qwen3.8-27b-4bit \
  --enable-dflash2 \
  --dflash2-drafter-path /path/to/Qwen3.8-27B-DFlash2 \
  --dflash2-block-size 5
# or
fusion-mlx serve --model qwen3.8-27b-4bit --spec-decode dflash2 \
  --dflash2-drafter-path z-lab/Qwen3.8-27B-DFlash2
```

### Constraints

- **Target family**: `qwen3_8` dense (non-MoE). The auto-router routes the
  `qwen3_8` family to `dflash2` first, with a `suffix` (n-gram) fallback.
- **block_size ≤ 5** for MLX quantized targets. The official draft config
  ships `block_size=8`; larger verify widths are matmul-inefficient on
  quantized weights, so fusion-mlx caps at 5. `load_runtime` rejects
  `block_size > 5`.
- **Drafter path**: a local directory (e.g. `~/.fusion-mlx/models/Qwen3.8-27B-DFlash2`)
  or an HF repo id. The bridge short-circuits `huggingface_hub.snapshot_download`
  for local dirs so drafts load from disk (no re-download, honors the
  hf-mirror workflow). Requires `dflash` extra: `pip install 'fusion-mlx[dflash2]'`.

### Real-model validation (2026-08-21)

Target `Qwen3.8-27B-4bit` + draft `Qwen3.8-27B-DFlash2`, `block_size=5`,
greedy (`temperature=0`), prompt "The capital of France is", 64 tokens:

| Metric | Value |
|---|---|
| Throughput | **52.3 tok/s** (18 verify steps, 1.22s) |
| Baseline (target greedy, no spec) | 21.2 tok/s (3.02s) |
| Speedup | **2.47×** |
| Accept length (avg per verify step) | **3.556** (range 1–5) |
| Lossless | **PASS** — tail tokens identical to baseline from index 1; decoded content matches (only first-token leading-space differs, a dflash detokenizer join-space artifact) |

Memory: target + draft in-place load (~15 GB target + ~3.7 GB draft); M5 Max
64 GB OK. No forked server.

### Code map

- `fusion_mlx/speculative/dflash2/` — bridge package (`runtime.py`, `eligibility.py`, `engine/generator.py`)
- `fusion_mlx/scheduler/spec_decode.py` — `dflash2_spec_step`, `DFlash2SpecState`
- `fusion_mlx/speculative/auto_router.py` — `METHOD_DFLASH2`, `qwen3_8` routing
- `fusion_mlx/engines/batched.py` — in-place DFlash2 load branch
- `tests/unit/test_dflash2_{eligibility,runtime,integration}.py` — 27 tests

---

## Reference

- `fusion_mlx/speculative/registry.py` — method registry, canonical names and aliases
- `fusion_mlx/speculative/auto_router.py` — `SpecAutoRouter`, `RouteSignals`, `available_methods`
- `fusion_mlx/scheduler/spec_decode.py` — adaptive pause/resume hysteresis, `SPEC_MIN_ACCEPT_RATE`, `DraftStats`
- `tests/unit/test_spec_auto_router.py` — full decision-table coverage
- `tests/unit/test_dflash2_*.py` — DFlash2 bridge + integration tests
- `docs/cli-reference.md` — `serve` flags, including all spec-decode toggles
- `docs/configuration.md` — per-model spec settings
