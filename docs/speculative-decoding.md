# Speculative Decoding

fusion-mlx ships **five speculative-decoding algorithms** — SuffixDecoding,
DFlash, DSpark, MTP, and VLM MTP — selectable per serve. This is the single
largest differentiator vs. omlx / Ollama / vLLM-mac, none of which ship a
spec-decode path. This document is the authoritative reference: which method to
pick, how to activate it, the architectural constraint that governs selection,
and the auto-router that automates the choice.

> **Status legend** — ✅ shipping and verified · 🧪 shipped but PoC / workload-gated · 🗺️ library landed, CLI wiring is Phase B (see [Auto-router](#auto-router-specroute)).

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
fusion-mlx serve --model <model> --spec-decode {none,mtp,dflash,dspark}
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
--enable-dspark            # DSpark lossless block spec-decode (serial mode)
```

### Method-specific knobs

| Flag | Default | Purpose |
|---|---|---|
| `--suffix-max-draft` | 8 | Max draft tokens per verify step (SuffixDecoding). Verify cost grows linearly. |
| `--dflash-drafter-path` | empty | Override the per-alias DFlash drafter HF path; empty = use the registry binding. |
| `--dspark-drafter-path` | — | Path to a converted MLX DSpark draft (from `dspark-metal-convert`). Required with `--enable-dspark`. |
| `--dspark-draft-quant-bits` | 8 | Draft quantization bits; lower = faster drafter, lower acceptance. |
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
- **Per-request cross-method routing is engine work, not a flag.** This is
  why `--spec-route auto` is not yet a shipping CLI flag — see
  [Auto-router](#auto-router-specroute). The decision logic is landed and
  tested; wiring it into the engine's request setup path (so a running serve
  can switch method per request without a draft reload) is Phase B.
- **Runtime tuning is already per-request.** Within the *active* method, the
  scheduler's adaptive gating (below) pauses/resumes spec decode per request
  based on observed acceptance. That is gating, not cross-method routing.

If your workload spans categories (e.g. a mix of free-form chat and tool
calls), run the method that matches the *dominant* traffic and let the
adaptive gate disable it on the hostile minority.

---

## Adaptive gating (runtime)

Within the active method, `scheduler/spec_decode.py` already does per-method
**hysteresis** so a method that starts failing doesn't drag throughput down:

- Each method tracks its acceptance rate (`DraftStats.acceptance_rate` =
  accepted/proposed).
- If acceptance drops below `SPEC_MIN_ACCEPT_RATE`, the method is **paused**
  (`_spec_paused`) — subsequent requests decode without spec until the rate
  recovers, then it **resumes**.
- This is automatic and free; no flag required.

This gating is what makes SuffixDecoding safe to leave on by default for
agent workloads: on hostile inputs its D1-match gate and the scheduler's
pause hysteresis combine to fall back to plain decode, so it never regresses
below baseline.

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

### Status: 🗺️ library landed, CLI wiring is Phase B

The router is a **ready-to-call library** with full unit coverage. It is **not
yet wired** to a `--spec-route auto` CLI flag, because of the
[boot-time loading constraint](#the-boot-time-loading-constraint): a running
serve cannot today hot-swap draft state per request. Wiring the router into
the engine's request-setup path — so the chosen method is selected per request
*without* a draft-model reload where the methods overlap, or with a lazy
draft load where they don't — is the Phase B engine refactor. Until then,
`available_methods()` and `SpecAutoRouter.decide` are usable from the admin
panel, tests, and per-model settings as a recommendation/preview of the
upcoming behavior.

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

## Reference

- `fusion_mlx/speculative/registry.py` — method registry, canonical names and aliases
- `fusion_mlx/speculative/auto_router.py` — `SpecAutoRouter`, `RouteSignals`, `available_methods`
- `fusion_mlx/scheduler/spec_decode.py` — adaptive pause/resume hysteresis, `SPEC_MIN_ACCEPT_RATE`, `DraftStats`
- `tests/unit/test_spec_auto_router.py` — full decision-table coverage
- `docs/cli-reference.md` — `serve` flags, including all spec-decode toggles
- `docs/configuration.md` — per-model spec settings
