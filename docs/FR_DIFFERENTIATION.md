# FR Differentiation Analysis (2026-07-07)

**FR reviewed:** "Full-stack optimization enhancement for Mac MLX — complete
TurboQuant/DFlash/DSpark ecosystem, hardware adaptive scheduling & production
serving" (~20 P0/P1/P2 items).

**Verdict:** fusion-mlx is **already differentiated** on the FR's headline
themes. Roughly 60% of the FR's premises were either already implemented or
factually wrong about the codebase. This document records the verified
findings, what landed in response, and what is honestly deferred — so future
work targets real gaps instead of re-asking answered questions.

This is an engineering record, not marketing. Every claim below was checked
against the source.

---

## What fusion-mlx already differentiates on

The FR positioned fusion-mlx as needing to "catch up" on spec decode,
TurboQuant, and scheduling. In fact:

| FR theme | fusion-mlx status | upstream fusion-mlx / Ollama / vLLM-mac |
|---|---|---|
| Speculative decoding | **5 methods** (Suffix, DFlash, DSpark, MTP, VLM-MTP) | ❌ none ship a spec path |
| TurboQuant | Runtime KV-cache quant (`--kv-cache-turboquant {v4,k8v4}`) | upstream fusion-mlx has KV quant; others don't |
| Continuous batching | vLLM-style scheduler, 25 modules | upstream fusion-mlx yes; Ollama no |
| 2-bit quant recipes | quant2 family, up to +167% decode speed | none |
| Paged KV cache + SSD cold layer | yes | no |
| Hardware-adaptive scheduling | per-method adaptive gating, 4-tier memory enforcer | partial / none |

See `docs/speculative-decoding.md` for the spec-decode surface and
`README.md` for the full feature matrix.

---

## FR premises that were wrong (verified against code)

### 1. "TurboQuant is a weight format — ship an HF→TurboQuant convert CLI"

**Wrong.** TurboQuant in fusion-mlx is **runtime KV-cache quantization**, not
a weight format. The implementation is `fusion_mlx/turboquant_kv.py` →
`TurboQuantKVCache`, applied at serve time via `--kv-cache-turboquant {v4,k8v4}`.
`mlx_vlm.turboquant` exports only the KV-cache class — there is no
`convert`/`quantize` function to wrap.

A "TurboQuant weight convert CLI" cannot exist because TurboQuant isn't a
weight format. The FR conflated two unrelated things:

- **Weight quantization** (saved to disk): GGUF, Imatrix, MLX mxfp4/mxfp8,
  the quant2/mixed-bit recipes — these ARE weight formats and ARE convertible.
- **Runtime KV-cache quantization** (in memory): TurboQuant — a serve-time
  knob, not a disk format.

What **did** land (correctly scoped): a `fusion-mlx convert` subcommand
wrapping `mlx_lm.convert` for the *weight* formats. See `docs/cli-reference.md`
→ `convert`, and the implementation in `fusion_mlx/cli_convert.py`.

### 2. "Per-request spec-decode method routing is a stopgap"

**Wrong scope.** All spec-decode methods load state at **boot time**: DFlash
and DSpark load draft models, MTP needs a converted checkpoint, SuffixDecoding
builds its tree as tokens stream. Therefore per-request *cross-method* routing
requires an **engine refactor**, not a stopgap flag. A boot-time-only
"resolver" would be too weak (Rule 2 — minimum code that solves the problem),
and a per-request hot-swap flag can't work without draft-state reload.

This is documented in detail in `docs/speculative-decoding.md` →
[The boot-time loading constraint](speculative-decoding.md#the-boot-time-loading-constraint).

---

## What landed (this work, 2026-07-07)

### A.1 — Spec auto-router library ✅

`fusion_mlx/speculative/auto_router.py`: `SpecAutoRouter`, a deterministic
pure-Python decision function that picks a spec-decode method from cheap
signals (prompt length, model MTP capability, prior acceptance rate). Full
unit coverage in `tests/unit/test_spec_auto_router.py` (20 tests across fresh
selection, hysteresis, abandon, degenerate fallback, determinism, threshold
tuning, registry integration).

- Canonical method names verified against `registry.py` (`ddtree`/`mtp`/
  `suffix`/`dspark`, NOT the aliases `dflash`/`ngram`).
- Decision order: abandon failing method → hysteresis keep → long-doc→DFlash
  → model-native MTP → n-gram default → degenerate fallback.
- Public thresholds (`long_doc_threshold`, `abandon_accept`, `keep_accept`)
  tunable without touching decision code.

### B.1 — `fusion-mlx convert` CLI ✅

`fusion_mlx/cli_convert.py` + `cli.py` wiring: wraps `mlx_lm.convert` with
model-alias resolution, `--quant-bits {2,3,4,6,8}`, `--quant-group-size`,
`--quant-mode`, `--dtype`, `--dequantize`, `--upload-repo`,
`--trust-remote-code`, `-o/--out`. Corrects the FR's TurboQuant premise by
documenting the weight-vs-runtime-KV distinction in `docs/cli-reference.md`.
11 unit tests in `tests/unit/test_cli_convert.py`.

---

## What is honestly deferred

### A.2 / Phase B — spec-decode auto wiring

**Boot-time auto: ✅ landed** (`--spec-decode auto`, commit d6d85e84).
`fusion_mlx/speculative/auto_resolve.py` wires `SpecAutoRouter` into the CLI
serve path: at boot it probes the model's MTP eligibility and lets the router
pick `mtp` (MTP-eligible) or `suffix` (safe default, zero GPU cost). The
router library from A.1 now has its first caller. 7 unit tests in
`tests/unit/test_spec_auto_resolve.py`; E2E verified on Qwen3.5-9B-4bit
(banner: `Spec-decode: auto → suffix`).

**Per-request routing: 🗺️ Phase B (engine refactor).** Switching the spec
method per request — not just at boot — requires the engine refactor dictated
by the boot-time loading constraint: the request-setup path must select the
method per request without a draft-model reload where methods overlap, or with
a lazy draft load where they don't. Boot-time auto lands the zero-config
happy path; per-request cross-method routing remains real engine work.

### B.2 — LoRA hot-swap

**Boot-time LoRA: ✅ landed** (`--lora-path`, Phase B LoRA slice 1).
`serve --model <X> --lora-path <adapter-dir>` applies a PEFT LoRA adapter at
boot via `mlx_lm.load(adapter_path=...)`. Threads through `server.load_model`
→ `_pending_single_model` → `BatchedEngine.__init__` → `_load_model_sync`.

**Multi-model per-model LoRA: ✅ landed** (Phase B LoRA slice 2). `lora_path`
is now a `ModelSettings` field registered in `MODEL_SPECIFIC_PROFILE_FIELDS`,
so in `--model-dir` mode each alias/profile can specify its own adapter. The
engine pool threads `model_settings.lora_path` into `BatchedEngine.__init__`
at all three construction sites. 11 unit tests in `tests/unit/test_lora_path.py`.

**Runtime hot-swap: ✅ landed (path A' — adapter-keyed engine cache).**
Per-request LoRA adapter hot-swap is now served via the OpenAI-compatible
`adapters` field on `ChatCompletionRequest` / `CompletionRequest` and the
Anthropic `MessagesRequest` (mlx-lm server-compatible). `get_engine(model, adapter_path=...)` derives an engine
entry keyed by `(model_id, adapter_path)` (`f"{model_id}::lora::{adapter_path}"`
when an adapter is set, plain `model_id` otherwise), lazily creating a derived
`EngineEntry` that clones the base model's path/type/size/config and loads a
fresh `BatchedEngine(... lora_path=adapter_path)`. Each adapter thus gets its
own loaded instance, reused across requests via the existing LRU; release
decrements the derived entry's `in_use`. An `FUSION_MAX_ADAPTER_ENGINES` cap
(default 4) LRU-evicts idle derived adapter engines. 13 unit tests in
`tests/unit/test_lora_hotswap.py`; Anthropic route threading covered by 5
tests in `tests/unit/test_anthropic_adapter_wiring.py`.

> Premise correction: an earlier draft of this section claimed "mlx_lm fuses
> the adapter into weights at load time, so hot-swap needs an unload+reload
> path." That is wrong — `mlx_lm.load(adapter_path=...)` wraps `LoRALinear`
> (base weight + LoRA delta as separate, removable layers); it does **not**
> fuse. Runtime hot-swap is therefore viable without an unload+reload of the
> base, and path A' ships it.

### Phase C — kernel work 🗺️

Fused GDN (GatedDeltaNet) megakernel and fused dequant SDPA. These are
upstream/kernel-blocked: `mx.compile` was measured non-viable, and the fused
projection paths already ship for QKV/gate. Not promised on a timeline.

---

## How to use this document

When a future FR or review re-asks "does fusion-mlx do spec decode /
TurboQuant / scheduling," point at the **Already differentiates** table and
the spec-decoding doc. When someone proposes a TurboQuant convert CLI or a
per-request spec-routing stopgap, point at the **wrong premises** section —
both were checked against the source and re-scoped. The remaining real gaps
are per-request spec routing (Phase B engine refactor) and Phase C kernels, in
that order. (Runtime LoRA hot-swap, formerly listed here, landed via path A'.)
---

## Changelog

- **2026-07-07 (update 4)** — Landed runtime LoRA hot-swap (B.2 path A',
  adapter-keyed engine cache): OpenAI-compatible `adapters` field on
  chat/completion requests routes to a derived `EngineEntry` keyed by
  `(model_id, adapter_path)`, lazily loaded with
  `BatchedEngine(... lora_path=...)` and reused via the existing LRU;
  `FUSION_MAX_ADAPTER_ENGINES` (default 4) caps idle derived adapter engines.
  13 unit tests in `tests/unit/test_lora_hotswap.py`. Corrected the false
  "mlx_lm fuses the adapter at load time" premise — it wraps `LoRALinear`,
  not fuses, so hot-swap is viable without a base reload.
- **2026-07-07 (update 3)** — Landed multi-model per-model LoRA (Phase B LoRA
  slice 2): `lora_path` is now a `ModelSettings` field in
  `MODEL_SPECIFIC_PROFILE_FIELDS`, so each alias/profile in `--model-dir` mode
  can specify its own adapter; engine pool threads it into engine construction.
  Runtime hot-swap remains Phase B.
- **2026-07-07 (update 2)** — Landed boot-time `--lora-path` (B.2 boot-time
  slice, Phase B LoRA slice 1): `serve --model <X> --lora-path <adapter>`
  applies a PEFT LoRA adapter at boot via `mlx_lm.load(adapter_path=...)`.
  Runtime hot-swap and multi-model per-model LoRA remain Phase B.
- **2026-07-07 (update)** — Landed boot-time `--spec-decode auto` (A.2
  boot-time slice, commit d6d85e84): `SpecAutoRouter` now wired into the CLI
  serve path with MTP-eligibility probing. Per-request routing remains Phase B
  engine work.
- **2026-07-07** — Initial analysis. Landed A.1 (spec auto-router library) and
  B.1 (`convert` CLI). Re-scoped A.2 to Phase B engine work. Corrected the
  TurboQuant weight-format premise.
