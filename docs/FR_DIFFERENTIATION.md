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

| FR theme | fusion-mlx status | omlx / Ollama / vLLM-mac |
|---|---|---|
| Speculative decoding | **5 methods** (Suffix, DFlash, DSpark, MTP, VLM-MTP) | ❌ none ship a spec path |
| TurboQuant | Runtime KV-cache quant (`--kv-cache-turboquant {v4,k8v4}`) | omlx has KV quant; others don't |
| Continuous batching | vLLM-style scheduler, 25 modules | omlx yes; Ollama no |
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

### A.2 / Phase B — `--spec-route auto` engine wiring 🗺️

The router logic is landed and tested (A.1). Wiring it to a `--spec-route auto`
CLI flag requires the engine refactor dictated by the boot-time loading
constraint: the request-setup path must select the method per request without a
draft-model reload where methods overlap, or with a lazy draft load where they
don't. This is real engine work, scoped as Phase B. Until then the router is
usable as a library / admin-panel recommendation.

### B.2 — LoRA hot-swap 🗺️

Per-request LoRA adapter hot-swap. Scoped as Phase B alongside spec routing
(both touch the engine's per-request setup path).

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
are A.2/B engine wiring, B.2 LoRA, and Phase C kernels, in that order.

---

## Changelog

- **2026-07-07** — Initial analysis. Landed A.1 (spec auto-router library) and
  B.1 (`convert` CLI). Re-scoped A.2 to Phase B engine work. Corrected the
  TurboQuant weight-format premise.
