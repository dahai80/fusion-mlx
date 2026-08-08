# Roadmap

> Last updated 2026-08-08 (v0.8.12). Drives from the catch-up plan at
> [`/Users/dahai/fusion/architecture/fusion-mlx-enhance.md`](file:///Users/dahai/fusion/architecture/fusion-mlx-enhance.md).
>
> fusion-mlx is **Apple Silicon only** and bets on the MLX ecosystem.
> We do not compete with llama.cpp on cross-platform/ggml breadth or
> with rapid-mlx on raw LLM throughput. We compete on **full-modality
> local serving** (image/video/STS/NER/reranker + training + Ollama
> compat) — the capabilities the others structurally don't have.

## Strategic moats (defend & amplify)

These are landed today and unmatched by llama.cpp / rapid-mlx / oMLX:

- **Full-modality engines** — 11 engine classes (LLM/VLM/Embedding/
  Reranker/STT/TTS/STS/ImageGen/VideoGen/NER/OCR).
- **Video generation** — 10 native MLX backends + VACE-14B E2E + IP-Adapter
  /ControlNet/AnimateDiff adapters.
- **Ollama protocol compat** — the only MLX server with `/api/generate`
  `/api/chat` `/api/tags` drop-in.
- **Speculative decoding** — 10 methods incl. EAGLE3 (1.445x measured).
- **Training** — LoRA/DPO/GRPO/Reward + in-place swap + HF→MLX wizard.
- **Paged KV + SSD cold tier** + 3-tier priority scheduling.

## Status legend

- ✅ done · 🚧 in progress · 📋 planned · ⛔ won't do (out of scope)

## Near-term (Phase 0–1, 0–4 weeks)

| Item | Status | Note |
|------|--------|------|
| Classifier Alpha → Beta | 🚧 | this PR |
| Governance docs (RELEASE/CONTRIBUTING/ROADMAP) | 🚧 | this PR |
| README badges + scope/maturity statement | 🚧 | this PR |
| Test-debt cleanup (301 quarantined → rescue/mark/delete) | 📋 | target active ≥600 |
| Public benchmark harness + ≥10 model reports | 📋 | `benchmarks/`, reuse `admin/benchmark` |
| Model compatibility matrix (public) | 📋 | "runs / limited / no" per model |
| GGUF load guard | ✅ | `engine/gguf_guard.py` (#423, v0.8.12) |
| GGUF→MLX load bridge | 📋 | guard done; runtime weight-mapping bridge next |
| tool_calling parser expansion (Gemma4/Hermes/Mistral/MiniMax/ui_tars) | 📋 | split `tool_calling.py` per family |

## Mid-term (Phase 2, 1–3 months)

| Item | Status | Note |
|------|--------|------|
| Resumable streaming (`/v1/stream` + lookup) | 📋 | like llama.cpp stream_session |
| Spec-decoding metrics (draft accept rate) | 📋 | `/metrics` or `spec_routes` |
| DFly/DSpark maturity convergence | 📋 | ~1.5KB vs dflash 29KB — fill or mark experimental |
| MLA/DSA dedicated KV path | 📋 | for DeepSeek/GLM, in `cache/paged_cache.py` |
| Telemetry framework | 📋 | consent/emit/queue/redact/schema |
| Dependency extras split (`[full]` default) | 📋 | text-only saves ~322MB |
| Sigstore / PEP 740 attestation | 📋 | PyPI provenance |

## Long-term (Phase 3, 3–6 months)

| Item | Status | Note |
|------|--------|------|
| Video benchmark + E2E tests + docs | 📋 | make "video tier-1" a verifiable claim |
| Full-modality as headline positioning | 📋 | README/landing rewrite |
| Ultra-low-bit quant (1.5–2bit / TQ / imatrix) evaluation | 📋 | build or bridge or document the boundary |
| homebrew-core inclusion | 📋 | replace self-maintained tap; needs tests+docs+stable API |

## Out of scope (won't do)

| Item | Why |
|------|-----|
| ⛔ Cross-platform backends (CUDA/Linux/Windows) | MLX-native bet; Apple Silicon is the scope |
| ⛔ Self-built 140-arch model enum / HF converter | follow MLX upstream + GGUF bridge |
| ⛔ Self-built GGUF quantization | llama.cpp is the standard; we load, not quantize |
| ⛔ MXFP8 mixed-precision training | mlx-lm 0.31.3 has no fp8 train path (#425 upstream-blocked); fail-visible raise stays |
| ⛔ Compete on raw LLM throughput vs rapid-mlx | their moat; we win on modality/training/Ollama |

## Supported models (live matrix)

A "runs / limited / no" table per model is a near-term goal (Phase 1).
Until then, fusion-mlx supports whatever **mlx-lm 0.31.3 / mlx-vlm
0.5.0** support, plus vendored model code under `fusion_mlx/patches/`
and `fusion_mlx/models/`. GGUF files are rejected with a clear error
pointing at `mlx-community` repos or `POST /v1/convert`.
