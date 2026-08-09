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
| Classifier Alpha → Beta | ✅ | `pyproject` (1d42d34) |
| Governance docs (RELEASE/CONTRIBUTING/ROADMAP) | ✅ | 1d42d34 |
| README badges + scope/maturity statement | ✅ | 1d42d34 |
| Test-debt cleanup (quarantined → rescue) | 🚧 | 15 reactivated, 8742→9105 items (cf8c1ea); more to rescue |
| tool_parser coverage (boundary bugs) | ✅ | 21 parsers + mlx-lm native; ui_tars 24→6 fail (c4d9af6) |
| Public benchmark harness + ≥10 model reports | 📋 | `benchmarks/`, reuse `admin/benchmark` |
| Model compatibility matrix (public) | ✅ | live table below (this commit) |
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

Status: ✅ **Tested** (has a fusion alias, covered by the test suite) ·
🟡 **Custom patch** (runs via vendored `fusion_mlx/patches/` — cutting-edge
arch, may carry caveats) · 🟦 **Upstream** (supported by mlx-lm 0.31.3 /
mlx-vlm 0.5.0, runs but no fusion alias — not individually tested by us) ·
❌ **No** (GGUF rejected with a clear error; or arch not in upstream/vendored).

GGUF files are rejected at load by `engine/gguf_guard.py` (#423, v0.8.12)
with an error pointing at `mlx-community` repos or `POST /v1/convert`.

### Text LLMs (mlx-lm)

| Family | Status | Tool parser | Spec decode | Alias example |
|--------|--------|-------------|-------------|---------------|
| Qwen3 / 3.5 / 3.6 / 3-Coder | ✅ | hermes / qwen3_coder_xml | most | `qwen3.6-27b-4bit` |
| DeepSeek-R1 | ✅ | deepseek | ✅ | `deepseek-r1-7b-4bit` |
| DeepSeek-V3 / V4 | ✅ + 🟡 patch | deepseek / deepseek_v3 | ✅ | `deepseek-v4-27b` |
| Gemma 3 / 4 | ✅ | gemma4 / hermes | ✅ | `gemma-4-4b-4bit` |
| Llama 3 / 4 | ✅ + 🟡 patch (`llama4_attention`) | llama | ✅ | `llama4-8b-4bit` |
| GLM-4 / GLM-MoE | ✅ + 🟡 patch (`glm_moe_dsa`) | glm47 | ✅ | `glm-4-9b-4bit` |
| Phi-3.5 / 4 | ✅ | hermes | ✅ | `phi-4-4bit` |
| Mistral / Magistral / Ministral | ✅ | hermes | ✅ | `mistral-24b-4bit` |
| MiniMax-M2.5 | ✅ + 🟡 patch (`minimax_m3_sparse_attention`) | minimax | ✅ | `minimax-m2.5-4bit` |
| Kimi-K2 | ✅ | kimi | ✅ | `kimi-k2-4bit` |
| Nemotron | ✅ | hermes | ✅ | `nemotron-30b-4bit` |
| gpt-oss | ✅ | harmony | ✅ | `gpt-oss-20b-mxfp4-q8` |
| Hermes 3 | ✅ | hermes | ✅ | `hermes3-8b-4bit` |
| Granite 4 | ✅ | hermes | — | `granite-4-4bit` |
| Bonsai / Devstral / SmolLM3 / Nanbeige / VibeThinker / Qwopus | ✅ | hermes | varies | `smollm3-3b-4bit` |
| Other mlx-lm arches (119 total) | 🟦 | native `tokenizer.tool_parser` | — | no alias |

### Vision LLMs (mlx-vlm 0.5.0)

| Family | Status | Note |
|--------|--------|------|
| Qwen2-VL / 2.5-VL / 3-VL / 3.5 / 3-Omni | 🟦 | mlx-vlm native |
| Llama 4 / mllama | 🟦 + 🟡 patch | `llama4_attention` |
| Gemma 3 / 3n / 4 | 🟦 | mlx-vlm native |
| GLM-4V / GLM-4V-MoE / GLM-OCR | 🟦 | mlx-vlm native |
| InternVL / Idefics2-3 / Pixtral / Molmo / Phi3-V / Phi4MM | 🟦 | mlx-vlm native |
| MiniCPM-V 4.6 / MiniCPM-o | 🟦 | mlx-vlm native |
| Kimi-K25 / Kimi-VL | 🟦 | mlx-vlm native |
| DeepSeek-VL-V2 / DeepSeekOCR / Florence2 / PaliGemma | 🟦 | mlx-vlm native |
| SAM3 / RFDetr / OCR family | 🟦 | mlx-vlm native |
| Qwen3.6 nested visual | 🟡 patch | `qwen3_6_nested_visual` |

### Specialized modalities

| Family | Status | Engine | Alias example |
|--------|--------|--------|---------------|
| bge-m3 (embedding) | ✅ | Embedding | `bge-m3-4bit` |
| xlm-roberta (reranker/NER) | ✅ | Reranker / NER | — |
| Diffusion-Gemma (text-diffusion) | ✅ | LLM-diffusion | `diffusion-gemma-26b-4bit` |
| UI-TARS (computer-use agent) | ✅ | LLM + `ui_tars` parser | `ui-tars-1.5-7b-4bit` |
| TurboQuant attention | 🟡 patch | quantized-arch accel | — |
| Step3p7 | 🟡 patch | vendored arch | — |

> **Counts**: 81 fusion aliases across ~24 families; mlx-lm 0.31.3 ships
> 119 LLM arches and mlx-vlm 0.5.0 ~60 VLM arches. Aliases = the subset
> we register, document, and run in CI. Upstream-only arches run but are
> not individually verified — add an alias + test to promote one to ✅.
