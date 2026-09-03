<div align="center">

# fusion-mlx

**Unified local model serving for Apple Silicon**

Drop-in replacement for Ollama / vLLM - runs natively on Metal via MLX

[![Version](https://img.shields.io/pypi/v/fusion-mlx?label=version&color=blue)](https://pypi.org/project/fusion-mlx/)
[![Python](https://img.shields.io/badge/Python-3.11+-3776AB.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-Apache--2.0-green.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/Tests-431%20files%20%7C%208742%20items-success.svg)](tests/)
[![CI](https://github.com/dahai80/fusion-mlx/actions/workflows/ci.yml/badge.svg)](https://github.com/dahai80/fusion-mlx/actions/workflows/ci.yml)
[![GitHub stars](https://img.shields.io/github/stars/dahai80/fusion-mlx?style=social)](https://github.com/dahai80/fusion-mlx/stargazers)

[English](README.md) | [Chinese](README_CN.md)

[Get Started](#quick-start) · [Download App](https://github.com/dahai80/fusion-mlx/releases) · [Benchmarks](https://bench.dpdns.org/) · [Documentation](docs/)

</div>

> **Scope & maturity**: macOS / Apple Silicon only (MLX-native). Beta —
> single-maintainer project, [seeking contributors](CONTRIBUTING.md). See
> [ROADMAP.md](ROADMAP.md) for the full-modality plan and supported models.

---

## Why fusion-mlx?

fusion-mlx doesn't just port existing runtimes to Metal - it builds capabilities
that are only possible on Apple Silicon's unified memory (UMA) and that the
x86+CUDA stack structurally cannot match. These are **landed and running today**:

- **UMA Radix text-KV cache (#178)** - radix-tree + LRU + pin/unpin over
  diffusion text encoders (UMT5/CLIP) with zero-copy reuse. Repeated prompts
  across multi-shot pipelines encode once; `/v1/cache/stats` surfaces it.
- **DSpark speculative decode, vendored for MLX (#190)** - 1.47× validated
  end-to-end on real 14B (`serve --enable-dspark`); the speculative win the LLM
  side already has.
- **DFlash2 block-diffusion spec decode (z-lab `dflash` pkg)** - 2.47× validated
  on real `Qwen3.8-27B-4bit` (`serve --enable-dflash2`); accept avg 3.56,
  greedy lossless; reads target hidden states, no forked server.
- **Speculative denoise (#177) — FALSIFIED, default off**: the diffusion analog of
  speculative decoding was tested on real 14B DiT and honestly **falsified**
  (0% acceptance, 0.42× slower, quality breaks). The machinery remains
  env-gated off for future research; the negative result is documented in
  `SPECULATIVE_DENOISE.md`.
- **Fusion-ComfyUI Stage API + `on_step` (#170-172)** - 10 stage methods across
  text-encoder / DiT / VAE plus a thread->async `on_step` bridge; native ComfyUI
  integration no other MLX server offers.
- **SkyReels-V3 full family + upstream arch fixes (#164/#168/#193)** -
  R2V/V2V/A2V/A2W all run end-to-end on real 14B weights; fixed upstream config
  bugs (cross_attn_type routing, norm affine) that otherwise broke the model.
- **Flux2 Klein + `mx.compile` (#166)** - 1.9× (1.56s/step) with raw-diffusers
  Flux2 auto-detect.
- **SD3-Medium native MLX (#369)** - from-scratch MMDiT + AutoencoderKL +
  CLIP-L/CLIP-G/T5-XXL tri-encoder pipeline (reuses mflux T5+CLIP-L). See
  [docs/sd3-image.md](docs/sd3-image.md).
- **SDXL native MLX (#371)** - from-scratch UNet2DConditionModel +
  AutoencoderKL + dual CLIP-L/OpenCLIP-G encoders (2048 cross-attn dim),
  Euler-discrete epsilon scheduler. Covers sdxl / cosxl / sdxs variants.
  See [docs/sdxl-image.md](docs/sdxl-image.md).
- **Stable Cascade native MLX (#370)** - from-scratch Würstchen 3-stage
  pipeline: prior → decoder (unified `StableCascadeUNet` by
  `switch_level`) → VQGAN, DDPM-Würstchen scheduler, CLIP-ViT-bigG.
  See [docs/cascade-image.md](docs/cascade-image.md).
- **Windows CUDA backend node (#365)** - optional vLLM-powered OpenAI-compatible
  node for heavy LLM inference (DeepSeek 70B / Qwen 72B FP8) on Windows CUDA,
  self-registering `platform=windows-cuda` over mDNS for fusion-gateway
  platform routing. See [docs/cuda-node.md](docs/cuda-node.md).
- **Metal Flash Attention (MFA) (#86)** - vendored Metal kernels for DiT
  attention (LTX-2, Wan2).

**Phase-2 LANDED: UMA Radix *Latent* cache** - the radix cache extends
from text KV to video frame latents. Phase-1: repeat I2V requests reuse the
input-image's VAE-encoded latent with zero-copy `mx.array` pointer sharing,
skipping the VAE load + forward (LTX-2, Wan2.2). Phase-2: multi-shot
pipeline's previous tail-frame latent is reused as the next shot's first-frame
latent, skipping VAE decode→re-encode on UMA. `session_id` parameter on
`/v1/videos/generate` enables multi-shot continuity. See
[cache/LATENT_CACHE.md](fusion_mlx/cache/LATENT_CACHE.md).
Env: `FUSION_SESSION_TAIL_CACHE=1` (default OFF until E2E validated).

**Benchmark** (Qwen3.6-27B, Apple M2 Ultra 137GB):

| Quantization | Model Size | bpw | Decode Speed | vs mxfp8 | vs mixed_3_4 |
|---|---|---|---|---|---|
| mxfp8 | 26 GB | 8.0 | 18.5 tok/s | baseline | - |
| mxfp4 | 13 GB | 4.0 | 32.3 tok/s | **+75%** | - |
| mixed_4_6 | 15 GB | 4.85 | 29.0 tok/s | **+57%** | - |
| mixed_3_4 | 12 GB | 3.68 | 36.2 tok/s | **+96%** | baseline |
| mixed_2_6 | 10 GB | 3.25 | 39.3 tok/s | **+112%** | +9% |
| mixed_2_4 | 9.3 GB | 2.95 | 42.8 tok/s | **+131%** | +18% |
| quant2 | 8.5 GB | 2.72 | 45.1 tok/s | **+144%** | +25% |
| quant2-g128 | 7.8 GB | 2.46 | 48.2 tok/s | **+161%** | +33% |
| quant2-all | 7.5 GB | 2.37 | 48.5 tok/s | **+162%** | **+34%** |
| quant2-flat | 7.1 GB | 2.25 | 49.4 tok/s | **+167%** | +36%* |

*\*quant2-flat: max speed but 2-bit embeddings degrade quality. Use quant2-all for best quality/speed tradeoff.*

Key optimizations: quant2/quant2_128/quant2_flat ultra-aggressive 2-bit quantization recipes, mixed-bit quantization (bandwidth reduction), greedy decode fast path (skip logsumexp for argmax), fused QKV/gate projections, fused decode sampler, async_eval double-buffering, GatedDeltaNet linear attention fast path, StreamingJSONEncoder, B=1 fast path.

## Features

- **9 engine types** - LLM, VLM, Embedding, Reranker, STT, TTS, STS, ImageGen (Flux 2), VideoGen (LTX-2, Wan2, SkyReels-V3)
- **OpenAI + Anthropic API** - one server, two API flavors, fully compatible
- **Continuous batching** - vLLM-style scheduler with chunked prefill, preemption, priority queues
- **Speculative decoding** - SuffixDecoding, DFlash, DSpark, MTP, VLM MTP, Eagle3 (2–5× faster generation)
- **TurboQuant KV** - 4-bit KV cache quantization, 4× less memory traffic
- **40+ quant formats** - GGUF (Q2_K -> Q8_0), Imatrix (IQ1_M -> IQ4_XS), TurboQuant (TQ1_0/TQ2_0), MLX (mxfp4/mxfp8/6bit/4bit/8bit/F16/BF16/F32)
- **Paged KV cache** - SSD cold layer, block-aware prefix caching with COW sharing
- **Fused sampler** - skip logsumexp, eliminate GPU sync, batched sampling
- **SmartRouter** - phase-aware routing with benchmark-based backend selection and EMA smoothing
- **Priority scheduling** - REALTIME / BATCH / BACKGROUND queues with Metal command queue priorities
- **4-tier memory enforcer** - safe / balanced / aggressive / custom hard limits with deadlock-free eviction
- **Multi-model concurrency** - EnginePool with LRU eviction, pinning, and TTL
- **MCP tool support** - list, discover, and execute MCP tools via API; auto-discovers fusion-plugin-server on PATH via stdio transport
- **LoRA / DORA fine-tuning** - train adapters on Apple Silicon via mlx_lm; job queue, SSE progress, adapter management
- **Admin web panel** - model management, live chat, HuggingFace downloads, online quantization
- **macOS native app** - SwiftUI with menu bar, auto-update, benchmark, fine-tune, model management, **hardware-aware setup wizard**
- **SkyReels-V3 video generation** - Pure-MLX port of the strongest open-source video model; all three branches (R2V/V2V/A2V) run end-to-end on real weights, with M5 Max dFlash attention + NF4 quantization keeping a 19B model at 720P under 14 GB resident memory
- **PyTorch -> MLX full-model converter** - `convert_skyreels_v3.py` one-shot converts SkyReels-V3's three branches (DiT + T5 + VAE + CLIP + audio) PyTorch weights to MLX safetensors, supporting bfloat16/float16/float32 + NF4 quantization with incremental per-shard writes to avoid unified-memory spikes
- **UMA Radix Latent cache** - repeat I2V requests skip the VAE-encode (model load + forward) via zero-copy `mx.array` reuse on Apple Silicon unified memory; extends the #178 radix cache from text KV to video frame latents (Phase-1: input-image latents, LTX-2 + Wan2.2). The UMA advantage the discrete-GPU CUDA stack cannot replicate. See [cache/LATENT_CACHE.md](fusion_mlx/cache/LATENT_CACHE.md)
- **Distributed pipeline parallelism** - split a transformer forward at a layer boundary across nodes; `/distributed/load_shard`, `/distributed/pipeline_step`, `/distributed/decode`, `/distributed/sync_weights` endpoints with bit-exact activation serialization (base64 `.npy`), final norm + lm_head decode (#630), and path-traversal confinement. See [docs/distributed-pipeline.md](docs/distributed-pipeline.md)
- **Public API boundary** - CI guard (`scripts/check_public_api_boundary.py`) that stops internal `fusion_mlx.*` modules from leaking into the public `fusion_mlx` import surface, with a grandparented whitelist for existing comfyui pairs. See [docs/public-api-boundary.md](docs/public-api-boundary.md)

### Advanced Feature Recommendations

When you launch the macOS app for the first time, the **6-step Welcome wizard** auto-detects your Mac hardware and recommends optimal settings:

| Use Case | Recommended Models (selectable list) | DFlash | DSpark | TurboQuant | Max Context |
|----------|--------------------------------------|--------|--------|------------|-------------|
| 🤖 Agent (OpenClaw) | DeepSeek-V4-Flash, Qwen3.6-27B | ✅ | ❌ | ✅ (≥64GB) | 65K |
| 💻 Coding | Qwen3.5-9B, DeepSeek-Coder-V2 | ❌ | ✅ | ✅ (≥64GB) | 131K |
| 💬 Chat | Qwen3.5-9B, Gemma-4-31B | ❌ | ❌ | ✅ (≥64GB) | 32K |

Recommendations are based on real-time hardware detection (CPU cores, unified memory, GPU bandwidth, disk space). All settings are editable with validation warnings for out-of-range values.

## Quick Start

### Install

```bash
# Option 1: curl one-liner (auto-detects RAM, recommends model)
curl -fsSL https://raw.githubusercontent.com/dahai80/fusion-mlx/main/scripts/install.sh | bash

# Option 2: Homebrew (one-line install)
brew install dahai80/fusion-mlx/fusion-mlx

# Option 3: uv (fastest)
uv tool install fusion-mlx

# Option 4: pip
pip install fusion-mlx
```

### First Run

```bash
# 1. Check your environment (no model load, <5s)
fusion-mlx doctor

# 2. Chat right away — spawns a server, picks a model by RAM
fusion-mlx chat

# 3. Or serve a specific model on port 11434
fusion-mlx serve qwen3.5-9b-4bit

# 4. List all available model aliases
fusion-mlx models

# 5. Upgrade to the latest version
fusion-mlx upgrade
```

### Chat API

```bash
curl http://localhost:11434/v1/chat/completions \
   -H "Content-Type: application/json" \
   -d '{
     "model": "Qwen3-4B-Q4_K_M",
     "messages": [{"role": "user", "content": "What is 2+2?"}],
     "max_tokens": 64
   }'
```

OpenAI Python client:

```python
from openai import OpenAI
client = OpenAI(base_url="http://localhost:11434/v1", api_key="local")
resp = client.chat.completions.create(
    model="Qwen3-4B-Q4_K_M",
    messages=[{"role": "user", "content": "What is 2+2?"}],
    max_tokens=64,
)
print(resp.choices[0].message.content)
```

Anthropic API:

```python
import anthropic
client = anthropic.Anthropic(base_url="http://localhost:11434/v1", api_key="local")
resp = client.messages.create(
    model="Qwen3-4B-Q4_K_M",
    max_tokens=64,
    messages=[{"role": "user", "content": "What is 2+2?"}],
)
print(resp.content[0].text)
```

## CLI Reference

All subcommands support `--help` for full flag documentation. Shell tab completion is available (see [Tab Completion](#tab-completion)).

### Core Commands

| Command | Description |
|---------|-------------|
| `fusion-mlx serve <model>` | Start OpenAI/Anthropic-compatible server |
| `fusion-mlx chat [model]` | Interactive chat REPL (alias: `run`) |
| `fusion-mlx models` | List available model aliases |
| `fusion-mlx models --cached` | List only locally-downloaded models (alias: `ls`) |
| `fusion-mlx info <model>` | Show per-model profile (parsers, capability gates) |
| `fusion-mlx bench <model>` | Run benchmark |
| `fusion-mlx convert <model>` | Convert HuggingFace model to MLX format |
| `fusion-mlx doctor` | Check environment health (Python, packages, HF cache, network) |

### Model Management

| Command | Description |
|---------|-------------|
| `fusion-mlx pull <model>` | Download a model to HuggingFace cache (no server needed) |
| `fusion-mlx rm <model>` | Remove a cached model (`-y` skips confirmation) |
| `fusion-mlx ps` | List running fusion-mlx servers |

### Server Lifecycle

Managed background server control (macOS app / Homebrew):

| Command | Description |
|---------|-------------|
| `fusion-mlx start` | Start as a managed background server |
| `fusion-mlx stop` | Stop the managed background server |
| `fusion-mlx restart` | Restart the managed background server |

All accept `--timeout <seconds>` (default 60). `start`/`restart` also accept `--no-wait`.

### Chat REPL

```bash
# Default model (qwen3.5-4b-4bit)
fusion-mlx chat

# Specific model with reasoning mode
fusion-mlx chat qwen3.5-9b-4bit --think

# Custom system prompt and temperature
fusion-mlx chat qwen3.5-9b-4bit --system "You are a poet." --temperature 0.9

# Connect to an existing server instead of spawning one
fusion-mlx chat qwen3.5-9b-4bit --port 11434
fusion-mlx chat qwen3.5-9b-4bit --base-url http://192.168.1.100:11434
```

| Flag | Description |
|------|-------------|
| `--think` | Enable thinking/reasoning mode (default: off) |
| `--system <prompt>` | System prompt prepended to conversation |
| `--max-tokens <N>` | Max tokens per response (default: 2048; 4096 with --think) |
| `--temperature <T>` | Sampling temperature (default: 0.7) |
| `--port <PORT>` | Connect to existing server on 127.0.0.1:PORT |
| `--base-url <URL>` | Connect to existing server at URL |
| `--ready-timeout <S>` | Seconds to wait for spawned server (default: 600) |
| `--response-timeout <S>` | Seconds to wait per response (default: 600) |

### Serve

```bash
# Single model
fusion-mlx serve qwen3.5-9b-4bit --port 11434

# Multi-model server (auto-discovers all models in directory)
fusion-mlx serve --model-dir ~/.cache/huggingface

# macOS app style
fusion-mlx serve --base-path ~/.fusion-mlx

# With speculative decoding
fusion-mlx serve qwen3.5-9b-4bit --enable-dspark

# With KV cache quantization (4-bit, 4× less memory traffic)
fusion-mlx serve qwen3.5-9b-4bit --kv-cache-turboquant
```

### Bench

```bash
# Freeform benchmark
fusion-mlx bench qwen3.5-9b-4bit --num-prompts 10 --max-tokens 100

# Standardized community benchmark (submit to bench.dpdns.org)
fusion-mlx bench qwen3.5-9b-4bit --submit

# Validation tiers: smoke / speed / harness / all
fusion-mlx bench qwen3.5-9b-4bit --tier smoke
fusion-mlx bench qwen3.5-9b-4bit --tier speed
fusion-mlx bench qwen3.5-9b-4bit --tier all
```

| Flag | Description |
|------|-------------|
| `--submit` | Run standardized B=1 benchmark and submit to community leaderboard |
| `--tier <tier>` | Validation tier: smoke / speed / harness / all |
| `--base-url <URL>` | Attach to already-running server (for --tier) |
| `--num-prompts <N>` | Number of prompts (default: 10) |
| `--max-tokens <N>` | Max tokens per prompt (default: 100) |
| `--kv-cache-quantization` | Quantize KV cache to reduce memory (8-bit default) |
| `--kv-cache-quantization-bits` | 4 or 8 (default: 8) |
| `--use-paged-cache` | Use paged KV cache (experimental) |
| `--enable-prefix-cache` | Enable prefix caching (default: on) |
| `--disable-prefix-cache` | Disable prefix caching |

### Convert

```bash
# Convert with 4-bit quantization
fusion-mlx convert qwen3.5-9b --quant-bits 4 -o ./qwen3.5-9b-4bit

# Convert and upload to HuggingFace
fusion-mlx convert mlx-community/Qwen3.5-9B --quant-bits 8 --upload-repo me/my-repo
```

This is **weight** quantization saved to disk, distinct from TurboQuant KV-cache compression (`--kv-cache-turboquant`), which is a runtime knob.

### Upgrade

Auto-detects your install method (brew / pip / install.sh) and runs the correct upgrade command:

```bash
fusion-mlx upgrade          # interactive confirmation
fusion-mlx upgrade -y       # skip confirmation
fusion-mlx upgrade --dry-run  # show what would run, then exit
```

### Agent Integrations

```bash
# List all available agent integrations
fusion-mlx agents

# Auto-configure an agent to use fusion-mlx
fusion-mlx agents hermes --setup
fusion-mlx agents codex --setup --model Qwen3-4B

# Test an agent integration
fusion-mlx agents hermes --test
```

### Share (SSH Tunnel)

Expose your local server behind a public URL:

```bash
fusion-mlx share
```

Creates an SSH tunnel to `fusionmlx.com`, giving you a shareable public URL for your local server. Useful for testing webhooks, sharing demos, or remote access.

### Telemetry

Anonymous usage telemetry is **opt-in** — nothing is sent unless you explicitly enable it.

```bash
fusion-mlx telemetry status    # check current state
fusion-mlx telemetry enable    # opt in
fusion-mlx telemetry disable   # opt out
fusion-mlx telemetry preview   # see exactly what would be sent
fusion-mlx telemetry reset     # delete consent + client-id (re-prompts next run)
```

Per-run override: `fusion-mlx --no-telemetry serve ...` disables telemetry for that invocation.

### Tab Completion

Shell tab completion is powered by [argcomplete](https://github.com/kislyuk/argcomplete). After installing fusion-mlx:

```bash
# Bash
eval "$(register-python-argcomplete fusion-mlx)"

# Zsh
autoload -U bashcompinit && bashcompinit
eval "$(register-python-argcomplete fusion-mlx)"

# Fish
register-python-argcomplete fusion-mlx > ~/.config/fish/completions/fusion-mlx.fish
```

Then `fusion-mlx chat gemma-4-<TAB>` completes model aliases instantly.

## Mirror Configuration

For users in regions where HuggingFace is slow or blocked (e.g. mainland China), fusion-mlx supports configuring a mirror source for model downloads. No manual environment variable export needed.

### Via config file (recommended)

Edit `~/.fusion-mlx/settings.json` and set the `huggingface.endpoint` field:

```json
{
  "huggingface": {
    "endpoint": "https://hf-mirror.com"
  }
}
```

`start.sh` automatically reads this config and sets `HF_ENDPOINT` for model downloads. Run `start.sh tune` to generate the config with the default mirror pre-filled.

### Via environment variable

```bash
# One-time override
HF_MIRROR=https://hf-mirror.com fusion-mlx pull Qwen3-4B

# Persistent (add to ~/.zshrc or ~/.bashrc)
export HF_MIRROR=https://hf-mirror.com
```

### Priority order

1. `HF_MIRROR` environment variable (highest)
2. `huggingface.endpoint` in `~/.fusion-mlx/settings.json`
3. Built-in default: `https://hf-mirror.com`

## Supported Models

| Type | Engine | Example Models |
|------|--------|----------------|
| LLM | `BatchedEngine` | Qwen, Llama, Mistral, Gemma, DeepSeek, Kimi |
| VLM | `VLMBatchedEngine` | Qwen2-VL, LLaVA, InternVL |
| Embedding | `EmbeddingEngine` | BGE, E5, GTE |
| Reranker | `RerankerEngine` | Cohere, Jina rerankers |
| STT | `STTEngine` | Whisper, VibeVoice-ASR |
| TTS | `TTSEngine` | Kokoro, VibeVoice |
| ImageGen | `ImageGenEngine` | Flux 2, SD3-Medium, SDXL, Stable Cascade |
| VideoGen | `VideoGenEngine` | LTX-2, Wan2, SkyReels-V3 (pure-MLX ports) |

## Quantization Formats

| Category | Formats |
|----------|---------|
| GGUF/GGML | Q2_K, Q3_K_S/M/L, Q4_0, Q4_1, Q4_K_S/M, Q5_0, Q5_1, Q5_K_S/M, Q6_K, Q8_0, Q8_K |
| Imatrix | IQ1_M, IQ2_S, IQ2_XS, IQ2_XXS, IQ3_M, IQ3_S, IQ4_NL, IQ4_XS |
| TurboQuant | TQ1_0, TQ2_0 |
| MLX-native | mxfp4, mxfp8, 6bit (ParoQuant), 4bit, 8bit, F16, BF16, F32 |
| MLX Recipes | mixed_3_4, mixed_2_6, mixed_2_4, mixed_3_6, mixed_4_6, quant2_all, quant2, quant2_128, quant2_flat (see below) |
| NVFP4 (read-only) | NVFP4 (E2M1 + E4M3 block scale) - NVIDIA 4-bit checkpoints dequantized to bf16 at load (#179) |

> **NVFP4** is a format-compatibility bridge, not a speed path: NVIDIA NVFP4 weights (4-bit E2M1, 2 per byte, with E4M3 block scales) are detected and dequantized to bf16 during `safetensors` load, so externally-quantized NVFP4 DiT checkpoints run without a separate conversion step. The 4-bit storage win is not retained at inference. Detection is conservative (uint8 weight + sibling block-scale with 1-scale-per-16-elements) and is a silent no-op on non-NVFP4 checkpoints.

### Quantization Recipes

MLX recipe quantization provides pre-tuned mixed-bit plans that maximize decode speed for Apple Silicon. Both modes produce standard mlx-lm safetensors compatible with any MLX runtime.

The macOS app offers a mode toggle between:

- **oQ Online** - sensitivity-based per-layer quantization (original mode)
- **MLX Recipe** - pre-tuned quantization plans via `mlx_lm.convert --quant-recipe <name>`

| Recipe | Label | BPW | Speed vs mxfp8 | Category |
|--------|-------|-----|-----------------|----------|
| mixed_3_4 | Mixed 3/4-bit | 3.68 | +96% | recommended |
| mixed_2_6 | Mixed 2/6-bit | 3.25 | +112% | recommended |
| mixed_2_4 | Mixed 2/4-bit | 2.95 | +131% | aggressive |
| mixed_3_6 | Mixed 3/6-bit | 4.0 | +75% | balanced |
| mixed_4_6 | Mixed 4/6-bit | 4.85 | +57% | conservative |
| quant2_all | quant2-all | 2.37 | +162% | recommended |
| quant2 | quant2 | 2.72 | +144% | aggressive |
| quant2_128 | quant2-g128 | 2.46 | +161% | aggressive |
| quant2_flat | quant2-flat | 2.25 | +167% | experimental |
| mxfp4 | MLX FP4 | 4.0 | +75% | conservative |
| mxfp8 | MLX FP8 | 8.0 | baseline | conservative |

**Recommended**: `mixed_3_4` or `quant2_all` for best quality/speed tradeoff. **Conservative**: `mixed_4_6` or `mxfp4` when quality is priority. **Aggressive**: `mixed_2_4` or `quant2` when maximizing speed on constrained memory.

## API Compatibility

| API | Endpoints | Status |
|-----|-----------|--------|
| OpenAI Chat | `/v1/chat/completions`, `/v1/models` | ✅ Fully compatible |
| OpenAI Legacy | `/v1/completions` | ✅ Supported |
| Anthropic Messages | `/v1/messages`, `/v1/count_tokens` | ✅ Fully compatible |
| Audio | `/v1/audio/transcriptions`, `/v1/audio/speech` | ✅ Supported |
| Images | `/v1/images/generate`, `/v1/images/super-resolution` | ✅ Generate (Flux 2, SD3-Medium, SDXL, Stable Cascade); Super-resolution (RealESRGAN x4plus, pure MLX, #752) |
| Videos | `/v1/videos/generate` | ✅ Supported (LTX-2, Wan2, SkyReels-V3; pure-MLX ports) |
| Embeddings | `/v1/embeddings` | ✅ Supported |
| Reasoning | `/v1/reasoning` | ✅ Explicit thinking step API (DeepSeek-R1, QwQ, etc.) |
| OCR | `/v1/ocr` | ✅ 4 dedicated OCR engines (DeepSeek-OCR, DOTS-OCR, GLM-OCR) |
| Sessions | `/v1/sessions/{id}/stats`, `/v1/sessions/{id}/context` | ✅ Per-session token usage + context cap (#226) |
| MCP | `/v1/mcp/tools`, `/v1/mcp/servers`, `/v1/mcp/execute` | ✅ Supported |
| Model Manager | `/admin/api/model-manager/models`, `.../load`, `.../unload`, `.../status` | ✅ Scoped-key model lifecycle (#302) |
| Embedding Mgr | `/admin/api/model-manager/embedding/*` | ✅ Pin/unpin/status for embedding models (#302) |
| OpenClaw Agent | `/v1/openclaw/agent/*` | ✅ Sessions, turns, tool calling, SSE streaming |
| Agent Graph | `/v1/agents/graphs`, `/v1/agents/run` | ✅ CRUD + export + run (in-memory) |
| Base Info | `/v1/base` | ✅ MLX runtime capability detection |
| Convert / Quantize | `/v1/convert`, `/v1/quantize` (+ `.../jobs/{id}`) | ✅ Async HF->MLX conversion + weight quantization |
| Watermark | `/v1/watermark/embed`, `/v1/watermark/verify` | ✅ Weight-tensor LSB watermark (#656) |

## Weight-Tensor Watermark (#656)

Secret-seeded LSB spread-spectrum watermark embedded into model weight tensors.
Tamper-resistant: a redistributor who ships only the weights still carries the
watermark. Hub-aligned signature `sha256(f"{secret}:{model}::{json.dumps(payload, sort_keys=True)}")[:32]`
is shared with Fusion-Model-Hub, which validates provenance without importing MLX.

**Prerequisite:** set a non-default secret env var. The route returns `503`
on a default/empty secret:

```bash
export FMH_WATERMARK_SECRET="<high-entropy-secret>"
```

**Embed** — writes a watermarked copy to `output_path` (must be under
`~/.fusion-mlx/models/`). Admin + hub-source gated. Synchronous (returns
`output_path`, `signature`, `carrier_count` in the body):

```bash
curl http://localhost:8897/v1/watermark/embed \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer <admin-key>" \
  -d '{
    "model": "org/model",
    "payload": {"owner": "dahai80", "purpose": "provenance"},
    "secret": "<FMH_WATERMARK_SECRET or omit to use env>",
    "output_path": "~/.fusion-mlx/models/wm-out"
  }'
```

**Verify** — extracts and verifies the embedded payload. Returns `200` with
`verified: false` (not an error) when the payload is absent/corrupted:

```bash
curl http://localhost:8897/v1/watermark/verify \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer <admin-key>" \
  -d '{
    "model": "~/.fusion-mlx/models/wm-out",
    "secret": "<FMH_WATERMARK_SECRET or omit to use env>"
  }'
```

Quantized (int4/int8) weight tensors are config-driven skipped during
embed/verify — only floating-point tensors carry the watermark.

## OCR — Dedicated Document Recognition

fusion-mlx provides 4 purpose-built OCR engines via the `/v1/ocr` endpoint:

| Engine | model_type | Best For | Default Prompt |
|--------|-----------|----------|---------------|
| DeepSeek-OCR | `deepseekocr` | General documents, tables | "Convert the document to markdown." |
| DeepSeek-OCR v2 | `deepseekocr_2` | Improved accuracy, CJK | "Convert the document to markdown." |
| DOTS-OCR | `dots_ocr` | Clean markdown output | "Convert this page to clean Markdown while preserving reading order." |
| GLM-OCR | `glm_ocr` | Chinese text recognition | "Text Recognition:" |

```bash
# OCR via API
curl http://localhost:8897/v1/ocr \
  -H "Content-Type: application/json" \
  -d '{
    "model": "deepseekocr",
    "image": "data:image/png;base64,<BASE64>",
    "output_format": "markdown"
  }'

# OCR with local file path
curl http://localhost:8897/v1/ocr \
  -H "Content-Type: application/json" \
  -d '{
    "model": "dots_ocr",
    "image": "/path/to/document.png",
    "output_format": "text"
  }'

# OCR via Python
import requests
resp = requests.post("http://localhost:8897/v1/ocr", json={
    "model": "glm_ocr",
    "image": "https://example.com/invoice.jpg",
    "output_format": "json"
})
print(resp.json()["results"][0]["text"])
```

Output formats: `text` (plain), `markdown` (default), `json` (`{"text": "..."}`).

Each engine uses temperature=0 and optimized generation defaults (max_tokens, repetition_penalty) for deterministic OCR output.

## Tool Calling & Structured Output

### 21 Tool Parsers — Full Coverage for Every Major Model

fusion-mlx ships 21 tool-call parsers, matching or exceeding every other MLX runtime:

| Parser | Models | Streaming |
|--------|--------|-----------|
| hermes | Hermes-series | ✅ |
| llama | Llama 3.x | ✅ |
| qwen | Qwen 2.x/3.x | ✅ |
| deepseek | DeepSeek-V2/V3 | ✅ |
| deepseek_v3 | DeepSeek-V3 native | ✅ |
| deepseekv31 | DeepSeek-V3.1 | ✅ |
| harmony | OpenAI harmony | ✅ |
| gemma4 | Gemma 4 | ✅ |
| mistral | Mistral/Mixtral | ✅ |
| granite | IBM Granite | ✅ |
| minimax | MiniMax | ✅ |
| kimi | Moonshot Kimi | ✅ |
| glm47 | GLM-4.7 | ✅ |
| nemotron | NVIDIA Nemotron | ✅ |
| functionary | Functionary | ✅ |
| seed_oss | Seed-OSS | ✅ |
| ui_tars | UI-TARS | ✅ |
| xlam | xLAM | ✅ |
| qwen3coder | Qwen3-Coder | ✅ |
| auto | Auto-detect from model config | ✅ |
| 3gap_stream | 3-gap streaming | ✅ |

### Grammar-Constrained Decoding — Dual Backend

| Backend | Install | Priority |
|---------|---------|----------|
| **llguidance** | `pip install fusion-mlx[llguidance]` | Default (AUTO) |
| **xgrammar** | `pip install fusion-mlx[grammar]` | Fallback |

### Usage

```bash
# JSON schema enforcement (OpenAI-compatible)
curl -X POST /v1/chat/completions -d '{
  "model": "my-model",
  "messages": [...],
  "response_format": {"type": "json_schema", "json_schema": {"schema": {"type": "object", "properties": {"name": {"type": "string"}}}}},
  "grammar_backend": "auto"
}'

# vLLM-compatible structured_outputs
curl -X POST /v1/chat/completions -d '{
  "model": "my-model",
  "messages": [...],
  "structured_outputs": {"json_schema": "{\"type\":\"object\",\"properties\":{\"answer\":{\"type\":\"string\"}}}"},
  "grammar_backend": "llguidance"
}'

# Regex, choice, grammar
"structured_outputs": {"regex": "[A-Z][a-z]+"}
"structured_outputs": {"choice": ["yes", "no", "maybe"]}
"structured_outputs": {"grammar": "root ::= [a-z]+", "format": "lark"}
```

### Backend Selection

- `"auto"` (default): prefers llguidance → xgrammar → no constraint
- `"llguidance"`: uses llguidance exclusively
- `"xgrammar"`: uses xgrammar exclusively

## Model Aliases

```bash
fusion-mlx serve --model claude-4.6-sonnet   # -> Qwen3.5-27B-Claude-4.6-Opus-Distilled-MLX-6bit
fusion-mlx serve --model gpt-4o               # -> Qwen3-32B-A3B-Think-2512-MLX
```

## Low-Resource Mac (8–24 GB)

fusion-mlx is the only MLX runtime that **runs 27B models on 8 GB Macs**:

| RAM | Recommended Model | Quant | Resident Memory | Speed |
|-----|-------------------|-------|-----------------|-------|
| 8 GB | Qwen3-4B | 4-bit | ~3.5 GB | Full speed |
| 16 GB | Qwen3.5-9B | 6-bit | ~8 GB | Full speed |
| 16 GB | Qwen3.6-27B | **quant2-flat** | **~7.1 GB** | **1.67× faster** than mxfp8 |
| 24 GB | Qwen3.6-27B | mxfp8 | ~18 GB | Full speed |
| 32 GB | Qwen3.6-27B | 6-bit | ~22 GB | Full speed |
| 64 GB+ | Qwen3-72B | 4-bit | ~42 GB | Full speed |

`install.sh` auto-detects your RAM via `sysctl hw.memsize` and recommends the best model. The macOS app Welcome Wizard does the same with a 6-step guided setup.

**quant2-flat** is unique to fusion-mlx — 2-bit weight quantization that keeps a 27B model under 8 GB while being **faster** than higher-precision formats.

## Drop-in Ollama Replacement

fusion-mlx exposes **both** OpenAI and Anthropic APIs — something Ollama cannot do:

| Feature | Ollama | fusion-mlx |
|---------|--------|------------|
| OpenAI Chat API | ❌ (custom only) | ✅ `/v1/chat/completions` |
| Anthropic Messages API | ❌ | ✅ `/v1/messages` |
| Streaming (SSE) | ✅ | ✅ |
| SSE keepalive | ❌ | ✅ (anti-timeout ping) |
| Context scaling | ❌ | ✅ (auto-cap max_tokens) |
| Tool calling | ✅ | ✅ (21 parsers) |
| Structured output | ❌ | ✅ (llguidance + xgrammar) |
| Embeddings | ✅ | ✅ |
| Image generation | ❌ | ✅ (Flux 2) |
| Video generation | ❌ | ✅ (LTX-2, Wan2, SkyReels-V3) |
| STT / TTS | ❌ | ✅ |
| OCR (dedicated engines) | ✅ | ✅ (4 OCR engines + `/v1/ocr` API) |
| Model aliases | ✅ | ✅ (`serve --model gpt-4o`) |
| Profile syntax | ✅ (`modelfile`) | ✅ (`model:profile` zero-mem) |
| Continuous batching | ❌ | ✅ (vLLM-style scheduler) |
| Prefix KV cache | ❌ | ✅ (block-aware + COW + SSD) |
| Homebrew install | ✅ | ✅ (`brew install dahai80/fusion-mlx/fusion-mlx`) |

```bash
# Point any OpenAI-compatible tool at fusion-mlx
export OPENAI_API_BASE=http://localhost:8897/v1

# Or use Anthropic SDK directly
export ANTHROPIC_BASE_URL=http://localhost:8897/v1

# Or use Ollama SDK / Open WebUI directly
export OLLAMA_HOST=http://localhost:8897
```

### Ollama-Compatible API

fusion-mlx now exposes Ollama-compatible endpoints so tools like **Open WebUI**, **LibreChat**, and the `ollama` CLI work out of the box:

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/generate` | POST | Text generation (prompt-based) |
| `/api/chat` | POST | Chat with message array |
| `/api/tags` | GET | List local models |
| `/api/version` | GET | Server version |

```bash
# Chat via Ollama API
curl http://localhost:8897/api/chat \
  -d '{"model": "qwen3", "messages": [{"role": "user", "content": "Hello!"}]}'

# Generate text
curl http://localhost:8897/api/generate \
  -d '{"model": "qwen3", "prompt": "Write a haiku about code"}'

# List models
curl http://localhost:8897/api/tags
```

### Model Profiles (`model:profile` syntax)

Switch sampling presets without loading a separate model — zero extra memory:

```bash
# Use the "creative" profile for qwen3 — high temperature, more tokens
curl http://localhost:8897/v1/chat/completions \
  -d '{"model": "qwen3:creative", "messages": [...]}'

# Same for Anthropic API
curl http://localhost:8897/v1/messages \
  -d '{"model": "qwen3:creative", "messages": [...]}'
```

Profiles are configured in the admin panel (**Model Settings → Profiles → Expose as model**).
Request-level parameters always take precedence over profile defaults.

## Integrations

```bash
# Claude Code - use fusion-mlx as your local Anthropic API
# Includes SSE keepalive (anti-timeout), context scaling (auto-cap max_tokens),
# and auto-compact window (CLAUDE_CODE_AUTO_COMPACT_WINDOW)
fusion-mlx launch claude

# Codex CLI (OpenAI) - configures ~/.codex/config.toml
fusion-mlx launch codex --model Qwen3-4B

# Hermes Agent - configures ~/.hermes/config.yaml
fusion-mlx launch hermes --model Qwen3-4B

# OpenCode - configures ~/.config/opencode/opencode.json
fusion-mlx launch opencode --model Qwen3-4B

# OpenClaw - batch agent processing
fusion-mlx launch openclaw --model Qwen3-4B

# ComfyUI - image generation with Flux 2
fusion-mlx launch comfyui

# GitHub Copilot
fusion-mlx launch copilot

# Qwen Code - env-var based
fusion-mlx launch qwen-code --model Qwen3-4B

# OpenHands - env-var based
fusion-mlx launch openhands --model Qwen3-4B

# Kilo Code - env-var based
fusion-mlx launch kilo-code --model Qwen3-4B

# Factory Droid - env-var based
fusion-mlx launch factory-droid --model Qwen3-4B

# Kimi Code (Moonshot) - env-var based
fusion-mlx launch kimi-code --model Qwen3-4B

# PydanticAI - configures ~/.pydantic-ai/config.json
fusion-mlx launch pydantic-ai --model Qwen3-4B

# smolagents (HuggingFace) - configures ~/.smolagents/config.json
fusion-mlx launch smolagents --model Qwen3-4B
```

## Pipeline Stage API & Step Callbacks (Fusion-ComfyUI)

For ComfyUI-style integrations that need per-stage control of the generation
pipeline (rather than a single `generate()` call), the image and video engines
expose a streaming stage API plus a per-step progress callback.

### Stage API (#170)

`ImageGenEngine` and `VideoGenEngine` expose paired load / run / unload methods
so a host can hold the text encoder, DiT, and VAE independently and free memory
between stages (`gc.collect()` + `mx.metal.clear_cache()` + active-memory log):

| Stage | Load | Run | Unload |
|---|---|---|---|
| Text encoder | `load_text_encoder()` | `encode_text(prompt) -> {"embed","text_ids"}` | `unload_text_encoder()` |
| DiT | `load_dit()` | `denoise(latent, pos_embed, neg_embed, steps, cfg, seed[, num_frames][, control][, inpaint_mask=, init_latent=])` | `unload_dit()` |
| VAE | `load_vae()` | `decode(latent)` / `decode_tiled(latent, tile_size=256)` | `unload_vae()` |
| VAE encoder (#652/#653) | `load_vae_encoder()` | `encode(pixels) -> latent` (Surface A, #653) / `encode_control(image=, width=, height=, num_frames=, control_video=, control_mask=, reference_images=, camera_conditions=, controlnet_image=, control_type=, controlnet_strength=) -> ControlState \| None` (Surface B, #653) | `unload_vae_encoder()` |

> **Wan2 conditioning (#652):** `encode_control()` encodes I2V / VACE / camera
> conditioning up front into a `ControlState` that the staged `denoise(control=...)`
> threads into `run_denoise` bit-exactly mirroring the monolith `generate_video`.
> Dispatches on `model_type`: VACE → `control_hidden_states`; I2V-14B → channel-concat
> `y_i2v`; TI2V-5B → `z_img` + `i2v_mask` (mask-blend); Fun-Camera → `y_camera`.
> Pure T2V (`image=None`, no camera) returns `None` — the pure-noise path is untouched.
> VACE / i2v paths require `load_vae_encoder()` first (raise otherwise); camera skips
> the gate. The `vae_encoder` flag is inject-on-load / pop-on-unload, not pre-declared.

> **#653 surfaces (Wan2 + SkyReels):**
> - **Surface A (VAE encode):** `encode(pixels) -> 5D latent` is now wired on 9 backends
>   (Wan2, SkyReels, ltx_video_legacy, svd, cosmos, hunyuanvideo, cogvideox, minimax_h3,
>   ltx2_5). `ltx2` / `opensora` / `uniworld` defer to follow-up issues. Numpy-bridges on
>   the caller thread, rebuilds `mx.array` + `mx.eval` on the worker thread (#630 stream
>   invariant).
> - **Surface B (ControlNet):** `encode_control(controlnet_image=<path>, control_type="canny",
>   controlnet_strength=1.0)` builds a `ControlNet` adapter + preprocessed control latent,
>   returned on `ControlState.controlnet_adapter`/`.controlnet_latent`. `denoise(control=...)`
>   injects per-step residuals into the DiT block loop (Wan2 `wan_2.py`; SkyReels
>   `pipelines/__init__.py`).
> - **Surface C (inpaint mask):** `denoise(..., inpaint_mask=<5D mask>, init_latent=<5D latent>)`
>   re-composites `mask*latents + (1-mask)*init` after each denoise step — `mask=1` reactive,
>   `mask=0` frozen (restored to `init_latent` bit-exactly). Orthogonal to the TI2V mask blend.

Latents flow as unpacked `(batch, c, h, w)` `mx.array` across all stages
(matches mflux `prepare_latents` output and `decode_packed_latents` input;
`h`/`w` derive from the array shape, no extra size params).

> **MLX stream constraint:** latents/embeds must be engine-native - created by
> `encode_text` or another stage running in the single image-executor thread
> (`max_workers=1`, `_init_mlx_thread`). Arrays created in a caller thread hit
> `RuntimeError: There is no Stream(gpu, 0) in current thread` on the per-step
> `mx.eval`. Stage-to-stage flow stays native because the executor is
> single-threaded.

`unload_*` drops the submodule reference to `None`; mflux loads all stages in
`__init__`, so reloading a single unloaded stage requires re-instantiating the
engine (the load methods raise `RuntimeError` with that guidance).

Video backends inherit `NotImplementedError` defaults for the stage API (issue
#170 phase 2); Surface A (`encode`) is now wired on 9 backends (#653), Surface B
(ControlNet) + Surface C (inpaint) on Wan2 + SkyReels (#653). The 10 other
denoise-less backends defer Surface B+C to per-family follow-up issues (#653
follow-ups) — their generate paths are monolithic (no separable `run_denoise`).
`Wan2Backend` additionally implements the full I2V / VACE / camera conditioning
stage surface (#652).

### Step callback (#171)

`generate()` (image) and `VideoGenEngine.generate()` accept
`on_step: Callable[[int, int], Awaitable[None]] | None`, fired as
`on_step(step, total_steps)` after each denoise step. The async callback is
bridged onto the synchronous mflux denoise loop via
`asyncio.run_coroutine_threadsafe` (fire-and-forget; errors logged, never
block generation). Image uses a real per-step subscriber on `flux.callbacks`;
video wires it through `VideoGenParams.on_step`.

### Model registry listing (#172)

`list_available_models()` in `fusion_mlx/model_registry.py` now returns the
full set of discoverable models additively (registered + discovered), so hosts
can enumerate models without a separate discovery call.

## Admin Panel

Access at `http://localhost:11434/admin`:

- **Models** - load / unload / pin models dynamically, ParoQuant compat detection
- **Chat** - live chat interface for testing any model
- **Downloads** - HuggingFace / ModelScope model downloads with progress tracking
- **Quantization** - online quantization (oQ) pipeline
- **Benchmarks** - throughput and accuracy benchmarking
- **Fine-Tune** - LoRA / DORA adapter training with live progress, job queue, adapter management
- **Monitoring** - real-time memory, performance, and request metrics
- **Settings** - global / per-model configuration, sub-API key management

## macOS App

Native SwiftUI app with menu bar integration:

- One-click model launch and server control
- Quantization mode toggle: **oQ Online** (sensitivity-based) / **MLX Recipe** (pre-tuned plans)
- **Fine-Tune screen** - LoRA / DORA training with advanced config, live progress, adapter management
- Throughput & accuracy benchmarking
- Auto-update from GitHub Releases
- Model management and downloads
- Live server status in menu bar

Download from [GitHub Releases](https://github.com/dahai80/fusion-mlx/releases).

## Fine-Tuning (LoRA / DORA)

Train LoRA or DORA adapters on any loaded model using `mlx_lm.tuner` under the hood.

### In-place adapter serving (#389)

By default each served adapter reloads the full base model. Set
`FUSION_LORA_INPLACE_SWAP=1` (plus `FUSION_LORA_ALLOWED_DIRS`) to keep one base
engine resident and swap adapters in place in milliseconds — no second base
copy, correct for quantized bases. See
[docs/lora-inplace-swap.md](docs/lora-inplace-swap.md).

### API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/admin/api/fine-tune/jobs` | Create a training job |
| GET | `/admin/api/fine-tune/jobs` | List all jobs |
| GET | `/admin/api/fine-tune/jobs/{id}` | Get job details |
| GET | `/admin/api/fine-tune/jobs/{id}/stream` | SSE progress stream |
| POST | `/admin/api/fine-tune/jobs/{id}/cancel` | Cancel a running job |
| DELETE | `/admin/api/fine-tune/jobs/{id}` | Delete a job record |
| GET | `/admin/api/fine-tune/adapters` | List saved adapters |
| DELETE | `/admin/api/fine-tune/adapters` | Delete an adapter |
| POST | `/admin/api/fine-tune/adapters/{model_id}/{adapter_name}/serve` | Serve adapter via EnginePool |
| POST | `/admin/api/fine-tune/adapters/{model_id}/{adapter_name}/unload` | Unload adapter engine |
| GET | `/admin/api/fine-tune/models` | List fine-tunable models |
| POST | `/admin/api/fine-tune/logprob` | Score prompt+completion logprob (#363) |
| POST | `/admin/api/fine-tune/grpo/jobs` | Create a GRPO RL training job (#363) |
| GET | `/admin/api/fine-tune/grpo/jobs` | List GRPO jobs |
| GET | `/admin/api/fine-tune/grpo/jobs/{id}` | Get GRPO job details |
| GET | `/admin/api/fine-tune/grpo/jobs/{id}/stream` | SSE GRPO progress stream |
| POST | `/admin/api/fine-tune/grpo/jobs/{id}/cancel` | Cancel a GRPO job |
| DELETE | `/admin/api/fine-tune/grpo/jobs/{id}` | Delete a GRPO job record |
| POST | `/admin/api/fine-tune/dpo/jobs` | Create a DPO preference training job (#399) |
| POST | `/admin/api/fine-tune/orpo/jobs` | Create an ORPO preference training job (#399) |
| GET | `/admin/api/fine-tune/dpo/jobs` | List DPO/ORPO jobs |
| GET | `/admin/api/fine-tune/dpo/jobs/{id}` | Get DPO/ORPO job details |
| GET | `/admin/api/fine-tune/dpo/jobs/{id}/stream` | SSE DPO/ORPO progress stream |
| POST | `/admin/api/fine-tune/dpo/jobs/{id}/cancel` | Cancel a DPO/ORPO job |
| DELETE | `/admin/api/fine-tune/dpo/jobs/{id}` | Delete a DPO/ORPO job record |

### Quick Example

```bash
# Create a LoRA training job
curl -X POST http://localhost:11434/admin/api/fine-tune/jobs \
  -H "Content-Type: application/json" \
  -d '{
    "model_id": "qwen3.5-9b",
    "dataset": " ~/data/my-dataset.jsonl",
    "adapter_name": "my-lora",
    "config": {
      "fine_tune_type": "lora",
      "lora_rank": 8,
      "lora_alpha": 16.0,
      "lora_layers": 16,
      "learning_rate": 1e-5,
      "batch_size": 4,
      "iters": 100,
      "max_seq_length": 2048
    }
  }'

# Stream progress (SSE)
curl -N http://localhost:11434/admin/api/fine-tune/jobs/{job_id}/stream

# List saved adapters
curl http://localhost:11434/admin/api/fine-tune/adapters

# Serve a trained adapter for inference
curl -X POST http://localhost:11434/admin/api/fine-tune/adapters/qwen3.5-9b/my-lora/serve

# Unload adapter when done
curl -X POST http://localhost:11434/admin/api/fine-tune/adapters/qwen3.5-9b/my-lora/unload
```

### MXFP8 Training (#425)

Mixed-precision training via the `mxfp8` config flag. MLX 0.32.0 has no
native fp8 dtype (`float8_e4m3fn` / `float8_e5m2` are absent), so real fp8
GEMM compute is impossible on this stack. Rather than fail visibly,
`mxfp8: true` self-implements by routing to the existing **QLoRA 8-bit
path**: the frozen base model is quantized to 8-bit (group_size=64) and a
LoRA adapter is trained on top. Honest semantics = **8-bit-base LoRA**
(memory saving), not fp8 compute.

When `mxfp8: true` is set, `validate()` forces `quantize_base=true`,
`quant_bits=8`, and `fine_tune_type="qlora"`. This honors the downstream
`fusion-trainer` `use_mxfp8=True` switch with a working training path
today; when a future MLX release adds real fp8, a native fp8 compute path
can activate behind the same flag.

**Constraint:** `mxfp8` is incompatible with `fine_tune_type="full"`
(full fine-tuning unfreezes the base, so there is no frozen base to
quantize) — the request fails visibly with a `ValueError`. Use
`lora` / `dora` / `qlora` with `mxfp8`.

```bash
# Train with mxfp8 (routes to QLoRA 8-bit: 8-bit frozen base + LoRA)
curl -X POST http://localhost:11434/admin/api/fine-tune/jobs \
  -H "Content-Type: application/json" \
  -d '{
    "model_id": "qwen3.5-9b",
    "dataset": " ~/data/my-dataset.jsonl",
    "adapter_name": "mxfp8-lora",
    "config": {
      "mxfp8": true,
      "lora_rank": 8,
      "lora_alpha": 16.0,
      "lora_layers": 16,
      "learning_rate": 1e-5,
      "batch_size": 4,
      "iters": 100,
      "max_seq_length": 2048
    }
  }'
# validate() forces: quantize_base=true, quant_bits=8, fine_tune_type="qlora"
# logs: "mxfp8=True: routing to QLoRA 8-bit path (8-bit frozen base + LoRA)"
```

### Logprob Scoring (#363 Phase 1)

Score a completion under a prompt (teacher-forcing single forward pass).
Optionally score under a LoRA adapter via `adapter_path`.

```bash
curl -X POST http://localhost:11434/admin/api/fine-tune/logprob \
  -H "Content-Type: application/json" \
  -d '{
    "model_id": "qwen3.5-9b",
    "prompt": "The capital of France is",
    "completion": " Paris",
    "adapter_name": "my-lora"
  }'
# -> { "logprob": -1.28, "token_count": 1, "per_token": [-1.28] }
```

### GRPO Reinforcement Learning (#363 Phase 2)

Group Relative Policy Optimization: trains a LoRA policy with PPO-clipped
loss and group-normalized advantages. The reference (base) model is loaded
on demand and evicted after each step to bound memory.

```bash
curl -X POST http://localhost:11434/admin/api/fine-tune/grpo/jobs \
  -H "Content-Type: application/json" \
  -d '{
    "model_id": "qwen3.5-9b",
    "prompts": ["Solve: 2+2=", "Translate to French: hello"],
    "adapter_name": "grpo-math",
    "config": {
      "group_size": 4,
      "iters": 50,
      "batch_size": 2,
      "learning_rate": 1e-5,
      "lora_layers": 16,
      "lora_rank": 8,
      "lora_alpha": 16.0,
      "max_completion_len": 64,
      "clip_ratio": 0.2,
      "reward_endpoint": "http://localhost:8000/reward",
      "temperature": 1.0
    }
  }'

# Stream GRPO progress (SSE)
curl -N http://localhost:11434/admin/api/fine-tune/grpo/jobs/{job_id}/stream
```

#### GRPO Config

| Field | Default | Description |
|-------|---------|-------------|
| `group_size` | 4 | Completions sampled per prompt for advantage normalization |
| `iters` | 50 | Training iterations |
| `batch_size` | 2 | Prompts per iteration |
| `learning_rate` | 1e-5 | AdamW learning rate (LoRA params only) |
| `lora_layers` | 16 | Number of layers to convert to LoRA |
| `lora_rank` | 8 | LoRA rank |
| `lora_alpha` | 16.0 | LoRA scale (alpha) |
| `lora_dropout` | 0.0 | LoRA dropout |
| `max_completion_len` | 64 | Max tokens sampled per completion |
| `clip_ratio` | 0.2 | PPO ratio clip |
| `reward_endpoint` | "" | HTTP reward server (`POST {prompt, completions} -> {rewards}`); length-based fallback if empty |
| `temperature` | 1.0 | Sampling temperature (0 = greedy) |
| `seed` | 0 | MLX RNG seed |
| `reward_timeout` | 30.0 | Reward endpoint timeout (seconds) |

### DPO / ORPO Preference Alignment (#399)

Direct Preference Optimization (DPO) and Odds Ratio Preference Optimization
(ORPO) align a LoRA policy to human preference pairs `{prompt, chosen, rejected}`.

- **DPO** trains against a frozen reference model (base, no adapter) loaded
  on demand and evicted after each step — same memory-bounded pattern as
  GRPO. Loss: `-log σ(β·((π_θ(w)−π_ref(w)) − (π_θ(l)−π_ref(l))))`.
- **ORPO** folds the reference into an odds-ratio penalty (SFT NLL on chosen
  + `λ·log σ(log p_w − log p_l)`) — **no reference model**, lower memory.

```bash
# DPO
curl -X POST http://localhost:11434/admin/api/fine-tune/dpo/jobs \
  -H "Content-Type: application/json" \
  -d '{
    "model_id": "qwen3.5-9b",
    "preference_pairs": [
      {"prompt": "What is MLX?", "chosen": "MLX is Apple's array framework.", "rejected": "idk"},
      {"prompt": "What is LoRA?", "chosen": "Low-rank adaptation.", "rejected": "a fruit"}
    ],
    "adapter_name": "dpo-pref",
    "config": {"iters": 50, "batch_size": 2, "beta": 0.1, "lora_layers": 16}
  }'

# ORPO (same shape, different endpoint)
curl -X POST http://localhost:11434/admin/api/fine-tune/orpo/jobs \
  -H "Content-Type: application/json" \
  -d '{
    "model_id": "qwen3.5-9b",
    "preference_pairs": [
      {"prompt": "What is MLX?", "chosen": "MLX is Apple's array framework.", "rejected": "idk"}
    ],
    "config": {"iters": 50, "lambda_odds": 1.0, "lora_layers": 16}
  }'

# Stream progress (SSE): dpo_step / orpo_step {iter, total_iters, loss, reward_margin, acc_chosen}
curl -N http://localhost:11434/admin/api/fine-tune/dpo/jobs/{job_id}/stream
```

#### DPO / ORPO Config

| Field | Default | Description |
|-------|---------|-------------|
| `method` | `dpo` | `dpo` (uses ref model) or `orpo` (odds-ratio, no ref); forced by endpoint |
| `iters` | 50 | Training iterations |
| `batch_size` | 2 | Preference pairs per iteration |
| `learning_rate` | 1e-5 | AdamW learning rate (LoRA params only) |
| `lora_layers` | 16 | Number of layers to convert to LoRA |
| `lora_rank` | 8 | LoRA rank |
| `lora_alpha` | 16.0 | LoRA scale (alpha) |
| `lora_dropout` | 0.0 | LoRA dropout |
| `beta` | 0.1 | DPO temperature (ignored by ORPO) |
| `lambda_odds` | 1.0 | ORPO odds-ratio penalty weight (ignored by DPO) |
| `max_seq_length` | 1024 | Max prompt + completion tokens (truncated) |
| `seed` | 0 | MLX RNG seed |

### Key Behaviors

- **1 concurrent job** — Apple Silicon memory constraints; additional jobs queue automatically
- **Model eviction** — training evicts the target model from the inference pool; it reloads after completion
- **Adapter storage** — `~/.fusion-mlx/adapters/{model_id}/{adapter_name}/` with `adapters.safetensors` + `adapter_config.json`
- **Adapter serving** — hot-swap trained adapters into the EnginePool for inference without restart; `serve` loads, `unload` frees
- **SSE progress** — real-time metrics: train/val loss, learning rate, tok/s, peak memory, ETA
- **Job persistence** — jobs survive server restarts (stored in `~/.fusion-mlx/fine_tune_jobs.json`); stale RUNNING/QUEUED jobs auto-cancelled on reload
- **macOS App** — dedicated Fine-Tune screen with configuration form, dataset file picker, SSE live progress bar, job list, and adapter management

## Model Manager API (#302)

Non-admin API for model lifecycle management. Authenticated via scoped API keys (`model_mgr_*` prefix).

### Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/admin/api/model-manager/models` | List all models with load status, size, pinned flag, type |
| POST | `/admin/api/model-manager/models/{model_id}/load` | Load a model into the EnginePool |
| POST | `/admin/api/model-manager/models/{model_id}/unload` | Unload a model (fails if not loaded) |
| GET | `/admin/api/model-manager/models/{model_id}/status` | Single model status |
| GET | `/admin/api/model-manager/embedding/status` | List all embedding models with status |
| POST | `/admin/api/model-manager/embedding/{model_id}/pin` | Pin embedding model (prevent eviction) |
| POST | `/admin/api/model-manager/embedding/{model_id}/unpin` | Unpin embedding model |

### Scoped API Key

```bash
# Generate a model-manager key
curl -X POST http://localhost:11434/admin/api/keys \
  -H "Authorization: Bearer <admin-key>" \
  -d '{"role": "model_manager"}'
# Returns: {"key": "model_mgr_...", "role": "model_manager"}

# Use it to list models
curl http://localhost:11434/admin/api/model-manager/models \
  -H "Authorization: Bearer model_mgr_..."
```

### Capabilities Field

The `/v1/models` endpoint now includes a `capabilities` array derived from each model's alias profile:

```json
{
  "id": "qwen3-72b",
  "capabilities": ["dflash", "dflash2", "dspark", "spec_decode", "moe"]
}
```

Derived from: `supports_dflash` (→`dflash`), `supports_dflash2` (→`dflash2`), `supports_dspark`, `supports_spec_decode`, `tool_call_parser`, `reasoning_parser`, `supports_mllm` (→`vision`), `is_audio` (→`audio`), `is_moe` (→`moe`), `is_hybrid` (→`hybrid`).

The CLI `models` command also displays a unified `Capabilities` column instead of the previous 4 separate columns.

## Security

fusion-mlx is the link endpoint in a 3-tier chain: App -> Gateway -> MLX. By default it binds to `127.0.0.1` (loopback only), so it is not exposed on the LAN. The controls below harden access when it must listen on a wider interface or sit behind a gateway (#342-#346).

### Environment variables

| Variable | Default | Effect |
|----------|---------|--------|
| `server.host` (config) / `--host` | `127.0.0.1` | Bind address. `0.0.0.0` exposes the server on all interfaces - only do this behind a gateway. |
| `FUSION_ROUTE_ENFORCE` | `true` | When `true` (default since v0.7.0, #349), requests missing the `X-Fusion-Route` header are rejected with `403`. Accepted as an explicit opt-in (redundant with the default, kept for backward compatibility). |
| `FUSION_ROUTE_WARN_ONLY` | `false` | Dev/standalone override (#349). When `true`, restores phase-1 warn-only behavior: a missing `X-Fusion-Route` is logged at WARN and allowed. Set this for standalone local-server use without a gateway. |
| `FUSION_ALLOW_ANONYMOUS` | `false` | Dev override. When `true`, requests without an API key are allowed. Does **not** bypass a configured `api_key` - a matching key is still required when one is set. |
| `FUSION_ROUTE_TOKEN` | _(unset)_ | #352. Optional shared secret for cross-host gateway→MLX auth. When set, `X-Fusion-Route`'s value must equal this token (constant-time compare); missing/mismatched → `403 invalid_route_token`. When unset, `X-Fusion-Route` keeps its provenance-only presence check (#343). The token is enforced even under `FUSION_ROUTE_WARN_ONLY=true` (stricter wins). |
| `FUSION_TENANT_ISOLATION` | `false` | #756. Multi-tenant mode for deployments behind fusion-gateway. When `true`, the backend half of the gateway contract is enforced: the `X-Fusion-Route` **value** must equal `gateway-decision` (a bare presence check is not enough — a direct-port caller could inject any value), and `X-Fusion-Tenant` must be present and non-empty. Missing/wrong route value → `403 invalid_route_origin`; missing tenant → `403 missing_tenant`. Per-tenant state (sessions, stats, context caps) is scoped by composing the tenant into the per-caller principal. `FUSION_ROUTE_TOKEN` (stricter) takes precedence when both are set. Default OFF: single-tenant dev is unaffected. |
| `FUSION_MLX_ALLOWED_READ_DIRS` | _(unset)_ | #633. Colon-separated list of extra directories appended to the path-traversal read allow-list (`~/.fusion-mlx/models`, `~/.fusion-mlx/cache`, `/tmp`, `/var/tmp`). Lets scene-continuity condition images (i2va first-frame, l2va last-frame) from custom output dirs (e.g. fusion-comfyui) pass `is_safe_local_path` without writing to `/tmp`. |
| `FUSION_INSTANCE_ID` | _(unset → `<hostname>:<pid>`)_ | #754. Stable instance identity so a gateway/CLI doing health-driven failover can distinguish replicas. Operator-set (e.g. `mlx-node-1`); when unset a stable `<hostname>:<pid>` fallback is derived once per process. Surfaced on `/health`, `/healthz`, and the `/v1/drain` responses. |
| `FUSION_SESSION_STATE_DIR` | _(unset → in-memory only)_ | #754. Directory for `SessionTracker` JSON snapshot persistence. When set, per-session token-usage stats are periodically snapshotted to `<dir>/sessions.json` (atomic tmp-write + `os.replace`) and rehydrated on startup so a failover/restart does not lose cumulative context. Off by default: single-process dev keeps the original in-memory-only behavior. |
| `FUSION_CODE_SANDBOX` | `false` | #743. Opt-in gate for the code-sandbox reward endpoint (`POST /admin/api/fine-tune/reward/code`). When `on`/`1`/`true`, model-generated code + dataset tests are executed under a macOS `sandbox-exec` deny-by-default profile to score GRPO completions by unittest pass rate. When unset (default), untrusted code execution is OFF and the endpoint returns `503` fail-visible. Mirrors the trainer-side `FUSION_CODE_SANDBOX_TRUSTED` posture — never on by default. |

### Server-side HA — drain, instance identity, health richness (#754)

For deployments running multiple fusion-mlx replicas behind a gateway (or a CLI doing health-driven failover), four bounded primitives let the server participate in graceful failover without a shared coordination store — each instance runs independently.

- **Drain for graceful failover.** `POST /v1/drain` (admin-gated, `verify_api_key_or_x_api_key`) flips a runtime `draining` flag. While draining, `/healthz` and `/health/ready` return `503` so a gateway routes new requests away, but in-flight work is **not** aborted — models stay loaded so queued requests finish. `DELETE /v1/drain` clears the flag and restores serving. Both are idempotent. The routes do not unload models; use them to take an instance out of rotation cleanly before maintenance, then bring it back.
- **Instance identity.** Every `/health`, `/healthz`, and drain response carries `instance_id` (and `version`) so a gateway can tell replicas apart and route around a draining one. Set `FUSION_INSTANCE_ID` to an operator-chosen name; unset it for the auto-derived `<hostname>:<pid>` fallback.
- **Rich `/health`.** `/health` now returns `version`, `instance_id`, `draining` (bool), and a single `status` field that is `"healthy"`, `"preloading"`, or `"draining"` — one field for a gateway's routing rules to branch on, with `ready` false during preload or drain.
- **Persistent session state.** Set `FUSION_SESSION_STATE_DIR` to a directory and the in-memory per-session token tracker snapshots to `sessions.json` (debounced ~5 s, atomic replace) and rehydrates on restart, so a failover does not lose cumulative per-session usage context. Off by default.

**Run mode:** multi-instance coordination is *each instance independent* — there is no leader election or shared request queue. The real HA primitive here is the drain flag (graceful failover) + persistent session state (context survival) + health richness (gateway routing). Client-side failover (retry the next replica on a 503/connection error) is handled by fusion-cli (#48); gateway routing rules live in fusion-gateway.

### Code-sandbox reward endpoint (#743)

For GRPO fine-tuning on coding tasks, the reward signal must come from **running** the model's generated code against a dataset test suite — but executing untrusted model output on the operator's machine is unsafe. This endpoint centralizes the isolation in fusion-mlx so the trainer stays a pure HTTP delegation layer.

`POST /admin/api/fine-tune/reward/code` (admin-gated, `require_admin`) accepts:

```json
{
  "code": "def add(a, b):\n    return a + b\n",
  "tests": "import unittest\n\nclass TestSolution(unittest.TestCase):\n    def test_add(self):\n        self.assertEqual(add(1, 2), 3)\n",
  "timeout": 30
}
```

and returns the pass-rate reward plus diagnostic capture:

```json
{
  "reward": 1.0,
  "passed": 1,
  "total": 1,
  "timed_out": false,
  "stdout": "...",
  "stderr": "...",
  "error": ""
}
```

- **Isolation.** The combined `{code}\n{tests}` module is run under `python -m unittest -v solution` inside a macOS `sandbox-exec` deny-by-default profile: network denied (no exfil/C2), home tree read-only (no `~/.ssh`, no dotfile tampering), only the per-run work dir writable, and **process-fork denied** so a poisoned completion (`import os; os.system(...)`, `subprocess.run`, `os.fork`) cannot spawn a sibling process. The interpreter itself runs; further forks are refused.
- **Reward.** `reward = passed / total` parsed from the unittest verbose output. `Ran N tests` gives the total; `... ok` markers give the passed count (robust to the word "FAIL" in an assertion body). A collection error (no summary line) yields `total=0` → `reward=0.0`.
- **Timeout.** `timeout` (seconds, default `30`, must be positive) is the only wall-clock backstop against infinite loops — `unittest` itself has no time cap. A timeout returns `timed_out: true`, `reward: 0.0`, `error: "timed out after Ns"`.
- **Fail-visible gate.** The endpoint is OFF by default. Set `FUSION_CODE_SANDBOX=on` to enable; unset it returns `503` (not a silent `200 reward 0.0`, which would read as "code ran, scored 0"). On a non-macOS host or a macOS host without `sandbox-exec`, the runner refuses to execute unsandboxed and returns a `503` with a `sandbox-exec not found` message. Missing/invalid body → `400`; runner exception → `500`.
- **Why here, not the trainer.** Keeping the isolation primitive in fusion-mlx means one hardened surface for all training clients (the trainer, the CLI, future tooling) instead of N per-trainer sandboxes drifting apart. The trainer sends `{code, tests}` over HTTP and gets a reward number back — it never touches untrusted code directly.

**Run mode:** single-process, one sandboxed python child per request (process-fork is denied, so the child cannot fan out). No persistent sandbox pool; each request gets a fresh temp work dir that is removed after the run. macOS-only — the deny-by-default profile is `sandbox-exec` (no Linux/containerd equivalent is shipped).

### CORS (#641 / #675)

CORS is opt-in via environment variables. With no `FUSION_MLX_CORS_*` env set, the server mounts a wildcard `*` origin policy (friendly single-machine default: a local browser frontend can call the API with no extra config) with credentials disabled and `POST,GET,OPTIONS` methods. Lock down for multi-tenant / browser-facing deployments with the variables below.

| Variable | Default | Effect |
|----------|---------|--------|
| `FUSION_MLX_CORS_ALLOW_ORIGINS` | _(unset → `*`)_ | CSV origin allowlist, e.g. `https://chat.openai.com,https://claude.ai`. A whitespace-only value (e.g. `" , ,, "`) is treated as an operator templating bug and **fails closed**: no CORS middleware is mounted and preflight returns `405` (with a WARNING log). Explicit `*` is a valid single-machine choice. |
| `FUSION_MLX_CORS_ALLOW_METHODS` | `POST,GET,OPTIONS` | CSV method allowlist for preflight. `*` expands to all methods. A whitespace-only value warns and falls back to the default (rather than silently broadening). |
| `FUSION_MLX_CORS_ALLOW_HEADERS` | env-path: `content-type,authorization,x-rapid-mlx-internal`; CLI-path: `*` | CSV request-header allowlist. On the **env-driven** origins path the default is narrowed (F-091); on the legacy `--cors-origins` CLI path the default stays wide-open `*` for back-compat. A whitespace-only value warns and falls back to the path-appropriate default. Allowlist custom headers (`OpenAI-Organization`, `X-Requested-With`, …) here. |
| `FUSION_MLX_CORS_MAX_AGE` | `3600` | Preflight result cache lifetime in seconds. A non-integer or negative value warns and falls back to `3600`. Replaces Starlette's silent 600 s default. |
| `FUSION_MLX_CORS_ALLOW_CREDENTIALS` | `false` | Opt-in credentials. `true`/`1`/`yes`/`on` enables `Access-Control-Allow-Credentials: true` so cookie / `Authorization`-bearing fetches succeed. Wildcard `*` origins force `false` per the fetch spec. |

**#675 migration notes (reverses #641 behavior):**

- **Credentials are now opt-in (default `false`).** `#641` auto-enabled credentials on any explicit origin (`allow_credentials=bool(origins)`). If you relied on cookies with an explicit origin allowlist, set `FUSION_MLX_CORS_ALLOW_CREDENTIALS=true`.
- **Env-path headers are narrowed.** If you serve origins via `FUSION_MLX_CORS_ALLOW_ORIGINS` and your browser client sends custom headers, allowlist them with `FUSION_MLX_CORS_ALLOW_HEADERS`. The `--cors-origins` CLI flag path is unchanged (still `*`).

### API key precedence (#632 / #636)

The effective API key is resolved once at startup with a fixed priority, and synced to every read path (the `verify_api_key` middleware, the admin auth, and the config singleton) so they agree:

**CLI `--api-key`  >  `FUSION_MLX_API_KEY` env  >  `settings.json` `auth.api_key`**

- `--api-key <X>` on the command line wins over both env and `settings.json`.
- `FUSION_MLX_API_KEY` wins over `settings.json`.
- `settings.json` `auth.api_key` is the fallback for bare `serve` launches.
- If none are set, anonymous access is governed by `FUSION_ALLOW_ANONYMOUS` (rejected by default).

This applies to **every serve path**: single-model (`serve <model>`), audio (`serve kokoro`), and `serve --model-dir <dir>` / `serve --base-path <dir>`. #636 fixed the `--model-dir` / `--base-path` path, which previously dropped `--api-key` and fell back to `settings.json`, so `/v1/*` rejected the CLI key with `401 Invalid API key` while enforcing the `settings.json` key.

`start.sh` resolves the key from `FUSION_MLX_API_KEY` (env) then `settings.json`, passes it as `--api-key`, and exports `FUSION_MLX_API_KEY` into the server process — so the env path also catches it for `--model-dir` launches.

### Keychain API-key storage (#770)

By default the API key is stored in `~/.fusion-mlx/settings.json` (mode `0o600`). For macOS deployments that prefer the Keychain over plaintext on disk, set `FUSION_KEYCHAIN=on`:

- The key is read from / written to the macOS Keychain (service `fusion-mlx.api-key`, account `fusion-mlx`) via the shipped `security` CLI — no extra dependency.
- Read priority becomes **CLI `--api-key` > `FUSION_MLX_API_KEY` env > Keychain > `settings.json` `auth.api_key`**.
- On startup, if a plaintext `api_key` exists in `settings.json` but the Keychain is empty, it is migrated into the Keychain and the plaintext field is cleared from disk.
- Admin settings writes route to the Keychain and leave the on-disk `api_key` field empty.
- On non-macOS, or if the `security` CLI is absent, every call fails visibly (logged) and the plaintext `settings.json` path is used unchanged. Default off.

### Rate limiting (#635)

`--rate-limit N` caps requests per minute per client (default `0` = disabled). `--rate-limit 0` explicitly disables the limiter; a positive value enables it. #635 fixed `--rate-limit 0` leaving the limiter active at its 60 rpm default on the `serve <model>` and `serve --model-dir` paths.

### Metrics

The `/metrics` endpoint exposes Prometheus-format series (requires `verify_management_access` — see Access policy below). Request counters:

- **`fusion_mlx_requests_total`** — total processed requests.
- **`fusion_mlx_requests_cancelled_total`** — client-disconnected requests, ticked from the live streaming `CancelledError` handler and the `/v1/responses` non-stream disconnect-wait (#645).
- **`fusion_mlx_prompt_tokens_total`** / **`fusion_mlx_completion_tokens_total`** — token throughput.

### Access policy

- **Route guard (#343):** routed requests should carry `X-Fusion-Route: gateway` so the server knows they came through the gateway. Exempt paths: `/`, `/health`, `/healthz`, `/readyz`, `/livez`, `/openapi.json`, `/docs`, `/redoc`, `/favicon.ico`, and `OPTIONS` preflight. Enforce is the default since v0.7.0 (#349): un-routed traffic is rejected with `403`. Set `FUSION_ROUTE_WARN_ONLY=true` to restore warn-only behavior for standalone use. The header is routing provenance only - it does **not** authenticate a caller (any client can set it). For cross-host deployments where the gateway is on a different machine, set `FUSION_ROUTE_TOKEN` (#352) to upgrade `X-Fusion-Route` from spoofable provenance to a shared-secret credential: its value must equal the token, else `403 invalid_route_token`.
- **Management endpoints (#344):** `/metrics` and `/v1/status` require `verify_management_access` - a valid API key or `FUSION_ALLOW_ANONYMOUS=true`. Since v0.7.0 (#350) loopback no longer exempts management endpoints: a same-host client (including a co-located gateway) must forward a valid API key, or set the dev override. `X-Fusion-Route` is not accepted as authentication.
- **Model lifecycle (#345):** `/v1/models/load` and `/v1/models/unload` require `X-Fusion-Source: model-hub` (or a loopback client); otherwise `403`. **#631:** the gui_compat router registers these paths first and shadows the engine-pool handler; its `unload` falls back to the engine pool (`_unload_pool_model`) when the model is loaded in the pool but not tracked in the gui database, so a loaded model is actually unloaded (no memory leak).
- **Anonymous access (#346):** rejected by default. Allow only for local dev via `FUSION_ALLOW_ANONYMOUS=true`. Since v0.7.0 (#350) loopback clients are no longer exempt - a same-host client (including a co-located gateway) must present a valid API key. A gateway must forward a valid API key; `X-Fusion-Route` alone does not authenticate.
- **Multi-tenant isolation (#756):** opt in with `FUSION_TENANT_ISOLATION=true` for deployments where fusion-gateway serves multiple tenants off one MLX backend. The gateway derives an authoritative tenant from the `api_key` → team binding and stamps it on every upstream request. The backend enforces the matching half:
    - **Origin gate.** `X-Fusion-Route`'s **value** must equal `gateway-decision` (not just present). Any other value → `403 invalid_route_origin`. This blocks a direct-port caller from injecting an arbitrary route header to bypass the gateway's tenant binding. `FUSION_ROUTE_TOKEN` (#352, shared-secret) is stricter and takes precedence when both are set.
    - **Tenant stamp.** `X-Fusion-Tenant` must be present and non-empty, else `403 missing_tenant`. A valid route without a tenant means a bypassed/misconfigured gateway and must not reach handlers (no tenant = unscoped state access).
    - **Per-tenant state.** Session stats and context caps (`/v1/sessions/*`, `/v1/context/budget`) are scoped by composing the tenant into the per-caller principal (`t:<tenant>:<bucket>`), so two tenants that share a bearer-key shape cannot read or mutate each other's sessions — a foreign-tenant lookup returns `404`, preserving the existing IDOR non-disclosure guarantee cross-tenant.
    - **`X-Space-Id` is non-authoritative.** It is a client-supplied passthrough the gateway may forward, but it is deliberately ignored for tenant derivation. A spoofed `X-Space-Id` must not cross tenant boundaries.
    - Default OFF: single-tenant and standalone dev deployments keep the legacy behavior. Tenant→model ACL (restricting which models a tenant may load) is out of scope here and tracked as a follow-up; the current isolation covers per-tenant **state** + origin/tenant stamping only.

```bash
# Multi-tenant deployment behind fusion-gateway
FUSION_TENANT_ISOLATION=true fusion-mlx serve --model-dir ~/.fusion-mlx/models
# Gateway stamps on every upstream request:
#   X-Fusion-Route: gateway-decision
#   X-Fusion-Tenant: <team-from-api-key-binding>
```

```bash
# Bind loopback only (default)
fusion-mlx serve --model qwen3.5-4b-4bit --host 127.0.0.1 --port 11434

# Standalone local server (no gateway): opt into warn-only route guard
FUSION_ROUTE_WARN_ONLY=true fusion-mlx serve --model qwen3.5-4b-4bit

# Behind a gateway (default since v0.7.0: enforce X-Fusion-Route)
fusion-mlx serve --model qwen3.5-4b-4bit

# Cross-host gateway (#352): shared-secret on X-Fusion-Route value
FUSION_ROUTE_TOKEN=$(cat /etc/fusion/gateway.token) fusion-mlx serve --model qwen3.5-4b-4bit
# Gateway then sends: X-Fusion-Route: <same-token>
```

### Unix Domain Socket (UDS) listen mode (#351)

For gateway deployments, UDS provides **transport-layer physical isolation** on top of the auth chain (#349/#350): MLX listens on a Unix socket instead of a TCP port, so only a process with filesystem access to the socket file can connect. A same-host process without access to the socket path cannot reach MLX at all.

- Trigger: `--host unix:/path/to.sock` (the `unix:` prefix selects UDS mode).
- The socket is created with owner-only `0600` permissions *before* it accepts connections (no race window where it is world-connectable).
- No TCP port is opened in UDS mode; `--port` is ignored.
- Backward compatible: `--host 127.0.0.1` (the default) keeps TCP loopback behavior unchanged.
- The gateway connects over the socket, e.g. `curl --unix-socket /path/to.sock http://localhost/health`.
- `fusion-mlx ps` shows the socket path in the `ADDR` column so UDS servers are discoverable for stop/status.

```bash
# UDS listen mode - only filesystem access to the socket can reach MLX
fusion-mlx serve --model qwen3.5-4b-4bit --host unix:/run/fusion-mlx.sock

# Gateway-side health check over the socket
curl --unix-socket /run/fusion-mlx.sock http://localhost/health

# Via start.sh (sets --host, drops --port, health-checks over the socket)
FUSION_HOST=unix:/run/fusion-mlx.sock ./start.sh start
```

> UDS is orthogonal to the #349/#350 auth chain: even over the socket, a valid API key is still required when one is configured. UDS removes the *transport* reachability; auth removes *request* authorization. Use both for defense in depth.

## Performance

Benchmarks on Apple M5 Max (128 GB RAM, 40 GPU cores), MLX 0.32.0.dev - 2026-07-04.
Single-stream decode, Qwen3.6-27B-mxfp8 (100 tokens, 5 warmup steps):

| Engine | TG mean (tok/s) | median | std | CV | step (ms) |
|---|---|---|---|---|---|
| fusion-mlx | 18.46 | 18.52 | 0.18 | 1.0% | 54.17 |
| fusion-mlx | 18.49 | 18.53 | 0.18 | 1.0% | 54.09 |

Ratio 0.998 - full parity. Speculative decoding is auto-gated off for GatedDeltaNet hybrid models to preserve coherence.

Prefill throughput (tok/s):

| Prompt tokens | 64 | 128 | 256 | 512 | 1024 | 2048 |
|---|---|---|---|---|---|---|
| tok/s | 421 | 657 | 733 | 669 | 692 | 722 |

Batched decode, fusion-mlx (aggregate / per-request tok/s):

| Batch size | 1 | 2 | 4 |
|---|---|---|---|
| Aggregate TG | 18.09 | 17.75 | 16.61 |
| Per-request TG | 18.09 | 8.87 | 4.15 |

> Earlier README figures (TG 29.8 tok/s, concurrent 36.0 tok/s) were measured with speculative decoding enabled, which corrupted output on this hybrid recurrent model. The numbers above are coherent (spec decode auto-gated off) and reflect real usable throughput. M5 Max coherent ceiling for 27B mxfp8 is ~18.5 tok/s.

Submit your own benchmarks at [bench.dpdns.org](https://bench.dpdns.org/).

### Paged KV Cache

A paged KV cache backend (`fusion_mlx/custom_kernels/`) that stores KV blocks
in a flat physical pool and addresses them through a per-request `block_table`,
eliminating the per-step `mx.concatenate` of the default `KVCache`. Three
independent opt-in phases, all default **OFF**:

- **Phase 1 — `FUSION_PAGED_KV=on`**: paged KV cache backend. Decode reads K/V
  from the flat pool via `block_table` indirection. Bit-exact vs upstream
  `KVCache` for greedy decode (fp16 tol).
- **Phase 2 — `FUSION_PAGED_FUSED_KERNEL=on`**: fused decode-attention Metal
  kernel (llama-family: llama / qwen2 / qwen3). Reads K/V directly from the
  physical block pool through `block_table` indirection inside the kernel, so
  the per-step concat is eliminated entirely on the decode path. Prefill keeps
  the concat path. Default off until the tiled kernel lands (see Limitations).
  Perf report: `~/fusion/audit/paged-kv-phase2-perf-report.md`.
- **Phase 3 — `FUSION_PAGED_POOL=on` + `FUSION_PAGED_POOL_NUM_BLOCKS=<cap>`**:
  shared `FusionPagedKVPool` for continuous batching. One physical pool, shared
  free-list, per-request `block_table`. Bounds memory by the cap; when the pool
  is exhausted a new request is rejected with `503` (no head-of-line blocking).
  Sequential-per-request submission through the shared pool is validated
  bit-exact with no cross-contamination. Perf report:
  `~/fusion/audit/paged-kv-phase3-concurrency-perf-report.md`.

| Variable | Default | Effect |
|----------|---------|--------|
| `FUSION_PAGED_KV` | `off` | Phase 1 paged KV cache backend. |
| `FUSION_PAGED_FUSED_KERNEL` | `off` | Phase 2 fused decode-attention Metal kernel (llama-family). |
| `FUSION_PAGED_POOL` | `off` | Phase 3 shared `FusionPagedKVPool` for continuous batching. |
| `FUSION_PAGED_POOL_NUM_BLOCKS` | `256` | Pool cap (block count) for `FUSION_PAGED_POOL`. |

Limitations:

- The Phase 2 fused kernel allowlist covers llama-family attention (llama /
  qwen2 / qwen3). Gemma / Mistral sliding-window attention is not yet supported
  — non-llama-family follow-up issue tracked.
- The Phase 2 naive scalar kernel is slower than the concat path; a
  Steel-style tiled/threadgroup-optimized kernel is required before the fused
  path can be enabled by default. Follow-up issue tracked.
- The Phase 3 shared pool rejects with `503` when the cap is exhausted; LRU
  eviction (freeing the least-recently-used request's blocks) is deferred.
  Follow-up issue tracked.
- True simultaneous batched decode (B>1) through `BatchedEngine` is not yet
  wired end-to-end: `FusionPagedRequestCache.merge` (B=N stack) is unit-tested
  but the `model_settings` -> `FusionConfig.from_model_settings` plumbing for
  `fusion_paged_pool="on"` is not hooked into the engine. Sequential-per-request
  submission is the validated path. Follow-up issue tracked.

### Video Generation (SkyReels-V3)

Pure-MLX port of SkyReels-V3 (R2V / V2V / A2V), running end-to-end on real weights
(full 40-layer DiT forward, no stubs). Benchmarks on Apple M5 Max (128 GB, 40 GPU
cores), MLX 0.32.0, 2026-07-18, bfloat16, 5 frames 256P latent:

| Branch | Model | Weight size | Load (s) | DiT fwd (s/step) | Metal peak (GB) | FPS/step | Status |
|---|---|---|---|---|---|---|---|
| R2V | Reference-to-Video 14B | 28.6 GB (`transformer/`) | 6.84 | **0.092** | 75.3 | **54.3** | ✅ runs |
| V2V | Video Extension 14B | 75 GB (14+6+1 shards) | 3.11 | **0.329** | 82.7 | **15.2** | ✅ runs (mx.compile fusion 3.3×) |
| A2V | Talking Avatar 19B | 123 GB (18+6+1+1+1 shards) | 3.16 | **0.328** | 24.8 | **3.0** | ✅ runs (audio_cross_attn+norm_x rebuild + kv_linear transpose + mx.compile, 18× speedup) |

The PyTorch -> MLX conversion products (`convert_skyreels_v3.py`) total 24 GB
(R2V-14B), 75 GB (V2V-14B), 123 GB (A2V-19B) across sharded DiT/T5/VAE/CLIP/audio
safetensors.

**Performance knobs:**

```bash
# Reduce sampling steps (default 30; 720p 30->20 ≈ -33% wall-clock, UniPC order-2 stays stable)
FUSION_SKYREELS_STEPS=20 fusion-mlx serve --model SkyReels-V3-R2V-14B-MLX
# DiT weight quantization at load: w8a16 / w4 / nf4 (default off = full bf16)
FUSION_SKYREELS_QUANT=w8a16 fusion-mlx serve --model SkyReels-V3-R2V-14B-MLX
# Dynamic CFG: early steps run cond+uncond (b=2), late steps cond-only (b=1, ~half compute)
FUSION_SKYREELS_DYNAMIC_CFG=1 FUSION_SKYREELS_CFG_KEEP_RATIO=0.6 fusion-mlx serve --model SkyReels-V3-R2V-14B-MLX
# Toggle warmup precompile (default on)
FUSION_SKYREELS_WARMUP=0 fusion-mlx serve --model SkyReels-V3-R2V-14B-MLX
```

> **xfuser + mx.compile are fundamentally incompatible** (T1-3): `mx.compile` bakes
> the pre-attach `_fast_attn=None` into the trace, so xfuser is a runtime no-op
> (`fa_calls=0`); attaching after compile forces per-step recompile that cancels
> the compile win. Do not attempt to make xfuser effective under `mx.compile`. Use
> `FUSION_SKYREELS_STEPS` to cut wall-clock instead.

> Full bug-fix history (#139 weight loading, #144 R2V reshape, #148 video timeout,
> #149 progress logs, #154 Tier-1 tuning) and the T5/VAE end-to-end fix details
> are documented in [README_CN.md](README_CN.md).

### Video Backend Registry

The video generation API auto-detects the backend from the model name and routes
to the correct pure-MLX implementation. Supported backends:

| Backend | Key | Models | I2V | Status |
|---|---|---|---|---|
| LTX-2 | `ltx2` | LTX-2, LTX-2.3 | ✅ | ✅ shipped |
| Wan2 | `wan2` | Wan2.1, Wan2.2 (TI2V), VACE-14B | ✅ | ✅ shipped |
| SkyReels-V3 | `skyreels` | R2V/V2V/A2V 14B-19B | ✅ (R2V) | ✅ shipped |
| Legacy LTX-Video | `ltx_video_legacy` | LTX-Video 0.9.x | ✅ | ✅ shipped |
| SVD | `svd` | Stable Video Diffusion XT | ✅ | ✅ #212 |
| Cosmos | `cosmos` | 7B T2V + Predict2 2B I2V | ✅ (Predict2) | ✅ #213 |
| HunyuanVideo | `hunyuanvideo` | HunyuanVideo | ✅ | ✅ #214 |
| MiniMax-H3 | `minimax_h3` | H3 33B (FL2VA/Ref2VA) | — | ✅ #588 native audio |
| CogVideo | `cogvideo` | CogVideoX | — | stub (no MLX port) |

Aliases: `svd-xt`, `stable-video-diffusion`, `cosmos-1.0`, `predict2`,
`video2world`, `hunyuan-video`, `hunyuan_video`, `cogvideox`, `ltx-video`, `wan`.

> **Video DiT throughput (#367):** HunyuanVideo and Cosmos fuse the
> uncond+cond CFG pair into a single batched B=2 DiT forward (~2x
> throughput, no quality change). `cfg_scale <= 1.0` skips the uncond
> branch (single-forward shortcut). Step-level it/s INFO logging
> reports progress for ComfyUI and makes hangs vs. slow steps
> diagnosable. Wan2/VACE already used batched CFG.

> **Wan2 self-attn overflow (#500):** Wan2.2-TI2V-5B at large
> resolutions (e.g. 1280×704×121f → seq=27280) overflows Metal on a
> single `(B,H,seq,seq)` attention matrix, yielding all-NaN latents
> and a static video. When the transformer seq exceeds the safe
> threshold (16384), `generate_video` auto-enables Q-chunking
> (`FUSION_WAN2_ATTN_CHUNK=8192`) so users never hit the NaN by hand.
> A user-set `FUSION_WAN2_ATTN_CHUNK` is honored as-is (set `0` to
> force-off). If NaN latents still reach the VAE, decode now fails
> visibly (raises) instead of silently zeroing pixels into a static
> clip.

### VACE: Video-Conditioned Auxiliary Control (Wan2.1-VACE-14B)

VACE enables **Video-to-Video (V2V)** and **Audio-to-Video (A2V)** control on
Wan2.1-VACE-14B via `control_video`, `control_mask`, and `reference_images`.

| Parameter | Type | Description |
|---|---|---|
| `control_video` | `string` | Input video path/URL/data-URI to be controlled. Required for V2V. |
| `control_mask` | `string` | Mask video path/URL/data-URI. Black=conditioning region, white=generation region. Optional — defaults to all-white (full generation). |
| `reference_images` | `string[]` | Reference image paths/URLs/data-URIs for subject-driven conditioning. Optional. |

```bash
# V2V: control video + mask (partial edit)
curl -X POST /v1/videos/generate \
  -H "Authorization: Bearer $KEY" \
  -d '{"model":"Wan2.1-VACE-14B","prompt":"A cat walking",
       "control_video":"/path/to/input.mp4",
       "control_mask":"/path/to/mask.mp4"}'

# V2V: control video only (auto-generates all-white mask)
curl -X POST /v1/videos/generate \
  -d '{"model":"Wan2.1-VACE-14B","prompt":"A dog running",
       "control_video":"/path/to/input.mp4"}'

# V2V with reference images
curl -X POST /v1/videos/generate \
  -d '{"model":"Wan2.1-VACE-14B","prompt":"A landscape",
       "control_video":"https://example.com/vid.mp4",
       "control_mask":"data:video/mp4;base64,...",
       "reference_images":["/path/to/ref.png"]}'
```

All media params accept **local paths**, **http(s) URLs**, and **data: URIs**.
URLs and data-URIs are downloaded/decoded to temp files automatically.


### Radix Text-Encoding Cache (#178)

In multi-shot pipelines the same prompt is re-encoded across shots (UMT5-XXL:
24 layers, 4096-dim, hundreds of ms to seconds per encode). `UMT5Encoder.encode_text`
is wired to `DiffusionRadixCache` (radix tree + LRU byte budget + pin/unpin); a
repeat hit on the same `prompt+max_length` returns the cached `mx.array` by
zero-copy reference, dropping text-encoding latency to ~0 ms.

- Cache key: `f"umt5:{max_length}:{sha256(prompt)[:16]}"`, per-encoder instance
  (auto-invalidated on model reload, no stale embeddings).
- Zero-copy: `mx.array` is immutable; a hit returns the cached reference directly.
- Stub mode is not cached (avoids zero-tensor pollution).
- Default LRU byte budget 512 MB (~128 entries for UMT5-XXL `[1,512,4096]` bf16).
- Env `FUSION_DIFFUSION_TEXT_CACHE` (default `"1"` on, `"0"` off).

```bash
fusion-mlx serve --model SkyReels-V3-R2V-14B-MLX                          # default on
FUSION_DIFFUSION_TEXT_CACHE=0 fusion-mlx serve --model SkyReels-V3-R2V-14B-MLX  # off (debug)
```

**Phase-2 additions:**

- **CLIP encoder wiring**: `CLIPTextEncoder.encode_text` (Flux/SD path) is now
  cached the same way — key `f"clip:{max_length}:{sha256(text)[:16]}"` (list
  inputs joined by `NUL`). A cache hit returns before `_ensure_loaded()`, so the
  CLIP model never loads on repeat prompts — real value beyond skipping the
  forward. Stub mode is not cached.
- **Admin stats endpoint**: `GET /v1/cache/stats` (admin-guarded) aggregates
  every live cache via a module-level `weakref` registry. Response:
  `{"cache_type": "diffusion_text_encoding", "caches": [{name, hits, misses,
  evictions, insertions, leaf_count, total_bytes, max_bytes, hit_rate}, ...],
  "totals": {cache_count, hits, misses, evictions, insertions, total_bytes,
  hit_rate}}`. Caches belonging to unloaded encoders are auto-dropped (weakref).
  This reports the diffusion text-encoding cache, not the LLM KV/prefix cache.

> **Scope**: phase-1 = full-key UMT5 cache (same prompt -> 0 ms). Phase-2 = CLIP
> wiring + admin stats endpoint. Phase-3 = session tail cache (multi-shot latent
> reuse via `session_id` on `/v1/videos/generate`, env
> `FUSION_SESSION_TAIL_CACHE=1` default OFF until E2E validated). Token-level
> prefix KV sharing for T5/UMT5 is **semantically invalid** — T5 is a
> bidirectional encoder (hidden state at position `i` depends on the full
> sequence), so prefix-hidden-state reuse corrupts output, unlike causal decoder
> LLMs; full-key caching is the correct approach.

### Cross-Restart Prefix Cache Persistence (#257)

The paged SSD cache (`BoundarySnapshotSSDStore`) is extended to persist LLM
prefix KV across server restarts, so a prompt prefix encoded in a previous
process can be reused without re-prefill. This is the LLM KV/prefix-cache
counterpart to the diffusion text-encoding cache above (which is full-key only;
token-level prefix reuse is valid for causal decoder LLMs, unlike bidirectional
T5/UMT5).

- **Write-hook**: on prefill completion, a prefix-keyed snapshot is captured
  (chain hash `hash_k = sha256(hash_{k-1} || block_k_tokens || model_name)` over
  `paged_cache_block_size`-token blocks of `prompt_token_ids`) and persisted to
  `_prefix_snapshots/` - a sibling of the ephemeral `_boundary_snapshots/` dir
  that survives restart. Writes run off the inference thread, LRU-bounded by
  byte budget.
- **Read-hook**: on a paged-cache miss, the prompt prefix is looked up; on a hit
  the cached blocks are materialized via `store_cache` + `reconstruct_cache` so
  prefill skips the cached prefix and resumes from `remaining_tokens`.
- **Safety**: only sliceable KV caches fully materialize (middle blocks sliced
  per-block, last block via snapshot); non-sliceable or hybrid caches fail fast
  and fall back to a clean full prefill - a partial block table is never
  promoted. VLM image requests are skipped (vision tokens live outside
  `prompt_token_ids`).
- Opt-in, default **off** - does not affect the existing paged-cache path when
  disabled.

#### Session-agnostic prefix cache & fork reuse (#386)

The prefix cache is keyed purely by **token-prefix chain hash**
(`hash_k = sha256(hash_{k-1} || block_k_tokens || model_name)`) — it has **no
session-id dimension**. Block hashes
(`fusion_mlx/cache/paged_cache.py` `compute_block_hash`) and lookups
(`BlockAwarePrefixCache.fetch_cache`, `PagedCacheManager.find_shared_prefix`)
consider only the token stream and model name, never `session_id` or
`request_id`. The `session_id` request field (`openai_models.py`) is confined
to per-session usage stats (#226).

By design this means **forked sessions share prefix KV automatically**: two
requests with identical `messages[:N]` (and identical tools / system /
chat-template-kwargs) produce identical `prompt_token_ids[:K]`, both hit the
same chain-hash blocks, and the second request reuses the cached prefix up to
the divergence point with no re-prefill — regardless of whether they share a
`session_id`. Adding `session_id` to the cache key would *defeat* this
cross-session reuse and lower hit rates, so it is intentionally excluded.

A copy-on-write `fork_cache(source_request_id, new_request_id)` helper exists
in `BlockAwarePrefixCache` for an explicit fork API if one is needed in the
future, but the automatic token-prefix path already covers the fork case.

```bash
# Enable cross-restart prefix persistence (default off)
FUSION_MLX_BOUNDARY_PREFIX_PERSIST=1 fusion-mlx serve --model <model>
# Cap the on-disk prefix snapshot budget (default 20 GiB)
FUSION_MLX_BOUNDARY_PREFIX_MAX_BYTES=53687091200 fusion-mlx serve --model <model>
```

> On restart, persisted prefix snapshots are scanned from `_prefix_snapshots/`
> and warm-start eligible requests log `Prefix snapshot warm-start for <id>:
> recovered N tokens in M blocks`. Config fields `boundary_prefix_persist` /
> `boundary_prefix_max_bytes` live on `SchedulerConfig`.

### Model-load admission & KV headroom (#355)

Before admitting a model, fusion-mlx projects memory as
`projected = current_footprint + model_size + kv_headroom`:

- **`model_size`** uses the **last observed** post-load footprint when available
  (persisted across unload), falling back to the static weight estimate. Re-loading
  a previously-seen model is admitted against its real cost, not an underestimate.
- **`kv_headroom`** reserves space for the live KV cache + activations, so an
  admitted model does not immediately OOM under concurrent requests. This closes
  the #355 admission under-projection (the weights-only estimate ignored runtime KV
  growth).

| Variable | Default | Effect |
|----------|---------|--------|
| `FUSION_MLX_ADMISSION_KV_HEADROOM_GB` | `min(max_kv_cache_memory, 2 GiB)` (≈ 2 GiB) | KV bytes reserved in the admission projection. Float in GiB. `0` disables the headroom (admit on weights alone, pre-#355 behavior). Invalid values warn and fall back to the default. Tracks `SchedulerConfig.max_kv_cache_memory` (default 4 GiB), capped at 2 GiB. |

```bash
# Reserve 1.5 GiB for KV cache in the load-admission projection
FUSION_MLX_ADMISSION_KV_HEADROOM_GB=1.5 fusion-mlx serve --model qwen3.5-27b-mxfp8
# Disable the headroom (admit on model weights alone)
FUSION_MLX_ADMISSION_KV_HEADROOM_GB=0 fusion-mlx serve --model qwen3.5-4b-4bit
```

> When a model alone exceeds the ceiling (`model_size + kv_headroom > ceiling`),
> the server raises `ModelTooLargeError`. When the model fits alone but no LRU
> victim can be evicted to free `model_size + kv_headroom`, it raises
> `InsufficientMemoryError`. Both log the projected footprint breakdown
> (`current / effective / kv_headroom`) at WARN for diagnosis.

### Metal wired memory limit (`iogpu.wired_limit_mb`, #356)

Metal's wired-memory allocator is capped by macOS at roughly 75% of unified
memory (`max_recommended_working_set_size`). fusion-mlx's ceiling model is
`Ceiling = min(static_ceiling, dynamic_ceiling, metal_cap)`, so when `metal_cap`
is the Apple default, the configured `memory_guard_tier` ceiling cannot be
reached above that ~75% line even if `static_ceiling` allows it.

To let fusion-mlx use more of unified memory for model weights + KV cache, raise
the kernel wired limit with:

```bash
# N = desired wired-memory ceiling in MB. Example: 96 GiB on a 128 GB Mac.
sudo sysctl iogpu.wired_limit_mb=98304
```

Persist it across reboots by appending to `/etc/sysctl.conf`:

```bash
echo 'iogpu.wired_limit_mb=98304' | sudo tee -a /etc/sysctl.conf
```

How to choose `N`:
- Leave ~10% RAM for the OS and other apps: `N_mb ≈ (total_ram_gb * 0.9) * 1024`.
- fusion-mlx reads the live value via `sysctl -n iogpu.wired_limit_mb` on
  startup; no restart of the daemon is needed if you set it before `serve`.
- `0` (unset) is the safe default - Metal keeps the Apple cap and fusion-mlx
  still clamps against it; no crash, just a lower effective ceiling.

At startup, if `iogpu.wired_limit_mb` is unset, fusion-mlx logs an `INFO` line
naming the current Apple cap and the `sudo sysctl` command to raise it, so the
hint is visible at the default log level (previously `DEBUG`).

<!-- Video Adapters section: documents IP-Adapter, ControlNet, AnimateDiff adapters.
  Importers: fusion_mlx.video.adapters.{ip_adapter,controlnet,animatediff}
  Callers: SkyReelsPipelineConfig, VideoGenParams, VideoGenerateRequest
  API: POST /v1/videos/generate {ip_adapter_image, ip_adapter_scale, controlnet_image,
       controlnet_strength, control_type, animatediff_scale}
  User instruction: "Continue the conversation from where it left off" (README update was pending task) -->

### Short-Drama MLX Submodules (PuLID / LatentSync / MuseTalk)

Three zero-PyTorch model ports for short-drama generation pipelines. All pure
MLX + numpy/cv2/insightface(CPU ONNX). Fusion-mlx provides the model inference
layer; [fusion-comfyui](https://github.com/dahai80/fusion-comfyui) handles
full pipeline orchestration (PuLID→Flux→LatentSync/MuseTalk).

| Submodule | Purpose | Key Models | Input → Output |
|---|---|---|---|
| **pulid_mlx** | Identity-preserving image generation | IDFormer + EVA02-CLIP-L-14-336 (24-layer ViT) + PerceiverAttentionCA | face image → 2048-d ID embedding → Flux DiT injection |
| **latentsync_mlx** | Audio-driven lip sync | UNet3D (13-ch) + DDIM + SD1.5 VAE + Whisper | video + audio → lip-synced video |
| **musetalk_mlx** | Realtime talking head | UNet2D (8-ch) + SD-VAE + WhisperEncoder | face + audio → animated face frames |

**Import:**

```python
from fusion_mlx.video import PuLIDPipeline, LipsyncPipelineMLX, MuseTalkPipeline
```

**Architecture highlights:**

- **PuLID-MLX**: IDFormer (Perceiver-resampler, dim=1024, depth=10) fuses ArcFace (1280-d) +
  EVA-CLIP (5 × 1024-d hidden states) into 2048-d ID embedding. PerceiverAttentionCA
  injects into Flux DiT via cross-attention hooks. IDAttnProcessor supports ORTHO/ORTHO_v2
  regularization. EVA-CLIP uses VisionRotaryEmbeddingFast (2D RoPE), SwiGLU + subln.
- **LatentSync-MLX**: UNet3DConditionModel (InflatedConv2d/GroupNorm for 5D video tensors)
  with temporal motion modules. 13-channel input (noise4+mask1+masked4+ref4). Reuses
  MuseTalk's Whisper subpackage for audio encoding — no duplicate Whisper code.
- **MuseTalk-MLX**: Single-step inpainting at t=0 with 8-channel UNet2D. WhisperEncoder
  (4-layer) produces per-frame audio features (B, seq, 5, 384) → chunked windows.

**Weight conversion:** `latentsync_mlx/convert_weights.py` converts PyTorch checkpoints
to MLX safetensors. EVA-CLIP/PuLID weights can be loaded via `from_pretrained()` with
automatic `visual.` prefix stripping.

### Video Adapters (IP-Adapter / ControlNet / AnimateDiff)

Three pluggable video adapters modify the denoising process for conditioned generation:

| Adapter | Mechanism | API parameter | Default |
|---|---|---|---|
| **IP-Adapter** | CLIP-Vision image encoder + projection MLP → prepend image tokens to text context | `ip_adapter_image`, `ip_adapter_scale` | off |
| **ControlNet** | Parallel smaller DiT → per-block residuals injected into main DiT | `controlnet_image`, `controlnet_strength`, `control_type` | off |
| **AnimateDiff** | Temporal motion modules injected into DiT blocks (after self-attention) | `animatediff_scale` | 0 (off) |

**Usage (API):**

```bash
# IP-Adapter: subject-driven image-to-video
curl -X POST /v1/videos/generate -d '{
  "prompt": "a cat walking", "ip_adapter_image": "/path/to/cat.jpg", "ip_adapter_scale": 1.0
}'

# ControlNet: structural guidance (Canny/depth/pose)
curl -X POST /v1/videos/generate -d '{
  "prompt": "a person dancing", "controlnet_image": "/path/to/pose.png",
  "control_type": "pose", "controlnet_strength": 1.0
}'

# AnimateDiff: enhanced temporal coherence
curl -X POST /v1/videos/generate -d '{
  "prompt": "ocean waves", "animatediff_scale": 1.0
}'
```

All adapters use zero-initialized output projections (identity at start), are backward-compatible
(adapter not present = no behavior change), and can be combined simultaneously.

### Speculative Denoise (#177) — FALSIFIED

> ⚠️ **This approach does not work on real 14B DiT.** The hypothesis is falsified.
> The machinery stays landed (env-gated, default off) for future research only.

A diffusion analog of LLM speculative decoding: a layer-pruned draft DiT (first M
of N transformer blocks + shared head, same weights) predicts K=3-5 future velocity
steps; the full DiT verifies all K in a single batched forward (per-element
timesteps, native `t.ndim==1` support); the longest consistent prefix is accepted
and a bonus full step at divergence always advances ≥1 step. Target was 2-3× on 14B.

- Draft co-loading: `LayerPrunedDraft(dit, n_blocks=M)` reuses the same weights,
  no separate draft checkpoint (MLX quantization is not a speed path, see #166).
- Env: `FUSION_SPECULATIVE_DENOISE` (default `"0"` off), `FUSION_SPEC_K` (4),
  `FUSION_SPEC_EPSILON` (0.1), `FUSION_SPEC_DRAFT_BLOCKS` (default `num_layers//4`).

```bash
# env-gated, default off - does not affect the existing SkyReels-V3 generation path
fusion-mlx serve --model SkyReels-V3-R2V-14B-MLX
```

> **Phase-2 result (falsified)**: at safe epsilon (0.1) the acceptance rate is 0%
> for 25%-75% blocks kept; acceptance appears only at 95% blocks where draft cost
> ≈ full (0.42× slower) and quality breaks (maxdiff 0.097 vs 0.00073). The #177
> hypothesis is falsified on MLX SkyReels-V3 14B: DiT velocity fields need full
> depth and are not sub-network predictable (unlike LLM tokens). The machinery is
> correct (all-rejected spec == baseline Euler to 7e-4) and stays landed (env-gated,
> default off, zero prod risk) as infrastructure for a future distilled small draft.
> See `fusion_mlx/video/skyreels_v3/SPECULATIVE_DENOISE.md`.

- **Phase-3 stats surface (landed)**: `VideoBackend.last_denoise_stats()` +
  `GET /v1/videos/denoise-stats?model=<name>` expose the last spec run's
  acceptance stats (`macro_steps`, `accepted`, `avg_accept`, `full_forwards`,
  `draft_forwards`, `baseline_steps`, `speedup`, `available`, `enabled`,
  `config`). Additive and default-off: returns `available=false` with zeroed
  counters when spec is off or no run happened - honest feature surface for when
  a real distilled draft arrives (no per-step callback change, no break to the
  released Stage API / `on_step` contract).

### Metal Async Dispatch (#180)

Attempt to recover GPU idle during the serial denoise loop: each step's `mx.eval`
blocks the CPU until the GPU finishes, leaving the GPU idle while Python builds the
next step's graph. MLX 0.32 has no CommandBuffer API, so the path uses `mx.async_eval`
per step (non-blocking, still materializes and frees like `eval`) with a final
`mx.synchronize` before VAE decode.

- Env: `FUSION_ASYNC_DENOISE` (default `"0"` off) - the prod sync path is
  byte-identical when off, zero risk.
- Memory-safe per #146: `async_eval` materializes each step's latents and frees the
  forward graph (just non-blocking), so peak ≈ single-step working set, not 2×/30×.

```bash
# env-gated, default off - does not affect the existing SkyReels-V3 generation path
FUSION_ASYNC_DENOISE=1 fusion-mlx serve --model SkyReels-V3-R2V-14B-MLX
```

> **Result (no speedup)**: numerically bit-identical to the sync path and memory-safe
> (peak unchanged), but `mx.async_eval` adds overhead that exceeds the GPU-idle
> (CPU graph-build) it recovers. A tiny DiT is flat (0.994×); a medium DiT
> (12L/dim256) is 16.8% **slower** (0.832×) and degrades across runs (60->72 ms while
> sync stays 57-58 ms). The #180 hypothesis is falsified at small/medium scale; the
> real 14B E2E was skipped per this negative signal (#177 precedent). The machinery
> stays landed (env-gated, default off, zero prod risk) as infrastructure. See
> `scripts/bench_async_denoise.py`.

## Project Structure

```
fusion-mlx/
├── fusion_mlx/
│    ├── api/             # OpenAI, Anthropic, Audio, Images, Videos, MCP, OpenClaw routes
│    ├── cache/           # PagedCache, PagedSSDCache, PrefixCache
│    ├── custom_kernels/  # MFA, TurboQuant, KV cache, xfuser attention, FlashKDA
│    ├── engines/         # 9 engine types (LLM, VLM, Embedding, Reranker, STT, TTS, STS, ImageGen, VideoGen)
│    ├── integrations/    # 15 integrations: Claude Code, Codex, Hermes, OpenCode, OpenClaw, OpenHands, Kilo Code, Factory Droid, Kimi Code, PydanticAI, smolagents, Copilot, ComfyUI, Pi, Qwen Code
│    ├── parsers/         # Tool call parsers (Gemma, Harmony, Hermes, etc.)
│    ├── pool/            # EnginePool, MemoryEnforcer, ModelDiscovery, PriorityScheduler
│    ├── router/          # RequestRouter, CloudRouter, SmartRouter
│    ├── scheduler/       # 25-module scheduler (admission, batching, cache, step, etc.)
│    ├── speculative/     # SuffixDecoding, DFlash, DSpark, MTP, VLM MTP
│    ├── telemetry/       # Opt-in anonymous usage telemetry (consent, emit, queue, redact, transport)
│    ├── video/           # Pure-MLX video generation ports (LTX-2, Wan2, SkyReels-V3, PuLID, LatentSync, MuseTalk)
│    ├── share/           # SSH tunnel public sharing (fusionmlx.com)
│    ├── launch/          # One-shot IDE/agent config bootstrapper (15 adapters)
│    └── admin/           # Web panel routes, benchmarking, downloads, settings
├── apps/fusion-mac/      # SwiftUI macOS app (~80 Swift files)
├── docs/                 # API reference, architecture, CLI guide, configuration
├── examples/             # 12 working code examples
├── scripts/              # install.sh, benchmarks, weight conversion
├── tests/                # 1200+ tests (unit, GUI, integration, performance)
└── downstream/           # Sync scripts for fusion-mlx and Rapid-MLX forks
```

## DSpark Speculative Decoding (vendored from dspark-metal, 2026-07-22)

DSpark = DeepSeek DeepSpec block-level speculative decoding for text-only Qwen3
models. Unlike token-level spec decode, DSpark trains a lightweight draft (block7)
on the target model's 7th-layer hidden state, with online rejection sampling for
losslessness. fusion-mlx vendors upstream `stefanopineda/dspark-metal` (MIT) into
`fusion_mlx/speculative/dspark/engine/` with no pip dependency - the upstream repo
has been dormant 20+ days, so fusion-mlx evolves it independently.

- Engine: `fusion_mlx/speculative/dspark/engine/` (13 modules + LICENSE + NOTICE).
- Boundary: `runtime.py` loads the vendored engine via `from .engine import DSparkGenerator`;
  `eligibility.have_runtime()` probes the vendored path and is always available (no
  `pip install dspark-metal` needed).
- VLM extension (PR#2): `Qwen3VLTargetAdapter` extends DSpark to mlx-vlm targets;
  ctx_taps act on text positions only; mlx_vlm is lazy-loaded. 22 weight-free tests
  in `tests/unit/test_dspark_vlm_adapter.py`.
- Size binding: draft = target block 7, so `dspark_qwen3_{4b,8b,14b}_block7` must
  pair with the same-size Qwen3-{4B,8B,14B} (bf16/8bit+; 4-bit rejected by the gate).
- Convert: `python -m fusion_mlx.speculative.dspark.engine.convert <source> --target <target> -o <outdir>`
  (do not pass `--reuse-target-embeddings`).

> **E2E status**: vendoring (phase 1+2) landed, 40 dspark tests pass (1 skipped),
> arch-handler statically de-risked. Real-model E2E (convert + load_runtime +
> generate) is deferred pending download of matching Qwen3-4B/8B/14B targets via
> hf-mirror.

## DFlash2 Speculative Decoding (z-lab `dflash` pkg, 2026-08-21)

DFlash2 is z-lab's second-generation block-diffusion speculative decoder
(PyPI `dflash==0.1.0`). A single draft forward predicts a whole **block** of
candidate tokens; a `CandidateSelector` traces one coherent path, and
two-tap grouped-dynamic convolutions keep the draft from decaying across the
block. The draft reads the target's hidden states at fixed `target_layer_ids`,
so it is quality-matched to the target (not a separate small model).

Unlike DFlash v1 and DSpark, DFlash2 does **not** fork a dedicated server — it
loads in-place via `BatchedEngine` and participates in normal continuous
batching as a self-contained generator.

- Bridge: `fusion_mlx/speculative/dflash2/` (`runtime.py`, `eligibility.py`,
  `engine/generator.py`). The generator delegates the whole propose→verify→
  rollback loop to `dflash.model_mlx.stream_generate` (handles hidden-state
  capture/trim/verify/rollback internally) — lossless by construction.
- Target family: `qwen3_8` **dense** (non-MoE). The auto-router routes
  `qwen3_8` to `dflash2` first, with a `suffix` (n-gram) fallback.
- **block_size ≤ 5** for MLX quantized targets (official draft ships 8; larger
  verify widths are matmul-inefficient on quantized weights). `load_runtime`
  rejects `block_size > 5`.
- Install: `pip install 'fusion-mlx[dflash2]'` (pulls `dflash==0.1.0` on
  `darwin/arm64`; same `mlx`/`mlx-lm` family as fusion-mlx, zero conflict).
- Draft path: local dir (e.g. `~/.fusion-mlx/models/Qwen3.8-27B-DFlash2`) or
  HF id. The bridge short-circuits `snapshot_download` for local dirs (no
  re-download, honors the hf-mirror workflow).

```bash
fusion-mlx serve --model qwen3.8-27b-4bit \
  --enable-dflash2 \
  --dflash2-drafter-path ~/.fusion-mlx/models/Qwen3.8-27B-DFlash2 \
  --dflash2-block-size 5
```

> **E2E status (2026-08-21)**: real `Qwen3.8-27B-4bit` + `Qwen3.8-27B-DFlash2`,
> `block_size=5`, greedy. **52.3 tok/s vs 21.2 tok/s baseline = 2.47× speedup**;
> accept avg **3.556** (range 1–5, 18 verify steps); **lossless PASS** (tail
> tokens identical to baseline, content match; only first-token leading-space
> differs — a dflash detokenizer join-space artifact). 27 dflash2 tests green.
> See [docs/speculative-decoding.md](docs/speculative-decoding.md#dflash2-block-diffusion-z-lab-dflash-pkg).

## Eagle3 Speculative Decoding (draft-model, 2026-08-23)

Eagle3 is a draft-model speculative decoder: a small one-layer drafter reads
the target's multi-layer hidden states (captured at `capture_layers=[8,16,31]`,
projected through `Eagle3Model.fc`) and proposes K draft tokens the target
verifies in one forward pass. fusion-mlx ships a custom MLX-native Eagle3
model (`fusion_mlx/speculative/eagle3/`) instead of `mlx_lm.load()`, which
fails on Eagle3's non-standard weight keys.

```bash
FUSION_SPEC_METHOD=eagle3 fusion-mlx serve --model mlx-community/Meta-Llama-3.1-8B-Instruct-4bit
```

Phase-2 hardening (PR #609), all real-model verified on Eagle3 + Llama-3.1-8B:

- **Family compatibility guard** — `is_compatible()` does case-insensitive
  substring matching (llama3/qwen3) incl. local-path basename, and
  **disables spec decode** if the target family doesn't match the draft's
  (prevents silent garbage: EAGLE3-LLaMA3 vs a Qwen target).
- **Adaptive pause/resume** — fixes a dead-code resume path: once paused,
  `should_speculate()` now lets a re-probe step through every
  `SPEC_RESUME_CHECK_INTERVAL` (default 10) so acceptance can recover and
  un-pause. Tunable via `FUSION_SPEC_MIN_ACCEPT_RATE` / `FUSION_SPEC_ADAPTIVE_WINDOW`
  / `FUSION_SPEC_RESUME_CHECK_INTERVAL`.
- **Draft temperature** — default `0.0`→`0.1` (`FUSION_EAGLE3_DRAFT_TEMP`);
  small temp helps the draft distribution match the target's.
- **Multi-layer hidden capture** — `HiddenStateCapture` at `[8,16,31]`.

> **E2E status (2026-08-23)**: real Eagle3 + Llama-3.1-8B-Instruct-4bit,
> `FUSION_SPEC_METHOD=eagle3`. Compat guard `family=llama3 match`, hidden
> capture `layers=[8,16,31]`, draft `temp=0.1`, and the full pause→re-probe
> →stay-paused adaptive cycle all verified in the real-model log. 14
> Phase-2 unit tests green.
> See [docs/speculative-decoding.md](docs/speculative-decoding.md#eagle3-eagle3).

## Examples

| # | Example | Description |
|---|---------|-------------|
| 01 | `basic-chat.py` | Simple non-streaming chat |
| 02 | `streaming-chat.py` | SSE streaming responses |
| 03 | `anthropic-api.py` | Anthropic Messages API |
| 04 | `tool-calling.py` | Function calling with JSON schema |
| 05 | `multi-model.py` | Concurrent multi-model requests |
| 06 | `image-generation.py` | Flux 2 image generation |
| 07 | `speech-to-text.py` | Whisper STT via API |
| 08 | `text-to-speech.py` | Kokoro TTS with WAV output |
| 09 | `mcp-tools.py` | MCP tool discovery and execution |
| 10 | `python-sdk.py` | OpenAI Python client integration |
| 11 | `comfyui-workflow.py` | ComfyUI workflow execution |
| 12 | `openclaw-agent.py` | OpenClaw agent protocol |

## Documentation

- [API Reference](docs/api-reference.md) - All endpoints with request/response examples
- [Architecture](docs/architecture.md) - EnginePool, Scheduler (25 modules), Cache layers, SmartRouter
- [CLI Reference](docs/cli-reference.md) - All commands and flags
- [Configuration](docs/configuration.md) - Memory tiers, scheduler settings, TurboQuant, aliases, executor pools
- [Speculative Decoding](docs/speculative-decoding.md) - Suffix/DFlash/DSpark/MTP/VLM-MTP methods, selection guide, auto-router
- [Video Input](docs/video-input.md) - VLM video support: `video_url` API, frame extraction, Qwen native path, limits
- [FR Differentiation](docs/FR_DIFFERENTIATION.md) - Verified analysis of fusion-mlx's spec-decode/TurboQuant/scheduling differentiation

## whichllm Integration

The macOS app's **Welcome wizard** uses [whichllm](https://github.com/Andyyyy64/whichllm) for hardware-aware model recommendations. whichllm auto-detects your Mac's GPU, CPU, RAM and disk, then ranks the best local LLMs from HuggingFace that fit your system.

**Integrated features:**
- **Hardware detection** - Apple Silicon chip type, unified memory, GPU bandwidth, CPU cores, free disk (via `system_profiler`/`sysctl`)
- **Model recommendations** - Top-ranked models by quality score, speed (tok/s), VRAM fit, and benchmark evidence
- **Use-case optimization** - Different recommendations for Agent / Coding / Chat workloads
- **Mirror selection** - HuggingFace, HF Mirror, or ModelScope for Chinese users without VPN
- **Graceful fallback** - when whichllm is not installed, detection falls back to `ProcessInfo` + `sysctl` (built-in, no Python dependency)

**Bridge architecture:**
```
Swift App -> WhichLLMService -> PythonRuntime -> whichllm_bridge.py -> whichllm
            ↓ (fallback)
       ProcessInfo + sysctl (zero Python deps)
```

## Flux 2 Klein Switch (mx.compile denoise speedup, 2026-07-20)

`ImageGenEngine` switched from Flux1 to `Flux2Klein` (mflux 0.18.0). Flux2Klein
wraps denoise with `mx.compile(predict)` (`flux2_klein.py:281`); Flux1 has no such
compile. After warmup the first step drops 2.98 s -> a steady 1.56 s/step (1.9×).

**Performance** (M5 Max / FLUX.2-klein-base-4B bf16 / 1024×1024):

| Steps | Total | s/step |
|---|---|---|
| 4 | 6.8s | 1.59 |
| 8 | 13.6s | 1.70 |

First call includes 8.5s model load (9.6 G lazy load).

**Serving:** mflux Flux2 repos are diffusers format (`model_index.json`) with no
mflux `configuration.json` task manifest, so discovery misclassifies them as LLMs
and `BatchedEngine` fails to load. Add the manifest manually:

```bash
HF_ENDPOINT=https://hf-mirror.com hf download black-forest-labs/FLUX.2-klein-base-4B \
  --local-dir ~/.fusion-mlx/models/FLUX.2-klein-base-4B
echo '{"task":"text-to-image"}' > ~/.fusion-mlx/models/FLUX.2-klein-base-4B/configuration.json
fusion-mlx serve --model-dir ~/.fusion-mlx/models --port 11434
curl -s http://127.0.0.1:11434/v1/images/generate \
  -H 'Content-Type: application/json' \
  -d '{"model":"FLUX.2-klein-base-4B","prompt":"a cat","width":1024,"height":1024,"steps":4,"n":1}'
```

`_infer_flux2_config` parses the variant from the path name (`base+4b` ->
`flux2_klein_base_4b`, `base+9b` -> `flux2_klein_base_9b`, `4b` -> `flux2_klein_4b`,
`kv+9b` -> `flux2_klein_9b_kv`, default `flux2_klein_9b`). `negative_prompt` degrades
to a warning (Flux2Klein.generate_image has no such param).

### Flux2Klein Weight Quantization (FUSION_FLUX_QUANT, memory not speed)

`ImageGenEngine.__init__` reads `FUSION_FLUX_QUANT` -> `mflux.Flux2Klein(quantize=...)`.
Values: `w8a16`/`w8`/`int8`/`8` -> 8-bit, `w4`/`nf4`/`int4`/`4` -> 4-bit,
`off`/`0`/`none`/`bf16`/empty -> bf16 (default). Case-insensitive.

> **Measured result (M5 Max / FLUX.2-klein-base-4B / 1024×1024 / 4 steps)**:
> bf16 6.81 s (1.70 s/step) vs w8a16 8.20 s (2.05 s/step) - w8a16 is **20% slower**.
> The 4B model already fits unified memory at bf16, so int8 dequant overhead exceeds
> the bandwidth win and `mx.compile` already optimizes the bf16 path. **Quantization
> is not a speed optimization for Flux2Klein** - use it only for memory (9B ~18 G ->
> ~9 G, to fit 16 G Macs).

## Flux-1.lite-8B-MLX Deep Optimization (2026-07-19)

**Performance** (M5 Max 128 GB / MLX 0.32 / Q4):

| Metric | Baseline | block compile fusion | mlx-mfa Metal attn | real ceiling |
|---|---|---|---|---|
| step/s (512×512×4 steps) | 1.83 | 1.96 | **1.88** | 1.88-2.03 |
| Metal peak | 10.8 GB | 10.6 GB | 10.5 GB | 10.5 GB |
| 256×256 real ceiling | - | - | - | 4.62 step/s |

bench.dpdns.org uploads: id 27 (1.97), id 30 (1.96), id 31 (1.88), id 32 (1.88 mlx-mfa).

**Landed optimizations:**

1. **Block compile fusion** (`joint_transformer_block.py` + `single_transformer_block.py`)
   - `_compiled_call = mx.compile(self._call_raw)` compiles the whole block, fusing
     AdaLN+attn+FFN submodules into one compiled unit, eliminating cross-`nn.Module`
     call breaks.
   - `to_out` list -> `to_out_0` named attribute (MLX nn.Module does not capture list
     attrs) + `flux_weight_mapping.py` maps `to_out.0` -> `to_out_0`.

2. **mlx-mfa Metal Flash Attention** (`attention_utils.py::compute_attention`)
   - `mlx_mfa.flash_attention` replaces `mx.fast.scaled_dot_product_attention`,
     targeting the M5 Neural Engine tile.
   - `has_nax: True` confirms the Metal kernel fires; landed but flat (1.88 vs 1.88
     step/s) since swapping only SDPA does not cover the RoPE + QKV projection bottleneck.

3. **Fused QKV+RoPE+attn single-op fusion** - shelved: Q4 weights use a packed
   `(out, in/8)` layout, and manual `mx.matmul`/`mx.addmm` breaks `quantized_matmul`
   encapsulation (ValueError). Kept `nn.Linear.__call__` on `quantized_matmul`; the
   whole-block `mx.compile` already fuses it.

**Bottleneck diagnosis:**

- 256 vs 512 ratio 2.48× (theoretical 4×) -> mixed bandwidth+compute bound.
- transformer 80% main bottleneck / encode_prompt 10% / VAE 10%.
- schnell has no CFG support (`supports_guidance=False`); `guidance=4.0` is inert,
  single-branch is already optimal.
- Shape jitter costs 21.4%: steady 512×512 = 1.90 step/s, mixed sizes drop to 1.56.
- **Real ceiling clarified**: 512×512 at 1.88 step/s (M5 Max Q4 + mlx-mfa Metal attn +
  dual-layer compile fusion) is the reasonable ceiling under the hardware+Q4+op-stack
  triple constraint.

**Key lessons:**

1. MLX Q4 quantized weights cannot be manually matmul'd (packed `(out, in/8)` layout,
   must go through `nn.Linear.__call__`'s `quantized_matmul`). All hand-written
   single-kernel fusion is infeasible on Q4 models.
2. Compiling 60+ blocks whole degrades generally (op-graph accumulation triggers Metal
   Command Buffer spray); dual-layer compile (per-block + transformer loop) is optimal.
3. mlx-mfa prebuilt path: local source + scikit-build-core + nanobind +
   `pip install -e --no-build-isolation` triggers the CMake build producing `_ext.so`,
   avoiding uncontrollable PyPI wheel build times.

## FlashKDA — Kimi Delta Attention for Apple Silicon

FlashKDA ports the gated linear attention mechanism (KDA) from CUDA SM90+ to
Apple Silicon via MLX. Core recurrence: `h_t = g_t * h_{t-1} + beta_t * (k_t ⊗ v_t)`,
`o_t = q_t^T * h_t`. Constraint: K = V = 128.

- **Python reference** — always available, correct, used for validation
- **Metal kernel** — auto-selected when compiled; uses `simdgroup_matrix` for
  bf16 outer product and query-state multiply (CHUNK=16, matching CUDA K1/K2)

```python
from fusion_mlx.custom_kernels.flash_kda import fwd

out, state = fwd(q, k, v, g, beta, scale=1.0, A_log=a_log, dt_bias=dt_bias)
```

See [custom_kernels/flash_kda/](fusion_mlx/custom_kernels/flash_kda/) for details.

## Docker

Multi-stage Dockerfile for deployment on Linux (CPU) or as a base image:

```bash
docker compose up
# or
docker build -t fusion-mlx .
docker run -p 11434:11434 -v ~/.fusion-mlx/models:/home/fusion/.fusion-mlx/models:ro fusion-mlx
```

## License

Apache-2.0

## Acknowledgments

- [MLX](https://github.com/ml-explore/mlx) and [mlx-lm](https://github.com/ml-explore/mlx-lm) by Apple
- [mlx-vlm](https://github.com/Blaizzy/mlx-vlm) - Vision-language model inference on Apple Silicon
- [vllm-mlx](https://github.com/waybarrios/vllm-mlx) - oMLX started from vllm-mlx v0.1.0
- [fusion-mlx](https://github.com/dahai80/fusion-mlx) - Continuous batching and tiered KV caching
- [Rapid-MLX](https://github.com/raullenchai/Rapid-MLX) - Speculative decoding, multi-modal, cloud routing
- [dflash-mlx](https://github.com/bstnxbt/dflash-mlx) - Block diffusion speculative decoding
- [DeepSpec (DSpark)](https://github.com/deepseek-ai/DeepSpec) - Lossless block speculative decoding
- [mlx-embeddings](https://github.com/Blaizzy/mlx-embeddings) - Embedding model support
- [venvstacks](https://venvstacks.lmstudio.ai) - Portable Python environment layering for the macOS app
