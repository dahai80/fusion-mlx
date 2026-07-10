<div align="center">

# fusion-mlx

**Unified local model serving for Apple Silicon**

Drop-in replacement for Ollama / vLLM — runs natively on Metal via MLX

[![Version](https://img.shields.io/badge/v0.4.5-blue.svg)](https://github.com/dahai80/fusion-mlx/releases)
[![Python](https://img.shields.io/badge/Python-3.11+-3776AB.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-Apache--2.0-green.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/Tests-1200+-success.svg)](tests/)

[English](README.md) | [中文](README_CN.md)

[Get Started](#quick-start) · [Download App](https://github.com/dahai80/fusion-mlx/releases) · [Benchmarks](https://bench.dpdns.org/) · [Documentation](docs/)

</div>

---

## Why fusion-mlx?

| | fusion-mlx | omlx | Ollama |
|---|---|---|---|
| Continuous batching | ✅ | ✅ | ❌ |
| 2-bit quant recipes | ✅ up to +167% speed | — | — |
| TurboQuant KV (4-bit) | ✅ | ✅ (advanced) | ❌ |
| Speculative decoding | ✅ 4 methods | ❌ | ❌ |
| OpenAI + Anthropic API | ✅ Both | ✅ Both | ✅ Both |
| VLM with paged KV cache | ✅ | ❌ | ❌ |
| 40+ quant formats | ✅ | ~15 | ~10 |
| macOS native app | ✅ SwiftUI | ✅ | ✅ |
| 8 engine types | ✅ | 2 | 2 |
| Admin web panel | ✅ | ✅ | ❌ |

**Benchmark** (Qwen3.6-27B, Apple M2 Ultra 137GB):

| Quantization | Model Size | bpw | Decode Speed | vs mxfp8 | vs mixed_3_4 |
|---|---|---|---|---|---|
| mxfp8 | 26 GB | 8.0 | 18.5 tok/s | baseline | — |
| mxfp4 | 13 GB | 4.0 | 32.3 tok/s | **+75%** | — |
| mixed_4_6 | 15 GB | 4.85 | 29.0 tok/s | **+57%** | — |
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

- **9 engine types** — LLM, VLM, Embedding, Reranker, STT, TTS, STS, ImageGen (Flux 2), VideoGen (LTX-2)
- **OpenAI + Anthropic API** — one server, two API flavors, fully compatible
- **Continuous batching** — vLLM-style scheduler with chunked prefill, preemption, priority queues
- **Speculative decoding** — SuffixDecoding, DFlash, DSpark, MTP, VLM MTP (2–5× faster generation)
- **TurboQuant KV** — 4-bit KV cache quantization, 4× less memory traffic
- **40+ quant formats** — GGUF (Q2_K → Q8_0), Imatrix (IQ1_M → IQ4_XS), TurboQuant (TQ1_0/TQ2_0), MLX (mxfp4/mxfp8/6bit/4bit/8bit/F16/BF16/F32)
- **Paged KV cache** — SSD cold layer, block-aware prefix caching with COW sharing
- **Fused sampler** — skip logsumexp, eliminate GPU sync, batched sampling
- **SmartRouter** — phase-aware routing with benchmark-based backend selection and EMA smoothing
- **Priority scheduling** — REALTIME / BATCH / BACKGROUND queues with Metal command queue priorities
- **4-tier memory enforcer** — safe / balanced / aggressive / custom hard limits with deadlock-free eviction
- **Multi-model concurrency** — EnginePool with LRU eviction, pinning, and TTL
- **MCP tool support** — list, discover, and execute MCP tools via API
- **Admin web panel** — model management, live chat, HuggingFace downloads, online quantization
- **macOS native app** — SwiftUI with menu bar, auto-update, benchmark, model management, **hardware-aware setup wizard**

### Advanced Feature Recommendations

When you launch the macOS app for the first time, the **6-step Welcome wizard** auto-detects your Mac hardware and recommends optimal settings:

| Use Case | Recommended Models (selectable list) | DFlash | DSpark | TurboQuant | Max Context |
|----------|--------------------------------------|--------|--------|------------|-------------|
| 🤖 Agent (OpenClaw) | DeepSeek-V4-Flash, Qwen3.6-27B | ✅ | ❌ | ✅ (≥64GB) | 65K |
| 💻 Coding | Qwen3.5-9B, DeepSeek-Coder-V2 | ❌ | ✅ | ✅ (≥64GB) | 131K |
| 💬 Chat | Qwen3.5-9B, Gemma-4-31B | ❌ | ❌ | ✅ (≥64GB) | 32K |

Recommendations are based on real-time hardware detection (CPU cores, unified memory, GPU bandwidth, disk space). All settings are editable with validation warnings for out-of-range values.

## Quick Start

```bash
# Install
pip install fusion-mlx

# Start server
fusion-mlx serve --model-dir ~/.cache/huggingface

# Chat
curl http://localhost:8000/v1/chat/completions \
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
client = OpenAI(base_url="http://localhost:8000/v1", api_key="local")
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
client = anthropic.Anthropic(base_url="http://localhost:8000/v1", api_key="local")
resp = client.messages.create(
    model="Qwen3-4B-Q4_K_M",
    max_tokens=64,
    messages=[{"role": "user", "content": "What is 2+2?"}],
)
print(resp.content[0].text)
```

## Supported Models

| Type | Engine | Example Models |
|------|--------|----------------|
| LLM | `BatchedEngine` | Qwen, Llama, Mistral, Gemma, DeepSeek, Kimi |
| VLM | `VLMBatchedEngine` | Qwen2-VL, LLaVA, InternVL |
| Embedding | `EmbeddingEngine` | BGE, E5, GTE |
| Reranker | `RerankerEngine` | Cohere, Jina rerankers |
| STT | `STTEngine` | Whisper, VibeVoice-ASR |
| TTS | `TTSEngine` | Kokoro, VibeVoice |
| ImageGen | `ImageGenEngine` | Flux 2 |
| VideoGen | `VideoGenEngine` | LTX-2, Wan2 (pure-MLX ports) |

## Quantization Formats

| Category | Formats |
|----------|---------|
| GGUF/GGML | Q2_K, Q3_K_S/M/L, Q4_0, Q4_1, Q4_K_S/M, Q5_0, Q5_1, Q5_K_S/M, Q6_K, Q8_0, Q8_K |
| Imatrix | IQ1_M, IQ2_S, IQ2_XS, IQ2_XXS, IQ3_M, IQ3_S, IQ4_NL, IQ4_XS |
| TurboQuant | TQ1_0, TQ2_0 |
| MLX-native | mxfp4, mxfp8, 6bit (ParoQuant), 4bit, 8bit, F16, BF16, F32 |
| MLX Recipes | mixed_3_4, mixed_2_6, mixed_2_4, mixed_3_6, mixed_4_6, quant2_all, quant2, quant2_128, quant2_flat (see below) |

### Quantization Recipes

MLX recipe quantization provides pre-tuned mixed-bit plans that maximize decode speed for Apple Silicon. Both modes produce standard mlx-lm safetensors compatible with any MLX runtime.

The macOS app offers a mode toggle between:

- **oQ Online** — sensitivity-based per-layer quantization (original mode)
- **MLX Recipe** — pre-tuned quantization plans via `mlx_lm.convert --quant-recipe <name>`

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

### Converting models

Convert any HuggingFace model to MLX (optionally quantized) with the `convert` command — accepts a model alias or full HF repo:

```bash
fusion-mlx convert qwen3.5-9b --quant-bits 4 -o ./qwen3.5-9b-4bit
fusion-mlx convert mlx-community/Qwen3.5-9B --quant-bits 8 --upload-repo me/my-repo
```

This is **weight** quantization saved to disk, distinct from TurboQuant KV-cache compression (`--kv-cache-turboquant`), which is a runtime knob. See [CLI Reference](docs/cli-reference.md).

## API Compatibility

| API | Endpoints | Status |
|-----|-----------|--------|
| OpenAI Chat | `/v1/chat/completions`, `/v1/models` | ✅ Fully compatible |
| OpenAI Legacy | `/v1/completions` | ✅ Supported |
| Anthropic Messages | `/v1/messages`, `/v1/count_tokens` | ✅ Fully compatible |
| Audio | `/v1/audio/transcriptions`, `/v1/audio/speech` | ✅ Supported |
| Images | `/v1/images/generate` | ✅ Supported (Flux 2) |
| Videos | `/v1/videos/generate` | ✅ Supported (LTX-2, Wan2; pure-MLX ports) |
| Embeddings | `/v1/embeddings` | ✅ Supported |
| MCP | `/v1/mcp/tools`, `/v1/mcp/servers`, `/v1/mcp/execute` | ✅ Supported |
| OpenClaw Agent | `/v1/openclaw/agent/*` | ✅ Sessions, turns, tool calling, SSE streaming |

## Model Aliases

```bash
fusion-mlx serve --model claude-4.6-sonnet   # → Qwen3.5-27B-Claude-4.6-Opus-Distilled-MLX-6bit
fusion-mlx serve --model gpt-4o               # → Qwen3-32B-A3B-Think-2512-MLX
```

## Integrations

```bash
# Claude Code — use fusion-mlx as your local Anthropic API
fusion-mlx launch claude

# OpenClaw — batch agent processing
fusion-mlx launch openclaw --model Qwen3-4B

# ComfyUI — image generation with Flux 2
fusion-mlx launch comfyui

# GitHub Copilot
fusion-mlx launch copilot
```

## Admin Panel

Access at `http://localhost:8000/admin`:

- **Models** — load / unload / pin models dynamically, ParoQuant compat detection
- **Chat** — live chat interface for testing any model
- **Downloads** — HuggingFace / ModelScope model downloads with progress tracking
- **Quantization** — online quantization (oQ) pipeline
- **Benchmarks** — throughput and accuracy benchmarking
- **Monitoring** — real-time memory, performance, and request metrics
- **Settings** — global / per-model configuration, sub-API key management

## macOS App

Native SwiftUI app with menu bar integration:

- One-click model launch and server control
- Quantization mode toggle: **oQ Online** (sensitivity-based) / **MLX Recipe** (pre-tuned plans)
- Throughput & accuracy benchmarking
- Auto-update from GitHub Releases
- Model management and downloads
- Live server status in menu bar

Download from [GitHub Releases](https://github.com/dahai80/fusion-mlx/releases).

## Performance

Benchmarks on Apple M5 Max (128 GB RAM, 40 GPU cores), MLX 0.32.0.dev — 2026-07-04.
Single-stream decode, Qwen3.6-27B-mxfp8 (100 tokens, 5 warmup steps):

| Engine | TG mean (tok/s) | median | std | CV | step (ms) |
|---|---|---|---|---|---|
| fusion-mlx | 18.46 | 18.52 | 0.18 | 1.0% | 54.17 |
| omlx | 18.49 | 18.53 | 0.18 | 1.0% | 54.09 |

Ratio 0.998 — full parity. Speculative decoding is auto-gated off for GatedDeltaNet hybrid models to preserve coherence.

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

## Project Structure

```
fusion-mlx/
├── fusion_mlx/
│    ├── api/             # OpenAI, Anthropic, Audio, Images, Videos, MCP, OpenClaw routes
│    ├── cache/           # PagedCache, PagedSSDCache, PrefixCache
│    ├── engines/         # 8 engine types (LLM, VLM, Embedding, etc.)
│    ├── integrations/    # Claude Code, OpenClaw, ComfyUI, Copilot, Codex, etc.
│    ├── parsers/         # Tool call parsers (Gemma, Harmony, Hermes, etc.)
│    ├── pool/            # EnginePool, MemoryEnforcer, ModelDiscovery, PriorityScheduler
│    ├── router/          # RequestRouter, CloudRouter, SmartRouter
│    ├── scheduler/       # 25-module scheduler (admission, batching, cache, step, etc.)
│    ├── speculative/     # SuffixDecoding, DFlash, DSpark, MTP, VLM MTP
│    └── admin/           # Web panel routes, benchmarking, downloads, settings
├── apps/fusion-mac/      # SwiftUI macOS app (~80 Swift files)
├── docs/                 # API reference, architecture, CLI guide, configuration
├── examples/             # 12 working code examples
├── tests/                # 1200+ tests (unit, GUI, integration, performance)
└── downstream/           # Sync scripts for omlx and Rapid-MLX forks
```

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

- [API Reference](docs/api-reference.md) — All endpoints with request/response examples
- [Architecture](docs/architecture.md) — EnginePool, Scheduler (25 modules), Cache layers, SmartRouter
- [CLI Reference](docs/cli-reference.md) — All commands and flags
- [Configuration](docs/configuration.md) — Memory tiers, scheduler settings, TurboQuant, aliases, executor pools
- [Speculative Decoding](docs/speculative-decoding.md) — Suffix/DFlash/DSpark/MTP/VLM-MTP methods, selection guide, auto-router
- [Video Input](docs/video-input.md) — VLM video support: `video_url` API, frame extraction, Qwen native path, limits
- [FR Differentiation](docs/FR_DIFFERENTIATION.md) — Verified analysis of fusion-mlx's spec-decode/TurboQuant/scheduling differentiation

## whichllm Integration

The macOS app's **Welcome wizard** uses [whichllm](https://github.com/Andyyyy64/whichllm) for hardware-aware model recommendations. whichllm auto-detects your Mac's GPU, CPU, RAM and disk, then ranks the best local LLMs from HuggingFace that fit your system.

**Integrated features:**
- **Hardware detection** — Apple Silicon chip type, unified memory, GPU bandwidth, CPU cores, free disk (via `system_profiler`/`sysctl`)
- **Model recommendations** — Top-ranked models by quality score, speed (tok/s), VRAM fit, and benchmark evidence
- **Use-case optimization** — Different recommendations for Agent / Coding / Chat workloads
- **Mirror selection** — HuggingFace, HF Mirror, or ModelScope for Chinese users without VPN
- **Graceful fallback** — when whichllm is not installed, detection falls back to `ProcessInfo` + `sysctl` (built-in, no Python dependency)

**Bridge architecture:**
```
Swift App → WhichLLMService → PythonRuntime → whichllm_bridge.py → whichllm
            ↓ (fallback)
       ProcessInfo + sysctl (zero Python deps)
```

## License

Apache-2.0

## Acknowledgments

- [MLX](https://github.com/ml-explore/mlx) and [mlx-lm](https://github.com/ml-explore/mlx-lm) by Apple
- [mlx-vlm](https://github.com/Blaizzy/mlx-vlm) — Vision-language model inference on Apple Silicon
- [vllm-mlx](https://github.com/waybarrios/vllm-mlx) — oMLX started from vllm-mlx v0.1.0
- [omlx](https://github.com/jundot/omlx) — Continuous batching and tiered KV caching
- [Rapid-MLX](https://github.com/raullenchai/Rapid-MLX) — Speculative decoding, multi-modal, cloud routing
- [dflash-mlx](https://github.com/bstnxbt/dflash-mlx) — Block diffusion speculative decoding
- [DeepSpec (DSpark)](https://github.com/deepseek-ai/DeepSpec) — Lossless block speculative decoding
- [mlx-embeddings](https://github.com/Blaizzy/mlx-embeddings) — Embedding model support
- [venvstacks](https://venvstacks.lmstudio.ai) — Portable Python environment layering for the macOS app
