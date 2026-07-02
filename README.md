<div align="center">

# fusion-mlx

**Unified local model serving for Apple Silicon**

Drop-in replacement for Ollama / vLLM — runs natively on Metal via MLX

[![Version](https://img.shields.io/badge/v0.3.0-blue.svg)](https://github.com/dahai80/fusion-mlx/releases)
[![Python](https://img.shields.io/badge/Python-3.11+-3776AB.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-Apache--2.0-green.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/Tests-1200+-success.svg)](tests/)

[Get Started](#quick-start) · [Download App](https://github.com/dahai80/fusion-mlx/releases) · [Benchmarks](https://bench.dpdns.org/) · [Documentation](docs/)

</div>

---

## Why fusion-mlx?

| | fusion-mlx | omlx | Ollama |
|---|---|---|---|
| Continuous batching | ✅ | ✅ | ❌ |
| 2× concurrent throughput | ✅ 36 vs 17.9 tok/s | Baseline | — |
| TurboQuant KV (4-bit) | ✅ | ❌ | ❌ |
| Speculative decoding | ✅ 4 methods | ❌ | ❌ |
| OpenAI + Anthropic API | ✅ Both | OpenAI only | OpenAI only |
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

- **8 engine types** — LLM, VLM, Embedding, Reranker, STT, TTS, STS, ImageGen (Flux 2)
- **OpenAI + Anthropic API** — one server, two API flavors, fully compatible
- **Continuous batching** — vLLM-style scheduler with chunked prefill, preemption, priority queues
- **Speculative decoding** — SuffixDecoding, DFlash, MTP, VLM MTP (2–5× faster generation)
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
- **macOS native app** — SwiftUI with menu bar, auto-update, benchmark, model management
- **8 integrations** — Claude Code, OpenClaw, ComfyUI, Copilot, Codex, OpenCode, Pi, Hermes

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

## Quantization Formats

| Category | Formats |
|----------|---------|
| GGUF/GGML | Q2_K, Q3_K_S/M/L, Q4_0, Q4_1, Q4_K_S/M, Q5_0, Q5_1, Q5_K_S/M, Q6_K, Q8_0, Q8_K |
| Imatrix | IQ1_M, IQ2_S, IQ2_XS, IQ2_XXS, IQ3_M, IQ3_S, IQ4_NL, IQ4_XS |
| TurboQuant | TQ1_0, TQ2_0 |
| MLX-native | mxfp4, mxfp8, 6bit (ParoQuant), 4bit, 8bit, F16, BF16, F32 |

## API Compatibility

| API | Endpoints | Status |
|-----|-----------|--------|
| OpenAI Chat | `/v1/chat/completions`, `/v1/models` | ✅ Fully compatible |
| OpenAI Legacy | `/v1/completions` | ✅ Supported |
| Anthropic Messages | `/v1/messages`, `/v1/count_tokens` | ✅ Fully compatible |
| Audio | `/v1/audio/transcriptions`, `/v1/audio/speech` | ✅ Supported |
| Images | `/v1/images/generate` | ✅ Supported (Flux 2) |
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
- Throughput & accuracy benchmarking
- Auto-update from GitHub Releases
- Model management and downloads
- Live server status in menu bar

Download from [GitHub Releases](https://github.com/dahai80/fusion-mlx/releases).

## Performance

Benchmarks on Apple M5 Max (128 GB RAM):

| Model | Quant | PP (tok/s) | TG (tok/s) | TTFT (ms) |
|-------|-------|-----------|-----------|-----------|
| Qwen3.6-27B | mxfp8 | 264 | 29.8 | ~1000 |
| Qwen2.5-3B | Q4_K_M | 580 | 32 | ~500 |

Concurrent throughput (Qwen3.6-27B-mxfp8, 4 requests):

| Metric | fusion-mlx | omlx |
|---|---|---|
| Aggregate TG | 36.0 tok/s | 17.9 tok/s |
| Per-request TG | ~9 tok/s | ~9 tok/s |

Submit your own benchmarks at [bench.dpdns.org](https://bench.dpdns.org/).

## Project Structure

```
fusion-mlx/
├── fusion_mlx/
│    ├── api/             # OpenAI, Anthropic, Audio, Images, MCP, OpenClaw routes
│    ├── cache/           # PagedCache, PagedSSDCache, PrefixCache
│    ├── engines/         # 8 engine types (LLM, VLM, Embedding, etc.)
│    ├── integrations/    # Claude Code, OpenClaw, ComfyUI, Copilot, Codex, etc.
│    ├── parsers/         # Tool call parsers (Gemma, Harmony, Hermes, etc.)
│    ├── pool/            # EnginePool, MemoryEnforcer, ModelDiscovery, PriorityScheduler
│    ├── router/          # RequestRouter, CloudRouter, SmartRouter
│    ├── scheduler/       # 25-module scheduler (admission, batching, cache, step, etc.)
│    ├── speculative/     # SuffixDecoding, DFlash, MTP, VLM MTP
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

## License

Apache-2.0

## Acknowledgments

- [MLX](https://github.com/ml-explore/mlx) and [mlx-lm](https://github.com/ml-explore/mlx-lm) by Apple
- [mlx-vlm](https://github.com/Blaizzy/mlx-vlm) — Vision-language model inference on Apple Silicon
- [vllm-mlx](https://github.com/waybarrios/vllm-mlx) — oMLX started from vllm-mlx v0.1.0
- [omlx](https://github.com/jundot/omlx) — Continuous batching and tiered KV caching
- [Rapid-MLX](https://github.com/raullenchai/Rapid-MLX) — Speculative decoding, multi-modal, cloud routing
- [dflash-mlx](https://github.com/bstnxbt/dflash-mlx) — Block diffusion speculative decoding
- [mlx-embeddings](https://github.com/Blaizzy/mlx-embeddings) — Embedding model support
- [venvstacks](https://venvstacks.lmstudio.ai) — Portable Python environment layering for the macOS app
