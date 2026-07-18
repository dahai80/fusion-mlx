<div align="center">

# fusion-mlx

**Unified local model serving for Apple Silicon**

Drop-in replacement for Ollama / vLLM — runs natively on Metal via MLX

[![Version](https://img.shields.io/badge/v0.4.8-blue.svg)](https://github.com/dahai80/fusion-mlx/releases)
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

- **9 engine types** — LLM, VLM, Embedding, Reranker, STT, TTS, STS, ImageGen (Flux 2), VideoGen (LTX-2, Wan2, SkyReels-V3)
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
- **SkyReels-V3 视频生成** — 最强开源视频生成模型纯 MLX 移植，R2V/V2V/A2V 三大分支全部真实权重端到端跑通，M5 Max 专属 dFlash 注意力 + NF4 量化，19B 模型 720P 常驻内存 ≤ 14GB
- **PyTorch → MLX 全模型转换器** — `convert_skyreels_v3.py` 一键转换 SkyReels-V3 三分支 (DiT + T5 + VAE + CLIP + audio) PyTorch 权重到 MLX safetensors，支持 bfloat16/float16/float32 + NF4 量化，分 shard 增量写盘避统一内存冲高

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
| VideoGen | `VideoGenEngine` | LTX-2, Wan2, SkyReels-V3 (pure-MLX ports) |

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
| Videos | `/v1/videos/generate` | ✅ Supported (LTX-2, Wan2, SkyReels-V3; pure-MLX ports) |
| Embeddings | `/v1/embeddings` | ✅ Supported |
| MCP | `/v1/mcp/tools`, `/v1/mcp/servers`, `/v1/mcp/execute` | ✅ Supported |
| OpenClaw Agent | `/v1/openclaw/agent/*` | ✅ Sessions, turns, tool calling, SSE streaming |
| Agent Graph | `/v1/agents/graphs`, `/v1/agents/run` | ✅ CRUD + export + run (in-memory) |
| Base Info | `/v1/base` | ✅ MLX runtime capability detection |
| Convert / Quantize | `/v1/convert`, `/v1/quantize` (+ `.../jobs/{id}`) | ✅ Async HF->MLX conversion + weight quantization |

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

### Video Generation (SkyReels-V3)

Benchmarks on Apple M5 Max (128 GB RAM, 40 GPU cores), MLX 0.32.0 — 2026-07-18.

**真实权重端到端** (非骨架 stub, 完整 40 层 DiT 前向, bfloat16, 5 frames 256P latent):

| Branch | Model | 权重体积 | 加载 (s) | DiT fwd (s/step) | Metal 峰值 (GB) | FPS/step | 状态 |
|---|---|---|---|---|---|---|---|
| R2V | Reference-to-Video 14B | 24 GB (7 shards) | 6.84 | **0.092** | 75.3 | **54.3** | ✅ 跑通 |
| V2V | Video Extension 14B | 75 GB (14+6+1 shards) | 3.11 | **0.329** | 82.7 | **15.2** | ✅ 跑通 (mx.compile 融合 3.3×) |
| A2V | Talking Avatar 19B | 123 GB (18+6+1+1+1 shards) | 3.16 | **0.328** | 24.8 | **3.0** | ✅ 跑通 (audio_cross_attn+norm_x 重构 + kv_linear 转置 + mx.compile, 18×加速) |

**真 T5/VAE 端到端** (A2V-19B, 非 stub `text_emb`, 真实 UMT5Encoder.encode_text + WanVAE.decode, 5 frames 128×128 latent, bf16, 2026-07-18):

| 阶段 | 耗时 | 输出 shape | 说明 |
|---|---|---|---|
| T5 encode_text | 3.05 s | (1, 14, 4096) | 真前向, token_embedding 零占比 0% |
| DiT 30 步去噪 | 8.43 s (281 ms/step) | (1, 16, 5, 16, 16) | mx.compile 融合, Metal 峰值 24.8 GB |
| VAE decode | 0.30 s | (1, 3, 20, 128, 128) | 真前向, 零占比 0% (出真非零画面) |
| **端到端总耗** | **11.78 s** | 5 帧 128×128 | **0.42 FPS** (首版真端到端) |

**T5/VAE 修复关键** (解除真端到端阻塞的 4 个 bug):

| Bug | 位置 | 修复 |
|---|---|---|
| T5 加载路径漏 `t5/` 子目录 | weights:346 | 加 `t5_dir = model_path / "t5"` 分支 |
| `T5Encoder` 命名失配 (`embed_tokens`/`final_layer_norm`/`block`) | t5_encoder:204 | 改 `token_embedding`/`norm`/`blocks` 对齐真实权重 |
| `WanVAE` list 属性不入 MLX `_children` 致 `parameters()` 丢子层 | weights:181 | 加 `_inject_list_child_weights` 手动递归注入 |
| `nn.Conv2d` 权重布局 `(out,in,kh,kw)` vs MLX 期望 `(out,kh,kw,in)` | weights:233 | `_inject` 内 `transpose(0,2,3,1)` 自动转置 |

**MLX 全模型转换产物** (PyTorch → MLX safetensors, `convert_skyreels_v3.py`):

| 分支 | DiT shards | T5 shards | VAE | CLIP | audio | 总体积 |
|---|---|---|---|---|---|---|
| R2V-14B-MLX | 7 (24 GB) | — | — | — | — | 24 GB |
| V2V-14B-MLX | 14 (53 GB) | 6 (21 GB) | 484 MB | — | — | 75 GB |
| A2V-19B-MLX | 18 (97 GB) | 6 (21 GB) | 484 MB | 4.4 GB | 218 tensors | 123 GB |

**关键修复** (解除 SkyReels-V3 真实权重端到端阻塞的 9 个 bug):

| Bug | 位置 | 修复 |
|---|---|---|
| `load_pytorch_state_dict` 漏子目录扫 | convert:104 | 加 `rglob` 递归扫 transformer/vae/text_encoder |
| `_load_safetensors_dir` numpy 不认 bf16 | convert:131 | 改 `framework=pt` + torch.float32 upcast |
| `_write_sharded_safetensors` mlx bf16 buffer 错 | convert:424 | try/except 捕获后 upcast float32 |
| `total_size` 求和漏 bf16 fix | convert:449 | `_safe_nbytes` helper |
| `mx.power(theta, -arange/dim)` Invalid Dtype | common:140 | 改 `mx.exp(-k*mx.log(theta))` |
| `grid_sizes` pre-patch 尺寸错位 | common:211 | 加 `patch_scale` 反推真实 seq_len |
| `context` 用错 dim 而非 text_dim | bench_skyreels:340 | 改 `branch_cfg.text_dim` |
| `noise_pred = zeros_like` 跳过 DiT | bench_skyreels:366 | 去兜底, 真实前向失败立即抛错 |
| `rope_apply` 广播错位 (padded 长度) | common:242 | 用 `seq_len` 而非全序列 `s` 广播 |

> 历史骨架压测 (1983/1151/906 FPS) 是假象 — `bench_skyreels.py:366` 的 `noise_pred = mx.zeros_like(latent_input)` 跳过了整个 DiT 前向, 只测了空循环开销。上表 0.110/2.862 s/step 才是真实推理速度。

Submit your own benchmarks at [bench.dpdns.org](https://bench.dpdns.org/).

```
fusion-mlx/
├── fusion_mlx/
│    ├── api/             # OpenAI, Anthropic, Audio, Images, Videos, MCP, OpenClaw routes
│    ├── cache/           # PagedCache, PagedSSDCache, PrefixCache
│    ├── custom_kernels/  # MFA, TurboQuant, KV cache, xfuser attention
│    ├── engines/         # 8 engine types (LLM, VLM, Embedding, etc.)
│    ├── integrations/    # Claude Code, OpenClaw, ComfyUI, Copilot, Codex, etc.
│    ├── parsers/         # Tool call parsers (Gemma, Harmony, Hermes, etc.)
│    ├── pool/            # EnginePool, MemoryEnforcer, ModelDiscovery, PriorityScheduler
│    ├── router/          # RequestRouter, CloudRouter, SmartRouter
│    ├── scheduler/       # 25-module scheduler (admission, batching, cache, step, etc.)
│    ├── speculative/     # SuffixDecoding, DFlash, DSpark, MTP, VLM MTP
│    ├── video/           # Pure-MLX video generation ports (LTX-2, Wan2, SkyReels-V3)
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

## Flux-1.lite-8B-MLX 深度优化 (2026-07-19)

### 性能数据 (M5 Max 128GB / MLX 0.32 / Q4)

| 指标 | 原基线 | block 整编译融合 | mlx-mfa Metal attn 对接 | 真上限 |
|---|---|---|---|---|
| step/s (512×512×4步) | 1.83 | 1.96 | **1.88** | 1.88-2.03 |
| Metal 峰值 | 10.8 GB | 10.6 GB | 10.5 GB | 10.5 GB |
| 256×256 真上限 | — | — | — | 4.62 step/s |

bench.dpdns.org 上传记录: id 27 (1.97), id 30 (1.96), id 31 (1.88), id 32 (1.88 mlx-mfa Metal attn)

### 落地优化项

1. **block 整编译融合** (`joint_transformer_block.py` + `single_transformer_block.py`)
   - 加 `_compiled_call = mx.compile(self._call_raw)` 封装整块编译, 融合 AdaLN+attn+FFN 子模块为单编译单元
   - 消跨 `nn.Module` 子调用断融合, `__call__` 入口走 `_compiled_call` 编译版本
   - `to_out` list → `to_out_0` 命名属性 (MLX nn.Module 不收录 list 属性) + `flux_weight_mapping.py` 补 `to_out.0` → `to_out_0` 映射

2. **mlx-mfa Metal Flash Attention 内核对接** (`attention_utils.py::compute_attention`)
   - 用 `mlx_mfa.flash_attention` 替 `mx.fast.scaled_dot_product_attention`, 走 M5 Neural Accelerator 优 Tile
   - `has_nax: True` ✅ Metal 内核真触, head_dim=128 在 mlx_mfa 支持范围
   - 实测对接成功但收益持平 (1.88 vs 1.88 step/s), 因单替 SDPA 不覆盖 RoPE + QKV 投影主瓶颈

3. **Fused QKV+RoPE+attn 单算子图融合** (搁置)
   - 写 `_fused_qkv_rope_attn` 融合函数 + `JointAttention` 接入 + `mx.compile` 封装
   - 实测 Q4 量化权重是 `(out, in/8)` 压缩布局, 手动 `mx.matmul/mx.addmm` 破 `quantized_matmul` 封装报 ValueError
   - 回滚保 `nn.Linear.__call__` 走 `quantized_matmul`, 整 block `mx.compile` 已融合

### mlx-mfa 预编译路径 (避 PyPI wheel build 耗时不可控)

```bash
# 本地源 + scikit-build-core + nanobind 触发 CMake build 生成 _ext.so
pip install scikit-build-core nanobind
pip install -e /path/to/mlx_mfa-2.61.0/ --no-build-isolation
# 验证: has_nax() True = Metal 内核真触, False = fallback SDPA
python3 -c "from mlx_mfa import has_nax; print(has_nax())"
```

### 瓶颈诊断结论

- **256 vs 512 比值 2.48×** (理论算力 4×) → 混合瓶颈 (带宽+算力双优)
- **transformer 80%** 主瓶颈 / encode_prompt 10% / VAE 10%
- **schnell 天生不支持 CFG** (`supports_guidance=False`), `guidance=4.0` 是无效参数, 单分支已是最优
- **shape 抖动代价 21.4%**: 同 512×512 连续稳态 1.90 step/s, 不同尺寸交替降至 1.56 step/s
- **真上限已厘清**: 512×512 在 M5 Max Q4 + mlx_mfa Metal attn + 双层编译融合后 1.88 step/s 是硬件+Q4 量化+算子栈三元约束下的合理上限

### 关键经验沉淀

1. **MLX Q4 量化权重不可手动 matmul**: 权重是 `(out, in/8)` 压缩布局, 必走 `nn.Linear.__call__` 内部 `quantized_matmul`. 所有"手写 Fused 单内核"方案对 Q4 量化模型破封装不可行
2. **60+ block 整体 mx.compile 劣化是通用规律**: 算子图按 N× 累积触 Metal Command Buffer 雾溅, 双层编译 (block 整编译 + transformer 循环编译) 是最优路径
3. **mlx-mfa 预编译路径**: 本地源 + scikit-build-core + nanobind + `pip install -e --no-build-isolation` 成功触发 CMake build 生成 `_ext.so`, 避 PyPI wheel build 耗时不可控

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
