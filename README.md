<div align="center">

# fusion-mlx

**Unified local model serving for Apple Silicon**

Drop-in replacement for Ollama / vLLM - runs natively on Metal via MLX

[![Version](https://img.shields.io/badge/v0.4.8-blue.svg)](https://github.com/dahai80/fusion-mlx/releases)
[![Python](https://img.shields.io/badge/Python-3.11+-3776AB.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-Apache--2.0-green.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/Tests-1200+-success.svg)](tests/)

[English](README.md) | [Chinese](README_CN.md)

[Get Started](#quick-start) · [Download App](https://github.com/dahai80/fusion-mlx/releases) · [Benchmarks](https://bench.dpdns.org/) · [Documentation](docs/)

</div>

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
- **Speculative denoise for DiT (#177)** - the diffusion analog of speculative
  decoding: a draft DiT predicts K velocity steps, the full DiT batched-verifies
  in one forward. Machinery landed (env-gated, default off); Phase-2 honestly
  **falsified** the layer-pruned draft on real 14B (0% acceptance) - the
  negative result + `GET /v1/videos/denoise-stats` surface are themselves a first
  in open-source MLX.
- **Fusion-ComfyUI Stage API + `on_step` (#170-172)** - 10 stage methods across
  text-encoder / DiT / VAE plus a thread->async `on_step` bridge; native ComfyUI
  integration no other MLX server offers.
- **SkyReels-V3 full family + upstream arch fixes (#164/#168/#193)** -
  R2V/V2V/A2V/A2W all run end-to-end on real 14B weights; fixed upstream config
  bugs (cross_attn_type routing, norm affine) that otherwise broke the model.
- **Flux2 Klein + `mx.compile` (#166)** - 1.9× (1.56s/step) with raw-diffusers
  Flux2 auto-detect.
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
- **Speculative decoding** - SuffixDecoding, DFlash, DSpark, MTP, VLM MTP (2–5× faster generation)
- **TurboQuant KV** - 4-bit KV cache quantization, 4× less memory traffic
- **40+ quant formats** - GGUF (Q2_K -> Q8_0), Imatrix (IQ1_M -> IQ4_XS), TurboQuant (TQ1_0/TQ2_0), MLX (mxfp4/mxfp8/6bit/4bit/8bit/F16/BF16/F32)
- **Paged KV cache** - SSD cold layer, block-aware prefix caching with COW sharing
- **Fused sampler** - skip logsumexp, eliminate GPU sync, batched sampling
- **SmartRouter** - phase-aware routing with benchmark-based backend selection and EMA smoothing
- **Priority scheduling** - REALTIME / BATCH / BACKGROUND queues with Metal command queue priorities
- **4-tier memory enforcer** - safe / balanced / aggressive / custom hard limits with deadlock-free eviction
- **Multi-model concurrency** - EnginePool with LRU eviction, pinning, and TTL
- **MCP tool support** - list, discover, and execute MCP tools via API
- **Admin web panel** - model management, live chat, HuggingFace downloads, online quantization
- **macOS native app** - SwiftUI with menu bar, auto-update, benchmark, model management, **hardware-aware setup wizard**
- **SkyReels-V3 video generation** - Pure-MLX port of the strongest open-source video model; all three branches (R2V/V2V/A2V) run end-to-end on real weights, with M5 Max dFlash attention + NF4 quantization keeping a 19B model at 720P under 14 GB resident memory
- **PyTorch -> MLX full-model converter** - `convert_skyreels_v3.py` one-shot converts SkyReels-V3's three branches (DiT + T5 + VAE + CLIP + audio) PyTorch weights to MLX safetensors, supporting bfloat16/float16/float32 + NF4 quantization with incremental per-shard writes to avoid unified-memory spikes
- **UMA Radix Latent cache** - repeat I2V requests skip the VAE-encode (model load + forward) via zero-copy `mx.array` reuse on Apple Silicon unified memory; extends the #178 radix cache from text KV to video frame latents (Phase-1: input-image latents, LTX-2 + Wan2.2). The UMA advantage the discrete-GPU CUDA stack cannot replicate. See [cache/LATENT_CACHE.md](fusion_mlx/cache/LATENT_CACHE.md)

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

### Converting models

Convert any HuggingFace model to MLX (optionally quantized) with the `convert` command - accepts a model alias or full HF repo:

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
fusion-mlx serve --model claude-4.6-sonnet   # -> Qwen3.5-27B-Claude-4.6-Opus-Distilled-MLX-6bit
fusion-mlx serve --model gpt-4o               # -> Qwen3-32B-A3B-Think-2512-MLX
```

## Integrations

```bash
# Claude Code - use fusion-mlx as your local Anthropic API
fusion-mlx launch claude

# OpenClaw - batch agent processing
fusion-mlx launch openclaw --model Qwen3-4B

# ComfyUI - image generation with Flux 2
fusion-mlx launch comfyui

# GitHub Copilot
fusion-mlx launch copilot
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
| DiT | `load_dit()` | `denoise(latent, pos_embed, neg_embed, steps, cfg, seed[, num_frames])` | `unload_dit()` |
| VAE | `load_vae()` | `decode(latent)` / `decode_tiled(latent, tile_size=256)` | `unload_vae()` |

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
#170 phase 2); `LegacyLTXBackend` and `Wan2Backend` wire real per-step denoise,
`LTX2Backend` / `SkyReelsBackend` accept-but-log.

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

Access at `http://localhost:8000/admin`:

- **Models** - load / unload / pin models dynamically, ParoQuant compat detection
- **Chat** - live chat interface for testing any model
- **Downloads** - HuggingFace / ModelScope model downloads with progress tracking
- **Quantization** - online quantization (oQ) pipeline
- **Benchmarks** - throughput and accuracy benchmarking
- **Monitoring** - real-time memory, performance, and request metrics
- **Settings** - global / per-model configuration, sub-API key management

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

### Speculative Denoise (#177)

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

### Speculative Denoise (#177)

扩散模型版的 speculative decoding: 草稿 DiT (层剪枝, 跑前 M/N 个 transformer block + 共享 head) 顺序预测 K=3-5 步未来速度场, 完整 DiT 单次 batched forward 验证 K 步 (per-element timestep, Wan2/SkyReels DiT 原生支持 `t.ndim==1`), 接受最长一致前缀, 分歧处用完整速度场补一步 (bonus step, 永不卡住). 目标 14B 上 2-3x 加速.

- 草稿协同加载: `LayerPrunedDraft(dit, n_blocks=M)` 复用同一份权重, 无需独立 draft checkpoint (MLX 量化非速度路径, 见 #166; 暂无 1B/3B SkyReels draft).
- 验证: K 个 latent 在 K 个不同 timestep 上单次前向 (批 per-element timestep embedding), 接受 `||v_draft - v_full|| / ||v_full|| < epsilon` 的最长前缀.
- 1 阶 Euler 推测环 (UniPC 2 阶 corrector 需上一步 full 输出, 推测模式旁路).
- env: `FUSION_SPECULATIVE_DENOISE` (默认 `"0"` 关), `FUSION_SPEC_K` (默认 4), `FUSION_SPEC_EPSILON` (默认 0.1), `FUSION_SPEC_DRAFT_BLOCKS` (phase-2 已接线, 默认 `num_layers//4`).

```bash
# phase-1: 模块 + API + 合成 DiT 单元测试 (env-gated, 不改生产 denoise 环)
# phase-2: R2V DiT forward_partial 接线 + 真 14B 接受率 sweep (负向结论, 见下)
# 默认关, 不影响现有 SkyReels-V3 生成路径
fusion-mlx serve --model SkyReels-V3-R2V-14B-MLX
```

> **范围 (phase-1 + phase-2)**: phase-1 落地推测去噪算法 + 草稿协同加载 API (`DraftDiTMixin.forward_partial` / `LayerPrunedDraft`) + 合成 DiT 单元测试, env-gated, **零生产风险**. phase-2 落地 R2V DiT `forward_partial` 接线 + 真 14B 接受率 sweep. **phase-2 实测结论 (负向)**: 层剪枝 draft 在安全 epsilon(0.1) 下接受率 0% (保留 25%-75% blocks), 仅保留 95% blocks 时出现接受但 draft 成本≈full 无加速且质量劣化(maxdiff 0.097); 放宽 epsilon 到 0.5 无效 (draft 速度场误差远超 0.5). #177 假设在 MLX SkyReels-V3 14B 上证伪: DiT 速度场需完整深度, 不可由子网络预测 (异于 LLM token 预测). 机制正确 (全拒绝时 spec==baseline Euler, 误差 7e-4), 保持落地 (env-gated 默认关, 零生产风险) 作为未来蒸馏小 draft 的基础设施. phase-3: fusion-comfyUI Stage API 接入. 详见 `fusion_mlx/video/skyreels_v3/SPECULATIVE_DENOISE.md`.

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
│    ├── video/           # Pure-MLX video generation ports (LTX-2, Wan2, SkyReels-V3, PuLID, LatentSync, MuseTalk)
│    └── admin/           # Web panel routes, benchmarking, downloads, settings
├── apps/fusion-mac/      # SwiftUI macOS app (~80 Swift files)
├── docs/                 # API reference, architecture, CLI guide, configuration
├── examples/             # 12 working code examples
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
