# Architecture

fusion-mlx is a multi-modal inference server built on Apple MLX. It serves LLM, VLM, audio, and image generation models through a unified OpenAI-compatible API.

## High-Level Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                    FastAPI Server (uvicorn)                     │
│   ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐   │
│   │ OpenAI    │  │ Anthropic │  │  Audio   │  │   Images │   │
│   │ Routes    │  │  Routes   │  │  Routes  │  │   Routes │   │
│   └────┬─────┘  └────┬─────┘  └────┬─────┘  └─────┬────┘   │
│        │              │              │              │          │
│   ┌────▼──────────────▼──────────────▼──────────────▼─────┐   │
│   │         RequestRouter / SmartRouter (dispatch)          │   │
│   │  - Modality-based routing (text/image/audio/gen)        │   │
│   │  - Phase-aware split (prefill → decode on different    │   │
│   │    backends)                                             │   │
│   │  - Priority scheduling (REALTIME/BATCH/BACKGROUND)      │   │
│   │  - Cloud fallback for large uncached context            │   │
│   └──────────────────────┬─────────────────────────────────┘   │
│                          │                                       │
│   ┌─────────────────────▼──────────────────────────────────┐   │
│   │                 EnginePool (LRU + Memory)                │   │
│   │   ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐  │   │
│   │   │ Batched   │ │   VLM    │ │  Embed   │ │  Audio   │  │   │
│   │   │ Engine    │ │  Engine  │ │  Engine  │ │  Engine  │  │   │
│   │   └────┬─────┘ └────┬─────┘ └────┬─────┘ └────┬─────┘  │   │
│   └────────┼─────────────┼────────────┼────────────┼────────┘   │
│            │             │            │            │               │
│   ┌───────▼─────────────▼────────────▼────────────▼─────────┐   │
│   │              Scheduler (continuous batching)               │   │
│   │   - Waiting queue   - Running set   - Preemption         │   │
│   │   - Chunked prefill (512 tokens)   - KV cache mgmt      │   │
│   └─────────────────────┬───────────────────────────────────┘   │
│                          │                                        │
│   ┌─────────────────────▼───────────────────────────────────┐   │
│   │         Typed Executor Pools (thread isolation)           │   │
│   │   ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐      │   │
│   │   │  LLM    │ │  Image  │ │  Audio  │ │   IO    │      │   │
│   │   │ (1 wrk) │ │ (1 wrk) │ │ (2 wrk) │ │ (2 wrk) │      │   │
│   │   └─────────┘ └─────────┘ └─────────┘ └─────────┘      │   │
│   └──────────────────────────────────────────────────────────┘   │
│                          │                                        │
│   ┌─────────────────────▼───────────────────────────────────┐   │
│   │              MLX Thread (Metal kernels)                    │   │
│   │   - BatchGenerator   - Forward pass   - Sampling         │   │
│   └──────────────────────────────────────────────────────────┘   │
│                          │                                        │
│   ┌─────────────────────▼───────────────────────────────────┐   │
│   │         ProcessMemoryEnforcer (deadlock-free)            │   │
│   │   - Timeout-based lock acquisition (2s)                  │   │
│   │   - Mark-then-execute eviction fallback                  │   │
│   │   - Double gc.collect() around mx.clear_cache()         │   │
│   └──────────────────────────────────────────────────────────┘   │
└──────────────────────────────────────────────────────────────┘
```

## Request Flow

1. **API Route** — Client sends request to `/v1/chat/completions`
2. **Adapter** — `OpenAIAdapter` or `AnthropicAdapter` normalizes request to `InternalRequest`
3. **Router** — `RequestRouter` dispatches by modality, `SmartRouter` decides prefill/decode backends
4. **EnginePool** — Looks up or loads the appropriate engine by model name
5. **Engine** — `BatchedEngine` creates a `Request` with `SamplingParams`
6. **EngineCore** — Submits request to the `Scheduler` via typed executor pool
7. **Scheduler** — Manages waiting queue, running batch, and KV cache
8. **MLX Thread** — Runs `scheduler.step()` → `BatchGenerator` → model forward pass
9. **Output Collector** — Streams tokens back through `AsyncIterator` to the client

## Component Layers

### 1. API Layer (`fusion_mlx/api/`)

Handles HTTP request parsing, validation, and response formatting. Each API flavor (OpenAI, Anthropic, Audio, Images, OpenClaw) has its own router and adapter.

- **Routes** — FastAPI endpoint definitions with Pydantic models
- **Adapters** — Convert between API-specific formats and internal representations
- **Tool Calling** — JSON schema validation, tool dispatch, and output parsing
- **OpenClaw Agent Protocol** — Multi-turn sessions with TTL (1h), max cap (1000), LRU eviction

### 2. Engine Layer (`fusion_mlx/engines/`)

Eight engine types, each optimized for a specific modality:

| Engine | Modality | Executor Pool | Key Features |
|--------|----------|---------------|-------------|
| `BatchedEngine` | LLM text | llm (1 worker) | Continuous batching, streaming, tool calling |
| `VLMBatchedEngine` | Vision + text | io (2 workers) | Image/video understanding, MTP drafter |
| `EmbeddingEngine` | Text → vectors | llm (1 worker) | Batch embedding generation |
| `RerankerEngine` | Passage ranking | llm (1 worker) | Cohere/Jina compatible reranking |
| `STTEngine` | Audio → text | audio (2 workers) | Whisper, VibeVoice-ASR |
| `TTSEngine` | Text → audio | audio (2 workers) | Kokoro TTS, voice cloning, streaming WAV |
| `STSEngine` | Audio → audio | audio (2 workers) | Speech enhancement, source separation |
| `ImageGenEngine` | Text → images | image (1 worker) | Flux 2 diffusion model |

### 3. Pool Layer (`fusion_mlx/pool/`)

Manages model lifecycle, memory, and concurrency:

- **EnginePool** — Central model registry with LRU eviction
    - Auto-discovers models from HuggingFace cache directories
    - Maps model type to engine class (LLM → BatchedEngine, etc.)
    - Pins frequently-used models to prevent eviction
    - TTL-based expiration for idle models
    - Double `gc.collect()` pattern (before + after `mx.clear_cache()`)

- **ProcessMemoryEnforcer** — 4-tier memory protection, deadlock-free:
    - **Safe** — 25% of system RAM reserved for OS
    - **Balanced** — 50% reserved (default)
    - **Aggressive** — 75% reserved for models
    - **Custom** — User-specified byte limit
    - Timeout-based lock acquisition (2s) to avoid blocking during Metal allocation
    - Mark-then-execute eviction when lock is held by a loading coroutine

- **ModelDiscovery** — Scans directories for MLX-format models, estimates size and type

- **PriorityScheduler** — Metal command queue priorities per task type:
    - REALTIME (Claude Code) — highest priority, lowest latency
    - BATCH (OpenClaw agents) — throughput-oriented
    - BACKGROUND (embedding/reranking) — lowest priority

### 4. Cache Layer (`fusion_mlx/cache/`)

Three-tier caching for KV states:

1. **PagedCache** — Block-based KV cache in GPU memory
    - Fixed-size blocks (default 64 tokens)
    - Dynamic allocation with LRU eviction
    - Up to 1000 blocks by default

2. **PagedSSDCache** — SSD cold layer for evicted blocks
    - Spills inactive blocks to SSD when GPU memory is full
    - 20 GB default capacity
    - Transparent recovery when blocks are needed again
    - Mock-friendly dtype serialization for cross-environment testing

3. **BlockAwarePrefixCache** — Copy-on-write prefix sharing
    - Shared prefixes between concurrent requests
    - COW semantics — blocks are copied only when modified
    - Reduces redundant computation for common prompts

### 5. Scheduler (`fusion_mlx/scheduler/`)

Split into 21 focused modules (~500 lines each). Core features:

- **Waiting Queue** — New requests wait for batch slots
- **Running Set** — Active requests processed in parallel
- **Chunked Prefill** — 512-token chunks to avoid memory spikes and allow preemption
- **Preemption** — Low-priority requests can be swapped out under memory pressure
- **Speculative Decoding** — Integrates SuffixDecoding, DFlash, MTP, and VLM MTP
- **Mid-Prefill Save** — Periodic cache snapshots during long prefill steps

### 6. Speculative Decoding (`fusion_mlx/speculative/`)

Four methods to accelerate token generation:

| Method | How It Works | Speedup |
|--------|-------------|---------|
| SuffixDecoding | Reuses suffix patterns from previous generations | 1.5-2× |
| DFlash | Block-level diffusion — drafts groups of tokens | 2-3× |
| MTP | Multi-Token Prediction — native for Qwen3.5/3.6, DeepSeek | 2-5× |
| VLM MTP | External assistant drafter for VLM models | 1.5-2× |

### 7. Router (`fusion_mlx/router/`)

Three routing layers, applied in order:

- **RequestRouter** — Routes requests to the correct engine by modality:
    - Pure text → `BatchedEngine`
    - Text + images/videos → `VLMBatchedEngine`
    - Embedding requests → `EmbeddingEngine`
    - Audio → `STTEngine` / `TTSEngine` / `STSEngine`
    - Image generation → `ImageGenEngine`
    - Large uncached context → `CloudRouter` (both streaming and non-streaming)

- **SmartRouter** — Phase-aware routing with cross-engine handoff:
    - Prefill on omlx (strong matmul), decode on Rapid-MLX (lightweight KV)
    - Benchmark-based backend selection with EMA smoothing (alpha=0.7)
    - REALTIME tasks skip benchmark routing to avoid high-latency backends
    - Phase split threshold: 8192 uncached tokens with <50% cache hit rate
    - Cloud fallback at 32768 uncached tokens

- **CloudRouter** — Optional fallback to cloud providers via litellm:
    - Circuit breaker prevents local/cloud oscillation (5 consecutive failures → open)
    - Supports both streaming and non-streaming cloud calls
    - Custom API base/key for OpenAI-compatible providers

### 8. Integrations (`fusion_mlx/integrations/`)

Pre-built connectors for AI development tools:

- **Claude Code** — `fusion-mlx launch claude` sets up environment variables
- **OpenClaw** — Writes `~/.openclaw/config.yaml`
- **GitHub Copilot** — Copilot-compatible proxy
- **OpenAI Codex** — Codex CLI integration
- **ComfyUI** — ComfyUI node server (stub)

## Thread Model

```
Main Thread (asyncio)          Typed Executor Pools          MLX Thread
┌─────────────────────┐       ┌──────────────────┐       ┌──────────────────────┐
│ FastAPI request      │       │ LLM pool (1 wrk) │       │ scheduler.step()       │
│   ├─ parse request   │──────>│   ├─ mx.array()  │──────>│   ├─ BatchGenerator   │
│   ├─ create Request  │       │   ├─ mx.eval()   │       │   ├─ model forward()  │
│   ├─ add to queue    │       │ Image pool (1 wrk)│       │   ├─ sample token     │
│   ├─ wait on queue   │<──────│ Audio pool (2 wrk)│       │   └─ return Output   │
│   └─ yield tokens    │       │ IO pool (2 wrk)   │       └──────────────────────┘
└─────────────────────┘       └──────────────────┘
```

- ML operations run on dedicated typed executor pools (llm/image/audio/io)
- IO pool (2 workers) handles model loading to avoid blocking inference
- Audio pool (2 workers) allows concurrent STT + TTS
- Token generation flows back via `asyncio.Queue` through `RequestOutputCollector`
- All `run_in_executor` calls have timeout protection via `asyncio.wait_for()`

## Memory Management

```
System RAM (e.g., 64 GB)
├── 32 GB — OS / other apps (Balanced tier: 50%)
└── 32 GB — fusion-mlx budget
     ├── Model weights (GPU)
     ├── KV cache (PagedCache → PagedSSDCache → disk)
     └── Prefix cache (shared blocks with COW)
```

The `ProcessMemoryEnforcer` monitors process memory in real-time. When memory exceeds the budget, it triggers:

1. **Soft warning** — Log warning, signal admission pause
2. **Cache eviction** — Evict least-recently-used KV cache blocks to SSD
3. **Request preemption** — Swap out low-priority requests
4. **Request abort** — Abort in-flight requests when memory is critically low

**Deadlock prevention**: The enforcer uses a 2-second timeout when acquiring the pool lock. If the lock is held by a loading coroutine (which blocks during Metal allocation), the enforcer marks models for eviction via `abort_loading=True` rather than waiting. This prevents OOM crashes when memory pressure hits during a slow model load.

**GC strategy**: Double `gc.collect()` pattern around every `mx.clear_cache()`:
- First `gc.collect()` BEFORE `clear_cache()` — frees C++ Metal buffer wrappers
- Second `gc.collect()` AFTER `clear_cache()` — collects Python-side wrapper objects
