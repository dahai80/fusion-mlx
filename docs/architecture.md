# Architecture

fusion-mlx is a multi-modal inference server built on Apple MLX. It serves LLM, VLM, audio, and image generation models through a unified OpenAI-compatible API.

## High-Level Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                    FastAPI Server (uvicorn)                    │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌────────────┐  │
│  │ OpenAI   │  │ Anthropic │  │  Audio   │  │   Images   │  │
│  │ Routes   │  │  Routes   │  │  Routes  │  │   Routes   │  │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘  └─────┬──────┘  │
│       │              │              │              │           │
│  ┌────▼──────────────▼──────────────▼──────────────▼─────┐   │
│  │              RequestRouter (modality dispatch)          │   │
│  └──────────────────────┬─────────────────────────────────┘   │
│                         │                                      │
│  ┌─────────────────────▼──────────────────────────────────┐   │
│  │                 EnginePool (LRU + Memory)               │   │
│  │  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐  │   │
│  │  │ Batched  │ │   VLM    │ │  Embed   │ │  Audio   │  │   │
│  │  │ Engine   │ │  Engine  │ │  Engine  │ │  Engine  │  │   │
│  │  └────┬─────┘ └────┬─────┘ └────┬─────┘ └────┬─────┘  │   │
│  └───────┼─────────────┼────────────┼────────────┼─────────┘   │
│           │             │            │            │               │
│  ┌────────▼────────────▼────────────▼────────────▼─────────┐   │
│  │              Scheduler (continuous batching)              │   │
│  │  - Waiting queue  - Running set  - Preemption           │   │
│  │  - Chunked prefill  - KV cache management               │   │
│  └─────────────────────┬───────────────────────────────────┘   │
│                         │                                       │
│  ┌─────────────────────▼───────────────────────────────────┐   │
│  │              MLX Thread (Metal kernels)                   │   │
│  │  - BatchGenerator  - Forward pass  - Sampling           │   │
│  └─────────────────────────────────────────────────────────┘   │
└──────────────────────────────────────────────────────────────┘
```

## Request Flow

1. **API Route** — Client sends request to `/v1/chat/completions`
2. **Adapter** — `OpenAIAdapter` or `AnthropicAdapter` normalizes request to `InternalRequest`
3. **EnginePool** — Looks up or loads the appropriate engine by model name
4. **Engine** — `BatchedEngine` creates a `Request` with `SamplingParams`
5. **EngineCore** — Submits request to the `Scheduler` via dedicated MLX thread
6. **Scheduler** — Manages waiting queue, running batch, and KV cache
7. **MLX Thread** — Runs `scheduler.step()` → `BatchGenerator` → model forward pass
8. **Output Collector** — Streams tokens back through `AsyncIterator` to the client

## Component Layers

### 1. API Layer (`fusion_mlx/api/`)

Handles HTTP request parsing, validation, and response formatting. Each API flavor (OpenAI, Anthropic, Audio, Images) has its own router and adapter.

- **Routes** — FastAPI endpoint definitions with Pydantic models
- **Adapters** — Convert between API-specific formats and internal representations
- **Tool Calling** — JSON schema validation, tool dispatch, and output parsing

### 2. Engine Layer (`fusion_mlx/engines/`)

Eight engine types, each optimized for a specific modality:

| Engine | Modality | Key Features |
|--------|----------|-------------|
| `BatchedEngine` | LLM text | Continuous batching, streaming, tool calling |
| `VLMBatchedEngine` | Vision + text | Image/video understanding, MTP drafter |
| `EmbeddingEngine` | Text → vectors | Batch embedding generation |
| `RerankerEngine` | Passage ranking | Cohere/Jina compatible reranking |
| `STTEngine` | Audio → text | Whisper, VibeVoice-ASR |
| `TTSEngine` | Text → audio | Kokoro TTS, voice cloning, streaming WAV |
| `STSEngine` | Audio → audio | Speech enhancement, source separation |
| `ImageGenEngine` | Text → images | Flux 2 diffusion model |

### 3. Pool Layer (`fusion_mlx/pool/`)

Manages model lifecycle, memory, and concurrency:

- **EnginePool** — Central model registry with LRU eviction
  - Auto-discovers models from HuggingFace cache directories
  - Maps model type to engine class (LLM → BatchedEngine, etc.)
  - Pins frequently-used models to prevent eviction
  - TTL-based expiration for idle models

- **ProcessMemoryEnforcer** — 4-tier memory protection:
  - **Safe** — 25% of system RAM reserved for OS
  - **Balanced** — 50% reserved (default)
  - **Aggressive** — 75% reserved for models
  - **Custom** — User-specified byte limit

- **ModelDiscovery** — Scans directories for MLX-format models, estimates size and type

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

3. **BlockAwarePrefixCache** — Copy-on-write prefix sharing
   - Shared prefixes between concurrent requests
   - COW semantics — blocks are copied only when modified
   - Reduces redundant computation for common prompts

### 5. Scheduler (`fusion_mlx/scheduler.py`)

The heart of continuous batching, ~285 KB of logic:

- **Waiting Queue** — New requests wait for batch slots
- **Running Set** — Active requests processed in parallel
- **Chunked Prefill** — Long prompts are split into chunks to avoid memory spikes
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

- **RequestRouter** — Routes requests to the correct engine by modality:
  - Pure text → `BatchedEngine`
  - Text + images/videos → `VLMBatchedEngine`
  - Embedding requests → `EmbeddingEngine`
  - Audio → `STTEngine` / `TTSEngine` / `STSEngine`
  - Image generation → `ImageGenEngine`

- **CloudRouter** — Optional fallback to cloud providers for large contexts (>32K tokens)

### 8. Integrations (`fusion_mlx/integrations/`)

Pre-built connectors for AI development tools:

- **Claude Code** — `fusion-mlx launch claude` sets up environment variables
- **OpenClaw** — Writes `~/.openclaw/config.yaml`
- **GitHub Copilot** — Copilot-compatible proxy
- **OpenAI Codex** — Codex CLI integration
- **ComfyUI** — ComfyUI node server (stub)

## Thread Model

```
Main Thread (asyncio)          MLX Thread (ThreadPoolExecutor, 1 worker)
┌─────────────────────┐        ┌──────────────────────────────┐
│ FastAPI request     │        │ scheduler.step()              │
│  ├─ parse request   │        │  ├─ BatchGenerator.forward() │
│  ├─ create Request  │──────►│  ├─ model forward pass       │
│  ├─ add to queue    │        │  ├─ sample next token        │
│  ├─ wait on queue   │◄──────│  └─ return RequestOutput     │
│  └─ yield tokens    │        └──────────────────────────────┘
└─────────────────────┘
```

- All MLX operations run on a dedicated single-threaded worker to avoid Metal device conflicts
- Requests are submitted synchronously (no executor overhead for queueing)
- Token generation flows back via `asyncio.Queue` through `RequestOutputCollector`

## Memory Management

```
System RAM (e.g., 64 GB)
├── 32 GB — OS / other apps (Balanced tier: 50%)
└── 32 GB — fusion-mlx budget
    ├── Model weights (GPU)
    ├── KV cache (PagedCache → PagedSSDCache → disk)
    └── Prefix cache (shared blocks with COW)
```

The `ProcessMemoryEnforcer` monitors `mx.get_active_memory()` and `mx.get_cached_memory()` in real-time. When memory exceeds the budget, it triggers:

1. **Soft warning** — Log warning, continue processing
2. **Cache eviction** — Evict least-recently-used KV cache blocks to SSD
3. **Request preemption** — Swap out low-priority requests
4. **Request abort** — Abort requests when memory is critically low
