# Architecture

fusion-mlx is a multi-modal inference server built on Apple MLX. It serves LLM, VLM, audio, and image generation models through OpenAI- and Anthropic-compatible APIs.

## High-Level Architecture

```
┌──────────────────────────────────────────────────────────────────────┐
│                      FastAPI Server (uvicorn)                         │
│   ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐           │
│   │ OpenAI    │  │ Anthropic │  │  Audio   │  │   Images │           │
│   │ Routes    │  │  Routes   │  │  Routes  │  │   Routes │           │
│   └────┬─────┘  └────┬─────┘  └────┬─────┘  └─────┬────┘           │
│        │              │              │              │                  │
│   ┌────▼──────────────▼──────────────▼──────────────▼─────────────┐  │
│   │         RequestRouter / SmartRouter (dispatch)                  │  │
│   │  - Modality-based routing (text/image/audio/gen)               │  │
│   │  - Phase-aware split (prefill → decode on different backends)  │  │
│   │  - Priority scheduling (REALTIME/BATCH/BACKGROUND)             │  │
│   │  - Cloud fallback for large uncached context                   │  │
│   └──────────────────────┬────────────────────────────────────────┘  │
│                          │                                            │
│   ┌─────────────────────▼──────────────────────────────────────────┐  │
│   │                 EnginePool (LRU + Memory)                        │  │
│   │   ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐        │  │
│   │   │ Batched   │ │   VLM    │ │  Embed   │ │  Audio   │        │  │
│   │   │ Engine    │ │  Engine  │ │  Engine  │ │  Engine  │        │  │
│   │   └────┬─────┘ └────┬─────┘ └────┬─────┘ └────┬─────┘        │  │
│   └────────┼─────────────┼────────────┼────────────┼──────────────┘  │
│            │             │            │            │                    │
│   ┌───────▼─────────────▼────────────▼────────────▼────────────────┐  │
│   │         Scheduler (25 modules, continuous batching)              │  │
│   │   - Waiting queue   - Running set   - Preemption                │  │
│   │   - Chunked prefill   - TurboQuant KV   - Fused sampler         │  │
│   │   - Output Collector   - Stale request recovery                  │  │
│   └─────────────────────┬──────────────────────────────────────────┘  │
│                          │                                             │
│   ┌─────────────────────▼──────────────────────────────────────────┐  │
│   │         Typed Executor Pools (thread isolation)                  │  │
│   │   ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐            │  │
│   │   │  LLM    │ │  Image  │ │  Audio  │ │   IO    │            │  │
│   │   │ (1 wrk) │ │ (1 wrk) │ │ (2 wrk) │ │ (2 wrk) │            │  │
│   │   └─────────┘ └─────────┘ └─────────┘ └─────────┘            │  │
│   └──────────────────────────────────────────────────────────────────┘  │
│                          │                                              │
│   ┌─────────────────────▼──────────────────────────────────────────┐  │
│   │              MLX Thread (Metal kernels)                          │  │
│   │   - BatchGenerator   - Forward pass   - Fused sampler           │  │
│   └──────────────────────────────────────────────────────────────────┘  │
│                          │                                              │
│   ┌─────────────────────▼──────────────────────────────────────────┐  │
│   │         ProcessMemoryEnforcer (deadlock-free)                    │  │
│   │   - Timeout-based lock acquisition (2s)                          │  │
│   │   - Mark-then-execute eviction fallback                          │  │
│   │   - Double gc.collect() around mx.clear_cache()                 │  │
│   └──────────────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────────────────┘
```

## Request Flow

1. **API Route** — Client sends request to `/v1/chat/completions` or `/v1/messages`
2. **Adapter** — `OpenAIAdapter` or `AnthropicAdapter` normalizes request to `InternalRequest`
3. **Dispatch** — `RequestRouter` dispatches by modality, `SmartRouter` decides prefill/decode backends
4. **EnginePool** — Looks up or loads the appropriate engine by model name
5. **Engine** — `BatchedEngine` creates a `Request` with `SamplingParams`
6. **EngineCore** — Submits request to the `Scheduler` via typed executor pool
7. **Scheduler** — Manages waiting queue, running batch, KV cache, and continuous batching
8. **MLX Thread** — Runs `scheduler.step()` → `BatchGenerator` → model forward pass → fused sampler
9. **Output Collector** — `RequestOutputCollector` buffers and merges tokens, streams back via `AsyncIterator`

## Component Layers

### 1. API Layer (`fusion_mlx/api/`)

Handles HTTP request parsing, validation, and response formatting. Each API flavor has its own router and adapter.

**Layering note** - three distinct packages collaborate and were historically confused by near-identical names:
- `fusion_mlx/api/` - the **external** OpenAI/Anthropic-compatible API surface (`/v1/chat/completions`, `/v1/messages`, etc.), each flavor with its own router + adapter.
- `fusion_mlx/routes/` - **internal** FastAPI APIRouters (`cache`, `health`, `metrics`, `responses`) mounted on the same app; these are operational/admin endpoints, not the public API.
- `fusion_mlx/dispatch/` (see §7) - **request-dispatch logic** (`RequestRouter`/`SmartRouter`/`CloudRouter`), NOT a FastAPI router layer. Renamed from `router/` to `dispatch/` to disambiguate from `routes/`.

- **OpenAI Routes** — `/v1/chat/completions`, `/v1/completions`, `/v1/models`, `/v1/embeddings`
- **Anthropic Routes** — `/v1/messages`, `/v1/count_tokens` with streaming tool_use blocks
- **Audio Routes** — `/v1/audio/transcriptions`, `/v1/audio/speech`, `/v1/audio/process`
- **Image Routes** — `/v1/images/generate` (Flux 2)
- **MCP Routes** — `/v1/mcp/tools`, `/v1/mcp/servers`, `/v1/mcp/execute`
- **OpenClaw Agent Protocol** — Multi-turn sessions with TTL (1h), max cap (1000), LRU eviction
- **Adapters** — Convert between API-specific formats and internal representations
- **Tool Calling** — JSON schema validation, tool dispatch, output parsing, streaming blocks

### 2. Engine Layer (`fusion_mlx/engines/`)

Eight engine types, each optimized for a specific modality:

| Engine | Modality | Executor Pool | Key Features |
|--------|----------|---------------|-------------|
| `BatchedEngine` | LLM text | llm (1 worker) | Continuous batching, streaming, tool calling, thinking mode |
| `VLMBatchedEngine` | Vision + text | io (2 workers) | Image/video understanding, MTP drafter, paged KV cache |
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

3. **BlockAwarePrefixCache** — Copy-on-write prefix sharing
    - Shared prefixes between concurrent requests
    - COW semantics — blocks are copied only when modified
    - Reduces redundant computation for common prompts

### 5. Scheduler (`fusion_mlx/scheduler/`)

Decomposed into 25 focused modules (~400 lines each):

| Module | Purpose |
|--------|---------|
| `config.py` | Scheduler configuration |
| `core.py` | Core scheduler loop and state management |
| `types.py` | Request/Response type definitions |
| `sched_admission.py` | Request admission control under memory pressure |
| `sched_batch.py` | Batch formation and management |
| `sched_boundary.py` | Boundary condition handling |
| `sched_cache.py` | Cache-aware scheduling decisions |
| `sched_handoff.py` | Phase handoff (prefill → decode) |
| `sched_init.py` | Scheduler initialization |
| `sched_misc.py` | Utility scheduling operations |
| `sched_query.py` | Query scheduling and GPU OOM preflight guard |
| `sched_response.py` | Response processing and output collection |
| `sched_schedule.py` | Main scheduling loop (prefill, insert, decode) |
| `sched_specprefill.py` | Speculative prefill |
| `sched_step.py` | Step execution with stale request recovery |
| `sched_thinking.py` | Thinking/reasoning token scheduling |
| `sched_token.py` | Token-level scheduling and boundary |
| `sched_trim.py` | Context trimming for long conversations |
| `sched_vlm_mtp.py` | VLM multi-token prediction |
| `sched_vlm_mtp_batched.py` | Batched VLM MTP (~14 → ~27 tok/s per request) |
| `compiled_kv_cache.py` | Compiled KV cache operations |
| `monkeypatches.py` | Runtime patches for MLX compatibility |
| `sampler_fast_path.py` | Fused sampler — skip logsumexp, batched sampling |
| `helpers.py` | Shared utility functions |

**Key scheduling flows:**

- **Continuous batching** — Multiple requests share one GPU step, giving 2× aggregate throughput under concurrent load
- **Chunked prefill** — 512-token chunks to avoid memory spikes and allow preemption
- **Stale request recovery** — After prefill+insert, the first decode step may return empty responses; the scheduler detects and correctly recovers without losing tokens
- **TurboQuant KV** — 4-bit KV cache quantization reduces memory traffic ~4× for KV reads
- **Fused sampler** — Skips logsumexp when not needed, eliminates `.item()` GPU sync calls, auto-detects and applies batched sampling

### 6. Speculative Decoding (`fusion_mlx/speculative/`)

Four methods to accelerate token generation:

| Method | How It Works | Speedup |
|--------|-------------|---------|
| SuffixDecoding | Reuses suffix patterns from previous generations | 1.5-2× |
| DFlash | Block-level diffusion — drafts groups of tokens | 2-3× |
| MTP | Multi-Token Prediction — native for Qwen3.5/3.6, DeepSeek | 2-5× |
| VLM MTP | External assistant drafter for VLM models | 1.5-2× |

### 7. Dispatch (`fusion_mlx/dispatch/`)

Three routing layers, applied in order:

- **RequestRouter** — Routes requests to the correct engine by modality:
    - Pure text → `BatchedEngine`
    - Text + images/videos → `VLMBatchedEngine`
    - Embedding requests → `EmbeddingEngine`
    - Audio → `STTEngine` / `TTSEngine` / `STSEngine`
    - Image generation → `ImageGenEngine`
    - Large uncached context → `CloudRouter`

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

8 pre-built connectors for AI development tools:

| Integration | What it does |
|-------------|-------------|
| Claude Code | Sets `ANTHROPIC_BASE_URL` and `ANTHROPIC_AUTH_TOKEN` for local proxy |
| OpenClaw | Writes `~/.openclaw/config.yaml` with local server URL |
| GitHub Copilot | Copilot-compatible proxy |
| OpenAI Codex | Codex CLI integration |
| ComfyUI | ComfyUI node server for Flux 2 |
| OpenCode | OpenCode integration |
| Pi | Pi integration |
| Hermes | Hermes tool parser |

## Thread Model

```
Main Thread (asyncio)          Typed Executor Pools          MLX Thread
┌─────────────────────┐       ┌──────────────────┐       ┌──────────────────────┐
│ FastAPI request      │       │ LLM pool (1 wrk) │       │ scheduler.step()       │
│   ├─ parse request   │──────>│   ├─ mx.array()  │──────>│   ├─ BatchGenerator   │
│   ├─ create Request  │       │   ├─ mx.eval()   │       │   ├─ model forward()  │
│   ├─ add to queue    │       │ Image pool (1 wrk)│       │   ├─ fused sampler    │
│   ├─ wait on queue   │<──────│ Audio pool (2 wrk)│       │   └─ return Output   │
│   └─ yield tokens    │       │ IO pool (2 wrk)   │       └──────────────────────┘
└─────────────────────┘       └──────────────────┘
```

- ML operations run on dedicated typed executor pools (llm/image/audio/io)
- IO pool (2 workers) handles model loading to avoid blocking inference
- Audio pool (2 workers) allows concurrent STT + TTS
- Token generation flows back via `asyncio.Queue` through `RequestOutputCollector`
- All `run_in_executor` calls have timeout protection via `asyncio.wait_for()`

## Output Pipeline

```
BatchGenerator._next()
    → gen_responses (per-request token arrays)
    → _process_batch_responses()
        → RequestOutput (new_text, output_text, finished, finish_reason)
    → RequestOutputCollector._merge_outputs()
        → Concatenates new_text, merges cumulative output_text
    → EngineCore._engine_loop()
        → Distributes to per-request collectors via ctx.collector.put()
    → BatchedEngine.generate()
        → clean_special_tokens(output_text)
        → extract_thinking() splits reasoning vs regular content
    → API adapter formats response (OpenAI or Anthropic)
```

Key behaviors:
- **Stale recovery**: After prefill+insert, the first decode may return empty responses. The scheduler detects this (empty responses + just scheduled) and skips the stale reschedule, avoiding token loss.
- **Thinking extraction**: `extract_thinking()` splits `مایه...` tags into `reasoning_content` and regular `content` for both OpenAI and Anthropic APIs.
- **Streaming detokenization**: Tokens are decoded incrementally via the streaming detokenizer, avoiding full re-decode each step.

## Memory Management

```
System RAM (e.g., 128 GB)
├── 64 GB — OS / other apps (Balanced tier: 50%)
└── 64 GB — fusion-mlx budget
     ├── Model weights (GPU)
     ├── KV cache (PagedCache → PagedSSDCache → disk)
     ├── TurboQuant KV (4-bit compressed, ~4× less memory traffic)
     └── Prefix cache (shared blocks with COW)
```

The `ProcessMemoryEnforcer` monitors process memory in real-time. When memory exceeds the budget, it triggers:

1. **Soft warning** — Log warning, signal admission pause
2. **Cache eviction** — Evict least-recently-used KV cache blocks to SSD
3. **Request preemption** — Swap out low-priority requests
4. **Request abort** — Abort in-flight requests when memory is critically low

**GPU OOM preflight guard**: Before scheduling a prefill, the scheduler estimates the memory needed (model weights + KV cache + activation tensors) and refuses admission if it would exceed available Metal memory. This prevents Metal GPU OOM crashes.

**Deadlock prevention**: The enforcer uses a 2-second timeout when acquiring the pool lock. If the lock is held by a loading coroutine (which blocks during Metal allocation), the enforcer marks models for eviction via `abort_loading=True` rather than waiting.

**GC strategy**: Double `gc.collect()` pattern around every `mx.clear_cache()`:
- First `gc.collect()` BEFORE `clear_cache()` — frees C++ Metal buffer wrappers
- Second `gc.collect()` AFTER `clear_cache()` — collects Python-side wrapper objects
