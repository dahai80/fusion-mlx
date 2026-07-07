# Configuration

fusion-mlx is configured through CLI flags and the `ServerConfig` dataclass. All settings live in `fusion_mlx/config.py`.

## Memory Tiers

Control how much system RAM the server can use for model inference:

| Tier | RAM Reserved for OS | Model Budget | Use Case |
|------|---------------------|---------------|----------|
| `safe` | 75% | 25% | Shared workstations, background serving |
| `balanced` | 50% | 50% | **Default** — good for dedicated inference |
| `aggressive` | 75% | 25% | Maximum model capacity, dedicated machines |
| `custom` | User-defined | User-defined | Precise control via `--custom-limit-mb` |

```bash
# Use balanced (default)
fusion-mlx serve

# Use aggressive — maximize model capacity
fusion-mlx serve --memory-tier aggressive

# Custom: limit to 16 GB
fusion-mlx serve --memory-tier custom --custom-limit-mb 16384
```

## Typed Executor Pools

MLX operations run on dedicated thread pools to prevent cross-modality blocking:

| Pool | Workers | Purpose |
|------|---------|---------|
| `llm` | 1 | LLM inference, embedding, reranking — single worker to avoid Metal device conflicts |
| `image` | 1 | Image generation (Flux 2) — isolated from text inference |
| `audio` | 2 | STT, TTS, STS — concurrent audio processing |
| `io` | 2 | Model loading, file I/O — non-blocking loads |

All `run_in_executor` calls are wrapped with `asyncio.wait_for()` for timeout protection:
- Model loading: 120s
- Inference: 30s (LLM), 60s (audio), 120s (image)
- Sync/clear: 5s

## Scheduler Settings

The `SchedulerConfig` dataclass controls batching, caching, and decoding:

### Concurrency

| Setting | Default | Description |
|---------|---------|-------------|
| `max_num_seqs` | 256 | Maximum concurrent sequences |
| `max_num_batched_tokens` | 65536 | Max tokens per batch step |

### Batching

| Setting | Default | Description |
|---------|---------|-------------|
| `prefill_batch_size` | 8 | Max sequences to start per prefill step |
| `completion_batch_size` | 32 | Max sequences in decoding batch |
| `prefill_step_size` | 2048 | Tokens processed per prefill step |

### Chunked Prefill

Splits long prompts into smaller chunks to avoid memory spikes and allow preemption:

| Setting | Default | Description |
|---------|---------|-------------|
| `chunked_prefill` | `True` | Enable chunked prefill |
| `chunked_prefill_tokens` | 512 | Tokens per chunk (0 = disabled) |
| `mid_prefill_save_interval` | 8192 | Save cache snapshot every N tokens |

The 512-token default balances between prefill overhead and REALTIME request latency. At 512 tokens, a 4K prompt takes ~8 chunks, each yielding ~2ms for high-priority requests to interleave.

### TurboQuant KV Cache

4-bit KV cache quantization that reduces memory traffic ~4× for KV reads:

| Setting | Default | Description |
|---------|---------|-------------|
| `kv_cache_quant_enabled` | `True` | Enable quantized KV cache |
| `kv_cache_quant_bits` | 4 | Bits per value (4 or 8) |
| `kv_cache_quant_group_size` | 64 | Quantization group size |
| `kv_cache_quant_min_tokens` | 256 | Minimum tokens before quantizing |

TurboQuant is enabled by default and is a key contributor to fusion-mlx's 2× concurrent throughput advantage. It compresses V-only KV cache to 4-bit with minimal quality loss.

### Paged Cache

Block-based KV cache with dynamic allocation:

| Setting | Default | Description |
|---------|---------|-------------|
| `paged_cache_enabled` | `True` | Enable paged KV cache |
| `paged_cache_block_size` | 64 | Tokens per block |
| `paged_cache_max_blocks` | 1000 | Maximum blocks |

### Prefix Cache

Copy-on-write prefix sharing for common prompts:

| Setting | Default | Description |
|---------|---------|-------------|
| `prefix_cache_enabled` | `True` | Enable prefix cache |
| `prefix_cache_max_size` | 100 | Maximum cached prefixes |

## SmartRouter

Phase-aware routing with benchmark-based backend selection. Configured via `RouterConfig`:

| Setting | Default | Description |
|---------|---------|-------------|
| `phase_split_threshold` | 8192 | Split prefill/decode when uncached tokens exceed this |
| `cloud_fallback_threshold` | 32768 | Route to cloud when uncached tokens exceed this |
| `enable_benchmark_routing` | `True` | Use EMA-smoothed benchmarks to select backends |
| `ema_alpha` | 0.7 | EMA smoothing factor (higher = more weight on history) |
| `prefill_chunk_size` | 512 | Tokens per prefill chunk for soft-preemption |
| `default_priority` | `BATCH` | Default task priority when no `task_tag` is provided |
| `warmup_batch_sizes` | `[1, 4, 8]` | Batch sizes to pre-compile compute graphs for |

**Priority levels** (determined by `task_tag`):
- `REALTIME` — Claude Code, interactive tools. Skips benchmark routing for lowest latency.
- `BATCH` — OpenClaw agents, batch processing. Uses benchmark routing for highest throughput.
- `BACKGROUND` — Embedding, reranking, offline tasks. Lowest priority, preemptible.

**Phase split example**: A 16K-token prompt with 20% cache hit rate:
- Prefill runs on omlx (strong matmul for large batches)
- Decode runs on Rapid-MLX (lightweight KV operations)
- KV cache is zero-copy transferred via `PhaseHandoff`

## Model Aliases

Map friendly names to full model IDs:

```python
# Default aliases in config.py
DEFAULT_ALIASES = {
     "claude-4.6-sonnet": "BeastCode/Qwen3.5-27B-Claude-4.6-Opus-Distilled-MLX-6bit",
     "claude-4.5-sonnet": "Qwen/Qwen3-32B-A3B-Think-2512-MLX",
     "gpt-4o": "Qwen/Qwen3-32B-A3B-Think-2512-MLX",
     "gpt-4.5": "BeastCode/Qwen3.5-27B-Claude-4.6-Opus-Distilled-MLX-6bit",
}
```

Custom aliases via `aliases.json` in `~/.fusion-mlx/`:
```json
{
     "my-model": "Qwen/Qwen2.5-7B-Instruct-MLX"
}
```

## Cloud Router

Automatically route large-context requests to cloud providers:

| Setting | Default | Description |
|---------|---------|-------------|
| `cloud_router_enabled` | `False` | Enable cloud fallback |
| `cloud_router_api_key` | `""` | Cloud provider API key |
| `cloud_router_threshold` | 32768 | Token threshold to trigger cloud routing |
| `cloud_router_api_base` | `None` | Custom API base for OpenAI-compatible providers |

```python
config = ServerConfig(
    cloud_router_enabled=True,
    cloud_router_api_key="sk-...",
    cloud_router_threshold=16384,    # Route to cloud at 16K+ tokens
)
```

**Circuit breaker**: After 5 consecutive local inference failures, the circuit opens and all requests route to cloud. A single local success closes the circuit.

**Streaming support**: Both streaming and non-streaming requests are routed to cloud when the threshold is exceeded. The cloud router uses litellm for provider-agnostic calls.

## SSD Cache

Offload inactive KV cache blocks to SSD:

| Setting | Default | Description |
|---------|---------|-------------|
| `ssd_cache_enabled` | `False` | Enable SSD cold layer |
| `ssd_cache_dir` | `~/.fusion-mlx/ssd-cache` | Cache directory |
| `ssd_cache_max_bytes` | 20 GB | Maximum disk usage |

```bash
# Enable via CLI
fusion-mlx serve --enable-ssd-cache
```

## OpenClaw Sessions

Agent session storage configuration:

| Setting | Default | Description |
|---------|---------|-------------|
| `_SESSION_TTL_SECONDS` | 3600 (1h) | Seconds before inactive session expiry |
| `_SESSION_MAX_COUNT` | 1000 | Maximum concurrent sessions (LRU eviction) |

Sessions are evicted in LRU order when the cap is reached. The TTL timer resets on every turn, tool-result submission, or session GET.

## Per-Model Settings

Each model can have custom settings stored in `~/.fusion-mlx/settings/`:

```json
{
     "Qwen3-4B-Q4_K_M": {
         "pinned": true,
         "ttl_seconds": 3600,
         "stream_interval": 1,
         "specprefill_enabled": false,
         "turboquant_kv_enabled": true,
         "dflash_enabled": false,
         "mtp_enabled": false,
         "vlm_mtp_enabled": false
     }
}
```

| Setting | Description |
|---------|-------------|
| `pinned` | Prevent LRU eviction |
| `ttl_seconds` | Seconds before idle unload (0 = never) |
| `stream_interval` | Tokens between stream updates (1 = every token) |
| `specprefill_enabled` | Enable speculative prefill |
| `turboquant_kv_enabled` | Enable TurboQuant 4-bit V-only KV compression (default: true) |
| `dflash_enabled` | Enable DFlash speculative decoding |
| `mtp_enabled` | Enable native MTP (Qwen3.5/3.6, DeepSeek-V4) |
| `vlm_mtp_enabled` | Enable VLM MTP with gemma4_assistant drafter |

> The speculative-decoding settings above (`specprefill_enabled`, `dflash_enabled`, `mtp_enabled`, `vlm_mtp_enabled`) mirror the `serve` flags. For the full method matrix, selection guide, the boot-time loading constraint, and the `SpecAutoRouter` API, see [Speculative Decoding](speculative-decoding.md).

## Server Config Summary

```python
ServerConfig(
    host="0.0.0.0",
    port=8000,
    model_dir="~/.fusion-mlx/models",
    memory=MemoryConfig(tier="balanced"),
    scheduler=SchedulerConfig(
        chunked_prefill=True,
        chunked_prefill_tokens=512,
        kv_cache_quant_enabled=True,
        kv_cache_quant_bits=4,
        max_num_batched_tokens=65536,
    ),
    model_aliases=DEFAULT_ALIASES,
    admin_enabled=True,
    cloud_router_enabled=False,
)
```
