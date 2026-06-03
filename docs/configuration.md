# Configuration

fusion-mlx is configured through CLI flags and the `ServerConfig` dataclass. All settings live in `fusion_mlx/config.py`.

## Memory Tiers

Control how much system RAM the server can use for model inference:

| Tier | RAM Reserved for OS | Model Budget | Use Case |
|------|---------------------|---------------|----------|
| `safe` | 75% | 25% | Shared workstations, background serving |
| `balanced` | 50% | 50% | **Default** — good for dedicated inference |
| `aggressive` | 25% | 75% | Maximum model capacity, dedicated machines |
| `custom` | User-defined | User-defined | Precise control via `--custom-limit-mb` |

```bash
# Use balanced (default)
fusion-mlx serve

# Use aggressive — maximize model capacity
fusion-mlx serve --memory-tier aggressive

# Custom: limit to 16 GB
fusion-mlx serve --memory-tier custom --custom-limit-mb 16384
```

## Scheduler Settings

The `SchedulerConfig` dataclass controls batching, caching, and decoding:

### Concurrency

| Setting | Default | Description |
|---------|---------|-------------|
| `max_num_seqs` | 256 | Maximum concurrent sequences |
| `max_num_batched_tokens` | 8192 | Max tokens per batch step |

### Batching

| Setting | Default | Description |
|---------|---------|-------------|
| `prefill_batch_size` | 8 | Max sequences to start per prefill step |
| `completion_batch_size` | 32 | Max sequences in decoding batch |
| `prefill_step_size` | 2048 | Tokens processed per prefill step |

### Chunked Prefill

Splits long prompts into smaller chunks to avoid memory spikes:

| Setting | Default | Description |
|---------|---------|-------------|
| `chunked_prefill_tokens` | 0 (off) | Tokens per chunk (0 = disabled) |
| `mid_prefill_save_interval` | 8192 | Save cache snapshot every N tokens |

Enable for prompts > 4K tokens:
```python
scheduler_config = SchedulerConfig(chunked_prefill_tokens=1024)
```

### KV Cache Quantization

Compress KV cache to reduce GPU memory:

| Setting | Default | Description |
|---------|---------|-------------|
| `kv_cache_quant_enabled` | `False` | Enable quantized KV cache |
| `kv_cache_quant_bits` | 8 | Bits per value (4-8) |
| `kv_cache_quant_group_size` | 64 | Quantization group size |
| `kv_cache_quant_min_tokens` | 256 | Minimum tokens before quantizing |

### Paged Cache

Block-based KV cache with dynamic allocation:

| Setting | Default | Description |
|---------|---------|-------------|
| `paged_cache_enabled` | `False` | Enable paged KV cache |
| `paged_cache_block_size` | 64 | Tokens per block |
| `paged_cache_max_blocks` | 1000 | Maximum blocks |

### Prefix Cache

Copy-on-write prefix sharing for common prompts:

| Setting | Default | Description |
|---------|---------|-------------|
| `prefix_cache_enabled` | `True` | Enable prefix cache |
| `prefix_cache_max_size` | 100 | Maximum cached prefixes |

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

```python
config = ServerConfig(
    cloud_router_enabled=True,
    cloud_router_api_key="sk-...",
    cloud_router_threshold=16384,  # Route to cloud at 16K+ tokens
)
```

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

## Per-Model Settings

Each model can have custom settings stored in `~/.fusion-mlx/settings/`:

```json
{
  "Qwen2.5-3B-Instruct-4bit": {
    "pinned": true,
    "ttl_seconds": 3600,
    "stream_interval": 1,
    "specprefill_enabled": false,
    "turboquant_kv_enabled": false
  }
}
```

| Setting | Description |
|---------|-------------|
| `pinned` | Prevent LRU eviction |
| `ttl_seconds` | Seconds before idle unload (0 = never) |
| `stream_interval` | Tokens between stream updates (1 = every token) |
| `specprefill_enabled` | Enable speculative prefill |
| `turboquant_kv_enabled` | Enable TurboQuant V-only KV compression |

## Server Config Summary

```python
ServerConfig(
    host="0.0.0.0",
    port=8000,
    model_dir="~/.fusion-mlx/models",
    memory=MemoryConfig(tier="balanced"),
    scheduler=SchedulerConfig(),
    model_aliases=DEFAULT_ALIASES,
    admin_enabled=True,
    cloud_router_enabled=False,
)
```
