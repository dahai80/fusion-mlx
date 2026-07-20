# 配置指南

[English](configuration.md) | 中文

fusion-mlx 通过 CLI 参数和 `ServerConfig` 数据类进行配置。所有设置项定义在 `fusion_mlx/config.py` 中。

## 内存分级

控制系统 RAM 中可用于模型推理的内存比例：

| 分级 | 预留给 OS | 模型预算 | 适用场景 |
|------|-----------|---------|---------|
| `safe` | 75% | 25% | 共享工作站、后台服务 |
| `balanced` | 50% | 50% | **默认** — 专用推理场景 |
| `aggressive` | 25% | 75% | 最大化模型容量，专用机器 |
| `custom` | 自定义 | 自定义 | 通过 `--custom-limit-mb` 精确控制 |

```bash
# 使用 balanced（默认）
fusion-mlx serve

# 使用 aggressive — 最大化模型容量
fusion-mlx serve --memory-tier aggressive

# 自定义：限制为 16 GB
fusion-mlx serve --memory-tier custom --custom-limit-mb 16384
```

## 类型化执行器线程池

MLX 操作运行在专用线程池上，防止跨模态阻塞：

| 线程池 | Workers | 用途 |
|--------|---------|------|
| `llm` | 1 | LLM 推理、embedding、reranking — 单 worker 避免 Metal 设备冲突 |
| `image` | 1 | 图像生成 (Flux 2) — 与文本推理隔离 |
| `audio` | 2 | STT、TTS、STS — 并发音频处理 |
| `io` | 2 | 模型加载、文件 I/O — 非阻塞加载 |

所有 `run_in_executor` 调用都通过 `asyncio.wait_for()` 设置超时保护：
- 模型加载：120s
- 推理：30s (LLM)、60s (audio)、120s (image)
- 同步/清理：5s

## 调度器设置

`SchedulerConfig` 数据类控制批处理、缓存和解码行为：

### 并发

| 设置 | 默认值 | 说明 |
|------|--------|------|
| `max_num_seqs` | 256 | 最大并发序列数 |
| `max_num_batched_tokens` | 65536 | 每个批步的最大 token 数 |

### 批处理

| 设置 | 默认值 | 说明 |
|------|--------|------|
| `prefill_batch_size` | 8 | 每次预填充步最大启动序列数 |
| `completion_batch_size` | 32 | 解码批次中最大序列数 |
| `prefill_step_size` | 2048 | 每次预填充步处理的 token 数 |

### Chunked Prefill

将长 prompt 分割为更小的块，避免内存尖峰并允许抢占：

| 设置 | 默认值 | 说明 |
|------|--------|------|
| `chunked_prefill` | `True` | 启用分块预填充 |
| `chunked_prefill_tokens` | 512 | 每块 token 数（0 = 禁用） |
| `mid_prefill_save_interval` | 8192 | 每 N 个 token 保存缓存快照 |

512 token 的默认值平衡了预填充开销和 REALTIME 请求延迟。以 512 token 为例，4K 的 prompt 需要 ~8 个块，每块留出约 2ms 让高优先级请求交错执行。

### TurboQuant KV Cache

4-bit KV cache 量化，KV 读取内存流量降低约 4×：

| 设置 | 默认值 | 说明 |
|------|--------|------|
| `kv_cache_quant_enabled` | `True` | 启用量化 KV cache |
| `kv_cache_quant_bits` | 4 | 每个值的位数（4 或 8） |
| `kv_cache_quant_group_size` | 64 | 量化组大小 |
| `kv_cache_quant_min_tokens` | 256 | 启动量化前的最小 token 数 |

TurboQuant 默认启用，是 fusion-mlx 2× 并发吞吐量优势的关键因素。它将 V-only KV cache 压缩到 4-bit，质量损失极小。

### Paged Cache

分块式 KV cache，动态分配：

| 设置 | 默认值 | 说明 |
|------|--------|------|
| `paged_cache_enabled` | `True` | 启用分页 KV cache |
| `paged_cache_block_size` | 64 | 每块 token 数 |
| `paged_cache_max_blocks` | 1000 | 最大块数 |

### Prefix Cache

公共 prompt 的写时复制 (COW) 前缀共享：

| 设置 | 默认值 | 说明 |
|------|--------|------|
| `prefix_cache_enabled` | `True` | 启用前缀缓存 |
| `prefix_cache_max_size` | 100 | 最大缓存前缀数 |

## SmartRouter

阶段感知路由，基于基准测试的后端选择。通过 `RouterConfig` 配置：

| 设置 | 默认值 | 说明 |
|------|--------|------|
| `phase_split_threshold` | 8192 | 未缓存 token 超过此值时分流 prefill/decode |
| `cloud_fallback_threshold` | 32768 | 未缓存 token 超过此值时路由到云服务 |
| `enable_benchmark_routing` | `True` | 使用 EMA 平滑的基准测试选择后端 |
| `ema_alpha` | 0.7 | EMA 平滑因子（越高越重视历史数据） |
| `prefill_chunk_size` | 512 | 软抢占时每块 token 数 |
| `default_priority` | `BATCH` | 未提供 `task_tag` 时的默认任务优先级 |
| `warmup_batch_sizes` | `[1, 4, 8]` | 预编译计算图的批量大小 |

**优先级**（通过 `task_tag` 确定）：
- `REALTIME` — Claude Code、交互式工具。跳过基准路由以获取最低延迟。
- `BATCH` — OpenClaw agents、批处理。使用基准路由获取最高吞吐量。
- `BACKGROUND` — Embedding、reranking、离线任务。最低优先级，可被抢占。

**阶段分流示例**：一个 16K-token 的 prompt，缓存命中率 20%：
- Prefill 在 omlx 上运行（大批量下 matmul 性能强）
- Decode 在 Rapid-MLX 上运行（轻量 KV 操作）
- KV cache 通过 `PhaseHandoff` 零拷贝传输

## 模型别名

将简短名称映射到完整模型 ID：

```python
# config.py 中的默认别名
DEFAULT_ALIASES = {
     "claude-4.6-sonnet": "BeastCode/Qwen3.5-27B-Claude-4.6-Opus-Distilled-MLX-6bit",
     "claude-4.5-sonnet": "Qwen/Qwen3-32B-A3B-Think-2512-MLX",
     "gpt-4o": "Qwen/Qwen3-32B-A3B-Think-2512-MLX",
     "gpt-4.5": "BeastCode/Qwen3.5-27B-Claude-4.6-Opus-Distilled-MLX-6bit",
}
```

通过 `~/.fusion-mlx/aliases.json` 自定义别名：
```json
{
     "my-model": "Qwen/Qwen2.5-7B-Instruct-MLX"
}
```

## Cloud Router

自动将大上下文请求路由到云服务提供商：

| 设置 | 默认值 | 说明 |
|------|--------|------|
| `cloud_router_enabled` | `False` | 启用云回退 |
| `cloud_router_api_key` | `""` | 云服务 API key |
| `cloud_router_threshold` | 32768 | 触发云路由的 token 阈值 |
| `cloud_router_api_base` | `None` | 自定义 API base，兼容 OpenAI 协议的提供商 |

```python
config = ServerConfig(
    cloud_router_enabled=True,
    cloud_router_api_key="sk-...",
    cloud_router_threshold=16384,    # 16K+ token 时路由到云
)
```

**熔断器**：连续 5 次本地推理失败后，熔断器打开，所有请求路由到云。一次本地成功即关闭熔断器。

**流式支持**：超过阈值时，流式和非流式请求都会路由到云。Cloud Router 使用 litellm 实现提供商无关的调用。

## SSD Cache

将不活跃的 KV cache 块卸载到 SSD：

| 设置 | 默认值 | 说明 |
|------|--------|------|
| `ssd_cache_enabled` | `False` | 启用 SSD 冷数据层 |
| `ssd_cache_dir` | `~/.fusion-mlx/ssd-cache` | 缓存目录 |
| `ssd_cache_max_bytes` | 20 GB | 最大磁盘使用量 |

```bash
# 通过 CLI 启用
fusion-mlx serve --enable-ssd-cache
```

## OpenClaw 会话

Agent 会话存储配置：

| 设置 | 默认值 | 说明 |
|------|--------|------|
| `_SESSION_TTL_SECONDS` | 3600 (1小时) | 不活跃会话过期秒数 |
| `_SESSION_MAX_COUNT` | 1000 | 最大并发会话数（LRU 驱逐） |

达到上限时按 LRU 顺序驱逐会话。TTL 计时器在每轮对话、工具结果提交或会话 GET 时重置。

## 单模型设置

每个模型可以有自定义设置，存储在 `~/.fusion-mlx/settings/` 中：

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

| 设置 | 说明 |
|------|------|
| `pinned` | 防止 LRU 驱逐 |
| `ttl_seconds` | 空闲卸载前秒数（0 = 永不卸载） |
| `stream_interval` | 流式更新间隔 token 数（1 = 每个 token） |
| `specprefill_enabled` | 启用 speculative prefill |
| `turboquant_kv_enabled` | 启用 TurboQuant 4-bit V-only KV 压缩（默认: true） |
| `dflash_enabled` | 启用 DFlash speculative decoding |
| `mtp_enabled` | 启用原生 MTP（Qwen3.5/3.6、DeepSeek-V4） |
| `vlm_mtp_enabled` | 启用 VLM MTP（gemma4_assistant drafter） |

## 服务配置汇总

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
