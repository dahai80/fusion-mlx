# 架构详解

[English](architecture.md) | 中文

fusion-mlx 是基于 Apple MLX 构建的多模态推理服务。它通过 OpenAI 和 Anthropic 兼容的 API 提供 LLM、VLM、音频和图像生成模型的推理服务。

## 高层架构

```
┌──────────────────────────────────────────────────────────────────────┐
│                      FastAPI Server (uvicorn)                         │
│   ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐           │
│   │ OpenAI    │  │ Anthropic │  │  Audio   │  │   Images │           │
│   │ Routes    │  │  Routes   │  │  Routes  │  │   Routes │           │
│   └────┬─────┘  └────┬─────┘  └────┬─────┘  └─────┬────┘           │
│        │              │              │              │                  │
│   ┌────▼──────────────▼──────────────▼──────────────▼─────────────┐  │
│   │         RequestRouter / SmartRouter (调度)                      │  │
│   │  - 按模态路由 (text/image/audio/gen)                            │  │
│   │  - 阶段感知分流 (prefill → decode 在不同后端)                   │  │
│   │  - 优先级调度 (REALTIME/BATCH/BACKGROUND)                       │  │
│   │  - 大上下文云回退                                              │  │
│   └──────────────────────┬────────────────────────────────────────┘  │
│                          │                                            │
│   ┌─────────────────────▼──────────────────────────────────────────┐  │
│   │                 EnginePool (LRU + 内存管理)                      │  │
│   │   ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐        │  │
│   │   │ Batched   │ │   VLM    │ │  Embed   │ │  Audio   │        │  │
│   │   │ Engine    │ │  Engine  │ │  Engine  │ │  Engine  │        │  │
│   │   └────┬─────┘ └────┬─────┘ └────┬─────┘ └────┬─────┘        │  │
│   └────────┼─────────────┼────────────┼────────────┼──────────────┘  │
│            │             │            │            │                    │
│   ┌───────▼─────────────▼────────────▼────────────▼────────────────┐  │
│   │         Scheduler (25 模块, continuous batching)                │  │
│   │   - 等待队列   - 运行集合   - 抢占式调度                        │  │
│   │   - Chunked prefill   - TurboQuant KV   - Fused sampler        │  │
│   │   - Output Collector   - Stale request 恢复                    │  │
│   └─────────────────────┬──────────────────────────────────────────┘  │
│                          │                                             │
│   ┌─────────────────────▼──────────────────────────────────────────┐  │
│   │         类型化执行器线程池 (线程隔离)                            │  │
│   │   ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐            │  │
│   │   │  LLM    │ │  Image  │ │  Audio  │ │   IO    │            │  │
│   │   │ (1 wrk) │ │ (1 wrk) │ │ (2 wrk) │ │ (2 wrk) │            │  │
│   │   └─────────┘ └─────────┘ └─────────┘ └─────────┘            │  │
│   └──────────────────────────────────────────────────────────────────┘  │
│                          │                                              │
│   ┌─────────────────────▼──────────────────────────────────────────┐  │
│   │              MLX Thread (Metal 内核)                             │  │
│   │   - BatchGenerator   - Forward pass   - Fused sampler           │  │
│   └──────────────────────────────────────────────────────────────────┘  │
│                          │                                              │
│   ┌─────────────────────▼──────────────────────────────────────────┐  │
│   │         ProcessMemoryEnforcer (无死锁)                           │  │
│   │   - 超时锁获取 (2s)                                             │  │
│   │   - 先标记再执行的驱逐回退策略                                    │  │
│   │   - mx.clear_cache() 前后双重 gc.collect()                      │  │
│   └──────────────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────────────────┘
```

## 请求流转

1. **API 路由** — 客户端发送请求到 `/v1/chat/completions` 或 `/v1/messages`
2. **适配器** — `OpenAIAdapter` 或 `AnthropicAdapter` 将请求标准化为 `InternalRequest`
3. **路由器** — `RequestRouter` 按模态分发，`SmartRouter` 决定 prefill/decode 后端
4. **EnginePool** — 按模型名查找或加载对应的引擎
5. **引擎** — `BatchedEngine` 创建带 `SamplingParams` 的 `Request`
6. **EngineCore** — 通过类型化执行器线程池将请求提交给 `Scheduler`
7. **Scheduler** — 管理等待队列、运行批次、KV cache 和 continuous batching
8. **MLX Thread** — 执行 `scheduler.step()` → `BatchGenerator` → 模型前向传播 → fused sampler
9. **Output Collector** — `RequestOutputCollector` 缓冲并合并 token，通过 `AsyncIterator` 流式返回

## 组件分层

### 1. API 层 (`fusion_mlx/api/`)

处理 HTTP 请求解析、验证和响应格式化。每种 API 风格有各自的路由器和适配器。

- **OpenAI 路由** — `/v1/chat/completions`、`/v1/completions`、`/v1/models`、`/v1/embeddings`
- **Anthropic 路由** — `/v1/messages`、`/v1/count_tokens`，支持流式 tool_use 块
- **Audio 路由** — `/v1/audio/transcriptions`、`/v1/audio/speech`、`/v1/audio/process`
- **Image 路由** — `/v1/images/generate` (Flux 2)
- **MCP 路由** — `/v1/mcp/tools`、`/v1/mcp/servers`、`/v1/mcp/execute`
- **OpenClaw Agent 协议** — 多轮会话，TTL (1小时)、上限 (1000)、LRU 驱逐
- **适配器** — 在 API 特定格式和内部表示之间转换
- **工具调用** — JSON Schema 验证、工具分发、输出解析、流式块

### 2. 引擎层 (`fusion_mlx/engines/`)

8 种引擎类型，每种针对特定模态优化：

| 引擎 | 模态 | 执行器线程池 | 核心特性 |
|------|------|-------------|---------|
| `BatchedEngine` | LLM 文本 | llm (1 worker) | Continuous batching、流式、工具调用、thinking 模式 |
| `VLMBatchedEngine` | 视觉 + 文本 | io (2 workers) | 图像/视频理解、MTP drafter、paged KV cache |
| `EmbeddingEngine` | 文本 → 向量 | llm (1 worker) | 批量 embedding 生成 |
| `RerankerEngine` | 文段排序 | llm (1 worker) | Cohere/Jina 兼容重排序 |
| `STTEngine` | 音频 → 文本 | audio (2 workers) | Whisper、VibeVoice-ASR |
| `TTSEngine` | 文本 → 音频 | audio (2 workers) | Kokoro TTS、语音克隆、流式 WAV |
| `STSEngine` | 音频 → 音频 | audio (2 workers) | 语音增强、声源分离 |
| `ImageGenEngine` | 文本 → 图像 | image (1 worker) | Flux 2 扩散模型 |

### 3. 池化层 (`fusion_mlx/pool/`)

管理模型生命周期、内存和并发：

- **EnginePool** — 中心模型注册表，LRU 驱逐策略
    - 从 HuggingFace 缓存目录自动发现模型
    - 模型类型到引擎类的映射 (LLM → BatchedEngine 等)
    - 锁定常用模型防止驱逐
    - 基于 TTL 的空闲模型过期
    - 双重 `gc.collect()` 模式（`mx.clear_cache()` 前后各一次）

- **ProcessMemoryEnforcer** — 4 级内存保护，无死锁：
    - **Safe** — 25% 系统内存供模型使用
    - **Balanced** — 50% 供模型使用（默认）
    - **Aggressive** — 75% 供模型使用
    - **Custom** — 用户指定字节数上限
    - 超时锁获取 (2s)，避免 Metal 分配期间阻塞
    - 锁被加载协程持有时，先标记再执行的驱逐策略

- **ModelDiscovery** — 扫描目录发现 MLX 格式模型，估算大小和类型

- **PriorityScheduler** — Metal command queue 按任务类型分配优先级：
    - REALTIME (Claude Code) — 最高优先级，最低延迟
    - BATCH (OpenClaw agents) — 面向吞吐量
    - BACKGROUND (embedding/reranking) — 最低优先级

### 4. 缓存层 (`fusion_mlx/cache/`)

KV 状态的三级缓存：

1. **PagedCache** — GPU 内存中的分块 KV cache
    - 固定大小块（默认 64 token）
    - 动态分配，LRU 驱逐
    - 默认最多 1000 个块

2. **PagedSSDCache** — SSD 冷数据层，存放被驱逐的块
    - GPU 内存满时将不活跃块溢出到 SSD
    - 默认 20 GB 容量
    - 需要时透明恢复

3. **BlockAwarePrefixCache** — 写时复制 (COW) 前缀共享
    - 并发请求间共享前缀
    - COW 语义 — 仅在修改时复制块
    - 减少公共 prompt 的重复计算

### 5. 调度器 (`fusion_mlx/scheduler/`)

分解为 25 个聚焦模块（每个约 400 行）：

| 模块 | 职责 |
|------|------|
| `config.py` | 调度器配置 |
| `core.py` | 核心调度循环和状态管理 |
| `types.py` | 请求/响应类型定义 |
| `sched_admission.py` | 内存压力下的请求准入控制 |
| `sched_batch.py` | 批次构建和管理 |
| `sched_boundary.py` | 边界条件处理 |
| `sched_cache.py` | 缓存感知的调度决策 |
| `sched_handoff.py` | 阶段交接 (prefill → decode) |
| `sched_init.py` | 调度器初始化 |
| `sched_misc.py` | 辅助调度操作 |
| `sched_query.py` | 查询调度和 GPU OOM 预检 |
| `sched_response.py` | 响应处理和输出收集 |
| `sched_schedule.py` | 主调度循环 (prefill、insert、decode) |
| `sched_specprefill.py` | Speculative prefill |
| `sched_step.py` | 步进执行，含 stale request 恢复 |
| `sched_thinking.py` | Thinking/reasoning token 调度 |
| `sched_token.py` | Token 级调度和边界处理 |
| `sched_trim.py` | 长对话上下文裁剪 |
| `sched_vlm_mtp.py` | VLM 多 token 预测 |
| `sched_vlm_mtp_batched.py` | 批量 VLM MTP (~14 → ~27 tok/s 每请求) |
| `compiled_kv_cache.py` | 编译的 KV cache 操作 |
| `monkeypatches.py` | MLX 兼容性运行时补丁 |
| `sampler_fast_path.py` | Fused sampler — 跳过 logsumexp，批量采样 |
| `helpers.py` | 共享工具函数 |

**核心调度流程：**

- **Continuous batching** — 多个请求共享一个 GPU 步骤，并发负载下聚合吞吐量提升 2×
- **Chunked prefill** — 512 token 分块，避免内存尖峰，允许抢占
- **Stale request 恢复** — prefill+insert 后的首个 decode 步骤可能返回空响应；调度器检测并正确恢复，不丢失 token
- **TurboQuant KV** — 4-bit KV cache 量化，KV 读取的内存流量降低约 4×
- **Fused sampler** — 不需要时跳过 logsumexp，消除 `.item()` GPU 同步调用，自动检测并应用批量采样

### 6. Speculative Decoding (`fusion_mlx/speculative/`)

4 种加速 token 生成的方法：

| 方法 | 原理 | 加速比 |
|------|------|--------|
| SuffixDecoding | 复用之前生成的后缀模式 | 1.5-2× |
| DFlash | 块级扩散 — 成组起草 token | 2-3× |
| MTP | 多 Token 预测 — Qwen3.5/3.6、DeepSeek 原生支持 | 2-5× |
| VLM MTP | VLM 模型的外部辅助 drafter | 1.5-2× |

### 7. 路由器 (`fusion_mlx/router/`)

三层路由，按顺序应用：

- **RequestRouter** — 按模态将请求路由到正确的引擎：
    - 纯文本 → `BatchedEngine`
    - 文本 + 图像/视频 → `VLMBatchedEngine`
    - Embedding 请求 → `EmbeddingEngine`
    - 音频 → `STTEngine` / `TTSEngine` / `STSEngine`
    - 图像生成 → `ImageGenEngine`
    - 大量未缓存上下文 → `CloudRouter`

- **SmartRouter** — 阶段感知路由，支持跨引擎交接：
    - Prefill 在 omlx 上运行（强 matmul 性能），decode 在 Rapid-MLX 上运行（轻量 KV 操作）
    - 基于基准测试的后端选择，EMA 平滑 (alpha=0.7)
    - REALTIME 任务跳过基准路由，避免高延迟后端
    - 阶段分流阈值：8192 未缓存 token 且缓存命中率 <50%
    - 32768 未缓存 token 时云回退

- **CloudRouter** — 可选的云服务回退，通过 litellm 实现：
    - 熔断器防止本地/云震荡（5 次连续失败 → 断开）
    - 同时支持流式和非流式云调用
    - 支持自定义 API base/key，兼容 OpenAI 协议的提供商

### 8. 集成 (`fusion_mlx/integrations/`)

8 个预构建的 AI 开发工具连接器：

| 集成 | 功能 |
|------|------|
| Claude Code | 设置 `ANTHROPIC_BASE_URL` 和 `ANTHROPIC_AUTH_TOKEN` 用于本地代理 |
| OpenClaw | 写入 `~/.openclaw/config.yaml` 配置本地服务地址 |
| GitHub Copilot | Copilot 兼容代理 |
| OpenAI Codex | Codex CLI 集成 |
| ComfyUI | Flux 2 的 ComfyUI 节点服务 |
| OpenCode | OpenCode 集成 |
| Pi | Pi 集成 |
| Hermes | Hermes 工具解析器 |

## 线程模型

```
主线程 (asyncio)           类型化执行器线程池           MLX Thread
┌─────────────────────┐       ┌──────────────────┐       ┌──────────────────────┐
│ FastAPI 请求处理      │       │ LLM pool (1 wrk) │       │ scheduler.step()       │
│   ├─ 解析请求         │──────>│   ├─ mx.array()  │──────>│   ├─ BatchGenerator   │
│   ├─ 创建 Request    │       │   ├─ mx.eval()   │       │   ├─ model forward()  │
│   ├─ 加入队列         │       │ Image pool (1 wrk)│       │   ├─ fused sampler    │
│   ├─ 等待队列         │<──────│ Audio pool (2 wrk)│       │   └─ return Output   │
│   └─ yield token     │       │ IO pool (2 wrk)   │       └──────────────────────┘
└─────────────────────┘       └──────────────────┘
```

- ML 操作运行在专用类型化执行器线程池 (llm/image/audio/io) 上
- IO 线程池 (2 workers) 处理模型加载，避免阻塞推理
- Audio 线程池 (2 workers) 允许并发 STT + TTS
- Token 生成通过 `asyncio.Queue` 经 `RequestOutputCollector` 流回
- 所有 `run_in_executor` 调用通过 `asyncio.wait_for()` 设置超时保护

## 输出流水线

```
BatchGenerator._next()
    → gen_responses (每个请求的 token 数组)
    → _process_batch_responses()
        → RequestOutput (new_text, output_text, finished, finish_reason)
    → RequestOutputCollector._merge_outputs()
        → 拼接 new_text，合并累积 output_text
    → EngineCore._engine_loop()
        → 通过 ctx.collector.put() 分发到各请求的 collector
    → BatchedEngine.generate()
        → clean_special_tokens(output_text)
        → extract_thinking() 分离推理内容和常规内容
    → API 适配器格式化响应 (OpenAI 或 Anthropic)
```

关键行为：

- **Stale 恢复**：prefill+insert 后，首次 decode 可能返回空响应。调度器检测此情况（空响应 + 刚调度）并跳过 stale 重调度，避免 token 丢失。
- **Thinking 提取**：`extract_thinking()` 将 `mentare...` 标签拆分为 `reasoning_content` 和常规 `content`，同时支持 OpenAI 和 Anthropic API。
- **流式反 token 化**：通过流式反 token 化器增量解码 token，避免每步全量重新解码。

## 内存管理

```
系统 RAM (例如 128 GB)
├── 64 GB — OS / 其他应用 (Balanced 级: 50%)
└── 64 GB — fusion-mlx 预算
     ├── 模型权重 (GPU)
     ├── KV cache (PagedCache → PagedSSDCache → 磁盘)
     ├── TurboQuant KV (4-bit 压缩, 内存流量降低约 4×)
     └── Prefix cache (COW 共享块)
```

`ProcessMemoryEnforcer` 实时监控进程内存。当内存超出预算时，依次触发：

1. **软警告** — 记录警告日志，发出准入暂停信号
2. **缓存驱逐** — 将最近最少使用的 KV cache 块驱逐到 SSD
3. **请求抢占** — 换出低优先级请求
4. **请求中止** — 内存严重不足时中止进行中的请求

**GPU OOM 预检**：在调度 prefill 之前，调度器估算所需内存（模型权重 + KV cache + 激活张量），若超出可用 Metal 内存则拒绝准入。这防止了 Metal GPU OOM 崩溃。

**死锁预防**：内存守护器获取池锁时使用 2 秒超时。如果锁被加载协程持有（在 Metal 分配期间阻塞），守护器通过 `abort_loading=True` 标记模型待驱逐，而非等待。

**GC 策略**：每次 `mx.clear_cache()` 前后双重 `gc.collect()`：
- 第一次 `gc.collect()` 在 `clear_cache()` **之前** — 释放 C++ Metal buffer 包装器
- 第二次 `gc.collect()` 在 `clear_cache()` **之后** — 回收 Python 侧包装器对象
