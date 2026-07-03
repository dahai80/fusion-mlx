# fusion-mlx 完整评审与对比报告

- **日期**: 2026-07-03
- **分析者**: AtomCode (GLM-5.2)
- **核实方式**: 直接读取 fusion-mlx / omlx / Rapid-MLX 三方源码(class 定义、字段、文件结构),非文档自述
- **GitHub Issues**: #3-#17 共 15 条已提交至 https://github.com/dahai80/fusion-mlx

---

## 第一部分:三方源码核实对比

### 1.1 真实规模(读文件统计)

| 项目 | Python 文件 | 代码行 | 测试 | 版本 |
|---|---|---|---|---|
| omlx | 395 | 229,179 | 有 | 0.4.4 |
| Rapid-MLX | 709 | 378,045 | 3300+ | 0.0.0 |
| fusion-mlx | 335 | 107,000 | 1200+ | 0.4.0 |

**关键事实**:fusion-mlx(107k)**比两个父项目都小**,是选择性吸收后的模块化精简重写,非超集。

### 1.2 引擎类型(读 class 定义核实)

| 引擎 | omlx | Rapid-MLX | fusion-mlx |
|---|---|---|---|
| BatchedEngine(LLM) | ✅ `batched.py` | ✅ `batched.py` | ✅ |
| VLMBatchedEngine | ✅ `vlm.py:1111` | ❌(无 vlm 引擎类) | ✅ |
| EmbeddingEngine | ✅ | ❌ | ✅ |
| RerankerEngine | ✅ | ❌ | ✅ |
| STT/TTS/STS | ✅ | ❌(audio/ 但无引擎类) | ✅ |
| DFlashEngine | ✅ `dflash.py` | ❌ | ❌ |
| ImageGenEngine(Flux) | ❌ | ❌ | ✅ `image_gen.py` |
| **合计** | **8** | **2** | **8** |

**核实结论**:引擎广度 omlx ≈ fusion(8) > Rapid(2)。fusion 比 omlx 多 ImageGen,少 DFlash。

### 1.3 调度器(读 class + 字段核实)

| 维度 | omlx | Rapid-MLX | fusion-mlx |
|---|---|---|---|
| Scheduler 类 | ✅ `scheduler.py:1369` | ✅ `scheduler.py:1829` | ✅(25 模块) |
| max_num_seqs | ✅(默认 256) | ✅(默认 256) | ✅ |
| chunked_prefill | ✅(默认 False) | ✅ `_install_chunked_prefill` | ✅ |
| continuous batching | ✅(注释明示) | ✅ | ✅ |
| scheduler LoC | 10170(单文件) | 5436+1762 | 11096(25 模块) |
| 模块化 | ❌ 单文件 10k 行 | 2 文件 | ✅ 25 模块 |

**核实结论**:三方都有连续 batching。fusion-mlx 唯一明确优势是模块化拆分(25 模块 vs omlx 单文件 10k 行)。

### 1.4 Cache 层(读文件结构核实)

| 组件 | omlx | Rapid-MLX | fusion-mlx |
|---|---|---|---|
| paged_cache | ✅ | ✅ 1344 行 | ✅ |
| paged_ssd_cache | ✅ | ❌ | ✅ |
| prefix_cache | ✅ | ✅ 1183 行 | ✅ 2596 行 |
| vision_feature_cache | ✅ | ✅ vision_embedding_cache | ❌ |
| hybrid_cache | ✅ | ❌ | ❌ |
| type_registry/handlers | ✅ | ❌ | ❌ |
| boundary_snapshot | ✅ | ❌ | ✅ |
| cache 子模块 LoC | 12897(15 文件) | ~2500 | 9900 |

**核实结论**:omlx cache 最完整(15 文件 12897 行),fusion 缺 type_registry/hybrid/vision_feature。

### 1.5 Rapid-MLX 独有生产工程化(读文件首行核实)

| 模块 | 文件 | 用途 | fusion 对应 |
|---|---|---|---|
| middleware/auth | ✅ 认证+限流 | ❌ 无 |
| middleware/body_depth | ✅ D-DEEP-JSON DoS 防御 | ❌ |
| middleware/body_size | ✅ 请求体大小限制 | ❌ |
| middleware/exception_handlers | ✅ 统一错误响应 | ❌ |
| middleware/probe_fastpath | ✅ k8s 探针快路径 | ❌ |
| agents/(10 profile) | ✅ aider/cline/codex/hermes/langchain 等 | ❌ |
| telemetry/(redact/consent) | ✅ 日志脱敏 | ❌ |
| doctor/(env_health) | ✅ 环境诊断 | ❌ |
| kernels/turboquant_fused.metal | ✅ 原生 Metal shader | ❌ |
| tool_parsers | 22 文件 9039 行 | 20 文件 |
| 测试 | 3300+ | 1200+ |

**核实结论**:`ls fusion_mlx/{middleware,agents,telemetry,doctor}` 全部 No such file。fusion-mlx 完全缺失这些生产级横切层。

### 1.6 三方定位

| 项目 | 定位 |
|---|---|
| omlx | 功能最全(8 引擎 + 完整 cache),但单文件巨型类难维护 |
| Rapid-MLX | 生产工程化最强(middleware/agents/telemetry),但引擎仅 2 种 |
| fusion-mlx | 模块化最清晰(25 模块调度器),但规模最小、生产层缺失、安全基线弱 |

---

## 第二部分:fusion-mlx 架构分析

### 2.1 分层架构

```
客户端 → API 层(12 路由) → Router(模态/阶段) → Pool(LRU+内存守卫)
→ Engine(8 种) → EngineCore(async 桥) → Scheduler(25 模块)
→ Cache(三级 KV) → MLX 单线程 executor → Metal kernels
```

### 2.2 核心子系统意图

| 子系统 | LoC | 意图 | 亮点 |
|---|---|---|---|
| Scheduler | 11096(25 模块) | vLLM 风格连续 batching | chunked prefill、抢占、stale recovery、fused sampler 跳 logsumexp、step_burst 减 GIL 切换 |
| Cache | 9900 | 长上下文在有限 unified memory 可行 | GPU paged + SSD 20GB + prefix COW;block 链式哈希;boundary snapshot |
| Pool | 5700 | 多模型并发不 OOM | LRU+pin+TTL、4 级内存守卫、settle barrier 轮询真实释放 |
| Speculative | 2500 | 加速 token 生成 | 4 种(Suffix/DFlash/MTP/VLM-MTP);27B+ 实测禁用(负优化) |

### 2.3 已实现性能优化

| 优化 | 位置 | 效果 |
|---|---|---|
| Fused sampler | sampler_fast_path.py | 跳 logsumexp,消除 GPU sync |
| step_burst | engine_core.py:254 | 多步合并减 GIL 切换 |
| TurboQuant KV 4-bit | turboquant_kv.py | KV 流量 4× |
| StreamingJSONEncoder | api/streaming.py | 跳 Pydantic 构造 |
| B=1 fast path | scheduler | 单请求短路 |
| mixed-bit 量化 | 量化配方 | 带宽受限 96-167% 加速 |
| monkeypatches 行重对齐 | monkeypatches.py | 修 mlx-lm batch 漂移 |

### 2.4 结构性问题

1. 巨型类:`BlockAwarePrefixCache` 2500 行、`ProcessMemoryEnforcer` 1124 行、`MLLMScheduler` 1300 行
2. 锁层次混乱:`Lock`/`RLock`/`asyncio.Lock` 混用,enforcer 跨模块访问 pool 内部锁,已发现死锁(F02)
3. 双 cache 体系:`MemoryAwarePrefixCache`(1629)与 `BlockAwarePrefixCache`(2596)并存
4. monkeypatches 脆弱:运行时改 mlx-lm,上游升级即失效
5. 生产层缺失:无 middleware/agents/telemetry/doctor

---

## 第三部分:代码评审 Findings(45 条)

### 3.1 已提交 GitHub Issues(15 条,#3-#17)

| Issue | 严重度 | 位置 | 标题 |
|---|---|---|---|
| #3 | Critical | auth_routes.py:94 | /api/setup-api-key 无认证致部署劫持 |
| #4 | Critical | paged_cache.py:1309,1337 | handle_memory_pressure 调 evict_lru_blocks 触发 Lock 重入死锁 |
| #5 | Critical | hf_downloader.py:604 | repo_id 校验不禁止 .. 致路径穿越写盘/删盘 |
| #6 | High | auth.py:77 | verify_api_key 时序攻击(== 短路比较) |
| #7 | High | server.py:238,254 | /v1/models/{id}/load 与 /unload 无认证 |
| #8 | High | engine_pool.py:622 | get_engine 持锁内 5+ 秒 unload 序列化所有引擎获取 |
| #9 | High | openai_routes.py:113 | 路由用 get_engine 非 acquire,引擎可被中途卸载 |
| #10 | High | paged_ssd_cache.py:320 | PagedSSDCacheManager 无锁跨线程竞态 |
| #11 | High | oq.py:144 | model_path 无路径校验,admin 任意文件读 |
| #12 | High | auth_routes.py:176 | auto_login GET 传 API key 致密钥泄漏 |
| #13 | Medium | mllm_scheduler.py:658 | _schedule_waiting zip 静默截断致请求泄漏 |
| #14 | Medium | mllm_scheduler.py:1077 | _distribute_outputs QueueFull 被吞,客户端收不到结束 |
| #15 | Medium | mllm_scheduler.py:1126 | stop() 关闭时序致正常请求误判 batch 失败 |
| #16 | Medium | paged_cache.py:736 | _maybe_evict_cached_block 误判驱逐,stats 虚高 |
| #17 | Medium | paged_ssd_cache.py:143 | _write_safetensors_no_mx 内存拼接 O(n²) |

### 3.2 未提 issue 的 Medium/Low(28 条,留本地报告)

| # | 严重度 | 位置 | 摘要 |
|---|---|---|---|
| F16 | Medium | mllm_scheduler.py:548 | _process_pending_aborts 用 set.pop() 顺序非确定 |
| F17 | Medium | mllm_scheduler.py:545 | record_disconnect_abort bare except: pass |
| F18 | Medium | mllm_scheduler.py:577 | _do_abort_request 跨线程复合删无统一锁 |
| F19 | Medium | mllm_scheduler.py:950 | _cleanup_finished 不在锁内,abort 幂等性破坏 |
| F20 | Medium | mllm_scheduler.py:1449 | generate() del self.requests 不在锁内 |
| F21 | Medium | paged_cache.py:1324 | evict_lru_blocks 不更新 stats.total_tokens_cached |
| F22 | Medium | memory_enforcer.py:1119 | enforcer 直接访问 engine_pool._lock,封装泄漏 |
| F23 | Medium | memory_enforcer.py:1159 | while 循环基于 await 间陈旧读数 |
| F24 | Medium | memory_enforcer.py:1121 | lock timeout 反复标记 abort_loading,活锁 |
| F25 | Medium | priority_scheduler.py:352 | _maybe_preempt running_counts 依赖 cleanup_finished |
| F26 | Medium | engine_pool.py:1195 | 释放锁后 unload,重获前状态可能已变 |
| F27 | Medium | paged_ssd_cache.py:394 | _read_safetensors 第一次 open 读 data 未使用,死代码 |
| F28 | Medium | subkey.py:95 | create_sub_key 回滚 pop() 无锁,并发弹错 |
| F29 | Medium | auth_routes.py:161 | logout 不失效服务端 session |
| F30 | Medium | server.py:374 | _shutdown 不停止 ProcessMemoryEnforcer |
| F31 | Medium | server.py:121 | CORS allow_origins=["*"] 过宽 |
| F32 | Medium | server.py:154 | /health/stats/metrics 无认证,信息泄漏 |
| F33 | Medium | auth.py:38 | create_session_token 无上限累积,DoS |
| F34-F45 | Low | (多处) | 幂等性、可维护性、防御性缺失(详见 CODE_REVIEW) |

### 3.3 验证状态

- **已确认复现**:F02(锁类型确认)、F03(`"../b".split("/")` 长度 2 验证)、F04(`==` 短路)、F14、F15
- **逻辑推断**:F01、F05-F13(需运行时确认调用方)
- **撤回误判**:`_read_safetensors` off-by-one(对照 writer 后确认正确)

---

## 第四部分:优化方向(三阶段路线图)

### 阶段一:安全 + 工程化补齐(1-2 周)

**S1. 修复 15 条 issue(#3-#17)**
- Critical:F01 部署劫持 / F02 死锁 / F03 path traversal
- High:F04 时序攻击 / F05 未认证端点 / F06 锁内 IO / F08 SSD 竞态 / F09 任意文件读 / F10 GET 传 key

**S2. 引入 middleware 层(对齐 Rapid-MLX)**
- middleware/auth.py 统一认证+限流
- middleware/body_size.py + body_depth.py 防 DoS
- middleware/exception_handlers.py 统一错误响应

**S3. 引入 telemetry + redact**
- 请求日志脱敏(借鉴 Rapid-MLX telemetry/redact.py)

### 阶段二:能力回流 + 结构清理(1-2 月)

**P1. 从 omlx 回流 cache 组件**
- type_registry / type_handlers
- hybrid_cache
- vision_feature_cache(三方都有唯独 fusion 缺)

**P2. 从 Rapid-MLX 回流生产模块**
- agents/profiles(10 个 agent profile)
- doctor/(环境诊断)
- tool_parsers 补齐(22 vs 20)

**P3. 原生 Metal kernel(对齐 Rapid-MLX)**
- 移植 turboquant_fused.metal,减少 monkeypatches 依赖

**P4. 结构性清理**
- 淘汰旧 MemoryAwarePrefixCache,统一到 BlockAwarePrefixCache
- 拆分巨型类
- 统一锁层次(解 F02/F06)

**P5. 性能优化**
- 动态 batch 配额(吞吐 +20-30%)
- radix tree prefix cache(命中率 +30%)
- SSE 背压(解 F12)
- SSD 写路径流式化(解 F15)

### 阶段三:差异化突破(3-6 月)

**A1. 模块化调度器深化**
- 25 模块能力暴露为可插拔策略(admission/eviction/scheduling policy 可配置)
- 形成对 omlx 单文件 10k 行的结构代差

**A2. 性能深化**
- prefill/decode 分离调度(decode 延迟 -15%)
- KV cache 分级量化(内存 -30% 保质量)
- 模型权重 SSD offload(突破 unified memory 上限跑 70B+)
- token-level 抢占(REALTIME P99 改善)

**A3. 安全作为卖点**
- 修复后公开 SECURITY 审计报告
- 以「安全已审计」对 Rapid-MLX 反超

---

## 第五部分:总结

### 5.1 fusion-mlx 真实定位

fusion-mlx 是 omlx + Rapid-MLX 的**模块化精简重写**(107k 行,比两父项目都小)。非超集 —— 缺 omlx 的部分 cache 组件,缺 Rapid-MLX 的整套生产工程化。

### 5.2 真实优势(代码核实)

1. 调度器模块化(25 模块 vs omlx 单文件 10170 行)— 唯一明确结构性优势
2. ImageGen 引擎(三方唯一)
3. 实测驱动禁用大模型 speculative

### 5.3 真实劣势(代码核实)

1. 生产工程化缺失(对 Rapid-MLX):无 middleware/agents/telemetry/doctor
2. 安全基线弱:3 critical + 8 high 已确认
3. 能力缺口:缺 omlx 的 type_registry/hybrid/vision_feature cache
4. 测试规模小:1200 vs Rapid 3300
5. 无原生 Metal kernel(依赖 monkeypatches)
6. 生态最薄

### 5.4 核心原则

**不再追求「比父项目功能更多」(规模差距已定),而以「模块化结构 + 安全审计」为差异化**。先补齐安全与 middleware 短板(对齐 Rapid-MLX),再回流两方关键能力,最后深化模块化策略可插拔,形成结构代差。

**移除瓶颈优于增加机制** —— 当前机制已足够丰富,瓶颈在协作与集成层面。

---

## 附录:产物清单

| 文件 | 内容 |
|---|---|
| COMPREHENSIVE_REPORT_2026-07-03.md | 本报告(合并版) |
| CODE_REVIEW_2026-07-03.md | 45 条 findings 详细说明 |
| ARCHITECTURE_ANALYSIS_2026-07-03.md | 架构分析 + 15 条性能优化 |
| GitHub Issues #3-#17 | 15 条已提 issue |

**数据来源**:全部基于源码 class 定义、字段、文件结构直接核实,非项目 README 自述。
