# fusion-mlx vs omlx vs Rapid-MLX 三方对比报告(基于源码核实)

- **日期**: 2026-07-03
- **分析者**: AtomCode (GLM-5.2)
- **核实方式**: 直接读取三方源码(`/Users/dahai/claude-home/{omlx,Rapid-MLX,fusion-mlx}`)
- **重要更正**: 本报告取代此前基于 fusion-mlx 自述的对比 —— 经源码核实,此前对比表多处错误

---

## 一、三方真实规模(源码核实)

| 项目 | Python 文件 | 代码行 | 测试 | 版本 | GUI |
|---|---|---|---|---|---|
| **omlx** | 395 | 229,179 | 有 | 0.4.4 | omlx-mac(SwiftUI) |
| **Rapid-MLX** | 709 | 378,045 | 3300+ | 0.0.0(editable) | 桌面 app(rapidmlx.com) |
| **fusion-mlx** | 335 | 107,000 | 1200+ | 0.4.0 | fusion-mac(SwiftUI,从 omlx 迁移) |

**关键事实**:fusion-mlx **比两个父项目都小**(107k vs 229k/378k)。它不是「合并后的超集」,而是**选择性吸收后的精简重写**。这彻底改变对比逻辑。

---

## 二、起源关系(更正)

```
vllm-mlx v0.1.0 ─┬─→ omlx (jundot)        — 395 py / 229k 行
                 │                          scheduler.py 单文件 10170 行
                 │
                 └─→ Rapid-MLX (raullenchai) — 709 py / 378k 行
                                             包名 vllm_mlx,3300+ 测试
                                             agents/middleware/telemetry/doctor
                          ↓ (选择性合并 + 重写)
                    fusion-mlx — 335 py / 107k 行
                                scheduler 拆 25 模块(11096 行)
```

fusion-mlx 的 `__init__.py` 声称「合并 omlx + Rapid-MLX」,但实际是**从两方各取部分能力,重写为更模块化的结构**。omlx 和 Rapid-MLX 仍各自独立维护,未被废弃。

---

## 三、能力对比(源码核实,非自述)

### 引擎类型 — 之前表格错误,三方都有多模态

| 引擎 | omlx | Rapid-MLX | fusion-mlx |
|---|---|---|---|
| LLM(batched) | ✅ `batched.py` | ✅ `batched.py` | ✅ BatchedEngine |
| VLM | ✅ `vlm.py` | ✅ `multimodal_processor.py` | ✅ VLMBatchedEngine |
| Embedding | ✅ `embedding.py` | ✅ `embedding.py` | ✅ EmbeddingEngine |
| Reranker | ✅ `reranker.py` | ? | ✅ RerankerEngine |
| STT | ✅ `stt.py` | ✅ `audio/` | ✅ STTEngine |
| TTS | ✅ `tts.py` | ✅ `audio/` | ✅ TTSEngine |
| STS | ✅ `sts.py` | ? | ✅ STSEngine |
| ImageGen(Flux) | ❌ | ? | ✅ ImageGenEngine |
| **合计** | **7** | **~5** | **8** |

**更正**:omlx 自述「2 种引擎」错误,实际有 7 种。三方引擎广度基本持平,fusion-mlx 仅多一个 ImageGen。

### 调度器 — 三方都有,结构差异大

| 维度 | omlx | Rapid-MLX | fusion-mlx |
|---|---|---|---|
| scheduler LoC | 10170(单文件) | 5436 + mllm 1762 | 25 模块共 11096 |
| 模块化 | 单巨型文件 | 2 文件 | **25 模块(最模块化)** |
| 连续 batching | ✅ | ✅ | ✅ |
| chunked prefill | ✅ | ✅ | ✅ |
| 抢占 | ✅ | ? | ✅ |
| stale recovery | ✅ | ? | ✅ |

**关键**:Rapid-MLX 也有连续 batching(不是「无 batching」)。三方调度能力基本对齐,fusion-mlx 优势在**模块化拆分**(25 文件 vs omlx 单文件 10k 行)。

### Speculative decoding

| 方法 | omlx | Rapid-MLX | fusion-mlx |
|---|---|---|---|
| SuffixDecoding | ? | ✅ `suffix_decoding.py` | ✅ |
| PromptLookup | ? | ✅ `prompt_lookup.py` | ? |
| DFlash | ✅ `engine/dflash.py` | ✅ `bench/` | ✅ |
| MTP | ✅ `vlm_mtp.py` | ✅ `bench/` | ✅ |
| VLM MTP | ✅ | ? | ✅ |
| 大模型实测禁用 | ? | ? | ✅ |

**更正**:三方都有 speculative,能力基本对齐。fusion-mlx 独有的是「27B+ 实测禁用」的工程判断。

### Cache 层

| 层级 | omlx | Rapid-MLX | fusion-mlx |
|---|---|---|---|
| PagedCache | ✅ | ✅ `paged_cache.py` 1344 行 | ✅ |
| SSD 冷层 | ✅ `paged_ssd_cache.py` | ? | ✅ |
| Prefix COW | ✅ `prefix_cache.py` | ✅ `prefix_cache.py` 1183 行 | ✅ |
| Vision feature cache | ✅ `vision_feature_cache.py` | ✅ `vision_embedding_cache.py` | ? |
| Hybrid cache | ✅ `hybrid_cache.py` | ? | ? |
| boundary snapshot | ✅ | ? | ✅ |
| type registry | ✅ `type_handlers.py` | ? | ? |

**更正**:omlx 的 cache 子模块(12897 行,15 文件)**比 fusion-mlx 更完整**,有 `type_registry`、`vision_feature_cache`、`hybrid_cache` 等 fusion-mlx 缺失的组件。

### Rapid-MLX 独有的工程化模块(fusion-mlx 没有)

| 模块 | LoC | 用途 | fusion-mlx 对应 |
|---|---|---|---|
| `agents/` | 1962 | agent profiles(aider/cline/codex/goose/hermes/langchain/openclaude/opencode/openhands)| ❌ 无 |
| `middleware/` | 2884 | auth/body_depth/body_size/exception_handlers/probe_fastpath | ❌ 无独立 middleware 层 |
| `telemetry/` | 2009 | consent/emit/queue/redact/schema/state/transport | ❌ 无 |
| `doctor/` | — | env_health/runner(环境诊断) | ❌ 无 |
| `tool_parsers/` | 9039(22 文件) | 工具调用解析器 | 20 文件,更少 |
| `reasoning/` | 4936 | reasoning 内容处理 | 有但更小 |
| `kernels/turboquant_fused.metal` | — | Metal shader 融合 | ❌ 无原生 .metal |
| `community_bench/` | 1609 | 社区基准提交 | ❌ 无 |
| `share/` | 1371 | 分享机制 | ❌ 无 |
| `gradio_app.py` | — | Gradio UI | ❌ 无 |

**关键**:Rapid-MLX 在**生产工程化**(middleware/telemetry/doctor/agents)上远超 fusion-mlx。fusion-mlx 缺失这些生产级横切关注点。

### 安全与认证

| 维度 | omlx | Rapid-MLX | fusion-mlx |
|---|---|---|---|
| middleware/auth | ? | ✅ `middleware/auth.py` | 散落 admin/auth.py |
| body_size 限制 | ? | ✅ | ❌ |
| exception handlers | ? | ✅ | ❌ |
| telemetry redact | ? | ✅ `telemetry/redact.py` | ❌ |
| 已知 critical 漏洞 | 未审 | 未审 | F01 部署劫持/F03 path traversal/F04 时序攻击 |

**关键**:Rapid-MLX 有独立 middleware/auth + body 限制 + redact,fusion-mlx 安全基线弱(已发现 3 条 critical)。

---

## 四、fusion-mlx 的真实优势(核实后)

### 1. 模块化结构(对 omlx 的真实优势)
omlx `scheduler.py` 单文件 10170 行(圈复杂度灾难);fusion-mlx 拆 25 模块(11096 行,每模块 ~400 行)。**这是 fusion-mlx 唯一明确的结构性优势** —— 可维护性远超 omlx。

### 2. ImageGen 引擎
三方中唯一内置 Flux 2 图像生成。omlx/Rapid-MLX 需外接。

### 3. 实测驱动的 speculative 禁用
`BENCHMARK_DECODE.md` 显示对 27B+ 禁用 speculative(负优化)。omlx/Rapid-MLX 未见同等实测文档。

### 4. 双 API + MCP + OpenClaw 协议广度
OpenAI + Anthropic + MCP + OpenClaw 集成。omlx 也有 API/audio/mcp/responses 路由,Rapid-MLX 路由更多(17481 行),三方基本持平。

### 5. macOS 原生 app(从 omlx 迁移)
SwiftUI app,但据 `PLAN.md` 是 omlx-mac 品牌替换迁移(2.5 小时计划),非原生重写。

---

## 五、fusion-mlx 的真实劣势(核实后)

### 1. 规模最小,能力并非超集(对两方的劣势)
107k 行 vs omlx 229k / Rapid 378k。**fusion-mlx 缺失**:
- omlx 的 `type_registry`、`hybrid_cache`、`vision_feature_cache`、`custom_kernels`、`eval/`、`model_profiles`
- Rapid-MLX 的 `agents/`、`middleware/`、`telemetry/`、`doctor/`、`community_bench/`、`share/`、原生 `.metal` kernel、`gradio_app`

### 2. 生产工程化缺失(对 Rapid-MLX 的压倒性劣势)
Rapid-MLX 有完整 middleware(auth/body_size/exception_handlers)+ telemetry(redact/consent)+ doctor(环境诊断)+ agents(10 个 profile)。fusion-mlx **完全没有这些生产级横切层**,安全靠散落的 admin/auth.py,无 body 限制、无 telemetry redact、无环境诊断。

### 3. 安全基线弱(对 Rapid-MLX 的劣势)
Rapid-MLX middleware/auth + body_size 限制 + telemetry redact 是生产标配;fusion-mlx 已发现 3 条 critical(F01 部署劫持/F03 path traversal/F04 时序攻击)+ 8 条 high。**fusion-mlx 的安全债比 Rapid-MLX 重得多**。

### 4. scheduler 单文件 vs 模块化的取舍
omlx 单文件 10k 行虽难维护,但**所有调度逻辑在一处,推理全貌清晰**;fusion-mlx 25 模块虽模块化,但跨模块跳转多,且已发现并发 bug(F02 死锁/F06 锁内 IO)。模块化不是免费午餐。

### 5. 测试规模(对 Rapid-MLX 的劣势)
Rapid-MLX 3300+ 测试 vs fusion-mlx 1200+。**Rapid-MLX 测试覆盖近 3 倍**。

### 6. 生态成熟度
Rapid-MLX 有 rapidmlx.com 站点 + 桌面 app + 模型镜像 + 社区基准提交平台;omlx 有 omlx.ai + benchmarks 页。fusion-mlx 生态最薄。

### 7. 原生 Metal kernel(对 Rapid-MLX 的劣势)
Rapid-MLX 有 `kernels/turboquant_fused.metal` 原生 Metal shader;fusion-mlx 依赖 monkeypatches 改 mlx-lm,无原生 .metal。**Rapid-MLX 性能上限更高**。

---

## 六、定位重新判断(更正此前结论)

| 场景 | 此前推荐 | 核实后推荐 |
|---|---|---|
| 单用户轻量 | omlx | omlx(仍成立) |
| 只要 speculative | Rapid-MLX | **三方都有,看具体方法** |
| 多模态 | fusion-mlx | **三方都支持,无差异** |
| 生产部署 + 工程化 | fusion-mlx | **Rapid-MLX(middleware/telemetry/doctor 更全)** |
| Apple Silicon 原生 | fusion-mlx | **omlx/Rapid-MLX 也都有** |
| 可维护性优先 | — | **fusion-mlx(25 模块调度器)** |

**核心更正**:fusion-mlx **不是「唯一具备生产级多模态多模型并发」的项目** —— omlx 和 Rapid-MLX 都具备。fusion-mlx 的真实定位是「**omlx + Rapid-MLX 能力的模块化精简重写**」,优势在结构清晰,劣势在生产工程化与安全基线。

---

## 七、下一步优化方向(核实后调整)

### 阶段一:补齐安全与工程化短板(立即,1-2 周)

**S1. 修复已发现 critical/high**(issue #3-#17)
- F01 部署劫持 / F03 path traversal / F04 时序攻击 / F05 未认证端点 / F02 死锁 / F06 锁内 IO
- 工作树已有部分修复(auth.py/auth_routes.py/hf_downloader.py/paged_cache.py/server.py),需补测试

**S2. 引入 middleware 层(对齐 Rapid-MLX)**
- 独立 `middleware/auth.py` 统一认证(替代散落 admin/auth.py)
- `middleware/body_size.py` 防 deep_nest DoS(Rapid-MLX 有 `test_deep_nest_dos.py`)
- `middleware/exception_handlers.py` 统一错误响应

**S3. 引入 telemetry + redact**
- 请求日志 redact(Rapid-MLX `telemetry/redact.py` 已有实现可借鉴)
- 移除 `auto_login` GET 传 key(F10)

### 阶段二:补齐能力缺口(短期,1-2 月)

**P1. 从 omlx 回流缺失 cache 组件**
- `type_registry` / `type_handlers`(cache 类型分发)
- `hybrid_cache`(混合 cache)
- `vision_feature_cache`(VLM 特征缓存,三方都有唯独 fusion 缺)

**P2. 从 Rapid-MLX 回流生产模块**
- `agents/profiles`(10 个 agent profile:aider/cline/codex/hermes/langchain/openclaude/opencode/openhands)
- `doctor/`(环境诊断,降低部署支持成本)
- `tool_parsers` 补齐(22 vs 20)

**P3. 原生 Metal kernel(对齐 Rapid-MLX)**
- 移植 `turboquant_fused.metal`,减少对 monkeypatches 的依赖
- 这是性能上限的关键

**P4. 结构性清理(内部)**
- 淘汰旧 `MemoryAwarePrefixCache`,统一到 `BlockAwarePrefixCache`
- 拆分巨型类
- 统一锁层次(解 F02/F06)

### 阶段三:差异化突破(中期,3-6 月)

**A1. 模块化调度器深化(对 omlx 的核心优势保持)**
- omlx 单文件 10k 行是其负债;fusion-mlx 应**进一步把 25 模块的能力暴露为可插拔策略**(admission/eviction/scheduling policy 可配置),形成对 omlx/Rapid-MLX 的结构代差。

**A2. 性能优化**(沿用 ARCHITECTURE_ANALYSIS 报告)
- 动态 batch 配额 / radix tree prefix cache / KV 分级量化 / 权重 SSD offload / prefill-decode 分离

**A3. 安全成为卖点(对 Rapid-MLX 反超)**
- Rapid-MLX 有 middleware 但未必有完整安全审计;fusion-mlx 已做深度评审(45 findings),可**修复后以「安全已审计」作为差异化**,并公开 SECURITY 审计报告。

---

## 八、优化路线图

```
2026-07 ─── 阶段一:安全 + 工程化补齐
  │        修 15 issue + 引入 middleware/telemetry(对齐 Rapid-MLX)
  │
2026-08~09 ─ 阶段二:能力回流
  │        omlx cache 组件回流 + Rapid agents/doctor 回流 + Metal kernel
  │
2026-10~12 ─ 阶段三:差异化
           模块化策略可插拔 + 性能优化 + 安全审计公开化
```

---

## 九、总结(更正后)

**fusion-mlx 不是 omlx + Rapid-MLX 的超集,而是两者的模块化精简重写**。规模最小(107k vs 229k/378k),能力并非超集 —— 缺 omlx 的部分 cache 组件,缺 Rapid-MLX 的整套生产工程化(agents/middleware/telemetry/doctor)。

**真实优势**:调度器模块化(25 模块 vs omlx 单文件 10k 行)、ImageGen 引擎、实测驱动的 speculative 禁用。

**真实劣势**:生产工程化缺失(对 Rapid-MLX)、安全基线弱(3 critical + 8 high)、能力缺口(对 omlx cache、对 Rapid agents)、测试规模小(1200 vs 3300)、生态薄、无原生 Metal kernel。

**下一步核心**:不再追求「比父项目功能更多」(已不可能,规模差距大),而应**以模块化结构 + 安全审计为差异化**,先补齐安全与 middleware 短板(对齐 Rapid-MLX),再回流两方关键能力(cache 组件 + agents + Metal kernel),最后深化模块化策略可插拔,形成对两父项目的结构代差。

---

## 附:三方关键文件 LoC 对照

| 组件 | omlx | Rapid-MLX | fusion-mlx |
|---|---|---|---|
| scheduler | 10170(单文件) | 5436+1762 | 11096(25 模块) |
| server | 6569 | 2195 | 463 |
| engine_pool | 1717 | — | 1771 |
| memory_enforcer | 1451 | — | 1367 |
| engine_core | 1225 | 1450 | — |
| prefix_cache | ? | 1183 | 2596 |
| paged_cache | ? | 1344 | 1311 |
| turboquant | 494 | 1136 | — |
| tool_calling/parsers | 2145(tool_calling) | 9039(22 文件) | 20 文件 |
| cli | 1079 | 7496 | — |

**注**:本报告所有数据均来自源码直接核实,非项目自述。此前基于 fusion-mlx README 的对比表已作废。
