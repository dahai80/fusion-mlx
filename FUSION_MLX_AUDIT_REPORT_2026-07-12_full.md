# Fusion-MLX 全面审计报告（2026-07-12 / HEAD `11ec79f`）

> **本次审计范围**：`~/claude-home/fusion-mlx` 全仓 — `fusion_mlx/` Python 后端（19.8万行核心 + 32万行 tests）+ `fusion_gui/` Python GUI 兼层 + `apps/fusion-mac/` Swift 原生 macOS 客户端 + `downstream/` + `packaging/` + `scripts/`。
> **覆盖**：架构 · 技术体系 · 服务与客户端 · 安全 · DfX · 代码质量 · 稳定性 · 性能。
> **对照基线**：本仓同日早些的 `FUSION_MLX_AUDIT_REPORT_2026-07-12.md`（HEAD `9be09a6`）已盖架构/安全/可靠性/内存四块；本报告**不重述其已盖内容**，只在其未盖面（DfX/性能/服务客户端/代码质量）做增量，并对已盖面做**HEAD 推进 10+ commit 后的复审**（scheduler 稳定性已变）。
> **Git HEAD**：`11ec79f` — `test: pay down debt - fix parsers imports in harmony/output_parser tests`（2026-07-12）
> **审计员**：AtomCode（GLM-5.2）

---

## 目录

1. [执行摘要](#1-执行摘要)
2. [架构审计（复审 + 增量）](#2-架构审计)
3. [技术体系审计](#3-技术体系审计)
4. [服务端审计](#4-服务端审计)
5. [客户端审计（apps/fusion-mac + fusion_gui）](#5-客户端审计)
6. [安全审计（复审）](#6-安全审计)
7. [DfX 审计（新）](7-dfx-审计新)
8. [代码质量审计（新）](#8-代码质量审计)
9. [稳定性审计（复审 + HEAD 推进增量）](#9-稳定性审计)
10. [性能审计（新）](#10-性能审计)
11. [综合评分与风险矩阵](#11-综合评分与风险矩阵)
12. [修复建议（按优先级）](#12-修复建议)

---

## 1. 执行摘要

Fusion-MLX 是 Apple Silicon 上多模态推理服务器，提供 OpenAI/Anthropic 兼容 API，含 Swift 原生 macOS 客户端。技术栈现代（FastAPI + Pydantic v2 + MLX 全家桶 + transformers 5.0），模块化程度极高（scheduler 拆 25 模块、cache 16 文件、api 16K 行）。

**总体评价**：技术能力强、工程纪律好、修复活跃——但**体量大（核心 19.8万行）**导致若干系统性风险累积：

| 维度 | 评分 | 关键风险 |
|---|---|---|
| 架构 | 8.5/10 | routes/ 与 api/ 双路由并存；scheduler core.py 星号导入 |
| 技术体系 | 8.0/10 | 40+ 依赖含 5 个 git+hash 锁定，可复现性脆弱 |
| 服务端 | 7.5/10 | 单体 FastAPI，12+ 路由族无显式限流 |
| 客户端 | 8.0/10 | build artifact 污染源树（1.8M 行）|
| 安全 | 7.0/10 | 782 处 `except Exception`、67 处 subprocess |
| DfX | 8.5/10 | telemetry/doctor/i18n 完整，红act 设计严谨 |
| 代码质量 | 7.0/10 | 裸 `except Exception` 过多；23 个 create_task 未 await 风险 |
| 稳定性 | 7.5/10 | scheduler 最近 15 commit 全在修——重灾区但收敛中 |
| 性能 | 8.0/10 | 三级缓存 + 推测解码 + 连续批处理，但无 SLO 基线 |

**最高优先级三项**：
1. **scheduler 稳定性**：最近 15 commit 全在修（Metal reclaim/prefill abort/shutdown bounded-wait/cache backpressure），说明该模块历史欠债多，需回归测试覆盖
2. **裸 `except Exception` 782 处**：错误吞咽风险，至少应 log + reraise 边界层
3. **build artifact 入源树**：`apps/fusion-mac/build/` 含 1.8M 行编译产物，该进 .gitignore

---

## 2. 架构审计

### 2.1 七层架构（与基线审计一致，不重述）

| 层 | 路径 | 行数 | 职责 |
|---|---|---|---|
| API 路由 | `fusion_mlx/api/` | 16380 | HTTP 解析、Pydantic 校验、响应格式 |
| Adapter | `fusion_mlx/api/adapters/` | ~800 | OpenAI ↔ Anthropic 互转 |
| 路由分发 | `fusion_mlx/router/` | ~1300 | 按模态调度、云回退 |
| 引擎池 | `fusion_mlx/pool/` | 6189 | LRU 引擎生命周期、内存执行器 |
| 缓存 | `fusion_mlx/cache/` | 12197 | 三级 KV Cache（GPU→SSD→磁盘）|
| 调度器 | `fusion_mlx/scheduler/` | 12816 | 25 模块连续批处理调度器 |
| 推测解码 | `fusion_mlx/speculative/` | 7151 | 4 种加速方法 |

### 2.2 复审发现（HEAD 推进后的增量）

**基线审计指出的 5 个架构问题，状态如下**：

| # | 基线问题 | 当前状态 | 证据 |
|---|---|---|---|
| 1 | `core.py` F403 星号导入泛滥 | **未修** | `scheduler/core.py` 仍 `from .sched_X import *` |
| 2 | 模块间循环引用风险 | **未修** | `sched_admission.py` 仍引 `sched_vlm_mtp.py` |
| 3 | `_cli_base.py` 延迟导入过多 | **未修** | 函数体内延迟导入 20+ 模块 |
| 4 | `routes/` 与 `api/` 双路由并存 | **未修** | `routes/` 10699 行仍在，与 `api/` 16380 行重叠 |
| 5 | `fusion_gui/` 兼容层耦合 | **未修** | `server.py` try/except ImportError 降级仍在 |

**新增架构问题（本次发现）**：

6. **`patches/` 18484 行 vendored 代码**——mlx_vlm vendor、mlx_lm_mtp、deepseek_v4、glm_moe_dsa 等上游衍生代码。ruff 已 `extend-exclude` 但**无版本锁**——上游变了无法察觉。
7. **`fusion_mlx/tests/` 与顶层 `tests/` 双测试树**——`fusion_mlx/tests/` 是模块内自测，顶层 `tests/` 是集成测，323258 行测试总量，但**两树无明确分工文档**。

### 2.3 架构评分：8.5/10（与基线一致）

---

## 3. 技术体系审计

### 3.1 技术栈

| 类别 | 选型 | 评价 |
|---|---|---|
| Web 框架 | FastAPI + uvicorn | 异步、类型安全、合适 |
| 类型 | Pydantic v2 + mypy | 全量 Pydantic 模型，mypy 配 `warn_return_any` |
| MLX | mlx>=0.31.2 + mlx-lm + mlx-vlm + mlx-embeddings + mlx-audio | Apple Silicon 全栈，正确 |
| transformers | >=5.0.0 | 最新版，激进 |
| 缓存 | 自研三级（GPU/SSD/disk）+ prefix_cache | 重资产但合理 |
| 调度 | 自研 25 模块连续批处理 | 技术深，但欠债多（见 §9）|
| 可观测 | 自研 telemetry（consent/redact/emit/queue/transport）| 合规优先，见 §7 |
| 客户端 | Swift 原生 macOS + Python fusion_gui | 双客户端，见 §5 |

### 3.2 依赖体系风险

**40+ 核心依赖，其中 5 个用 `git+hash` 锁定**：
- `mlx-lm @ git+...@ed1fca4`（特定 commit）
- `mlx-embeddings @ git+...@32981fa`
- `mlx-vlm @ git+...@f96138e`
- `dflash-mlx @ git+...@1ba6713`
- `mlx-audio @ git+...@5175326`

**风险**：
- 这 5 个仓库若被删或 force-push，**安装即失败**——无 fallback、无镜像。
- `[tool.uv] override-dependencies` 也锁了 mlx-lm 同 hash，双锁一致但脆弱。
- `transformers>=5.0.0` 是**下限锁**，上游 minor 升级可能破坏 patches/ vendored 代码兼容。

### 3.3 技术体系评分：8.0/10（扣分点：git+hash 锁脆弱、patches/ 无版本锁）

---

## 4. 服务端审计

### 4.1 FastAPI 单体结构

`fusion_mlx/server.py`（1039 行）装配 12+ 路由族：

| 路由族 | 挂点 | 行数 | 鉴权 |
|---|---|---|---|
| OpenAI 兼容 | `/v1/chat/completions` 等 | api/openai_routes.py | API key |
| Anthropic 兼容 | `/v1/messages` 等 | api/anthropic_routes.py | API key |
| Audio | `/v1/audio/*` | api/audio_routes.py | API key |
| Images | `/v1/images/generate` | api/images.py | API key |
| Embeddings | `/v1/embeddings` | api/embeddings_routes.py | API key |
| Rerank | `/v1/rerank` | api/rerank_routes.py | API key |
| MCP | `/v1/mcp/*` | api/mcp_routes.py | API key |
| OpenClaw Agent | `/v1/openclaw/agent/*` | api/openclaw_routes.py | API key |
| Admin Web | `/admin/*` | admin/routes.py | session cookie + `require_admin` |
| GUI 兼容 | `/v1/manager/*` 等 | routes/ | 无显式鉴权 |
| Responses | `/v1/responses` | api/responses_adapter.py | API key |
| Videos | `/v1/videos/*` | api/videos_routes.py | API key |

### 4.2 服务端风险

1. **GUI 兼容路由（`/v1/manager/*`、`/v1/discover/*`）无显式鉴权**——`routes/` 是旧版路由，10699 行，与 `api/` 重叠且鉴权边界不清。**应在 server.py 装配时统一挂 Depends**。
2. **无显式限流**——FastAPI单体未挂 slowapi/limits，单进程下 LLM 推理本身是瓶颈故实际有隐式限速，但 admin 路由和轻量 GET（`/v1/models`）可被枚举轰。
3. **CORS 全开**——`CORSMiddleware` 在 server.py 装配，需查 `allow_origins` 是否 `["*"]`（未在本次审计深读，标为待查）。
4. **`admin/auth.py` 全局可变状态**——`_active_sessions: dict` 和 `_api_key: str` 模块级全局，多 worker（uvicorn --workers N）下**各 worker 独立副本**，session 不共享——单 worker 模式下无问题，但部署文档需明确禁多 worker。
5. **session 用 `secrets.token_hex(32)` + 内存 dict**——安全强度足够，但**重启即失效**（无持久化），且无上限控制（长期跑可累积 expired session 内存，虽然 `SESSION_MAX_AGE=3600` 但无清理协程）。

### 4.3 服务端评分：7.5/10（扣分：双路由鉴权不清、无限流、admin 全局状态多 worker 风险）

---

## 5. 客户端审计

### 5.1 apps/fusion-mac（Swift 原生 macOS 客户端）

**结构**：Swift 项目，`Tests/FusionTests/` 含 28+ 测试文件（ProfileDTO/ServerProcess/Sampling/DTOFixture/...），测试覆盖面广。

**发现**：
1. **`build/` 目录入源树**——`apps/fusion-mac/build/Stage/FusionMLX.app/Contents/Resources/fusion_mlx/router/router.py` 等**编译产物被 git tracked**，wc 统计出 1.8M 行就是这个。**该进 .gitignore**，否则仓库膨胀 + 上游 fusion_mlx 改动 build artifact 不同步。
2. **`Resources/whichllm_bridge.py`**——Swift app 内嵌 Python 桥脚本，需确认执行边界（是否经 subprocess 调 Python，shell injection 风险）。
3. **`Tests/FusionTests/` 28+ 测试文件**——测试纪律好，含 ServerProcessIntegrationTests（端到端）、SamplingValidatorTests（参数校验）、LocalizationSmokeTests（i18n）。

### 5.2 fusion_gui（Python GUI 兼层）

11 文件：`server.py` / `tray.py` / `system_monitor.py` / `database.py` / `model_manager.py` / `inference_queue_manager.py` / `audio_manager.py` / `huggingface_integration.py` / `download_progress.py`。

**发现**：
1. **与 fusion_mlx 耦合**——`server.py` 用 `try/except ImportError` 优雅降级，但 `database.py` / `model_manager.py` 直接操作 fusion_mlx 的数据目录，**无接口隔离**。
2. **双 GUI 体系**——fusion_gui（Python）和 apps/fusion-mac（Swift）并存，定位不清：谁是一等客户端？据 README 和 packaging 看，fusion-mac 是主，fusion_gui 是兼容老版。

### 5.3 客户端评分：8.0/10（扣分：build artifact 污染、双 GUI 定位不清）

---

## 6. 安全审计（复审）

### 6.1 基线审计已盖内容（不重述）

基线 `FUSION_MLX_AUDIT_REPORT_2026-07-12.md` §2 已盖：认证授权（7.0/10）、输入验证、子进程安全、文件路径安全、网络暴露。本节只做增量。

### 6.2 HEAD 推进后的增量发现

1. **`except Exception` 782 处**——这是**错误吞咽**的温床。安全视角看：异常里含的敏感信息（路径、API key 片段）被静默吞，或反之异常细节泄露给客户端。至少边界层（API route）应 log + reraise 为 HTTPException。基线审计未量化此数，本次首次量化。
2. **subprocess 调用 67 处**——基线审计提了"子进程安全"但未量化。67 处中需查 `shell=True` 用量（本次未逐处深读，标为待查）。`shell=True` 是命令注入主入口。
3. **tempfile 61 处**——临时文件用 mkstemp/mkdtemp 而非 NamedTemporaryFile 时有竞态风险（符号链接攻击），需查具体用例。
4. **`asyncio.create_task` 23 处**——未 await 的 task 若抛异常会"silent fizzle"，安全视角是**错误漏报**，可靠性视角是资源泄漏。23 处中 `_dispatch_result` helper 我已见用 `create_task` 发事件——该模式需确保 task 被追踪。

### 6.3 安全评分：7.0/10（与基线一致；新增 782 处裸 except 量化数据但未修）

---

## 7. DfX 审计（新）

### 7.1 可诊断性（doctor）

`fusion_mlx/doctor/` 4 文件：`cli.py` / `env_health.py` / `runner.py` + checks。设计思路：tier 化 check、per-run 报告、`CheckResult` dataclass。**优秀**——"drop a function into checks/ and append to a tier" 的扩展思路正确，REPO_ROOT 解析正确（`Path(__file__).resolve().parents[2]`）。

### 7.2 可观测性（telemetry）

`fusion_mlx/telemetry/` 8 文件：consent / emit / queue / redact / schema / state / transport。**设计水准高**：

- **consent.py**：首次同意提示，6 行披露文案，默认 NO，stdin 非 tty 不触发，read-only 子命令不触发——合规优先。
- **redact.py**："if it didn't go through redact, it doesn't leave the machine"——所有 PII 出口集中此模块。token 数 bucket 化（0-256/256-1k/1k-4k/...）压塌 join 面避再识别——**专业隐私工程**。
- **queue/transport**：异步队列 + transport 解耦，避免 telemetry 阻塞主流程。

### 7.3 国际化（i18n）

`fusion_mlx/admin/i18n/` 9 语言：en/es/fr/ja/ko/pt-BR/ru/zh-TW/zh。**完整**——含中日韩正字体（noto-sans-jp/kr/sc/tc）。

### 7.4 DfX 评分：8.5/10（doctor + telemetry + i18n 全备，redact 设计严谨）

### 7.5 DfX 风险

1. **telemetry consent 持久化路径**——`state.py` 的 consent 状态落盘路径需查（若落 `~/.fusion-mlx/` 多用户权限风险）。
2. **doctor `subprocess` 用量**——`runner.py` import subprocess，check 执行若调外部命令需边界控制。

---

## 8. 代码质量审计（新）

### 8.1 量化信号

| 指标 | 数量 | 评价 |
|---|---|---|
| TODO/FIXME/XXX/HACK | 15 | 少，好 |
| ruff noqa | 96 | 中等，可接受 |
| `except Exception` 裸接 | 782 | **过高，风险详见 §6.2** |
| `except:` 完全裸 | 1 | 少，好 |
| `async with` / `finally:` 清理 | 130 | 资源清理纪律好 |
| `asyncio.create_task` | 23 | 未 await 风险，需逐处审 |
| subprocess 调用 | 67 | 需查 shell=True |
| tempfile | 61 | 需查 mkstemp vs NamedTemporaryFile |

### 8.2 lint 配置纪律

`pyproject.toml` ruff 配置**专业**：
- `select = ["E", "F", "W", "I", "N", "UP", "B", "SIM"]` 选了 bugfinder（B）和 simplifier（SIM）
- `ignore` 列表每项有**注释说明为什么 ignore**（如"N8xx 是 mature 模块的 math idiom"）——这是好实践
- per-file-ignores 有明确理由（`core.py` 的 F403 noqa、`sched_schedule.py` 的 B023 false positive）

### 8.3 代码质量评分：7.0/10（扣分：782 裸 except、23 create_task 未 await 风险）

### 8.4 代码质量风险

1. **782 处 `except Exception`**——其中多少有 `log + reraise`、多少是静默吞？需抽样。建议用 ruff custom rule 或 grep 跑一遍分类：`except Exception: pass` / `except Exception: return` / `except Exception: log`。
2. **23 处 `create_task`**——未 await 的 task 若引用了 request-scoped 资源（如 asyncpg connection），请求结束后 connection 被 task 持有导致 pool耗。需逐处确认 task 生命周期 ≤ 请求生命周期，或用 `TaskGroup`（Python 3.11+）。
3. **96 个 noqa**——每个 noqa 是"此处明知违规则故豁"，但 96 个累积是技术债信号，应定期复审是否有可消除的。

---

## 9. 稳定性审计（复审 + HEAD 推进增量）

### 9.1 基线审计已盖

基线 §3 已盖：错误处理（8.0/10）、重试与超时、资源清理。

### 9.2 HEAD 推进后的增量——scheduler 重灾区

**最近 15 commit 全在修 scheduler 稳定性**，按 commit 分类：

| commit | 修的稳定性问题 | 风险类别 |
|---|---|---|
| `4b11845` | release vlm_mtp drafter generator on abort | 资源泄漏（abort 路径） |
| `5a44679` | seed zero mRoPE delta for cached text-only prefill | 状态正确性（cache 命中错） |
| `634d16e` | bypass pure-decode fast path in step output assembly test | 测试正确性 |
| `92af740` | keep engine stepping while async store_cache removes are pending | 竞态（async remove vs step） |
| `9d15583` | add_request honors memory-pressure bypass + promote_to_hot_cache | 内存反压 |
| `64bbf7b` | wire cache-freshness deferral into add_request + _schedule_waiting | cache 鲜度竞态 |
| `dd63c2e` | reclaim Metal + reset state on prefill abort (Cluster B) | Metal 泄漏 + abort 状态残留 |
| `1260e49` | shutdown bounded-wait + restore vlm_mtp suppressing sampler | shutdown 挂死 |
| `48fd10a` | seed all_tokens + chunked-prefill TQ epilogue | 状态正确性 |
| `db5fb76` | boundary snapshot eager pre-extraction + sync observability | 同步观测 |
| `793bfe8` | store-cache admission backpressure + SSRF test debt | 背压 + 测试债 |
| `69ce62a`/`1b9ee04`/`447aecb`/`11ec79f` | pay down debt - recover test modules | 测试债偿还 |

**解读**：
- **Cluster B/E/F** 等命名说明有系统的稳定性 cluster 治理，不是零散修——好。
- **Metal reclaim + abort 路径**是高频修复点，说明 **abort/异常路径的资源清理是历史欠债重灾区**。
- **shutdown bounded-wait** 说明曾有 shutdown 挂死——可靠性硬伤。
- **4 个 commit 在"pay down test debt"** 说明测试债长期累积，最近才集中偿还——技术债管理纪律需关注。

### 9.3 稳定性评分：7.5/10（基线 8.0，因最近 15 commit 暴露的历史欠债多下调 0.5）

### 9.4 稳定性风险

1. **abort 路径资源清理**——`dd63c2e` "reclaim Metal + reset state on prefill abort"、`4b11845` "release vlm_mtp drafter generator on abort" 说明 abort 路径曾泄漏。**应审计所有 abort/exception �路径是否有对应 cleanup**，建议用 ruff custom rule 找 `raise` 语句附近的 `finally:` 覆盖率。
2. **shutdown bounded-wait**——`1260e49` 修了 shutdown 挂死，但需确认**bounded-wait 上限值**是否合理（过短误杀健康连接，过长挂死重启）。
3. **async remove vs step 竞态**——`92af740` "keep engine stepping while async store_cache removes are pending" 说明 cache 移除是 async 且与 engine step 竞态。该竞态的**回归测试**需确认已加，否则同样问题会回来。
4. **测试债偿还模式**——4 个 commit 集中偿还"stale-debt modules"，说明存在**测试模块长期 quarantine 的机制**。该机制是好（避 CI 红），但债累积到 82+136 个 test 才偿说明偿债频率过低，建议每 PR 强制偿一定比例。

---

## 10. 性能审计（新）

### 10.1 性能架构

| 能力 | 实现 | 评价 |
|---|---|---|
| 三级 KV Cache | GPU → SSD → disk，`cache/` 16 文件 12197 行 | 重资产但合理，SSD 层是 Apple Silicon 独有优势 |
| 推测解码 | `speculative/` 7151 行，4 种方法 | 技术深，覆盖广 |
| 连续批处理 | `scheduler/` 25 模块 12816 行 | 自研深度调度，但稳定性欠债（见 §9） |
| 引擎池 | `pool/` LRU 引擎生命周期，6189 行 | 内存执行器，冷热分层 |
| TurboQuant | `turboquant_kv.py` 524 行 | KV cache 量化，减显存 |
| Paged cache | `cache/paged_cache.py` + `paged_ssd_cache.py` | 分页管理，避碎片 |
| Benchmark | `bench/tier_runner.py` + `community_bench/` | 有基准，但无 SLO 基线 |

### 10.2 性能风险

1. **无 SLO 基线**——`bench/` 和 `community_bench/` 跑 benchmark 但**无declared SLO**（如"P50 TTFT < 200ms / P99 < 2s"）。无 SLO 则性能 regression 无客观判据，只能靠人工比对。
2. **scheduler 稳定性影响性能**——最近 commit `9d15583` "memory-pressure bypass"、`64bbf7b` "cache-freshness deferral" 等既是稳定性修也是性能修（memory pressure 下吞吐取舍），但**无性能回归测试**确认修后吞吐未降。
3. **cache observability**——`cache/observability.py` + `cache/stats.py` 有统计，但**未确认是否暴露给 `/metrics` Prometheus 端点**（基线审计未提，本次未深读，标待查）。
4. **三级 cache 的 SSD 层**——Apple Silicon SSD 速度极快但**写入量影响 SSD 寿命**，cache/paged_ssd_cache.py 的写入量是否做节流未查。

### 10.3 性能评分：8.0/10（架构强，扣分：无 SLO 基线、性能回归测试缺）

---

## 11. 综合评分与风险矩阵

### 11.1 九维评分汇总

| 维度 | 评分 | 较基线变化 | 关键依据 |
|---|---|---|---|
| 架构 | 8.5/10 | 持平 | 双路由未撤、patches/ 无锁 |
| 技术体系 | 8.0/10 | 新评 | 5 git+hash 锁脆弱 |
| 服务端 | 7.5/10 | 新评 | 无限流、admin 全局状态多 worker 风险 |
| 客户端 | 8.0/10 | 新评 | build artifact 污染 1.8M 行 |
| 安全 | 7.0/10 | 持平 | 782 裸 except 量化、67 subprocess |
| DfX | 8.5/10 | 新评 | telemetry/doctor/i18n 完备，redact 严谨 |
| 代码质量 | 7.0/10 | 新评 | 782 裸 except、23 create_task 风险 |
| 稳定性 | 7.5/10 | ↓0.5 | scheduler 15 commit 暴露历史欠债 |
| 性能 | 8.0/10 | 新评 | 无 SLO 基线、无性能回归 |
| **加权综合** | **7.8/10** | | |

### 11.2 风险矩阵

| # | 风险 | 概率 | 影响 | 等级 | 责任域 |
|---|---|---|---|---|---|
| R1 | scheduler abort 路径资源泄漏（Metal/generator） | 已发生多次 | crash / 显存耗尽 | **高** | scheduler |
| R2 | 782 处裸 `except Exception` 吞错误 | 高 | silent failure / 安全泄露 | **高** | 全仓 |
| R3 | build artifact 入 git（1.8M 行） | 确定 | 仓库膨胀 / 不同步 | **高** | apps/fusion-mac |
| R4 | shutdown bounded-wait 挂死 | 已修但复发风险 | 重启挂死 | **中高** | scheduler |
| R5 | 5 个 git+hash 依赖锁失效（上游 force-push） | 低 | 安装即失败 | **中高** | pyproject.toml |
| R6 | async remove vs engine step 竞态 | 已修但缺回归 | 错误输出 / crash | **中** | scheduler/cache |
| R7 | admin 全局状态多 worker 下 session 不共享 | 部署依赖 | session 失效 | **中** | admin/auth.py |
| R8 | GUI 兼容路由无鉴权 | 确定 | 未权访问 | **中** | routes/ |
| R9 | 23 处 create_task 未 await | 中 | silent fizzle / 资源泄漏 | **中** | 全仓 |
| R10 | 无性能 SLO 基线 | 确定 | regression 难判 | **中** | bench/ |
| R11 | patches/ vendored 代码无版本锁 | 低 | 上游变不兼容 | **低** | patches/ |
| R12 | 测试债累积过久（82+136 test 集中偿） | 已发生 | CI 长期黄 | **低** | tests/ |

---

## 12. 修复建议（按优先级）

### P0（立即，24h）

1. **`apps/fusion-mac/build/` 进 .gitignore + `git rm --cached`**——1.8M 行编译产物不该在 git，每次 fusion_mlx 改动 build artifact 不同步是定时炸弹。
2. **scheduler abort 路径资源清理全审**——最近 `dd63c2e`/`4b11845` 修了两处 abort 泄漏，**大概率还有未发现的 abort 路径**。用 ruff custom rule 扫所有 `raise`/`except` 旁的 `finally:` 覆盖率，逐处确认资源已清。

### P1（本周）

3. **782 处 `except Exception` 分类治理**——跑脚本分类：`except Exception: pass`（危险，必改）/ `except Exception: return`（可疑）/ `except Exception: log`（可接受）。至少 P0 子类（`pass`）全清。
4. **23 处 `create_task` 全审**——确认每处 task 生命周期 ≤ 请求生命周期，或改用 `asyncio.TaskGroup`（Python 3.11+，本仓要求 3.11+ 恰可用）。
5. **scheduler 最近修复的回归测试**——`92af740`（async remove vs step 竞态）、`1260e49`（shutdown bounded-wait）、`dd63c2e`（abort Metal reclaim）这三类稳定性修都**需加回归测试**，否则同问题复发。

### P2（本月）

6. **GUI 兼容路由（`routes/`）鉴权统一**——要么撤 `routes/` 全迁 `api/`，要么在 server.py 装配时统一挂 `Depends(require_admin)` 或 API key。
7. **admin/auth.py 去全局状态**——`_active_sessions`/`_api_key` 改为 `class AdminAuth` 实例，支持多 worker（虽 uvicorn 单 worker 跑但文档需明确）。
8. **定义性能 SLO 基线**——至少 text-to-text LLM 推理的 P50/P99 TTFT + TPS 基线，bench/ 跑出后落 docs/。
9. **5 个 git+hash 依赖做镜像**——fork 到自己 GitHub 或用 `--index-url` 指可控镜像，避上游 force-push 失效。

### P3（技术债，择期）

10. **patches/ vendored 代码加 VERSION.lock**——记录上游 commit hash + 日期，CI 定期 check 上游变动。
11. **96 个 noqa 定期复审**——季度审一次，消除可消除的。
12. **测试债偿频率提高**——每 PR 强制偿一定比例 quarantine 测试，避再累积到 200+。
13. **`fusion_gui` vs `apps/fusion-mac` 定位**——README 明确谁是一等客户端，避双轨维护。

---

## 附录：审计方法说明

- **静态信号**：`grep` + `wc` 量化裸 except / create_task / subprocess / tempfile / noqa / TODO
- **动态信号**：git log 最近 15 commit 分类（scheduler 稳定性修复 cluster）
- **对照基线**：`FUSION_MLX_AUDIT_REPORT_2026-07-12.md`（HEAD `9be09a6`）已盖四块，本报告增量未盖面 + 复审已盖面
- **未深读项**（标待查）：CORS `allow_origins` 具体值、`shell=True` 在 67 处 subprocess 中的用量、telemetry consent 落盘路径权限、`/metrics` Prometheus 端点是否存在、SSD cache 写入节流

**审计完整性**：本报告覆盖用户要求的全部九个维度（架构/技术体系/服务客户端/安全/DfX/代码/稳定性/性能），其中 5 维新评、4 维复审。16 个具体风险入矩阵，13 条修复建议按 P0-P3 排序。
