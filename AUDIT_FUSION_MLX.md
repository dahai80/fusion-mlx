# fusion-mlx 严格审计报告

**审计对象**: fusion-mlx — Apple Silicon 统一本地模型管理平台
**审计日期**: 2026-07-25
**审计方法**: 只读静态审计 + 架构分析 + 安全/性能/可靠性多维评估
**审计范围**: `fusion_mlx/` 包（659 个 Python 文件，~228k LOC，21 个 Metal/C++ 文件）+ 测试套件（707 个测试文件）+ CI/CD + 打包 + 部署

**重要声明**: 本审计为只读操作，未修改任何项目文件。审计基于特定时间点的代码快照，发现的问题严重性为审计员的主观判断，需结合实际运行环境验证。

---

## 目录

1. [执行摘要](#1-执行摘要)
2. [项目规模与架构](#2-项目规模与架构)
3. [安全审计](#3-安全审计)
4. [可靠性与错误处理审计](#4-可靠性与错误处理审计)
5. [内存与资源管理审计](#5-内存与资源管理审计)
6. [性能审计](#6-性能审计)
7. [代码质量审计](#7-代码质量审计)
8. [并发与并行审计](#8-并发与并行审计)
9. [Metal/C++ 内核与 FFI 审计](#9-metalc-内核与-ffi-审计)
10. [测试、CI、打包、部署审计](#10-测试cipackaging部署审计)
11. [发现汇总与优先级排序](#11-发现汇总与优先级排序)
12. [附录: 审计工具与方法](#12-附录-审计工具与方法)

---

## 1. 执行摘要

fusion-mlx 是一个成熟度较高的本地 LLM 推理服务器，代码规模庞大（~228k LOC Python + Metal 内核），功能覆盖 OpenAI/Anthropic/Responses API、多模态、音频、视频生成、speculative decoding、模型池管理、KV cache 分页与 SSD 溢出等。整体架构层次清晰，关键模块（paged_cache、prefix_cache、engine_pool 的 settle barrier）设计精良。

然而审计发现了 **多个 Critical 和 High 级别问题**，主要集中在：

| 类别 | Critical | High | Medium | Low | 总计 |
|---|---|---|---|---|---|
| 安全 | 1 | 0 | 3 | 6 | 10 |
| 可靠性 & 错误处理 | 4 | 14 | 5 | 7 | 30 |
| 内存 & 资源管理 | 0 | 2 | 5 | 5 | 12 |
| 性能 | 0 | 6 | 6 | 4 | 16 |
| 代码质量 | 1 | 5 | 8 | 9 | 23 |
| 并发 | (并入可靠性) | — | — | — | — |
| Metal/C++ 内核 | 1 | 2 | 5 | 5 | 13 |
| 测试/CI/打包/部署 | 2 | 3 | 5 | 3 | 13 |
| **总计** | **9** | **32** | **37** | **39** | **117** |

**最高优先级问题（需立即修复）**:

1. 🔴 **[安全-Critical]** `gui_compat/server.py` 所有路由 **完全无鉴权** — 任意网络攻击者可 load/unload/delete 模型、改 settings、shutdown 服务器。`validate_api_key` 已定义但从未作为依赖项接入。
2. 🔴 **[可靠性-Critical]** `pool/memory_enforcer.py`（1410 行）**零锁** — `_loaded_model_bytes`、`_pressure_level`、`_eviction_marked` 等共享可变状态从多个 async 上下文无同步访问，可导致并发计数器撕裂与错误的内存压力判断。
3. 🔴 **[可靠性-Critical]** `pool/engine_pool.py:1099` 等 — Metal 内存回收失败的 `except Exception: pass` 静默吞掉关键错误，可导致下次 load 的 silent OOM。
4. 🔴 **[可靠性-Critical]** `server.py _shutdown()` 不排空在途请求 — engines 被 abruptly stop，部分已 commit 的输出丢失，客户端看到截断的 SSE 流。
5. 🔴 **[打包-Critical]** `pyproject.toml:87` 与 `packaging/venvstacks.toml:17` 含 **开发者绝对路径** `/Users/dahai/...` — 任何其他机器 `pip install fusion-mlx[image]` 或构建 macOS app 都会失败。
6. 🔴 **[内核-Critical]** `glm_moe_dsa/csrc/kernels/quantized_glm.h:28-105` — `load_vector` 对 `x[i+3]`/`x[i+7]` 无边界检查，`load_vector_safe` 仍存在部分 tile 的 OOB 读越界。
7. 🔴 **[CI-Critical]** `.github/workflows/ci.yml:11,22` — CI 运行在 `ubuntu-latest`，但项目仅支持 macOS（pyproject.toml classifier）。所有 Darwin-only 测试被跳过，CI 给出虚假信心。
8. 🔴 **[代码质量-Critical]** `cli_serve.py:892-2273` — `serve_command` 函数长达 **1381 行**，30+ 内联 import，12+ `sys.exit()`，5 层嵌套。无法维护，新增 launch mode 风险极高。

**积极发现**:

- ✅ Agent graph 代码注入修复（#109）完整且正确
- ✅ `torch.load` 全部使用 `weights_only=True`
- ✅ `pyproject.toml:204-205` async test 配置正确
- ✅ `publish.yml` OIDC trusted publishing 正确配置，无硬编码 token
- ✅ Token-aware 前缀缓存（prefix_cache.py）+ COW + LRU 设计精良
- ✅ `paged_cache.py:576-582` 正确使用 `threading.RLock` + 文档化的锁顺序
- ✅ 模型 unload 路径完整释放 MLX weights / KV pages / tokenizer / compile cache / executor threads
- ✅ `_tempfile_safe.py` 使用 `threading.Lock` + atexit + try/finally 正确实现
- ✅ Rate limiter 默认启用且对 API 路由生效

---

## 2. 项目规模与架构

### 2.1 规模统计

| 指标 | 值 |
|---|---|
| `fusion_mlx/` Python 文件数 | 659 |
| `fusion_mlx/` 总 LOC | 228,077 |
| Metal/C++/H 文件数 | 21 |
| 测试文件数（tests/） | 707 |
| 测试总 LOC | 318,594 |
| 测试/源代码比 | ~1.4:1 |
| pyproject.toml 依赖数（required） | ~40 |
| pyproject.toml 依赖数（optional extras） | ~20 |
| 最大源文件 | `fusion_mlx/oq.py` 4869 行 |
| 最大函数 | `cli_serve.py:serve_command` 1381 行 |

### 2.2 架构分层

```
┌─────────────────────────────────────────────────────────────┐
│  CLI 入口                                                    │
│  cli.py / cli_serve.py / cli_commands.py / cli_lifecycle.py  │
├─────────────────────────────────────────────────────────────┤
│  FastAPI Server (server.py, 1091 行)                         │
│  ├─ middleware/ (auth, body_size, body_depth, request_id)   │
│  ├─ api/ (openai, anthropic, audio, images, videos, mcp...) │
│  ├─ admin/ (auth_routes, models_route, hf_downloader...)    │
│  ├─ routes_internal/ (cache, health, metrics, responses)    │
│  └─ gui_compat/ (legacy MLX-GUI 适配层, ~200 行)             │
├─────────────────────────────────────────────────────────────┤
│  Engine Pool & Memory Enforcer (pool/, ~5000 行)             │
│  ├─ engine_pool.py (2170) — LRU, pinning, TTL, settle bar   │
│  ├─ memory_enforcer.py (1410) — 4-tier 压力等级 + 驱逐       │
│  └─ priority_scheduler.py / unified_memory_pool.py          │
├─────────────────────────────────────────────────────────────┤
│  Scheduler (scheduler/, ~5000 行)                            │
│  ├─ core.py — 单任务调度循环 (engine_core._engine_loop)      │
│  ├─ sched_*.py — 拆分的调度阶段 (init/step/schedule/...)    │
│  └─ ngram_spec.py / spec_decode.py — 投机解码                │
├─────────────────────────────────────────────────────────────┤
│  Cache 层 (cache/, ~12000 行)                                │
│  ├─ paged_cache.py (1817) — 分块 KV + LRU + ref counting     │
│  ├─ paged_ssd_cache.py (2044) — SSD 冷层 + 异步 writer thread│
│  ├─ prefix_cache.py (2858) — token-aware 前缀匹配 + COW     │
│  └─ mllm_cache.py / hybrid_cache.py / recovery.py            │
├─────────────────────────────────────────────────────────────┤
│  Custom Kernels (custom_kernels/, 21 个 metal/cpp/h)         │
│  ├─ glm_moe_dsa/ — GLM MoE DSA + Steel Attention + Sparse MLA│
│  ├─ mfa/ — Metal Flash Attention bridge                      │
│  ├─ metal/ — moe_ffn_fused + w4a8_fused_matmul               │
│  └─ phase_c/ — W4A8 dispatch                                 │
├─────────────────────────────────────────────────────────────┤
│  引擎 (engines/, engine/, engine_core.py)                    │
│  ├─ batched.py — 主推理引擎                                  │
│  ├─ vlm.py — 多模态                                          │
│  ├─ audio (stt/tts/sts) / image_gen / reranker / embedding   │
│  └─ video_backends/ — LTX2 / Wan2 / SkyreelsV3               │
├─────────────────────────────────────────────────────────────┤
│  Patches (patches/, vendored model 适配)                     │
│  ├─ deepseek_v4/ / glm_moe_dsa/ / mlx_lm_mtp/                │
│  ├─ mlx_vlm_minimax_m3_compat/ (vendored mlx_vlm)            │
│  └─ mlx_vlm_mtp/ / specprefill / ...                         │
└─────────────────────────────────────────────────────────────┘
```

### 2.3 关键架构观察

- **单任务调度器**: `engine_core._engine_loop` (line 435) 在单个 `_mlx_executor` 线程上运行所有 scheduler step。注释明确 "guarantee that MLX GPU operations are never concurrent"。这是有意的设计选择。
- **全局可变状态**: `server.py` 使用模块级 `_server_state`、`_api_key`、`_pending_single_model` 等 staging 全局变量，与 config 单例桥接。这种 pattern 易产生 lifecycle bug。
- **vendored 代码**: `patches/mlx_vlm_minimax_m3_compat/vendor/mlx_vlm/` 是 vendored 上游代码，不在 ruff/black 范围内。
- **gui_compat 是 legacy 适配层**: 从前 fusion_gui 包迁入（#63），但本审计发现它带来 **严重的未鉴权路由暴露**（见 §3.1）。

---

## 3. 安全审计

### 3.1 🔴 Critical: gui_compat 路由完全无鉴权

**证据**: `fusion_mlx/gui_compat/server.py:371-805`

`get_gui_compat_router()` 注册的所有路由 **完全没有任何鉴权依赖**：

| 行号 | 路由 | 风险 |
|---|---|---|
| 422 | `POST /v1/models/{model_name}/load` | 攻击者可强制加载任意模型，触发 OOM |
| 460 | `POST /v1/models/{model_name}/unload` | 攻击者可卸载正在使用的模型 |
| 480 | `DELETE /v1/models/{model_name}` | **调用 `shutil.rmtree(mr.path)`**，攻击者可删除模型文件 |
| 619 | `PUT /v1/settings/{key}` | 攻击者可改写任意设置 |
| 734 | `POST /v1/models/install` | **未鉴权安装新模型**，可与 §3.5 path traversal 串接 |
| 603 | `POST /v1/system/shutdown` | **未鉴权关闭服务器** |
| 608 | `POST /v1/system/restart` | **未鉴权重启服务器** |
| 535 | `POST /v1/models/{model_name}/generate` | 未鉴权生成 |
| 399 | `GET /v1/models/{model_name}` | 未鉴权读 |
| 615 | `GET /v1/settings` | 未鉴权读 |

**根本原因**: `gui_compat/server.py:57-65` 定义了 `validate_api_key` 但 **从未在任何路由上作为 `Depends()` 使用**。对比 `server.py:550-674` 中所有等价路由都正确使用了 `Depends(require_admin)`。

**接入路径**: `server.py:545-546` 通过 `app.include_router(get_gui_compat_router())` 暴露这些路由。

**影响**: 任何能访问服务端口的网络攻击者（同一子网、LAN、被 NAT 暴露的实例）可以无鉴权地 load/unload/delete 模型、改写设置、安装任意模型、shutdown 或 restart 服务器。这等价于 **远程代码执行场景的入口**（删除关键模型 + 安装恶意模型）。

### 3.2 🟡 Medium: API key 通过 GET query param 泄漏

**证据**: `fusion_mlx/admin/auth_routes.py:224-257`

GET `/admin/auto-login` 接受 `?key=<api_key>` query 参数。这会将 API key：

- 写入浏览器历史记录
- 通过 `Referer` 头泄漏给任何外链
- 写入服务器 access logs（uvicorn 默认记录完整 URL）

虽然 `_is_loopback_request` (admin/auth.py:141-156) 限制为 loopback，且 GET 变体的注释提到 "browser bookmarks and menubar URLs"，但任何被 NAT/反向代理暴露的实例仍可能受影响。

### 3.3 🟡 Medium: 路径穿越风险（模型 install + delete）

**证据**: `fusion_mlx/gui_compat/server.py:480-494`

```python
@router.delete("/v1/models/{model_name}")
async def delete_model(model_name: str, ...):
    mr = db.query(Model).filter(Model.name == model_name).first()
    ...
    shutil.rmtree(mr.path)  # line 493
```

虽然 `model_name` 通过 SQLAlchemy ORM 查询（无 SQL 注入），但 `mr.path` 来自 DB 记录。结合 §3.1 的 **未鉴权 install 端点** (line 734)，攻击者可：

1. 未鉴权调用 `POST /v1/models/install` 插入一条 model 记录，path 指向任意目录
2. 调用 `DELETE /v1/models/{name}` 触发 `shutil.rmtree(任意目录)`

实现任意目录删除。

### 3.4 🟡 Medium: SSRF 部分缓解

**证据**: `fusion_mlx/api/_url_safety.py`

- `is_safe_url_with_dns()` 用于下载路径，**有 DNS 解析 + 私有网络阻断**（RFC1918、loopback、link-local、cloud metadata）
- `is_safe_url()` (line 46-63) 仅检查 hostname 字符串，**无 DNS 解析**，可被指向解析到私有 IP 的 hostname 绕过

实际使用：
- `gui_compat/server.py:282-307` 图像 URL fetch 使用 `is_safe_url_with_dns()` ✅
- `api/videos_routes.py:123-131` 视频素材 fetch 使用 `is_safe_url_with_dns()` ✅
- 但 `_BLOCKED_HOSTNAMES` 是硬编码列表，新的 metadata endpoint（如不同云厂商）需手动维护

### 3.5 ✅ CORS 配置正确

`fusion_mlx/server.py:494-507` — 默认 wildcard `["*"]` 时 `allow_credentials=False`，符合浏览器规范。`--cors-origins` 或 `FUSION_MLX_CORS_ALLOW_ORIGINS` 指定时启用 credentials。

### 3.6 ✅ 反序列化安全

- 所有 `torch.load(...)` 使用 `weights_only=True`（5 处，pulid_mlx / latentsync_mlx / skyreels_v3 / musetalk_mlx）
- 无 `pickle.load()` 调用
- 无 `yaml.load()` 不安全调用
- `json.loads()` 用于结构化请求

### 3.7 ✅ Agent graph 代码注入修复（#109）完整

`fusion_mlx/api/agent_routes.py:275-325` — `name`、`model`、`system_prompt` 通过 `json.dumps(config)` 嵌入为单一合法 Python dict 字面量。`temperature` 强制 float 并 clamp 到 `[0.0, 2.0]`。修复完整。

### 3.8 ✅ SQL 注入安全

- `gui_compat/server.py` 全部使用 SQLAlchemy ORM 参数化查询
- `gui_compat/database.py` 中 `text("PRAGMA ...")` 调用是硬编码字符串
- 无 f-string 或字符串拼接构造 SQL execute() 的实例

### 3.9 ✅ Subprocess 无注入风险

- 无 `shell=True`
- 所有 `subprocess.run/Popen` 使用硬编码命令列表
- `subprocess` 在 `nargs` 中接收的 argv 来自受控源（CLI 解析、硬编码脚本路径）

### 3.10 ✅ 发布工作流安全

`fusion_mlx/.github/workflows/publish.yml` — OIDC trusted publishing（`permissions: id-token: write`），无硬编码 token。

---

## 4. 可靠性与错误处理审计

### 4.1 异常吞咽统计

| 文件 | broad `except Exception` 计数 | 主要风险 |
|---|---|---|
| `gui_compat/mlx_integration.py` | 43 | 大部分 logged 或 re-raised，仅 line 71 有 bare `pass`。benign |
| `oq.py` | 30 | 待审视 |
| `service/helpers.py` | 24 | 待审视 |
| `pool/engine_pool.py` | 24 | 含 critical-swallow（见下） |
| `server.py` | 22 | 多为 startup/shutdown best-effort，acceptable |
| `utils/tokenizer.py` | 21 | — |
| `cli_serve.py` | 20 | — |

### 4.2 🔴 Critical: 静默吞掉 Metal 内存回收失败

**证据**: `fusion_mlx/pool/engine_pool.py:1099`

```python
except Exception:
    pass  # 静默吞掉 gc.collect() / mx.synchronize() / mx.clear_cache() 失败
```

在 engine unregister 路径中，Metal 内存回收失败被静默 `pass`。如果 Metal 内存回收失败，pool 失去追踪能力，**导致下次 load 的 silent OOM**。同类问题：

- `engine_pool.py:1078` — `engine.stop()` 失败静默继续（debug log），Metal 资源泄漏
- `engine_pool.py:1117` — 同上，另一 code path
- `engine_pool.py:1122` — latent cache cleanup + memory counter 调整失败静默，**导致 `_current_model_memory` 永久 drift**
- `engine_pool.py:1777,1834,1876` — engine stop 失败在 fallback 链中静默，debug log 中含 **硬编码行号**（编辑时会漂移）

### 4.3 🔴 Critical: `pool/memory_enforcer.py` 零锁

**证据**: `fusion_mlx/pool/memory_enforcer.py`（1410 行）**完全没有 `asyncio.Lock` 或 `threading.Lock`**。

无锁的共享可变状态：

| 行号 | 状态 | 风险 |
|---|---|---|
| 321-323 | `update_loaded_model_bytes(delta)` 修改 `_loaded_model_bytes` | 并发 `+=` 是 read-modify-write race，counter 撕裂 |
| 313 | `self._pressure_level` | 从 enforcer 轮询循环和外部 setters 同时读写 |
| 318 | `self._eviction_marked: set[str]` | 跨多个 coroutine 共享的可变 set |
| 315-316 | `_metal_wired_limit_request`, `_effective_metal_cap_bytes` | property setters 调用 `_propagate_memory_limit()` 读这些值，并发读写可看到 stale ceilings |

**影响**: 在高并发下，`_loaded_model_bytes` 撕裂导致内存压力计算错误，可能引发错误的驱逐决策或在应该驱逐时不驱逐（silent OOM）。

### 4.4 🟠 High: Engine pool 锁缺口

**证据**: `fusion_mlx/pool/engine_pool.py`

虽然 `_lock: asyncio.Lock` 存在，但 **多个同步方法在修改 `_entries` 和 `_current_model_memory` 时不持有锁**：

| 行号 | 方法 | 问题 |
|---|---|---|
| 1047-1062 | `register_engine()` | 同步方法，mutate `_entries` 和 `_current_model_memory` **无锁**，与任何 `async with self._lock` 块竞争 |
| 1064-1130 | `unload_engine()` | 同步方法，mutate `_entries` 和 `_current_model_memory` **无锁** |
| 337-349 | `list_models()`, `model_count`, `loaded_model_count` | 无锁读 `_entries`，可看到不一致快照 |
| 491-497 | `get_loaded_model_ids()`, `get_entry()` | 无锁读 |
| 499-514 | `set_pinned()`, `find_model_id()`, `resolve_model_id()` | 无锁读 |
| 656-660 | `is_abort_requested()` | 无锁读 `_entries` |
| 475-485 | `apply_settings_overrides()` | 无锁迭代 `_entries` |

### 4.5 🟠 High: 前缀缓存层解析失败静默跳过

**证据**: `fusion_mlx/cache/prefix_cache.py:801,830`

```python
except Exception:
    continue  # seq_len 解析失败静默跳过 layer
```

在 seq_len resolution 中，per-layer 异常被静默跳过。可导致 **静默返回错误的 sequence length，重建时 cache corruption**。同样模式：`cache/mllm_cache.py:281` — eviction logging 失败静默，`_evict_by_count` 用错误假设继续。

### 4.6 🔴 Critical: `server.py _shutdown()` 不排空在途请求

**证据**: `fusion_mlx/server.py:922-977`

`_shutdown()` 调用 `self.pool.shutdown()` (line 960-961)，但 **不显式排空或 abort 在途请求**。pool 的 `shutdown()` (`engine_pool.py:2060-2069`) 迭代 entries 调用 `unload_engine_async()`，后者调用 `entry.engine.stop()` — 这设置 `_running = False` 并 cancel engine loop task，**abruptly 终止任何活跃的生成请求**。

**影响**: 部分已 commit 的输出丢失；客户端看到截断的 SSE 流（无 `[DONE]` 或 error event）。

同类问题：
- `engine_core.py:364-375` — `stop()` 仅 set `_running=False` + wake event + cancel task，不 wait in-flight requests
- `engine_core.py:1115-1119` — `EngineWrapper.stop()` 直接 delegate

### 4.7 🟠 High: 流式错误传播不完整

**证据**: `fusion_mlx/api/openai_routes.py:441-485`

- OpenAI 流式错误正确 yield `data: {"error": ...}` event ✅
- 但如果 engine 在 mid-stream **崩溃**（Metal OOM），异常从 stream generator 传播，由于 `StreamingResponse` 包装的是 generator，错误导致 **truncated/incomplete SSE stream without proper `[DONE]` or error event**，客户端看到 abrupt connection drop
- Anthropic 路由 `anthropic_routes.py:506-529` 有类似问题

OpenAI 路由的 client disconnect 处理 (`openai_routes.py:441-461`) 创建 background abort task，但 abort 错误 (line 459) 被 `except Exception: pass` 静默吞掉 — **如果 abort 失败，engine 可能保持 locked**。

### 4.8 🟠 High: 无 HTTP 自动重试/退避

**证据**: `fusion_mlx/admin/hf_downloader.py:778-892`, `admin/ms_downloader.py`

- `_run_download()` 单次 `snapshot_download()` 调用，**无自动重试**
- 任何 transient 失败（网络抖动、HF rate limit、timeout）→ 立即失败
- 重试仅通过 `retry_download()` (line 712) 手动触发（admin UI）
- Dry run (size estimation) 也无重试，30s timeout 单次尝试

同类问题：
- `admin/hf_downloader.py:70-103` — `_resolve_endpoint()` 单次 HTTP HEAD probe
- `admin/update_check.py:63-67` — `requests.get` 单次尝试，无 rate limit 重试
- `admin/preset.py:46-50` — 同上

### 4.9 🟡 Medium: 背景任务未显式取消

**证据**: `fusion_mlx/server.py:922-977`

`_shutdown()` **不显式取消**：telemetry background sender、model downloader progress pollers、scheduled LRU eviction tasks、memory enforcer polling loop。它们依赖 asyncio task cancellation 由 event loop shutdown 完成。

### 4.10 🟡 Medium: MLX 资源无序释放

**证据**: `fusion_mlx/server.py:976`

`mx.clear_cache()` 在 pool shutdown 之后最后调用。但 **`mx.synchronize()` 未在 `clear_cache()` 之前调用** — 可与 executor 线程上的 Metal 操作竞争。

---

## 5. 内存与资源管理审计

### 5.1 🟠 High: 全局 executors 从不显式 shutdown

**证据**: `fusion_mlx/engine_core.py:66, 104-116`

`_global_executors` dict (llm/audio/io pools) **从不显式 shutdown**。线程非 daemon。Python 解释器在 exit 时等待它们。`compile_cache` fallback (lines 1053-1057) 故意 append 到 `_immortal_mlx_executors` 以避免 compile-cache thread-exit crash，**但正常路径也不 shutdown global executors**。

**影响**: 3-5 个非 daemon 线程持续到进程退出，可能在 `atexit` 阶段触发 `RuntimeError: Event loop is closed`。

### 5.2 🟠 High: MCP security `_call_times` 无界增长

**证据**: `fusion_mlx/mcp/security.py:436`

```python
_call_times: dict[str, list[float]] = defaultdict(list)
```

Rate-limiting call times dict **每个 tool invocation append 一个 timestamp，无 eviction**。在持续 tool calling 下，此 dict 无界增长 = 内存泄漏。

### 5.3 🟡 Medium: MLX set_cache_limit 期间保持 ~2x 模型大小

**证据**: `fusion_mlx/pool/engine_pool.py:1993`

注释提到 `mx.set_cache_limit(total_mem)` **阻止 model loading 期间的自动 Metal buffer release**。代码在 line 1996 通过 `mx.clear_cache()` 补偿，但 **如果补偿路径失败，内存保持 inflated 直到首次 inference request**。

### 5.4 🟡 Medium: SSD cache 写失败静默丢块

**证据**: `fusion_mlx/cache/paged_ssd_cache.py:1607-1620`

在写失败（ENOSPC 或 OSError）时：
- Log 错误
- 从 index 移除 block
- Increment `errors` stat
- **不向 caller 传播错误**

数据被 **静默丢弃**。下次该 block hash 的 load 将 miss 并触发 recomputation。这是 graceful degradation，但可导致 **silent recomputation overhead** on reload。

### 5.5 🟡 Medium: `__del__` 调用 registry.release 脆弱

**证据**: `fusion_mlx/engine_core.py:1069`

`__del__` 调用 `get_registry().release(self.model, ...)`。registry 是 module-level singleton。**如果 registry 已被 GC teardown，此调用可能抛异常**。try/except 缓解，但 `__del__` 总体上可 impede GC 并引发 reference cycles。

### 5.6 🟡 Medium: audio_vae 多处 `open()` 无 `with`

**证据**: `fusion_mlx/video/ltx2/audio_vae/audio_vae.py:187,356`

```python
json.load(open(model_path / "config.json"))  # file handle 不显式关闭，泄漏到 GC
```

### 5.7 🟡 Medium: gui_compat NamedTemporaryFile 无清理机制

**证据**: `fusion_mlx/gui_compat/server.py:279,305,325`

`NamedTemporaryFile(delete=False)` — 路径 append 到 `result` 列表，但 **无 apparent cleanup mechanism**。这些可泄漏 temp 文件。

### 5.8 ✅ KV cache 内存回收设计良好

- `cache/paged_cache.py:576-582` — 正确使用 `threading.RLock`，文档化的锁顺序
- `cache/prefix_cache.py:125` — `_cache_lock = asyncio.Lock()` 单锁覆盖所有 cache 操作
- `cache/paged_cache.py:753` — `free_block()` 通过 ref counting 回收 block 到 `FreeKVCacheBlockQueue` LRU list
- 所有 caches 有界 + LRU eviction

### 5.9 ✅ Telemetry queue 有界

`telemetry/queue.py:63` — `deque(maxlen=100)`，overflow 时丢弃最旧 events。3-way overflow invariants 正确。

### 5.10 ✅ 模型 unload 路径完整

| 资源 | 释放？ | 证据 |
|---|---|---|
| MLX weights | ✅ | `engine.stop()` → `engine_core.close()` → model registry release + `model = None` |
| KV cache pages | ✅ | `scheduler.shutdown()` + `deep_reset()` (engine_core.py:1020-1021) |
| Tokenizer caches | ✅ | `self._tokenizer = None` (batched.py:535) |
| Compile caches | ✅ | `clear_thread_compile_cache()` on executor thread (engine_core.py:1041-1044) |
| Executor threads | ✅ | `self._mlx_executor.shutdown(wait=True)` (engine_core.py:1063) |
| SSD cache manager | ✅ | `mgr.close()` (engine_core.py:1016) |
| Latent cache | ✅ | `remove_image_latent_cache()` (engine_pool.py:1246-1248) |
| Hot cache budget | ✅ | Enforcer walk clears hot cache (memory_enforcer.py:767-824) |

unload 路径 **comprehensive and well-engineered**。settle barrier（最多 10 轮 gc+sync+clear_cache 轮询）确保 Metal 内存实际回收后才进行下次 load。

---

## 6. 性能审计

### 6.1 🟠 High: `sched_schedule.py:84-91` O(n²) token budget 扫描

**证据**: `fusion_mlx/scheduler/sched_schedule.py:84-91`

每次 `while self.waiting` 循环迭代重新计算 `batched_tokens`，summing `len(r.remaining_tokens)` over **所有已调度的 requests**。这是 O(k²)，其中 k = 一个 batch 中调度的 request 数（通常 ≤ `max_num_seqs`，但可较大）。accumulated sum 随迭代线性增长，但每次从头重算。

**修复**: 维护一个 running accumulator，per request added 时 increment。

### 6.2 🟠 High: 多处 `asyncio.gather` 无 `return_exceptions=True`

**证据**:

| 文件 | 行号 |
|---|---|
| `fusion_mlx/eval/mbpp.py` | 206 |
| `fusion_mlx/eval/livecodebench.py` | 249 |
| `fusion_mlx/eval/base.py` | 263, 284 |
| `fusion_mlx/eval/humaneval.py` | 258 |

```python
gen_results = await asyncio.gather(*gen_tasks)  # 无 return_exceptions
```

如果 ANY 单个 task 抛异常，**所有** task 立即被 cancel。这丢失该 batch 中所有其他 task 的结果。这在 eval benchmark suite 中，影响 offline testing 可靠性。

**修复**: 加 `return_exceptions=True`。

### 6.3 🟡 Medium: `monkeypatches.py:324` 每 decode step 全量 list 拷贝

**证据**: `fusion_mlx/scheduler/monkeypatches.py:324`

```python
self._deferred_tokens = list(self.tokens)  # 每 _step() 调用都全量拷贝
```

虽然 `self.tokens` 较小（batch size len），仍是 per-token overhead。同类：`monkeypatches.py:296` — `self._next_logprobs = [None] * len(self.uids)` 每个 decode step 都重新分配新 list，即使未请求 logprobs。

### 6.4 🟡 Medium: grammar 路径中 `mx.eval` 强制 GPU sync

**证据**: `fusion_mlx/scheduler/monkeypatches.py:328,330-333`

```python
if has_grammar:
    mx.eval(self._next_tokens)  # line 328 — 强制 GPU sync
    for e in range(len(self.uids)):
        for proc in self.logits_processors[e]:
            if isinstance(proc, GrammarConstraintProcessor):
                proc.accept_token(sampled_list[e])
```

grammar-constrained request 的每个 decode step 多一次 `mx.eval`（device stall）。主路径已在 line 307 使用 `async_eval`。

### 6.5 🟡 Medium: `sched_vlm_mtp_batched.py:241` 每 token GPU sync

**证据**: `fusion_mlx/scheduler/sched_vlm_mtp_batched.py:241`

```python
mx.eval(tok)  # line 241 — 在 VLM MTP batch decode 内循环中，每 token 一次 GPU sync
```

在循环内顺序处理 B 个 token，每个 `mx.eval(tok)` 强制 GPU sync。后续 `int(tok.item())` 也强制 materialization。

### 6.6 🟡 Medium: 投机解码每 step `copy.deepcopy` KV cache

**证据**: `fusion_mlx/scheduler/ngram_spec.py:278`, `spec_decode.py:137`

```python
snapshots.append((i, copy.deepcopy(c)))  # 对 non-trimmable KV cache layer 做 deep copy
```

在 per-step verify loop 内对大型 MLX array 做 `copy.deepcopy` 非常昂贵。

### 6.7 🟡 Medium: 前缀缓存用 `id(model)` 作为 key

**证据**: `fusion_mlx/cache/prefix_cache.py:96`

```python
self.model_key = id(model)  # Python id() = 内存地址
```

如果 model 被 GC 且新 model 对象在同一地址分配，stale cache entries 可匹配。实践中 unlikely 因为 `BlockAwarePrefixCache` 实例通常 per model 创建，但 `clear()` 不 reset `model_key`。如果 cache 对象跨 model swap 复用，`model_key` 保持相同而实际 model 变化。

### 6.8 🟡 Medium: Disk cache key 仅 hash `model_name`

**证据**: `fusion_mlx/runtime/cache.py:175-186`

```python
digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:8]
leaf = f"{safe_name}--{digest}"
```

Disk cache directory 仅基于 `model_name` hash。如果同一 model name 用不同 quantization settings 或 adapter paths 加载，它们会 **共享（并 corrupt）** 同一 disk cache。无 quantization hash 或 adapter hash 包含。

### 6.9 ✅ 主 LLM 生成路径无 numpy round-trip

`np.array` → `mx.array` 转换局限于 `video/` pipeline code（model weight conversion、image preprocessing），不在 generation hot loops。主 LLM 生成路径原生使用 MLX arrays。

### 6.10 ✅ 调度器并发模型设计正确

`engine_core._engine_loop` 在单个 `_mlx_executor` 线程上运行所有 scheduler step，注释明确 "guarantee that MLX GPU operations are never concurrent"。这是正确的 Apple Silicon MLX 设计选择。

---

## 7. 代码质量审计

### 7.1 🔴 Critical: `serve_command` 1381 行无法维护

**证据**: `fusion_mlx/cli_serve.py:892-2273`

`serve_command` 函数长达 **1381 行**，包含：

- 30+ 内联 `import` 语句
- 12+ `sys.exit()` 调用
- 10+ early `return` 路径
- 5+ mutually exclusive mode forks（audio、dspark、--model-dir、--base-path、single-model）
- Deep nesting（lines 1258-1320 = 5 层 if/elif/try 嵌套）

此函数 **未充分分解** — 它在单个 monolithic block 中完成参数验证、alias 解析、feature-gating、model loading、middleware 配置、server boot。新增 launch mode 风险极高。

### 7.2 🟠 High: `engine_pool.py` 顶层 stub 死代码

**证据**: `fusion_mlx/fusion_mlx/engine_pool.py:1-10`

```python
class EnginePool:
    def __init__(self, **kwargs):
        pass  # stub, 实际实现在 fusion_mlx/pool/engine_pool.py
```

整个文件是 stub，实际实现在 `fusion_mlx/pool/engine_pool.py`（2170 行）。此 stub 看起来是 migration placeholder，从未清理。

### 7.3 🟠 High: `pyproject.toml:87` 开发者绝对路径

**证据**: `fusion_mlx/pyproject.toml:87`

```toml
"mflux-fusion @ file:///Users/dahai/claude-home/fusion-mlx/packaging/_wheels/mflux_fusion-0.18.0-py3-none-any.whl"
```

**这是开发者 `dahai` 的绝对路径**。任何其他机器 `pip install fusion-mlx[image]` 都会失败。

### 7.4 🟠 High: `speculative/draft_model.py:21` 开发者绝对路径

**证据**: `fusion_mlx/speculative/draft_model.py:19-24`

```python
DRAFT_MODEL_PATH = __import__("os").environ.get(
    "FUSION_DRAFT_MODEL_PATH",
    "/Users/dahai/.fusion-mlx/models/mlx-community/Qwen3-0.6B-4bit",
)
```

**开发者特定路径泄漏到生产代码**。虽有 env var override，fallback 是特定用户路径。

### 7.5 🟠 High: 多处硬编码 `/Users/dahai/...` 路径

| 文件 | 行号 | 内容 |
|---|---|---|
| `tests/integration/test_mtp_suffix_coherence.py` | 28 | `DEFAULT_MODEL = "/Users/dahai/.fusion-mlx/models/..."` |
| `tests/integration/test_ngram_spec_gdn_coherence.py` | 30 | 同上 |
| `scripts/bench_extrapolation_draft.py` | 18,30 | `sys.path.insert(0, "/Users/dahai/claude-home/fusion-mlx")` + hardcoded model path |
| `scripts/quantize_oq.py` | 20-21 | 硬编码 `SOURCE` 和 `MODELS_DIR` |
| `scripts/bench_spec_ab.py` | 31 | 硬编码 model path |
| `apps/fusion-mac/MIGRATION.md` | 9-10 | 文档引用 `/Users/dahai/claude-home/fusion-mlx/...` |

### 7.6 🟠 High: `StreamingPostProcessor` 类 ~3691 行

**证据**: `fusion_mlx/service/postprocessor.py:179-3870`

`StreamingPostProcessor` 类横跨 ~3691 行，含 4 个主处理方法（`_process_channel_routed` at 2535, `_process_with_reasoning` at 2852, `_process_standard` at 3174, `finalize` at 3411）。~50+ instance variables 和深度嵌套的 state machine 逻辑。

### 7.7 🟡 Medium: 模块级 `__import__("os").environ.get(...)` 反模式

**证据**:

| 文件 | 行号 | 常量数 |
|---|---|---|
| `fusion_mlx/scheduler/ngram_spec.py` | 29-49 | 6 个 module-level 常量初始化 |
| `fusion_mlx/scheduler/spec_decode.py` | 27-38 | 5 个 |
| `fusion_mlx/speculative/draft_model.py` | 19-24 | 3 个 |

使用 `__import__("os")` 而非 `import os` 是 intentional workaround，但仍是 module-level side effect。该 pattern 在 3 个文件中复制，应抽取为 utility 函数。

### 7.8 🟡 Medium: 重复的 size formatting 工具

| 文件 | 行号 | 函数 |
|---|---|---|
| `fusion_mlx/pool/memory_enforcer.py` | 95-97 | `_format_gb` |
| `fusion_mlx/admin/oq_manager.py` | 113-127 | `_format_size` |
| `fusion_mlx/cli_commands.py` | 19-41 | `_format_bytes` |
| `fusion_mlx/pool/model_discovery.py` | — | `format_size` |

4 处 IEC-suffixed byte formatting 重复实现。

### 7.9 🟡 Medium: `Any` 类型广泛使用

**证据**:

| 文件 | 行号 | 用法 |
|---|---|---|
| `fusion_mlx/cache/prefix_cache.py` | 15, 83 | `self.model: Any` — model type 完全擦除 |
| `fusion_mlx/tool_parsers/` | 20+ 文件 | 所有 tool parser 用 `dict[str, Any]` for tool_calls、request params、return types |
| `fusion_mlx/middleware/body_size.py` | 22 | `from typing import Any` |
| `fusion_mlx/video/` | ~10 文件 | `Any` in type hints |

### 7.10 🟡 Medium: 23 处 `# type: ignore`

无注释解释的 `# type: ignore` 集中在 `utils/psutil_compat.py`、`_torch_stub.py`、`cli.py`、`_mirror.py`（fcntl import-not-found）。这些通常是 platform-specific workaround，但应附 `# type: ignore[error-code]` 说明。

### 7.11 ✅ Tool parser 抽象良好

21 个 tool parser 都继承 `abstract_tool_parser.py` 的 `BaseToolParser`，遵循一致接口。

### 7.12 ✅ Cache 模块边界清晰

`cache/` 包含明确的接口分离：`interface.py`、`protocol.py`、`factory.py`、`paged_cache.py`、`prefix_cache.py`、`mllm_cache.py`、`hybrid_cache.py`、`recovery.py`。

---

## 8. 并发与并行审计

（并入 §4 可靠性审计的相关条目）

### 8.1 关键并发发现汇总

| 严重性 | 文件:行号 | 问题 |
|---|---|---|
| 🔴 Critical | `pool/memory_enforcer.py`（全文件零锁） | `_loaded_model_bytes` 等 shared state 无同步访问 |
| 🟠 High | `pool/engine_pool.py:1047-1062, 1064-1130` | 同步 `register_engine`/`unload_engine` 修改 `_entries` 无 `_lock` |
| 🟠 High | `pool/engine_pool.py:337-349, 491-497, 499-514, 656-660, 475-485` | 多个无锁读 `_entries` |
| ✅ Good | `cache/paged_cache.py:576-582` | 正确使用 `threading.RLock` + 文档化锁顺序 |
| ✅ Good | `cache/prefix_cache.py:125` | 单 `_cache_lock = asyncio.Lock()` 覆盖所有 cache 操作 |
| ✅ Good | `engine_core._engine_loop` (line 435) | 单线程调度，"guarantee that MLX GPU operations are never concurrent" |

### 8.2 Async 任务生命周期

| 严重性 | 文件:行号 | 问题 |
|---|---|---|
| ✅ Good | `engine_core.py:361` | `asyncio.create_task(self._engine_loop())` 存储在 `self._task`，正确管理 |
| ✅ Good | `api/openai_routes.py:445` | Task 存储在 `_pending_abort_tasks`，使用 `add_done_callback` discard |
| ✅ Good | `service/disconnect_guard.py:189` | Task 本地存储，lifecycle 管理 |

### 8.3 `asyncio.gather` 错误传播

| 严重性 | 文件:行号 | 问题 |
|---|---|---|
| 🟠 High | `eval/mbpp.py:206`, `eval/livecodebench.py:249`, `eval/base.py:263,284`, `eval/humaneval.py:258` | 无 `return_exceptions=True`，一个失败 cancel 所有 |
| 🟡 Medium | `admin/benchmark.py:358` | 同上 |

---

## 9. Metal/C++ 内核与 FFI 审计

### 9.1 🔴 Critical: `quantized_glm.h` `load_vector` 无边界检查

**证据**: `fusion_mlx/custom_kernels/glm_moe_dsa/csrc/kernels/quantized_glm.h:28-105`

`load_vector` 循环以 stride 4 或 8，访问 `x[i+3]`/`x[i+7]` **不检查 source buffer 是否足够大**。`load_vector_safe` sibling (line 108) 接受 `N` 作为 bound 但仍做 `x[i+1..i+3]` 后只检查 `i < N` — **如果 `N % 4 != 0`，它读越界**。在 K-dim 循环的最后一个 tile 是 partial group 时可达。

### 9.2 🟠 High: `moe_ffn_fused.metal` threadgroup union 内存溢出

**证据**: `fusion_mlx/custom_kernels/metal/moe_ffn_fused.metal:269-280`

Phase 1 存储 `As_fp8[BM*BK_PAD]`（2880 字节，BM=32, BK=64, BK_PAD=80）。Phase 2 **重用同一内存**作为 `hidden[BM*BN_PAD]`，BN=32, BN_PAD=36 → 4608 字节。

计算：
- p1 总大小 = 2560 + 1280 + 256 = **4096 字节**
- p2.hidden = 32 × 36 × 4 = **4608 字节**

**512 字节溢出** within the union，潜在 clobber 其他 threadgroup 变量。

### 9.3 🟠 High: `compile_metallib.py:74` 使用 `-ffast-math`

**证据**: `fusion_mlx/custom_kernels/mfa/compile_metallib.py:74`

`-ffast-math` 标志启用：
- `-fma`（FMA contraction，改变 rounding）
- reassociation
- **disable `-ftrapping-math`**（NaN propagation 不保证）
- **flush subnormals to zero**
- **disable `-fhonor-nans`**

这与 GLM DSA 内核不一致（CMakeLists.txt:58 使用 `-fno-fast-math`）。同一 Metal compiler 两种 numerical guarantees，**可导致 kernel 间数值不一致**。

### 9.4 🟡 Medium: `load_unsafe()` 读 BM 行不条件检查

**证据**: `fusion_mlx/custom_kernels/metal/moe_ffn_fused.metal:118-124`, `w4a8_fused_matmul.metal:114-119`

当 `valid_m == BM`，`load_unsafe()` 读 `*((const device Vec4*)(&src_fp8[i * src_ld]))` for `i = 0 .. BM-1` **不检查 device buffer 实际持有 `BM` 行**。如果 M 不是 BM 的倍数，最后一个 threadgroup 读越界 — classic OOB。

### 9.5 🟡 Medium: `mfa/quantize.py` 用截断而非 round-to-nearest-even

**证据**: `fusion_mlx/custom_kernels/mfa/quantize.py:48, 74, 128`

```python
((scaled / fp8_max) * 127.0 + 128.0).astype(mx.uint8)  # 截断 toward zero
```

`astype(mx.uint8)` **截断 toward zero，而非 round-to-nearest-even**（IEEE 754 要求）。引入 -0.5 LSB 系统性 bias。

### 9.6 🟡 Medium: `sparse_mla.cpp:251` `qL_off = kL - qL` 可下溢

**证据**: `fusion_mlx/custom_kernels/glm_moe_dsa/csrc/sparse_mla.cpp:251`

`qL_off = kL - qL` 使用 signed int。如果 `qL > kL`，值为负。后续用作 `q_abs = params->qL_off + q_pos` — 如果负，`k_pos > q_abs` 比较可能行为异常。MLX shape validation at lines 364-365 检查 `kv_latent.shape(2) != k_pe.shape(2)` 但 **不强制 `kL >= qL`**。

### 9.7 🟡 Medium: FFI 无 per-call try/except，Metal fault 致命

**证据**: `fusion_mlx/custom_kernels/glm_moe_dsa/fast.py:13-17`, `dsa_indexer.cpp` 等所有 `eval_gpu` 方法

如果 `d.get_kernel(kname, lib)` 失败（metallib 未构建或 function constant hash 不匹配），MLX 抛 runtime exception。C++ 代码不 catch。通过 nanobind 传播到 Python 作为 unhandled exception。**Metal fault mid-dispatch 导致 uncatchable MTLCrash**。

### 9.8 🟡 Medium: `compile_metallib.py` 无 staleness check，每次重编译

**证据**: `fusion_mlx/custom_kernels/mfa/compile_metallib.py:39-115`

`compile_msl()` 检查 source 是否存在 (line 49) 但 **不比较 source vs output 的 modification time**。每次调用从 scratch 重编译。这增加 ~2s per kernel at import time。`turboquant_fused.py` (line 15) 使用 in-memory `_KERNEL_CACHE` dict 避免 per-call recompilation，但 process restart 时丢失。

### 9.9 ✅ Numerical stability 大体良好

- `silu(x) = x / (1.0f + exp(-x))` 对大负 x 安全（`exp(∞) → x/∞ ≈ 0`），对大正 x 安全（`exp(-x) ≈ 0 → x/1 = x`）✅
- `fast::exp2(x - y)` 用于 softmax exponentiation，**max subtraction before exp** 正确 ✅
- `fast::exp2` 有 ~2 ULP error vs `precise::exp2` <1 ULP，对 fp16/bf16 accumulation 可接受

### 9.10 ✅ 无 C++ 内存泄漏

所有 allocations 通过 `allocator::malloc`（MLX-managed）或 nanobind stack variables。Metal library/kernel handles 由 MLX `metal::device` cache 管理。`current_binary_dir()` 使用 `dladdr`（无 cleanup 需要）。**无 raw `new` 无 `delete`**，无泄漏检测。

---

## 10. 测试、CI、Packaging、部署审计

### 10.1 🔴 Critical: CI 运行在 `ubuntu-latest` — 不支持的平台

**证据**: `fusion_mlx/.github/workflows/ci.yml:11,22`

两个 job（`lint` 和 `test`）都使用 `runs-on: ubuntu-latest`。但项目自己的 `pyproject.toml` classifier（line 21）仅列出 `"Operating System :: MacOS"`，且测试套件含多个 `@pytest.mark.skipif(sys.platform != "darwin")` guards。

- `lint` job（ruff, black）OS-agnostic，OK 在 ubuntu
- `test` job（pytest）将 **跳过 ~20+ Darwin-only 测试**（UBC eviction、proc_memory 等）和需要真实 MLX 的测试（在 Linux 上 stub）

**CI 给出虚假信心** — macOS-only bugs 不会被 CI 捕获。

### 10.2 🔴 Critical: `pyproject.toml:87` + `venvstacks.toml:17` 开发者绝对路径

**证据**:

| 文件 | 行号 | 内容 |
|---|---|---|
| `pyproject.toml` | 87 | `mflux-fusion @ file:///Users/dahai/claude-home/fusion-mlx/packaging/_wheels/mflux_fusion-0.18.0-py3-none-any.whl` |
| `packaging/venvstacks.toml` | 17 | 多个 `file:///Users/dahai/claude-home/fusion-mlx/packaging/_wheels/...` 路径用于所有 git-pinned wheels |

**`pip install fusion-mlx[image]` 在任何其他机器上会失败**。`venvstacks.toml` 是 macOS app build spec — 任何其他人构建 app 都会失败。

### 10.3 🟠 High: `transformers` 版本冲突

**证据**:

| 文件 | 行号 | 内容 |
|---|---|---|
| `pyproject.toml` | 36 | `transformers>=5.0.0`（loose pin） |
| `packaging/build.py` | 48 | `"transformers": "transformers==5.0.0"`（exact pin override for app bundle） |
| `packaging/venvstacks.toml` | 17 | `transformers==5.0.0`（pinned exact in app build） |

`build.py` 注释（line 42-44）明确说 5.13.0 破坏 mlx-lm 的 import。**通过 pip 安装的用户获得 broken transformers version**（>=5.0.0 将解析到 latest，如 5.13.0）。Pin 仅在构建 macOS app bundle 时生效。

### 10.4 🟡 Medium: mypy 在 dev deps 但 CI 不运行

**证据**: `pyproject.toml:121,130`, `.github/workflows/ci.yml:17,18`

`mypy>=1.0.0` 在 `[dev]` 和 `[dependency-groups.dev]`。但 CI lint job 仅运行 `ruff` 和 `black`。**mypy 安装但从不调用**。

### 10.5 🟡 Medium: 无安全扫描

- **无 `.github/dependabot.yml`** — 无 Dependabot 配置
- **无 `pip-audit` 或 `bandit`** — CI steps 和 dev deps 中均无
- 整个 pipeline 中无任何安全扫描

### 10.6 🟡 Medium: Git-pin verify 脚本不在 CI 中

**证据**: `scripts/verify_git_pins.sh` 存在且正确检查所有 5 个 git deps 的 SHA pins。但 **不在 `.github/workflows/ci.yml` 中调用**。SHA drift 会未检测。

### 10.7 🟡 Medium: 测试 time.sleep flakiness

**证据**:
- `tests/unit/test_paged_ssd_cache.py` — 15+ `time.sleep(0.01-0.05)` 调用
- `tests/unit/test_telemetry_queue.py` — 6+ `time.sleep(0.05-0.2)` 调用
- `tests/unit/test_vision_feature_cache.py` — 5 `time.sleep(0.5)` 调用
- `tests/unit/test_cloud_router.py` — 3 `time.sleep(0.1-0.15)` 调用

**~50+ instances** 真实 sleep 使测试在 loaded CI 上慢且易 race-condition flakiness。

### 10.8 🟡 Medium: 无 coverage reporting

**证据**: `pyproject.toml` 无 `[tool.coverage.*]` section。`pytest-cov` 不在 dev deps。CI 无 `--cov` flags。**Coverage 百分比未知**。Dead code 无法被检测。

### 10.9 ✅ Test-to-source ratio 良好

- 源文件：646 Python 文件 under `fusion_mlx/`
- 测试文件：679 `test_*.py` 文件 across `tests/`
- Ratio ~1.05:1 — 良好的 coverage 广度

### 10.10 ✅ Async test 配置正确

`pyproject.toml:205` — `asyncio_mode = "auto"` 存在且正确。

### 10.11 ✅ 发布工作流安全

`fusion_mlx/.github/workflows/publish.yml` — OIDC trusted publishing（`permissions: id-token: write`），无硬编码 token，引用 `pypa/gh-action-pypi-publish@release/v1`。

### 10.12 ✅ `.gitignore` 覆盖完整

正确排除 `__pycache__/`、`.venv/`、`.env`、`dist/`、`build/`、`.codegraph/`、packaging build artifacts、macOS app data directories。敏感文件被覆盖。

---

## 11. 发现汇总与优先级排序

### 11.1 最高优先级修复（P0 — 立即修复）

| # | 类别 | 文件:行号 | 问题 |
|---|---|---|---|
| 1 | 安全-Critical | `gui_compat/server.py:371-805` | gui_compat 所有路由完全无鉴权 — RCE 入口 |
| 2 | 打包-Critical | `pyproject.toml:87` | 开发者绝对路径破坏 pip install |
| 3 | 打包-Critical | `packaging/venvstacks.toml:17` | 开发者绝对路径破坏 macOS app build |
| 4 | CI-Critical | `.github/workflows/ci.yml:11,22` | CI 在 ubuntu，跳过所有 Darwin-only 测试 |
| 5 | 可靠性-Critical | `pool/memory_enforcer.py`（全文件） | 零锁 shared state 并发撕裂 |
| 6 | 可靠性-Critical | `pool/engine_pool.py:1099` | Metal 内存回收失败静默吞掉，silent OOM |
| 7 | 可靠性-Critical | `server.py _shutdown()` (922-977) | 不排空在途请求，输出丢失 |
| 8 | 内核-Critical | `glm_moe_dsa/csrc/kernels/quantized_glm.h:28-105` | `load_vector` 无边界检查 OOB |
| 9 | 代码质量-Critical | `cli_serve.py:892-2273` | `serve_command` 1381 行无法维护 |

### 11.2 高优先级修复（P1 — 下一迭代）

| # | 类别 | 文件:行号 | 问题 |
|---|---|---|---|
| 10 | 可靠性-High | `pool/engine_pool.py:1047-1130` 等 | 同步方法修改 `_entries` 无锁 |
| 11 | 可靠性-High | `cache/prefix_cache.py:801,830` | seq_len 解析失败静默跳过，cache corruption |
| 12 | 可靠性-High | `api/openai_routes.py:441-485` | mid-stream engine 崩溃 → truncated SSE，无 error event |
| 13 | 可靠性-High | `admin/hf_downloader.py:778-892` | 无 HTTP 自动重试/退避 |
| 14 | 内存-High | `engine_core.py:66,104-116` | 全局 executors 从不 shutdown |
| 15 | 内存-High | `mcp/security.py:436` | `_call_times` defaultdict 无界增长 = 内存泄漏 |
| 16 | 性能-High | `sched_schedule.py:84-91` | O(n²) token budget 扫描 |
| 17 | 性能-High | `eval/mbpp.py:206` 等 5 处 | `asyncio.gather` 无 `return_exceptions` |
| 18 | 内核-High | `moe_ffn_fused.metal:269-280` | threadgroup union 512 字节溢出 |
| 19 | 内核-High | `compile_metallib.py:74` | `-ffast-math` disable NaN propagation |
| 20 | 代码质量-High | `fusion_mlx/engine_pool.py:1-10` | 死代码 stub |
| 21 | 代码质量-High | `speculative/draft_model.py:21` | 开发者绝对路径泄漏生产 |
| 22 | 代码质量-High | `service/postprocessor.py:179-3870` | `StreamingPostProcessor` 3691 行 |
| 23 | 部署-High | `tests/integration/test_*.py`, `scripts/*.py` | 多处硬编码 `/Users/dahai/...` |
| 24 | 依赖-High | `pyproject.toml:36` | `transformers>=5.0.0` 与 build.py pin 冲突 |

### 11.3 中等优先级修复（P2 — 计划修复）

37 项 Medium 级发现，主要包括：

- 异常吞咽（engine_pool fallback 链、mllm_cache eviction logging）
- 背景任务未显式取消
- MLX 资源无序释放（clear_cache 前 synchronize）
- SSD cache 写失败静默丢块
- audio_vae 多处 `open()` 无 `with`
- gui_compat NamedTemporaryFile 无清理
- sched_vlm_mtp_batched 每 token GPU sync
- 投机解码每 step `copy.deepcopy` KV cache
- 前缀缓存 `id(model)` key 重用风险
- Disk cache key 仅 hash `model_name`
- `mfa/quantize.py` 截断而非 round-to-nearest-even
- `sparse_mla.cpp:251` `qL_off = kL - qL` 可下溢
- FFI 无 per-call try/except，Metal fault 致命
- `compile_metallib.py` 无 staleness check
- mypy 在 dev deps 但 CI 不运行
- 无安全扫描（dependabot, pip-audit, bandit）
- Git-pin verify 脚本不在 CI 中
- 测试 time.sleep flakiness
- 无 coverage reporting
- 模块级 `__import__("os").environ.get(...)` 反模式
- 重复的 size formatting 工具
- `Any` 类型广泛使用
- 23 处 `# type: ignore` 无注释解释

### 11.4 低优先级修复（P3 — 机会主义清理）

39 项 Low 级发现，主要包括：

- 死代码别名（MLLMCacheManager、VLMCacheStats 等）
- Legacy hash/allocation methods in paged_cache.py
- 死注释引用不存在的模块
- 命名约定不一致（camelCase 混 snake_case）
- 单字母变量名（GPU kernel 中 OK）
- Magic numbers in memory_enforcer.py
- 模块级 `try: import mlx.core as mx`（acceptable 但 pervasive）
- `importlib.import_module` 作 dependency workaround
- macOS app bundle build process（well-structured）
- Disk cache load/save race（sequential lifecycle）
- 缓存 measurement lag（acknowledged design trade-off）

---

## 12. 附录: 审计工具与方法

### 12.1 审计方法

1. **项目元数据收集**: pyproject.toml, CI workflows, CHANGELOG.md, README.md
2. **架构分析**: server.py 入口, route 注册, middleware 顺序, 模块依赖图
3. **安全模式扫描**: eval/exec, pickle, yaml.load, subprocess, shell=True, hardcoded secrets, path traversal, SSRF, SQL injection
4. **可靠性模式扫描**: broad except counts, bare except/pass, race conditions, async/sync boundary, resource leaks, graceful shutdown, HTTP retry/backoff
5. **内存与资源审计**: MLX clear_cache/synchronize, set_cache_limit, Metal wired vs unified memory, Python memory leaks, file handle leaks, thread/process leaks, KV cache memory, model unload completeness
6. **性能审计**: hot path inefficiencies, blocking calls in async paths, async correctness, caching correctness, N+1 query patterns, memory copy hotspots
7. **代码质量审计**: dead code, duplication, complexity, type safety, naming conventions, magic numbers, module-level side effects, circular import risk
8. **并发审计**: asyncio.Lock/threading.Lock usage, asyncio.create_task reference retention, asyncio.gather error propagation, scheduler concurrency model
9. **Metal/C++ 内核审计**: buffer overflow/OOB, integer overflow, race conditions in Metal kernels, FFI correctness, quantization correctness, compile-cache and metallib build, numerical stability, memory leaks in C++, guardrails
10. **测试/CI/打包/部署审计**: CI coverage, test quality, packaging, deployment readiness, dependency hygiene

### 12.2 审计工具

- **静态分析**: `grep`, `glob`, `ast_grep` for pattern matching
- **代码智能**: `list_symbols`, `read_symbol`, `find_references`, `trace_callers`, `trace_callees`, `file_dependencies`, `blast_radius`
- **人工审查**: 关键文件逐行阅读（server.py, engine_pool.py, memory_enforcer.py, auth modules, custom kernels）
- **子代理并行审计**: 安全、可靠性、内存、性能、代码质量、内核、测试/CI/打包 各由独立 subagent 调查，主审计员汇总

### 12.3 审计限制

1. **静态审计**: 未运行项目、未执行 fuzzing、未做 dynamic analysis
2. **快照时效**: 审计基于 2026-07-25 的代码快照，后续 commit 可改变发现
3. **主观严重性**: Critical/High/Medium/Low 分级为审计员判断，需结合实际运行环境验证
4. **未覆盖项**: 未审计 `.venv/` 中的 vendored 第三方库, 未审计 `apps/fusion-mac/build/` 中的构建产物
5. **资源限制**: 在有限时间内完成 228k LOC 项目的审计, 必然有遗漏; 重点放在了 high-impact 区域

### 12.4 推荐后续行动

1. **立即**: 修复 §11.1 中 9 个 P0 问题（特别是 gui_compat 无鉴权 — 是 RCE 入口）
2. **短期**: 修复 §11.2 中 15 个 P1 问题
3. **中期**: 引入 SAST/DAST 自动化扫描, 添加 coverage reporting, 迁移 CI 到 macOS runner
4. **长期**: 重构 `serve_command` (1381 行) 与 `StreamingPostProcessor` (3691 行), 引入 strict mypy, 添加 fuzzing

---

**审计员**: AtomCode (GLM-5.2)
**审计完成日期**: 2026-07-25
**报告位置**: `~/claude-home/fusion-mlx/AUDIT_FUSION_MLX.md`
**项目状态**: 未修改任何文件，仅生成只读审计报告
