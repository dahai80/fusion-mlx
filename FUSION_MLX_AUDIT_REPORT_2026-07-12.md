# Fusion-MLX 全面审计报告

**日期:** 2026-07-12  
**审计范围:** fusion_mlx (Python 后端) + apps/fusion-mac (Swift 原生 macOS 应用)  
**覆盖:** 技术架构、安全、可靠性、内存安全与泄漏  
**Git HEAD:** `9be09a6` — `fix/code-review-findings` 分支

---

## 目录

1. [技术架构审计](#1-技术架构审计)
2. [安全审计](#2-安全审计)
3. [可靠性审计](#3-可靠性审计)
4. [内存安全与泄漏审计](#4-内存安全与泄漏审计)
5. [综合风险评估](#5-综合风险评估)
6. [Mac App 专项审计报告（见附录A）](#附录a-mac-app-专项审计报告)

---

## 1. 技术架构审计

### 1.1 总体架构

fusion-mlx 是一个运行在 Apple Silicon 上的多模态推理服务器，提供 OpenAI/Anthropic 兼容的 API。架构分为七层：

| 层 | 路径 | 职责 | 规模 |
|---|---|---|---|
| API 路由 | `fusion_mlx/api/` | HTTP 请求解析、Pydantic 校验、响应格式化 | ~1200 文件 |
| Adapter 适配 | `fusion_mlx/api/adapters/` | OpenAI ↔ Anthropic ↔ 内部格式互转 | 4 文件 |
| 路由分发 | `fusion_mlx/router/` | 按模态调度、phase-aware 路由、云回退 | 中等 |
| 引擎池 | `fusion_mlx/pool/` | LRU 引擎生命周期管理、内存执行器 | 10 文件 |
| 缓存 | `fusion_mlx/cache/` | 三级 KV Cache（GPU→SSD→磁盘） | 16 文件 |
| 调度器 | `fusion_mlx/scheduler/` | 25 模块连续批处理调度器 | 25 文件 |
| 推测解码 | `fusion_mlx/speculative/` | 4 种加速方法 | 多文件 |

### 1.2 架构评分: 8.5/10

**优势:**

- **模块化程度极高** — 调度器拆分为 25 个专注模块（~400 行/模块），职责清晰
- **设计模式得当** — Adapter 模式、策略模式（路由）、工厂模式（缓存）运用合理
- **类型系统完整** — Pydantic v2 模型覆盖所有请求/响应，`populate_by_name` 支持 snake_case + camelCase 双输入
- **中间件链完整** — 请求 ID、Body 大小限制、Body 深度限制、异常处理、健康探针快速路径
- **线程模型清晰** — 类型化执行器池（LLM/Image/Audio/IO）隔离不同类型负载

**问题:**

1. **`F403` 星号导入泛滥** — `core.py` 从 20+ 子模块 `from .sched_X import *`，导致命名空间污染和 IDE 索引困难
2. **模块间循环引用风险** — `scheduler/core.py` 作为枢纽汇聚所有子模块，`sched_admission.py` 引用 `sched_vlm_mtp.py` → `VLMMTPDrafter`，创建了间接耦合
3. **`_cli_base.py` 中的延迟导入过多** — 在函数体内部延迟导入 20+ 模块，隐藏了导入错误并增加了冷启动延迟
4. **旧版路由（`fusion_mlx/routes/`）与新版 API（`fusion_mlx/api/`）并存** — 两个路由体系同时存在，存在功能重叠和潜在不一致
5. **`fusion_gui/` 仅作为兼容层** — 与 `fusion_mlx` 耦合高，但 `server.py` 中 `try/except ImportError` 处理优雅降级

### 1.3 依赖架构

核心依赖 40+ 项，包括：
- `mlx` + `mlx-lm` + `mlx-vlm` + `mlx-embeddings` + `mlx-audio` — Apple MLX 全家桶
- `transformers>=5.0` — 最新版 transformers
- `fastapi` + `uvicorn` + `pydantic` — 异步 Web 框架
- `huggingface-hub` — 模型下载

**值得注意:** 依赖全部指向 git commit hash（非发布版本），存在供应链风险。

---

## 2. 安全审计

### 2.1 评分: 7.0/10

### 2.2 认证与授权

**API 认证机制（`middleware/auth.py` + `admin/auth.py`）:**

- **双重认证体系** — 公共 API 和 Admin 面板使用不同的认证模块，但逻辑有重叠
- **Bearer Token** — `middleware/auth.py` 用 `HTTPBearer` 验证，Admin 用 `verify_api_key`
- **Session Cookie** — Admin 登录后设置 `httponly` + `samesite="lax"` cookie
- **Sub Key 机制** — 支持子密钥，仅存储 SHA-256 哈希，主密钥不能同时作为子密钥
- **Query 参数认证** — `admin/auth.py` 允许 `?key=` 或 `?api_key=` 查询参数认证（L144-150），**会在服务器日志和浏览器历史中泄露密钥**

**问题发现:**

| # | 严重性 | 描述 | 位置 |
|---|---|---|---|
| S1 | **高** | API 密钥认证默认为空密码策略 — 当 `_global_settings` 返回 None 或 `api_key` 为空时，`_verify_api_key_values` 对任何请求都返回 True（L192-198），且只在首次记录 debug 日志 | `middleware/auth.py:189-206` |
| S2 | **中** | Auto-login GET 端点将 API 密钥暴露在 URL 查询参数中 — `/admin/auto-login?key=<api_key>&redirect=...`，浏览器历史、代理日志、Referer 头都会泄露密钥 | `auth_routes.py:227-259` |
| S3 | **中** | Admin require_admin 允许通过 `?key=` 查询参数传递密钥，与 S2 同理泄露 | `admin/auth.py:144-150` |
| S4 | **低** | Session 存储在进程内字典 `_active_sessions` 中，无持久化，服务器重启后所有会话失效 | `admin/auth.py:18` |
| S5 | **低** | Sub Key 创建时主密钥以明文传输（`request.key`），但存储使用 SHA-256 哈希；网络传输需 HTTPS 保护 | `admin/subkey.py:47` |

### 2.3 输入验证

- **Pydantic v2 模型** — 所有 API 端点使用 Pydantic 模型做输入验证，类型安全
- **Body 大小限制** — `RequestBodyLimitMiddleware` 防止大请求攻击
- **Body 深度限制** — `RequestBodyDepthMiddleware` 防止嵌套深度攻击
- **异常处理中间件** — `install_exception_handlers` 捕获所有异常并返回 JSON 响应，防止信息泄露
- **CORS** — 已配置 `CORSMiddleware`

### 2.4 子进程安全

- **Mac App 子进程管理** — `ServerProcess.swift` 正确管理 `Process` 生命周期，SIGTERM→SIGKILL 安全终止
- **信号处理** — `SignalHandlers.swift` 安装 SIGTERM/SIGINT/SIGHUP/SIGQUIT 处理器，确保子进程被回收
- **Python 运行时路径** — `PythonRuntime.swift` 通过硬编码路径搜索，防止 PATH 注入
- **环境变量传递** — `makeEnvironment()` 基于 `ProcessInfo.processInfo.environment`，无注入机会

### 2.5 文件路径安全

- **`_tempfile_safe.py`** — 安全临时文件管理，`atexit` 注册清理，线程安全
- **`ShellEnvWriter.swift`** — shell 配置文件写入使用 `shellQuote()` 转义，防止注入
- **模型发现** — `model_discovery.py` 扫描目录，但未验证路径是否是符号链接（潜在 TOCTOU 问题）

### 2.6 网络暴露

- **CORS 配置** — 已配置，未看到具体来源白名单
- **Unix Domain Socket** — `AppControlServer.swift` 使用 `AF_UNIX` 套接字，权限 `0o700`，仅本地进程可访问，设计安全
- **控制服务端口** — `AppControlServer` 监听 Unix Socket（默认 `control.sock`），不暴露 TCP 端口

---

## 3. 可靠性审计

### 3.1 评分: 8.0/10

### 3.2 错误处理

| 组件 | 错误处理策略 | 评价 |
|---|---|---|
| API 路由 | 结构化异常处理，映射到 HTTP 状态码 | ✅ 优秀 |
| 异常处理中间件 | 捕获 `HTTPException`、`ValidationError`、`JSONDecodeError`、`RecursionError`、通用 `Exception` | ✅ 全面 |
| 调度器 | GPU OOM 预检、stale request 恢复、GPU 竞争检测 | ✅ 高级 |
| 引擎池 | `InsufficientMemoryError`、`ModelLoadingError`、`ModelNotFoundError` | ✅ 完善 |
| 缓存 | 格式校验、版本号 `_CACHE_FORMAT_VERSION` | ✅ 谨慎 |

**问题发现:**

| # | 严重性 | 描述 | 位置 |
|---|---|---|---|
| R1 | **中** | 调度器 `sched_step.py` 中 `_process_batch_responses` 的 `uid_to_request_id` 查找失败时静默跳过（`continue`），可能导致请求丢失 | `sched_response.py:98-100` |
| R2 | **中** | `EnginePool._current_ceiling()` 中 `try/except BLE001` 捕获所有异常返回 0，可能掩盖编程错误 | `engine_pool.py:197-200` |
| R3 | **低** | 多处 `except Exception` 未指定具体异常类型，可能掩盖错误 | 散见各处 |
| R4 | **低** | `managed_tempfile_path` 的异常处理路径中存在双重 `_ensure_atexit_registered` 调用 | `_tempfile_safe.py:92-110` |

### 3.3 重试与超时

| 组件 | 配置 | 评价 |
|---|---|---|
| `hf_downloader.py` | `_HF_API_TIMEOUT=10s`、`_STALL_TIMEOUT=300s` | ✅ 合理 |
| `ServerProcess` | 健康检查 5s 间隔、3 次失败→unresponsive、自动重启 5/10/20s 退避、最多 3 次 | ✅ 优秀 |
| `PortConflictResolver` | 连接超时 1s、健康检查 5s | ✅ 合理 |
| `AppControlServer` | 命令等待 30s 超时 | ✅ 合理 |
| `ProcessMemoryEnforcer` | 锁获取 2s 超时，防死锁 | ✅ 优秀 |

### 3.4 资源清理

- **`_tempfile_safe.py`** — `managed_tempfile_path` 使用 `atexit` + `finally` 双重保障
- **`PagedSSDCache`** — 版本号校验（v2/v3），兼容格式
- **`AppControlServer.deinit`** — 调用 `stop()` 清理 Unix socket
- **`ServerProcess.stop()`** — SIGTERM→SIGKILL 链，10s 优雅关闭
- **`EnginePool`** — `__aexit__` 未实现，但 `stop()` 方法存在

---

## 4. 内存安全与泄漏审计

### 4.1 评分: 7.5/10

### 4.2 Python 内存管理

**ProcessMemoryEnforcer（`pool/memory_enforcer.py`）— 核心内存看门狗:**

- 4 级内存保护（Safe/Balanced/Aggressive/Custom）
- 双重 `gc.collect()` 模式（`collect()` 前后各一次）
- Metal 内存限制应用（`_apply_metal_wired_limit`）
- 热缓存预算控制（`SharedHotCacheBudget`）
- 预填充内存预检（`_preflight_memory_check`）
- 紧急压力下的请求中止

**GPU OOM 预检防护:**

- 调度前估计内存需求（模型权重 + KV Cache + 激活张量）
- 如果超过可用 Metal 内存则拒绝准入
- 防止 Metal GPU OOM 崩溃

### 4.3 潜在内存泄漏分析

| # | 严重性 | 描述 | 位置 |
|---|---|---|---|
| M1 | **中** | `_active_sessions` 字典不限定大小，理论上可无限增长（但过期条目在 `verify_session` 时删除，实际风险低） | `admin/auth.py:18` |
| M2 | **中** | `RateLimiter._requests` 字典使用 HMAC 键，cleanup 只清理非活跃条目，在大量唯一客户端下可能膨胀 | `middleware/auth.py:34` |
| M3 | **低** | `PagedCacheManager` 中 `BlockHashToBlockMap` 使用 SHA-256 哈希键，块数量受限于 `max_blocks`，风险低 | `cache/paged_cache.py` |
| M4 | **低** | `PagedSSDCache` 的 `_pending_writes` 队列受 `max_pending_writes` 限制，不会无限增长 | `cache/paged_ssd_cache.py` |
| M5 | **低** | `EnginePool._entries` 对应 `_current_model_memory` 的双重记账，但 `actual_size` 更新可能滞后 | `engine_pool.py` |
| M6 | **低** | `mlx.core.clear_cache()` 的 `gc.collect()` 双重调用模式是正确的最佳实践 | 分散各处 |

### 4.4 Swift 端内存管理

- **ARC 自动引用计数** — Swift 标准内存管理，未发现循环引用
- `ServerProcess` 使用 `[weak self]` 在闭包中避免循环引用 ✅
- `FusionClient` 使用 `@MainActor` + `@Observable`，无强引用环 ✅
- `AppControlServer` 使用 `weak var handler` 防止委托循环 ✅
- **`SignalHandlers` 中的 `atexit` 闭包** — 使用 `static func` 引用共享单例，无泄漏风险 ✅

### 4.5 缓存层内存风险评估

**三层缓存架构:**
1. **PagedCache（GPU）** — 固定最大块数（默认 1000），LRU 驱逐 → 有界
2. **PagedSSDCache（SSD）** — 默认 20GB 容量，使用 `SharedHotCacheBudget` 管理 → 有界
3. **BlockAwarePrefixCache** — COW 语义共享前缀块 → 参考计数安全

**结论:** 缓存层设计有界，LRU + 预算控制防止无限增长。

---

## 5. 综合风险评估

### 5.1 风险矩阵

| 领域 | 评分 | 关键风险 | 优先级 |
|---|---|---|---|
| 技术架构 | 8.5/10 | 模块耦合（F403 星号导入）、新旧路由并存 | 低 |
| 安全 | 7.0/10 | API 密钥空密码策略、Auto-login GET 暴露密钥 | **高** |
| 可靠性 | 8.0/10 | 静默跳过请求、宽泛异常捕获 | 中 |
| 内存安全 | 7.5/10 | 会话字典无限增长、速率限制器字典膨胀 | 中 |

### 5.2 高优先级修复建议

1. **【安全】空 API 密钥策略** — `middleware/auth.py:192-198` 中当未配置 API 密钥时，对所有请求返回 True。应改为：当 `--api-key` 未设置但请求中包含 Bearer token 时，返回 401。当前行为意味着"无密钥 = 所有人通过"，而非"无密钥 = 无需认证"。

2. **【安全】Auto-login GET 端点** — `auth_routes.py:227-259` 的 `auto_login_get` 端点通过 query param 接受 API 密钥。应移除 GET 变体，仅保留 POST 变体，且密钥通过 POST body 传递。

3. **【安全】Query 参数认证** — `admin/auth.py:144-150` 允许通过 URL 查询参数传递密钥，应在生产环境禁用。

4. **【可靠性】静默跳过请求** — `sched_response.py:98-100` 中 `uid_to_request_id` 查找失败时仅 `continue`，应记录日志并考虑恢复机制。

5. **【内存】会话管理** — `admin/auth.py:18` 的 `_active_sessions` 字典应添加最大大小限制或 LRU 驱逐。

### 5.3 中优先级建议

1. 移除 `F403` 星号导入，改为显式导入
2. 统一 `routes/` 和 `api/` 两个路由体系
3. 为 `RateLimiter._requests` 添加最大条目限制
4. 在 `EnginePool` 中实现 `__aexit__` 以支持 `async with`
5. 锁定依赖版本（从 git commit hash 改为发布版本）

---

## 附录A: Mac App 专项审计报告

### 审计范围

**目标:** `apps/fusion-mac/` — Swift 原生 macOS 应用  
**技术栈:** Swift 6 + SwiftUI + AppKit + Foundation  
**代码行数:** ~3,500 行 Swift  
**构建工具:** Xcode 项目 (`FusionMLX.xcodeproj`)

### A.1 技术架构评分: 9.0/10

**架构:**

```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│  AppDelegate  │────>│ ServerProcess│────>│  Python      │
│  (NSApp)      │     │  (Process)   │     │  fusion_mlx  │
│              │     │              │     │  (子进程)     │
│  ┌──────────┐│     │  健康检查     │     └──────────────┘
│  │Menubar   ││     │  自动重启     │
│  │Controller││     │  端口冲突     │
│  └──────────┘│     │  解决         │
│              │     └──────────────┘
│  ┌──────────┐│     ┌──────────────┐
│  │AppView   ││────>│ FusionClient │────> HTTP /admin/api/*
│  │(SwiftUI) ││     │  (URLSession)│
│  └──────────┘│     └──────────────┘
│              │     ┌──────────────┐
│  ┌──────────┐│     │ AppControl   │
│  │Welcome   ││     │ Server       │
│  │Wizard    ││     │ (Unix Socket)│
│  └──────────┘│     └──────────────┘
└──────────────┘
```

**架构优势:**

- **MVVM 清晰** — `AppView/ViewModels/` 层将数据获取逻辑从视图分离
- **进程隔离** — Python 推理服务器作为独立子进程运行，Swift 应用通过 HTTP 通信
- **状态机驱动** — `ServerProcess` 使用 `State` 枚举（stopped/starting/running/stopping/unresponsive/failed）
- **无阻塞 UI** — 健康检查使用 `async/await`，UI 永远不阻塞
- **Unix Socket 控制** — `AppControlServer` 使用本地 socket 通信，安全且高效
- **Dock 图标智能切换** — 窗口打开时 `.regular`，关闭时 `.accessory`，优雅

### A.2 安全评分: 8.5/10

**优势:**

- ✅ `AppControlServer` 使用 Unix Domain Socket（`S_IRUSR \| S_IWUSR` 权限），仅当前用户可访问
- ✅ `ShellEnvWriter` 对 shell 文件写入使用 `shellQuote()` 转义
- ✅ `FusionClient` 使用 `URLSession` 的 cookie 管理，自动处理 session
- ✅ `SignalHandlers` 正确安装 POSIX 信号处理器，确保子进程被回收
- ✅ `PythonRuntime` 路径硬编码，无 PATH 注入
- ✅ `ServerProcess.reconfigure()` 在服务器运行时禁止修改配置

**问题发现:**

| # | 严重性 | 描述 | 位置 |
|---|---|---|---|
| AS1 | **中** | `FusionClient` 的 `login()` 方法在 401 时自动重试，将 API 密钥存储在内存中，一旦内存转储即可获取 | `FusionClient.swift:561-581` |
| AS2 | **低** | `AppControlServer.handle()` 中 `readRequest` 有 65536 字节上限，但无速率限制，可被本地进程洪泛 | `AppControlServer.swift:217-231` |
| AS3 | **低** | `PortConflictResolver.findOwnerPIDSync()` 调用 `lsof` 子进程，但未验证子进程执行路径 | `PortConflictResolver.swift:73-97` |

### A.3 可靠性评分: 9.0/10

**优势:**

- ✅ **自动重启机制** — 5/10/20s 指数退避，最多 3 次，60s 稳定期后重置计数器
- ✅ **健康检查循环** — 5s 间隔，3 次失败标记为 unresponsive，辅助健康检查（`/api/status`）防止误判
- ✅ **端口冲突检测** — `sync` + `async` 双重检测，支持 `lsof` 查找占用进程
- ✅ **优雅关闭** — 10s SIGTERM 等待，超时后 SIGKILL
- ✅ **Force Restart** — 立即 SIGKILL + 重新启动
- ✅ **`@unchecked Sendable`** — 正确标注线程安全边界
- ✅ **`[weak self]`** — 所有闭包中正确使用 weak 引用

**问题发现:**

| # | 严重性 | 描述 | 位置 |
|---|---|---|---|
| AR1 | **低** | `healthCheckInterval` 固定 5s，不支持动态调整 | `ServerProcess.swift:105` |
| AR2 | **低** | `maxAutoRestarts = 3` 硬编码，不支持运行时修改 | `ServerProcess.swift:108` |

### A.4 内存安全评分: 8.5/10

**Swift 内存管理（ARC）：**

| 组件 | 内存管理 | 风险 |
|---|---|---|
| `ServerProcess` | `process`、`logHandle`、`healthTask` 属性在 `stop()` 中置为 nil | ✅ 安全 |
| `MenubarController` | `statsPoller` 在 `start()`/`stop()` 中管理 | ✅ 安全 |
| `FusionClient` | `URLSession` 共享，`apiKey` 字符串在内存中 | ✅ 安全（但 AS1 相关） |
| `AppControlServer` | `deinit` 调用 `stop()` 清理 socket | ✅ 安全 |
| `SignalHandlers` | `sources` 数组在 `install()` 中取消旧源 | ✅ 安全 |

**循环引用检查:**

- `AppDelegate` → `[weak self]` 在闭包中 ✅
- `ServerProcess` → `[weak self]` 在 `terminationHandler` 中 ✅
- `AppControlServer` → `weak var handler` ✅
- `MenubarController` → 无 delegate 循环 ✅

**问题发现:**

| # | 严重性 | 描述 | 位置 |
|---|---|---|---|
| AM1 | **低** | `PortConflictResolver` 中 `SendableBox` 使用 `@unchecked Sendable` 绕过检查，但使用方式安全 | `PortConflictResolver.swift:173-176` |
| AM2 | **低** | `AppControlServer` 中 `ResponseBox` 使用 `@unchecked Sendable`，同上 | `AppControlServer.swift:258-260` |

### A.5 Mac App 综合评分: 8.8/10

**总结:**
Mac App 部分代码质量高，架构清晰，安全实践良好。主要优势在于清晰的进程管理状态机、完善的子进程生命周期管理、以及优秀的 Swift 并发实践。主要问题集中在 API 密钥在内存中的存储和自动重试机制。

**高优先级修复:**
1. 无

**中优先级建议:**
1. `PortConflictResolver.findOwnerPIDSync()` 使用完整路径 `/usr/sbin/lsof` 验证
2. `FusionClient` 中 API 密钥使用 `String` 的敏感数据擦除（`defer { key.withCString { memset(...) } }`）
3. 为 `AppControlServer` 添加本地速率限制

---

*报告生成: 2026-07-12 | 审计工具: 手动代码审查 + 静态分析*