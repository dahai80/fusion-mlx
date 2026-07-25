# fusion-mlx 多维度量化评分

**评分日期**: 2026-07-25
**评分对象**: fusion-mlx（659 个 Python 文件，~228k LOC，21 个 Metal/C++ 内核文件，707 个测试文件）
**评分依据**: 基于 `AUDIT_FUSION_MLX.md` 的审计发现，并经主审计员对关键事实的二次核查
**评分标准**: 每个维度 0–10 分，10 分为工业级最佳实践，5 分为可用但需改进，3 分以下为严重风险

---

## 综合评分卡

| 维度 | 评分 | 等级 | 简评 |
|---|---|---|---|
| 安全 | **3.0/10** | 🔴 高风险 | gui_compat 路由完全无鉴权 = RCE 入口；无安全扫描；MCP 工具调用记录无界增长 |
| 可靠性 & 错误处理 | **4.5/10** | 🟠 中高风险 | 关键路径静默吞咽 Metal 错误致 silent OOM；shutdown 不排空在途请求；无 HTTP 重试 |
| 内存 & 资源管理 | **7.0/10** | 🟢 良好 | 模型 unload 路径完整；KV cache 有界 + LRU；但全局 executors 不 shutdown |
| 性能 | **6.5/10** | 🟢 中等 | 调度器并发模型正确；主路径无 numpy round-trip；但 O(n²) 扫描 + 多处 GPU sync |
| 代码质量 | **5.0/10** | 🟠 中等 | `serve_command` 1382 行无法维护；广泛使用 `Any`；30+ 内联 import |
| 并发正确性 | **5.5/10** | 🟠 中等 | engine_pool 有 `_lock` 但同步方法绕过；memory_enforcer 自身无锁 |
| Metal/C++ 内核 | **5.5/10** | 🟠 中等 | Steel Attention 设计精良；但 `load_vector` OOB、threadgroup union 溢出、`-ffast-math` |
| 测试覆盖 | **5.0/10** | 🟠 中等 | 测试/源码比 ~1.05:1 良好；但无 coverage 报告、无安全扫描、CI 平台错误 |
| CI/CD 工程化 | **3.5/10** | 🔴 高风险 | CI 运行在 ubuntu（项目仅支持 macOS）；无 mypy/安全扫描/coverage；开发者绝对路径 |
| 文档完整性 | **6.0/10** | 🟢 中等 | README 详尽、CHANGELOG 规范；但关键设计决策散落代码注释中 |

**加权综合评分**: **5.2 / 10** — 🟠 **可用但存在显著风险，需在投入生产前优先修复 P0/P1 问题**

---

## 维度 1: 安全 — 3.0/10 🔴

### 扣分项

| 严重性 | 发现 | 扣分 |
|---|---|---|
| 🔴 Critical | `gui_compat/server.py` 所有路由完全无鉴权（`validate_api_key` 定义但从未接入 `Depends()`），任意网络攻击者可 load/unload/delete 模型、改 settings、**未鉴权 shutdown/restart 服务器**，结合未鉴权 `install` 端点 = RCE 入口 | -3.5 |
| 🟡 Medium | `admin/auth_routes.py:224-257` GET `/admin/auto-login` 接受 `?key=<api_key>`，key 泄漏至浏览器历史、Referer、access logs | -0.5 |
| 🟡 Medium | `gui_compat/server.py:480-494` 未鉴权 `DELETE /v1/models/{name}` 调用 `shutil.rmtree(mr.path)`，配合未鉴权 install = 任意目录删除 | -0.5 |
| 🟡 Medium | `_url_safety.py` `is_safe_url()` 无 DNS 解析，可被指向解析到私有 IP 的 hostname 绕过（`is_safe_url_with_dns` 用于下载路径） | -0.3 |
| 🟢 Low | 无 `.github/dependabot.yml`，无 `bandit`/`pip-audit` 安全扫描 | -0.3 |
| 🟢 Low | Rate limiter 仅对 API 路由生效，admin/gui_compat 路由无速率限制 | -0.2 |

### 加分项

| 发现 | 加分 |
|---|---|
| ✅ CORS 配置正确：wildcard 时 `allow_credentials=False`，符合浏览器规范 | +0.5 |
| ✅ Agent graph 代码注入修复（#109）完整：`json.dumps(config)` 嵌入 + temperature clamp | +0.3 |
| ✅ 所有 `torch.load` 使用 `weights_only=True`（5 处） | +0.3 |
| ✅ 无 `pickle.load`、无 `yaml.load` 不安全调用、无 `shell=True` | +0.3 |
| ✅ Publish workflow 使用 OIDC trusted publishing，无硬编码 token | +0.4 |
| ✅ SSRF 部分缓解：`is_safe_url_with_dns()` 用于下载路径，含 `_BLOCKED_HOSTNAMES`（cloud metadata） | +0.3 |
| ✅ API key 比较使用 `secrets.compare_digest`（常数时间） | +0.2 |
| ✅ SQL 全部使用 SQLAlchemy ORM 参数化查询 | +0.3 |

### 评语

**核心问题是 gui_compat 路由完全无鉴权**——这是一个完整的 RCE 入口（shutdown 服务器、删除任意模型文件、安装恶意模型）。审计员核查确认：`gui_compat/server.py` 中所有路由 handler 均无 `Depends(require_admin)` 或 `Depends(verify_api_key)`，`validate_api_key` 函数被定义但从未使用。考虑到项目默认 listen `0.0.0.0:8080`（可被 LAN/反向代理暴露），此风险的实际可利用性很高。

---

## 维度 2: 可靠性 & 错误处理 — 4.5/10 🟠

### 扣分项

| 严重性 | 发现 | 扣分 |
|---|---|---|
| 🔴 Critical | `pool/engine_pool.py:1099,1122` 静默 `except Exception: pass` 吞掉 Metal 内存回收失败（gc.collect/mx.synchronize/mx.clear_cache），导致 pool 失去追踪能力 → 下次 load 的 **silent OOM** | -1.5 |
| 🔴 Critical | `server.py _shutdown()` (922-977) 不排空在途请求，engines 被 abruptly stop，**部分已 commit 的输出丢失**，客户端看到截断的 SSE 流 | -1.0 |
| 🟠 High | `cache/prefix_cache.py:801,830` seq_len 解析失败静默 `continue`，可导致 **静默返回错误 sequence length，重建时 cache corruption** | -0.8 |
| 🟠 High | `api/openai_routes.py:441-485` mid-stream engine 崩溃（Metal OOM）→ truncated/incomplete SSE stream，**无 `[DONE]` 或 error event**，客户端看到 abrupt connection drop | -0.5 |
| 🟠 High | `admin/hf_downloader.py:778-892` 无 HTTP 自动重试/退避，transient 失败立即失败 | -0.4 |
| 🟡 Medium | 30+ 处 broad `except Exception`（oq.py 30, service/helpers.py 24, engine_pool.py 24, server.py 22） | -0.3 |
| 🟡 Medium | 背景任务未显式取消（telemetry、downloader pollers、LRU eviction、enforcer loop），依赖 event loop shutdown | -0.2 |

### 加分项

| 发现 | 加分 |
|---|---|
| ✅ 大部分 broad except 是 logged 或 re-raised（仅少数是 `pass`） | +0.4 |
| ✅ `engine_pool.py` 的 settle barrier（最多 10 轮 gc+sync+clear_cache 轮询）确保 Metal 内存实际回收 | +0.5 |
| ✅ `_tempfile_safe.py` 使用 `threading.Lock` + atexit + try/finally 正确实现 | +0.3 |

### 评语

可靠性问题集中在 **"静默吞咽关键错误"** 和 **"graceful shutdown 不完整"** 两点。前者会导致最难排查的 silent OOM，后者在生产环境强制重启时丢失用户请求。两者都是工业级生产系统不可接受的。

---

## 维度 3: 内存 & 资源管理 — 7.0/10 🟢

### 扣分项

| 严重性 | 发现 | 扣分 |
|---|---|---|
| 🟠 High | `engine_core.py:66,104-116` 全局 `_global_executors`（llm/audio/io）从不显式 shutdown，3-5 个非 daemon 线程持续到进程退出，可能在 atexit 阶段触发 `RuntimeError: Event loop is closed` | -0.8 |
| 🟠 High | `mcp/security.py:436` `_call_times: defaultdict(list)` 每个 tool invocation append 一个 timestamp，**无 eviction**，持续 tool calling 下无界增长 = 内存泄漏 | -0.6 |
| 🟡 Medium | `pool/engine_pool.py:1993` `mx.set_cache_limit` 期间保持 ~2x 模型大小在 Metal buffer pool，补偿 `clear_cache` 路径若失败内存保持 inflated | -0.4 |
| 🟡 Medium | `cache/paged_ssd_cache.py:1607-1620` SSD 写失败静默丢块，下次 load 该 block hash 将 miss 触发 recomputation | -0.3 |
| 🟡 Medium | `video/ltx2/audio_vae/audio_vae.py:187,356` 多处 `json.load(open(...))` 无 `with`，文件句柄泄漏至 GC | -0.2 |
| 🟡 Medium | `gui_compat/server.py:279,305,325` `NamedTemporaryFile(delete=False)` 无 apparent cleanup mechanism | -0.2 |

### 加分项

| 发现 | 加分 |
|---|---|
| ✅ 模型 unload 路径 **comprehensive and well-engineered**：MLX weights / KV pages / tokenizer / compile cache / executor threads 全部释放 | +1.0 |
| ✅ `cache/paged_cache.py:576-582` 正确使用 `threading.RLock`，文档化的锁顺序 | +0.5 |
| ✅ KV cache 有界 + LRU + ref counting：`paged_cache.py:753` `free_block()` 回收 block 到 `FreeKVCacheBlockQueue` | +0.4 |
| ✅ `telemetry/queue.py:63` `deque(maxlen=100)` 有界，overflow 时丢弃最旧 events | +0.3 |
| ✅ `_tempfile_safe.py` + `utils/video.py` `TempFileManager` atexit 清理 | +0.3 |
| ✅ 所有 caches 有界 + LRU eviction（paged_cache, prefix_cache, paged_ssd_cache） | +0.4 |

### 评语

内存管理整体设计精良，**模型 unload 路径的完整性尤其值得称道**。主要短板是几个未界定的增长容器（MCP `_call_times`、全局 executors 不 shutdown）。

---

## 维度 4: 性能 — 6.5/10 🟢

### 扣分项

| 严重性 | 发现 | 扣分 |
|---|---|---|
| 🟠 High | `sched_schedule.py:84-91` O(n²) token budget 扫描，每次 `while self.waiting` 循环重新 summing `len(r.remaining_tokens)` over 所有已调度 requests | -0.8 |
| 🟠 High | 5 处 `asyncio.gather` 无 `return_exceptions=True`（eval/mbpp.py:206, eval/livecodebench.py:249, eval/base.py:263,284, eval/humaneval.py:258），一个失败 cancel 所有 | -0.5 |
| 🟡 Medium | `monkeypatches.py:324` 每 decode step `list(self.tokens)` 全量拷贝 | -0.3 |
| 🟡 Medium | `monkeypatches.py:328` grammar 路径每 step `mx.eval` 强制 GPU sync | -0.3 |
| 🟡 Medium | `sched_vlm_mtp_batched.py:241` 每 token `mx.eval(tok)` GPU sync | -0.3 |
| 🟡 Medium | `ngram_spec.py:278, spec_decode.py:137` 投机解码每 step `copy.deepcopy` KV cache | -0.3 |
| 🟡 Medium | `cache/prefix_cache.py:96` `id(model)` 作为 cache key，model 被 GC 后复用地址 = stale match | -0.2 |

### 加分项

| 发现 | 加分 |
|---|---|
| ✅ 调度器并发模型设计正确：单 `_mlx_executor` 线程，"guarantee that MLX GPU operations are never concurrent" | +0.8 |
| ✅ 主 LLM 生成路径原生使用 MLX arrays，无 numpy round-trip | +0.5 |
| ✅ `engine_core.py` 正确使用 `loop.run_in_executor()` 卸载 MLX 阻塞调用 | +0.4 |
| ✅ KV cache 分块 + ref counting + COW 设计精良 | +0.3 |

### 评语

性能问题集中在 **"hot loop 中的同步开销"**（O(n²) 扫描、每 token GPU sync、每 step deepcopy）和 **"批处理失败传播"**（`asyncio.gather` 无 `return_exceptions`）。这些不会导致功能错误，但会显著影响吞吐量和延迟。

---

## 维度 5: 代码质量 — 5.0/10 🟠

### 扣分项

| 严重性 | 发现 | 扣分 |
|---|---|---|
| 🔴 Critical | `cli_serve.py:serve_command`（892-2273）**1382 行单函数**，30+ 内联 import、12+ `sys.exit()`、5 层嵌套、5+ mutually exclusive mode forks，无法维护 | -1.5 |
| 🟠 High | `service/postprocessor.py:StreamingPostProcessor`（179-3870）**~3691 行单类**，4 个主处理方法、~50+ instance variables、深度嵌套 state machine | -0.8 |
| 🟠 High | 广泛使用 `Any` 类型（`cache/prefix_cache.py:15,83` `self.model: Any`、20+ tool parsers 用 `dict[str, Any]`、`middleware/` 全部用 `Any`） | -0.6 |
| 🟠 High | 23 处 `# type: ignore` 无注释解释（`utils/psutil_compat.py`, `_torch_stub.py`, `cli.py` 等） | -0.4 |
| 🟡 Medium | 重复实现：4 处 IEC byte formatting、3 个文件复制 `__import__("os").environ.get(...)` 反模式 | -0.3 |
| 🟡 Medium | Magic numbers：`memory_enforcer.py:54-55` 4GB/24GB 硬编码、`config.py:189-195` 阈值 0.1/0.85/0.95/0.99 无解释、`spec_decode.py:27-38` 投机解码参数 | -0.3 |
| 🟡 Medium | 命名约定不一致：`mistral_tool_parser.py` 混用 snake_case (`_stream_old_format`) 和 camelCase (`needs_name_emit`) | -0.2 |

### 加分项

| 发现 | 加分 |
|---|---|
| ✅ Tool parser 抽象良好：21 个 parser 都继承 `BaseToolParser` 遵循一致接口 | +0.5 |
| ✅ Cache 模块边界清晰：`interface.py`/`protocol.py`/`factory.py`/`paged_cache.py`/`prefix_cache.py` 分离 | +0.4 |
| ✅ Ruff 配置合理（per-file-ignores + 注释解释 false positives） | +0.3 |
| ✅ 模块级 `try: import mlx.core as mx` 模式 pervasive 但 acceptable | +0.2 |

### 评语

代码质量核心矛盾是 **"巨型函数/类与良好抽象并存"**。`serve_command`（1382 行）和 `StreamingPostProcessor`（3691 行）是两个极端例子——前者新增 launch mode 风险极高，后者单类承担过多职责。但同时，cache 层、tool parser 层的抽象设计是教科书级的。

---

## 维度 6: 并发正确性 — 5.5/10 🟠

### 扣分项

| 严重性 | 发现 | 扣分 |
|---|---|---|
| 🔴 Critical | `pool/memory_enforcer.py` 自身无锁（核查确认：文件内无 `threading.Lock` 或 `asyncio.Lock` 定义），`update_loaded_model_bytes(delta)` 中 `self._loaded_model_bytes += delta` 是 read-modify-write race，从 `engine_pool.py:1089,1126,1939` 多处调用 | -1.5 |
| 🟠 High | `pool/engine_pool.py:1047-1130` 同步方法 `register_engine`/`unload_engine` 修改 `_entries` 和 `_current_model_memory` **不持有 `self._lock`** | -0.8 |
| 🟠 High | `pool/engine_pool.py:337-349,491-497,499-514,656-660,475-485` 多处无锁读 `_entries`，可看到不一致快照 | -0.4 |
| 🟡 Medium | 5 处 `asyncio.gather` 无 `return_exceptions=True`（见维度 4） | -0.3 |

### 加分项

| 发现 | 加分 |
|---|---|
| ✅ `cache/paged_cache.py:576-582` 正确使用 `threading.RLock` + 文档化锁顺序 | +0.6 |
| ✅ `cache/prefix_cache.py:125` 单 `_cache_lock = asyncio.Lock()` 覆盖所有 cache 操作 | +0.4 |
| ✅ `engine_core._engine_loop` 单线程调度，"guarantee that MLX GPU operations are never concurrent" | +0.5 |
| ✅ memory_enforcer 通过 `engine_pool._lock` 协作（17 处 lock 引用），关键操作在 `_lock_held` 内执行 | +0.5 |

### 评语

**核查修正**：子代理报告 memory_enforcer "零锁" 不完全准确。memory_enforcer 文件本身确实没有定义自己的锁（1410 行无 `threading.Lock`/`asyncio.Lock` 实例），但它通过 `engine_pool._lock` 协作（line 1156-1157 `asyncio.wait_for(self._engine_pool._lock.acquire(), timeout=2.0)`）。**真正的问题是 `update_loaded_model_bytes(delta)` 这个高频计数器更新没有自身锁保护**，而它从 `engine_pool.py` 的同步路径（`register_engine`/`unload_engine`/`unload_engine_async`）被多处调用，存在真实的并发撕裂风险。

---

## 维度 7: Metal/C++ 内核 — 5.5/10 🟠

### 扣分项

| 严重性 | 发现 | 扣分 |
|---|---|---|
| 🔴 Critical | `glm_moe_dsa/csrc/kernels/quantized_glm.h:28-105` `load_vector` 对 `x[i+3]`/`x[i+7]` 无边界检查，`load_vector_safe` (line 108) 仍 OOB 当 `N % 4 != 0`，K-dim 循环最后一个 partial tile 可达 | -1.5 |
| 🟠 High | `moe_ffn_fused.metal:269-280` threadgroup union 内存溢出：Phase 1 总大小 4096 字节，Phase 2 `hidden` 4608 字节，**512 字节溢出 within the union**，潜在 clobber 其他 threadgroup 变量 | -0.8 |
| 🟠 High | `compile_metallib.py:74` `-ffast-math` 标志：disable NaN propagation、flush subnormals、enable unsafe FMA contraction。与 GLM DSA 内核（`-fno-fast-math`）不一致 | -0.6 |
| 🟡 Medium | `moe_ffn_fused.metal:118-124, w4a8_fused_matmul.metal:114-119` `load_unsafe()` 读 BM 行不条件检查，M 不是 BM 倍数时最后一个 threadgroup OOB | -0.4 |
| 🟡 Medium | `mfa/quantize.py:48,74,128` `astype(mx.uint8)` 截断 toward zero 而非 round-to-nearest-even（IEEE 754 要求），引入 -0.5 LSB 系统性 bias | -0.3 |
| 🟡 Medium | `sparse_mla.cpp:251` `qL_off = kL - qL` 可下溢（qL > kL 时为负），MLX shape validation 不强制 `kL >= qL` | -0.2 |
| 🟡 Medium | FFI 无 per-call try/except：如果 `d.get_kernel(kname, lib)` 失败，Metal fault mid-dispatch 导致 uncatchable MTLCrash | -0.3 |
| 🟡 Medium | `compile_metallib.py:39-115` 无 staleness check，每次重编译（~2s per kernel at import time） | -0.2 |

### 加分项

| 发现 | 加分 |
|---|---|
| ✅ Numerical stability 大体良好：`silu(x) = x/(1+exp(-x))` 对大正/负 x 安全、softmax 用 max-subtraction before exp | +0.5 |
| ✅ 无 C++ 内存泄漏：所有 allocations 通过 `allocator::malloc`（MLX-managed）或 nanobind stack variables，无 raw `new` 无 `delete` | +0.4 |
| ✅ `bind10gs.cpp` 使用 nanobind `NB_DOMAIN mlx`，正确启用 MLX stable ABI | +0.3 |
| ✅ Steel Attention 设计精良，与 MLX steel attention kernels 数值一致 | +0.3 |

### 评语

内核问题集中在 **"边界检查缺失"** 和 **"编译标志不一致"**。`load_vector` 的 OOB 是经典 GPU kernel bug，在生产环境高负载下可能触发 Metal fault（且由于 §维度 7 的 FFI 无 try/except，是致命的）。`-ffast-math` 与 `-fno-fast-math` 在同一项目共存，意味着不同 kernel 间的数值保证不一致。

---

## 维度 8: 测试覆盖 — 5.0/10 🟠

### 扣分项

| 严重性 | 发现 | 扣分 |
|---|---|---|
| 🔴 Critical | `.github/workflows/ci.yml:11,22` CI 运行在 `ubuntu-latest`，但项目仅支持 macOS（pyproject.toml classifier），所有 Darwin-only 测试被跳过，CI 给出虚假信心 | -1.5 |
| 🟠 High | ~50+ instances of `time.sleep(...)` in test files（test_paged_ssd_cache.py 15+、test_telemetry_queue.py 6+、test_vision_feature_cache.py 5+、test_cloud_router.py 3+），真实 sleep 使测试在 loaded CI 上慢且易 race-condition flakiness | -0.5 |
| 🟠 High | Random seed 使用不一致：有些用 `np.random.seed`、有些用 `mx.random.seed`、许多完全不用 seed，非确定性测试结果 | -0.4 |
| 🟡 Medium | ~80+ skipped tests 无跟踪机制（feature gaps ~40、platform-gated ~20、MLX availability ~15、optional deps ~10、refactored paths ~10） | -0.3 |
| 🟡 Medium | 无 coverage reporting：`pyproject.toml` 无 `[tool.coverage.*]` section、无 `pytest-cov`、CI 无 `--cov` flags，**Coverage 百分比未知** | -0.3 |
| 🟡 Medium | Integration tests 极少：`tests/integration/` 仅 3 文件，E2E 仅 1 文件（skip-gated by `FUSION_E2E_ADAPTERS=1`） | -0.2 |

### 加分项

| 发现 | 加分 |
|---|---|
| ✅ Test-to-source ratio ~1.05:1（679 测试文件 vs 659 源文件）— 良好的 coverage 广度 | +0.6 |
| ✅ `pyproject.toml:205` `asyncio_mode = "auto"` 配置正确 | +0.3 |
| ✅ `.gitignore` 覆盖完整，敏感文件被排除 | +0.2 |

### 评语

测试的核心问题是 **"CI 平台错误导致大量测试被静默跳过"**——这意味着即使代码引入 macOS-only 回归，CI 也不会捕获。叠加无 coverage reporting 和大量无跟踪的 skipped tests，测试套件的实际保障能力被严重高估。

---

## 维度 9: CI/CD 工程化 — 3.5/10 🔴

### 扣分项

| 严重性 | 发现 | 扣分 |
|---|---|---|
| 🔴 Critical | `pyproject.toml:87` + `packaging/venvstacks.toml:17` 含开发者绝对路径 `/Users/dahai/claude-home/fusion-mlx/...`，任何其他机器 `pip install fusion-mlx[image]` 或构建 macOS app 都会失败 | -1.5 |
| 🔴 Critical | CI 在 ubuntu（见维度 8） | -1.0 |
| 🟠 High | `pyproject.toml:36` `transformers>=5.0.0` 与 `packaging/build.py:48` `transformers==5.0.0` 冲突，build.py 注释明确说 5.13.0 破坏 mlx-lm import，**pip 用户获得 broken transformers version** | -0.6 |
| 🟡 Medium | mypy 在 dev deps 但 CI 不运行（`.github/workflows/ci.yml:17,18` 仅 ruff/black） | -0.4 |
| 🟡 Medium | 无安全扫描（无 `.github/dependabot.yml`、无 `pip-audit`/`bandit`） | -0.3 |
| 🟡 Medium | `scripts/verify_git_pins.sh` 存在但不在 CI 中调用，SHA drift 未检测 | -0.3 |

### 加分项

| 发现 | 加分 |
|---|---|
| ✅ Publish workflow 使用 OIDC trusted publishing（`permissions: id-token: write`），无硬编码 token | +0.5 |
| ✅ Git-pinned deps SHAs 已 pin（mlx-lm、mlx-embeddings、mlx-vlm、dflash-mlx、mlx-audio），注释说明 "the commit SHA IS the supply-chain integrity check" | +0.4 |
| ✅ `.gitignore` 覆盖完整 | +0.2 |

### 评语

CI/CD 的核心问题是 **"开发者机器特定配置泄漏到项目级配置"**——pyproject.toml 的绝对路径会让任何贡献者的 `pip install -e .[image]` 失败，venvstacks.toml 的绝对路径会让任何贡献者的 macOS app build 失败。这是开源协作的致命障碍。叠加 CI 平台错误、无安全扫描、无 mypy 执行，整体工程化水平偏低。

---

## 维度 10: 文档完整性 — 6.0/10 🟢

### 扣分项

| 严重性 | 发现 | 扣分 |
|---|---|---|
| 🟡 Medium | `model_auto_config.py:114-115` 死注释引用不存在的模块 `# TODO: fusion_mlx/reasoning/think_detector.py does not exist yet` | -0.2 |
| 🟡 Medium | 关键设计决策散落代码注释中（如 `engine_pool.py` settle barrier、`memory_enforcer.py` 4-tier 压力等级），缺乏集中的架构文档 | -0.4 |
| 🟡 Medium | 多处硬编码 `/Users/dahai/...` 在 scripts/ 和 integration tests 中，且 `apps/fusion-mac/MIGRATION.md:9-10` 文档引用开发者绝对路径 | -0.3 |
| 🟢 Low | `CHANGELOG.md` 中部分条目缺少 issue/PR 链接追溯 | -0.1 |

### 加分项

| 发现 | 加分 |
|---|---|
| ✅ README.md（44KB）和 README_CN.md（49KB）详尽，含安装、使用、API 示例 | +0.6 |
| ✅ CHANGELOG.md 规范，遵循 Keep a Changelog 格式，含 issue/PR 引用（如 `#109`、`#110`） | +0.5 |
| ✅ 代码注释质量高：关键路径有详细注释解释 "why" 而非 "what"（如 `engine_pool.py:1993` 的 `set_cache_limit` 解释） | +0.4 |
| ✅ `pyproject.toml` 含清晰的 `[tool.ruff.lint.per-file-ignores]` 解释（如 core.py star-import hub、sched_schedule.py B023 false positive） | +0.3 |

### 评语

文档质量整体良好，尤其是 README 详尽度和 CHANGELOG 规范性。主要短板是缺乏集中的架构决策记录（ADR），关键设计决策散落代码注释中，新贡献者难以快速理解整体设计意图。

---

## 总评与推荐行动

### 综合评分: **5.2 / 10** — 🟠 可用但存在显著风险

fusion-mlx 是一个 **功能完整、设计精良、但工程化不足** 的项目。它的核心推理引擎、KV cache 分页、模型池管理是工业级的；但安全边界、CI 工程化、代码可维护性存在明显短板。

### 推荐优先级

| 优先级 | 行动 | 影响维度 |
|---|---|---|
| **P0 立即修复** | 1. 为 `gui_compat/server.py` 所有路由接入 `Depends(require_admin)` 或 `Depends(verify_api_key)` | 安全 → 8.0 |
| | 2. 替换 `pyproject.toml:87` + `venvstacks.toml:17` 的 `file:///Users/dahai/...` 为相对路径或 PyPI 依赖 | CI/CD → 6.0 |
| | 3. 迁移 CI 到 `macos-14`（Apple Silicon）runner | 测试 → 6.5 |
| | 4. 为 `memory_enforcer.update_loaded_model_bytes` 添加锁保护 | 并发 → 7.0 |
| | 5. 修复 `quantized_glm.h:load_vector` 边界检查 | 内核 → 7.0 |
| **P1 下一迭代** | 6. 修复 `engine_pool.py:1099,1122` 的 silent `except Exception: pass` | 可靠性 → 5.5 |
| | 7. 实现 `server.py _shutdown()` 的在途请求排空 | 可靠性 → 6.0 |
| | 8. 重构 `serve_command`（1382 行）为多个小函数 | 代码质量 → 6.0 |
| | 9. 添加 `pip-audit` + `bandit` + dependabot 安全扫描 | 安全 → 5.0 |
| | 10. 添加 `pytest-cov` + coverage reporting to CI | 测试 → 6.0 |
| **P2 计划修复** | 11. 修复 5 处 `asyncio.gather` 无 `return_exceptions` | 性能 → 7.0 |
| | 12. 修复 `moe_ffn_fused.metal` threadgroup union 512 字节溢出 | 内核 → 6.5 |
| | 13. 修复 `compile_metallib.py` `-ffast-math` 与 GLM DSA 不一致 | 内核 → 7.0 |
| | 14. 实现 `mcp/security.py _call_times` 的 eviction | 内存 → 7.5 |
| | 15. 修复 `transformers` 版本冲突（统一 pin） | CI/CD → 6.5 |

### 评分变更预测

修复 P0 后：综合评分 **5.2 → 7.5**（安全 +5.0、并发 +1.5、内核 +1.5、测试 +1.5、CI/CD +2.5）
修复 P0 + P1 后：综合评分 **5.2 → 8.2**（叠加可靠性 +1.5、代码质量 +1.0、CI/CD +0.5）

---

**评分员**: AtomCode (GLM-5.2)
**评分完成日期**: 2026-07-25
**报告位置**: `~/claude-home/fusion-mlx/AUDIT_SCORE.md`
**依据来源**: `AUDIT_FUSION_MLX.md`（完整审计报告）+ 主审计员对关键事实的二次核查
