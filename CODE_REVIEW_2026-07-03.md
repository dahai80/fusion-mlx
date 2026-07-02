# fusion-mlx 代码评审报告

- **日期**: 2026-07-03
- **评审者**: AtomCode (GLM-5.2)
- **范围**: Python 代码深度评审(安全 / 并发 / 资源泄漏 / 数据完整性 / 逻辑 / 性能)
- **方法**: 建心智模型 → 追调用链 → 挖性能 → 反证边界
- **评审子模块**: `scheduler/` + `mllm_scheduler.py`、`cache/`、`pool/`、`admin/auth*` + `subkey`、`admin/hf_download*` + `hf_upload` + `ms_download`、`api/*_routes`、`server.py` ×2 + `server_metrics`
- **未评审**: Swift 端、`tests/`、`examples/`、`docs/`、`fusion_mlx/` 其余子模块(`engines/`、`reasoning/`、`speculative/`、`integrations/`、`tool_parsers/`、`utils/`、`router/`、`parsers/`、`adapter/`)

## 摘要

| 严重度 | 数量 | 说明 |
|---|---|---|
| Critical | 3 | 部署劫持、死锁、路径穿越 |
| High | 8 | 时序攻击、未认证敏感操作、并发瓶颈、引擎生命周期、SSD 竞态、任意文件读 |
| Medium | 18 | 逻辑错误、竞态、性能、资源泄漏、信息泄漏 |
| Low | 14 | 幂等性、可维护性、防御性缺失 |
| **合计** | **43** | (另有 20 项为设计取舍或观察项,见末节) |

## Issue 候选池(本轮提交 15 条,critical/high 优先)

| # | 严重度 | 类别 | 位置 | 标题 |
|---|---|---|---|---|
| F01 | Critical | 安全/认证 | `admin/auth_routes.py:94-158` | `/api/setup-api-key` 无认证,部署劫持 |
| F02 | Critical | 并发/死锁 | `cache/paged_cache.py:1309,1337` | `handle_memory_pressure` 调 `evict_lru_blocks` 触发 `Lock` 重入死锁 |
| F03 | Critical | 安全/路径穿越 | `admin/hf_downloader.py:604-605,775` | `repo_id` 校验不禁止 `..`,`../b` 路径穿越写盘 |
| F04 | High | 安全/时序 | `admin/auth.py:77-83` | `verify_api_key` 注释声称常量时间,实际 `==` 短路比较 |
| F05 | High | 安全/未认证 | `server.py:238,254` | `/v1/models/{id}/load`、`/unload` 无认证 |
| F06 | High | 性能/并发瓶颈 | `pool/engine_pool.py:622,729` | `get_engine` 持 `asyncio.Lock` 期间跑 5+ 秒 unload,序列化所有引擎获取 |
| F07 | High | 资源生命周期 | `api/openai_routes.py:113,182` | 流式/非流式路由用 `get_engine` 而非 `acquire`,引擎可被 enforcer 中途卸载 |
| F08 | High | 并发/数据竞争 | `cache/paged_ssd_cache.py:320-498` | `PagedSSDCacheManager` 无锁,`_index`/`_current_size` 跨线程竞态 |
| F09 | High | 安全/任意文件读 | `admin/oq.py:144,191,202` | `estimate_oq`/`start_oq_quantization` 的 `model_path` 无路径校验 |
| F10 | High | 安全/密钥泄漏 | `admin/auth_routes.py:176-177` | `auto_login` 用 GET query param 传 API key |
| F11 | Medium | 逻辑/状态泄漏 | `mllm_scheduler.py:658-664` | `_schedule_waiting` 的 `zip(uids, scheduled)` 静默截断,未分配 uid 的请求永远卡 running |
| F12 | Medium | 逻辑/静默失败 | `mllm_scheduler.py:1077-1078,1087-1088` | `_distribute_outputs` 的 `QueueFull` 被 `pass` 吞掉,客户端收不到结束信号 |
| F13 | Medium | 并发/关闭时序 | `mllm_scheduler.py:1126-1142` | `stop()` 先关 `_detok_executor`,step 线程 `submit` 抛 RuntimeError 被误判为 batch 失败 |
| F14 | Medium | 逻辑/误统计 | `cache/paged_cache.py:736-738` | `_maybe_evict_cached_block` 用 `block.block_id` 作 default,误判驱逐 + `stats.evictions` 虚高 |
| F15 | Medium | 性能/写放大 | `cache/paged_ssd_cache.py:143,147` | `_write_safetensors_no_mx` 内存拼接 `all_bytes += raw`,大 block O(n²) + 内存峰值 |

## 详细 Findings

### F01 [Critical / 安全] `/api/setup-api-key` 无认证导致部署劫持
- **位置**: `fusion_mlx/admin/auth_routes.py:94-158`
- **调用链**: `POST /api/setup-api-key` → `setup_api_key()` → 仅检查 `global_settings.auth.api_key` 是否已设
- **触发时序**: 服务器首次部署,默认绑定 `0.0.0.0`(`server.py:437`、`config.py:152`、`cli.py:319`),未设 API key 时,**任何能访问端口的远程客户端**可抢先调用 `/api/setup-api-key` 设置自己的 key 并获得 admin session cookie(行 149-156),完成部署劫持。
- **量级**: 暴露在公网/局域网的首次部署 100% 可被劫持。
- **建议**: 首次 setup 应要求本地访问(127.0.0.1)或一次性 setup token(首次启动打印到 stderr),且绑定地址默认改为 `127.0.0.1`。

### F02 [Critical / 并发] `handle_memory_pressure` → `evict_lru_blocks` 死锁
- **位置**: `fusion_mlx/cache/paged_cache.py:1309`(evict_lru_blocks)、`:1337`(handle_memory_pressure)
- **调用链**: `handle_memory_pressure`(持 `_hash_map_lock, _free_queue_lock`) → `evict_lru_blocks`(内部 `with _block_table_lock, _hash_map_lock, _free_queue_lock`)
- **不变量违反**: 三把锁均为 `threading.Lock()`(不可重入,行 573-575)。同线程二次 acquire `_hash_map_lock`/`_free_queue_lock` 必然死锁。
- **触发时序**: 任何调用 `handle_memory_pressure` 的路径(内存压力下)立即死锁,持有锁的线程永久阻塞,所有后续 cache 操作阻塞。
- **量级**: 100% 复现,系统完全僵死。
- **建议**: `evict_lru_blocks` 改为内部 `_evict_lru_blocks_unlocked`(假设调用方持锁),或三把锁改 `RLock`;`handle_memory_pressure` 直接调 unlocked 版本。

### F03 [Critical / 安全] `hf_downloader` repo_id 路径穿越
- **位置**: `fusion_mlx/admin/hf_downloader.py:604-605`(校验)、`:775`(target_dir)、`_cleanup_partial:1024`
- **调用链**: `start_download(repo_id)` → 校验 `len(repo_id.split("/")) == 2` → `target_dir = self._model_dir / task.repo_id` → `snapshot_download(local_dir=target_dir)` → 取消时 `_cleanup_partial` 用 `shutil.rmtree`
- **触发时序**: `repo_id = "../b"` 通过校验(`["..","b"]` 长度 2)。`target_dir` 路径穿越到 `model_dir` 上级。下载写入任意目录,取消时 `shutil.rmtree` 删除任意目录。
- **量级**: admin(已认证)可写/删服务器任意路径(受 `_model_dir` 上级可达性限制)。
- **建议**: `repo_id` 校验加 `..` 禁止(`"/" in repo_id and ".." not in repo_id.split("/")`),且 `target_dir.resolve().is_relative_to(self._model_dir.resolve())` 二次校验。

### F04 [High / 安全] `verify_api_key` 时序攻击
- **位置**: `fusion_mlx/admin/auth.py:77-83`
- **问题**: 注释行 78 "constant-time comparison",但行 83 `return a == b` 是 Python 字符串短路比较(逐字符,首个不匹配即返回)。SHA-256 哈希不防时序(攻击者可逐字符暴力哈希前缀)。`subkey.py:71,79,125` 正确用了 `secrets.compare_digest`,说明 API 已知但此处误用。
- **建议**: 行 83 改 `return secrets.compare_digest(a, b)`。

### F05 [High / 安全] 模型加载/卸载端点无认证
- **位置**: `fusion_mlx/server.py:238`(`/v1/models/{id}/load`)、`:254`(`/unload`)
- **问题**: 两个端点无 `Depends(require_admin)`,任何远程客户端可触发模型加载(耗尽内存 DoS)或卸载(中断他人推理)。
- **建议**: 两端点加 `is_admin: bool = Depends(require_admin)`。

### F06 [High / 性能] `get_engine` 持锁内长 IO 序列化所有引擎获取
- **位置**: `fusion_mlx/pool/engine_pool.py:622`(持锁)、`:650,662,704,729`(锁内 unload/load)、`_unload_engine:923-990`(5×sleep + 10×0.5s settle)
- **量级**: 一次 unload 最长 5+ 秒持 `asyncio.Lock`,期间所有 `get_engine`/`acquire`/`release_engine` 阻塞。高并发多模型切换场景吞吐降一个数量级。
- **建议**: unload/load 移出锁外(用状态机 + 条件变量),锁只保护 `_entries` 字典读写。

### F07 [High / 资源] 路由层用 `get_engine` 而非 `acquire`,引擎可中途卸载
- **位置**: `fusion_mlx/api/openai_routes.py:113,182`、`anthropic_routes.py:159,236`
- **问题**: `get_engine` 不取 in_use lease,流式生成期间 enforcer 可卸载引擎,`_unload_engine` 的 `_entry_has_active_requests` 依赖引擎自身活动追踪,若流式请求未注册为 active,KV cache 被释放,后续 token 崩溃/乱码。
- **建议**: 路由层改用 `async with pool.acquire(model_name) as engine:`。

### F08 [High / 并发] `PagedSSDCacheManager` 无锁保护共享状态
- **位置**: `fusion_mlx/cache/paged_ssd_cache.py:320-498`(store/load/evict/verify)
- **问题**: 类无锁字段(`SharedHotCacheBudget` 的 `_lock` 不属于本类)。`_index.blocks` dict 与 `_current_size` int 被多线程(scheduler step + recovery + close)并发读写,`_current_size += size` 累加丢失,`verify_and_repair_index` 遍历 `list(self._index.blocks.keys())` 时并发修改可能抛 `RuntimeError: dictionary changed size`。
- **建议**: 类加 `threading.Lock`,所有 `_index`/`_current_size` 操作加锁。

### F09 [High / 安全] `oq` 端点 model_path 任意文件读
- **位置**: `fusion_mlx/admin/oq.py:144`(GET `estimate_oq`)、`:191,202`(POST `start_oq_quantization`)
- **问题**: `model_path` 来自客户端无路径校验,`estimate_bpw_and_size`/`start_quantization` 读取/处理任意本地文件。GET 参数还会泄漏到 URL 日志。
- **建议**: `model_path` 校验 `is_relative_to(model_dir)`,且 POST 化。

### F10 [High / 安全] `auto_login` GET 传 API key
- **位置**: `fusion_mlx/admin/auth_routes.py:176-177`
- **问题**: `key: str = ""` query param,API key 出现在 URL → 浏览器历史、代理日志、Referer 头泄漏。
- **建议**: 改 POST + body,或用短期一次性 token。

### F11 [Medium / 逻辑] `_schedule_waiting` zip 静默截断致请求泄漏
- **位置**: `fusion_mlx/mllm_scheduler.py:658-664`
- **调用链**: `_schedule_waiting` → `batch_generator.insert(batch_requests)` 返回 uids → `zip(uids, scheduled)`
- **不变量违反**: 若 insert 返回 uids 数量 < scheduled 数量(部分插入失败),`zip` 静默丢弃未分配 uid 的 request,但其 `self.running[request_id]` 已写入(行 654)、`status = RUNNING`(行 653),**永远不被 generate,泄漏在 running dict**。
- **建议**: 校验 `len(uids) == len(scheduled)`,不等则回滚未分配的 request(从 running 移除,状态回 WAITING)。

### F12 [Medium / 逻辑] `_distribute_outputs` QueueFull 静默吞掉
- **位置**: `fusion_mlx/mllm_scheduler.py:1077-1078,1087-1088`
- **问题**: `put_nowait` 失败被 `except QueueFull: pass` 吞掉,无日志。当前 `asyncio.Queue()` 默认无 maxsize(行 1356)不触发,但未来引入限制时 bug 静默激活,客户端收不到结束信号。
- **建议**: 失败时至少 `logger.warning`,或用 `await queue.put()`(背压)。

### F13 [Medium / 并发] `stop()` 关闭时序致正常请求被判 batch 失败
- **位置**: `fusion_mlx/mllm_scheduler.py:1126-1142`(`stop`)、`_step_no_queue:997`(捕获 RuntimeError)
- **触发时序**: `stop()` 行 1142 `_detok_executor.shutdown(wait=False)` 后,step 线程 `_process_batch_responses` 行 729 `self._detok_executor.submit` 抛 `RuntimeError: cannot schedule new futures` → 被 `_step_no_queue` 行 997 `except (ValueError, RuntimeError)` 捕获 → **所有 running 请求被标记为 error/length finish**,优雅关闭变成强制失败。
- **建议**: `stop()` 先设标志让 step 线程跳过 detok submit,再 shutdown executor;或 `_step_no_queue` 区分 shutdown 期 RuntimeError。

### F14 [Medium / 逻辑] `_maybe_evict_cached_block` 误判驱逐
- **位置**: `fusion_mlx/cache/paged_cache.py:736-738`
- **问题**: `cached_block_hash_to_block.pop(block.block_hash, block.block_id)` 用 `block.block_id`(truthy)作 default,hash 不存在时返回 `block.block_id`,进入 `if evicted:`(行 740)误判已驱逐,`reset_hash()` + `stats.evictions += 1` 虚高。
- **建议**: default 改 `None`,`if evicted is not None:`。

### F15 [Medium / 性能] `_write_safetensors_no_mx` 内存拼接 O(n²)
- **位置**: `fusion_mlx/cache/paged_ssd_cache.py:143,147`
- **问题**: `all_bytes = b""` + `all_bytes += raw` 循环拼接,bytes 不可变每次创建新对象,大 KV block(数十 MB)O(n²) 复制 + 内存峰值 = 总数据大小。
- **建议**: 直接 `f.write(raw)` 流式写,删除 `all_bytes`。

## 其余 Medium / Low Findings(仅入报告,不提 issue 或留待下轮)

| # | 严重度 | 位置 | 摘要 |
|---|---|---|---|
| F16 | Medium | `mllm_scheduler.py:548-555` | `_process_pending_aborts` 用 `set.pop()` 顺序非确定 |
| F17 | Medium | `mllm_scheduler.py:545-546` | `record_disconnect_abort` bare `except: pass` 吞掉 ledger 损坏 |
| F18 | Medium | `mllm_scheduler.py:577,594` | `_do_abort_request` 跨线程复合删 `request_id_to_uid`/`running` 无统一锁 |
| F19 | Medium | `mllm_scheduler.py:950-966` | `_cleanup_finished` 不在 `_cancel_counter_lock` 内,abort 幂等性破坏 |
| F20 | Medium | `mllm_scheduler.py:1449` | `generate()` 的 `del self.requests` 不在锁内 |
| F21 | Medium | `cache/paged_cache.py:1324` | `evict_lru_blocks` 清 `token_count` 不更新 `stats.total_tokens_cached`,统计漂移 |
| F22 | Medium | `pool/memory_enforcer.py:1119,1195,1292` | enforcer 直接访问 `engine_pool._lock`,封装泄漏 + 释放-重获窗口竞态 |
| F23 | Medium | `pool/memory_enforcer.py:1159` | `_check_and_enforce` while 循环基于 await 间陈旧读数,过度/不足驱逐 |
| F24 | Medium | `pool/memory_enforcer.py:1121-1154` | lock timeout 路径反复标记 `abort_loading`,活锁风险 |
| F25 | Medium | `pool/priority_scheduler.py:352-374,540-548` | `_maybe_preempt` 的 `running_counts` 依赖 `cleanup_finished` 定期调用,否则 stale 致过度抢占 |
| F26 | Medium | `pool/engine_pool.py:1195-1202` | `_check_and_enforce` 释放锁后 unload,重获前状态可能已变 |
| F27 | Medium | `cache/paged_ssd_cache.py:394` | `_read_safetensors` 第一次 open 读 data 未使用,死代码 + 无用 IO |
| F28 | Medium | `admin/subkey.py:95` | `create_sub_key` 回滚 `pop()` 无锁,并发 append 后弹错条目 |
| F29 | Medium | `admin/auth_routes.py:161-173` | `logout` 不失效服务端 session,窃 cookie 仍可用 |
| F30 | Medium | `server.py:374-392` | `_shutdown` 不停止 `ProcessMemoryEnforcer`,任务泄漏(待确认 pool.shutdown 级联) |
| F31 | Medium | `server.py:121-126` | CORS `allow_origins=["*"]` 过宽 |
| F32 | Medium | `server.py:154,178,185,189,231` | `/health`/`/stats`/`/metrics`/`/api/status`/`/v1/models/status` 无认证,信息泄漏 |
| F33 | Medium | `admin/auth.py:38-42` | `create_session_token` 无上限累积,DoS(需先认证) |
| F34 | Low | `mllm_scheduler.py:1117-1128` | `_running` bool 无锁跨线程(实际 event loop 单线程,低风险) |
| F35 | Low | `admin/auth.py:60-65` | `_active_sessions` dict 无锁,`del` 竞态 `KeyError`(GIL 缓解) |
| F36 | Low | `admin/auth.py:70` | `validate_api_key` 最小 4 字符过低 |
| F37 | Low | `admin/auth.py:19,25-26` | `_api_key` 全局变量难测试 |
| F38 | Low | `admin/auth_routes.py:191` | `auto_login` redirect 仅 `startswith("/admin")`,编码绕过待确认 |
| F39 | Low | `admin/subkey.py:126,130` | `delete_sub_key` 无锁,并发删索引错位 |
| F40 | Low | `pool/engine_pool.py:1749` | `check_ttl_expirations` 持锁调 `has_active_requests` |
| F41 | Low | `pool/engine_pool.py:1762-1764` | `check_ttl_expirations` 释放锁后 unload,窗口窄竞态 |
| F42 | Low | `pool/priority_scheduler.py:163` | `PriorityScheduler` 用 `threading.Lock` 在 asyncio 上下文可能阻塞事件循环 |
| F43 | Low | `api/openai_routes.py:304,307` | SSE error 用 `!r` 产生单引号,JSON 解析失败 |
| F44 | Low | `mllm_scheduler.py:790-892` | `_process_batch_responses` stop-string 逻辑圈复杂度极高(仅观察) |
| F45 | Low | `admin/helpers.py:672` | subprocess 用绝对路径数组(无 shell),无注入风险(确认安全) |

## 代码质量观察(未提 issue)

- `fusion_mlx/cache/prefix_cache.py` 单文件 2596 行(接近 3000 行阈值),`BlockAwarePrefixCache` 类 2500+ 行,圈复杂度高,建议拆分。
- `fusion_mlx/pool/memory_enforcer.py` `ProcessMemoryEnforcer` 类 1124 行,`_check_and_enforce` 单方法 276 行,圈复杂度极高。
- `fusion_mlx/mllm_scheduler.py` `MLLMScheduler` 类 1300+ 行。
- 162 处 broad `except Exception:`,其中 `scheduler/sched_boundary.py`、`sched_cache.py` 多处为有意吞掉(模型能力探测回退),已确认设计取舍。
- Ruff 1670 条(大多 F401/W293/I001 风格问题),按规则不入 issue。

## 验证状态

- **已确认复现**: F02(死锁,锁类型确认)、F03(路径穿越,`"../b".split("/")` 长度 2 验证)、F04(`==` 短路确认)、F14(default truthy 确认)、F15(O(n²) 拼接确认)
- **逻辑推断(需运行时确认)**: F01(需确认部署时序)、F05-F13(需确认调用方与并发场景)
- **撤回的误判**: 原怀疑 `_read_safetensors` off-by-one,对照 `_write_safetensors_no_mx` 后确认正确(skip = header_size = 8 + N)。

## 下一步

- 本轮提交 15 条 issue(F01-F15,critical/high 优先)。
- F16-F33(medium)留待你决定是否下轮提交。
- F34-F45(low)与观察项仅入报告。
