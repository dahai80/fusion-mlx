# B.2 Runtime LoRA Hot-Swap — 路径 A' 实现计划

分支: `feat/b2-lora-runtime-hotswap` (已从 main 创建)

## 目标 + 验证标准 (Rule 4)

per-request `adapters` 字段切换 LoRA adapter，**不 reload base model**（首次 load adapter engine 后复用 cache）。

成功标准:
1. `curl` 同一 base model + 两个不同 `adapters` → 命中两个独立 engine 实例
2. 重复同 adapter 请求 → 0 次 reload（命中 cache）
3. 内存压力下 LRU 淘汰衍生 adapter entry（不影响 base）
4. continuous batching 下各 adapter engine 互不干扰
5. CI lint+test 全绿；macOS 全套通过

## 核心设计 (路径 A' — adapter-keyed engine cache)

engine_pool 按 `(model_id, adapter_path)` 组合 key 索引**衍生 EngineEntry**，lazy 创建，复用现有 LRU + 内存预算机制。per-profile `lora_path`（commit 0642b8e0 已落地）是其**预注册特例**——A' 把它泛化为请求时 lazy 创建。

**纠正文档前提**: `docs/FR_DIFFERENTIATION.md` 称 "mlx_lm fuses the adapter into weights at load time" —— 已核实为误。`mlx_lm.load(adapter_path=...)` 内部调 `load_adapters()`，**包装 `LoRALinear`（base + LoRA delta 独立层），不融合**；`remove_lora_layers()` 可剥离。故 runtime hot-swap 技术可行。

## 改动清单 (Rule 3 — surgical)

| # | 文件 | 改动 |
|---|---|---|
| 1 | `fusion_mlx/api/models.py` | `ChatCompletionRequest`/`CompletionRequest` 加 `adapters: str \| None = None`（mlx_lm server 兼容字段） |
| 2 | `fusion_mlx/api/openai_routes.py` | `resolve_model_id(req.model)` 后传 `adapter_path=req.adapters` 给 `get_engine`；校验 path 存在 |
| 3 | `fusion_mlx/api/anthropic_routes.py` | 同上（messages API） |
| 4 | `fusion_mlx/pool/engine_pool.py` | `get_engine` 加 `adapter_path` 参数；衍生 entry lazy 创建；组合 key；内存预算/LRU 复用 |
| 5 | `EngineEntry` (engine_pool.py:54) | 加 `adapter_path: str \| None = None` + `base_model_id: str \| None = None` |
| 6 | `tests/unit/test_lora_hotswap.py` | 新增（macOS 跑，Linux 入 `_linux_skip`） |
| 7 | `docs/FR_DIFFERENTIATION.md` | 纠正 fuse 前提 + B.2 A' changelog |

`server.py resolve_model_id` **不变**（`adapters` 是独立请求字段，不编码进 `model`）。

## 关键实现细节

- **组合 key**: `model_id` if `not adapter_path` else `f"{model_id}::lora::{adapter_path}"`
- **衍生 entry 创建** (get_engine 内):
  - clone base EngineEntry 的 `model_path`/`estimated_size`/`config_model_type`/`engine_type`/`thinking_default`/`preserve_thinking_default`/`model_context_length`
  - 设 `adapter_path=<请求path>`, `base_model_id=<base model_id>`, `source_type="lora_adapter"`
  - model_settings: 从 `_settings_manager.get_settings(base_model_id)` clone，覆盖 `lora_path=adapter_path`
  - 注册 `_entries[组合key]`，走现有 load 路径（BatchedEngine 构造已支持 `lora_path=model_settings.lora_path`，见 engine_pool.py:1408/1461）
- **LRU/内存**: 衍生 engine `actual_size` 算入 `_current_model_memory`；`_find_lru_victim` 已按 `last_access` 淘汰，天然支持衍生 entry；衍生 entry 默认非 pinned
- **上限**: 加 `max_adapter_engines`（scheduler_config 或 env），防 N×base 内存爆炸；超限时 LRU 淘汰最旧 adapter entry
- **校验**: adapter_path 不存在 → 友好 400（`FileNotFoundError` 捕获），不污染 pool
- **日志** (Rule — 默认有日志): `logger.info("LoRA adapter engine cached: base=%s adapter=%s key=%s")` + cache hit/miss/evict
- **代码约束**: 4倍缩进、无 docstring、单文件 ≤3000 行、有日志

## defer (不在此 plan 范围)

- **base weights 共享**（N adapter 仍 N×base memory）—— 需改 `mlx_lm.load` 共享 base，Phase C 级风险
- **LoRA bank 权重指针切换**（路径 C）—— 改 LoRALinear 内部，违反 don't-break-released
- **scheduler 改动** —— A' 各 adapter engine 独立 batching，无需改
- **per-request spec routing** —— 独立 Phase B 项，与此并行 defer

## 风险 + 缓解

| 风险 | 缓解 |
|---|---|
| 内存 N×base (8B-4bit×3≈15GB) | LRU + `max_adapter_engines` 上限；文档说明 |
| 衍生 entry discovery clone 不完整 | 测试覆盖 base+adapter 隔离；clone 字段显式列出 |
| macOS app config sync | `adapters` 是请求级字段非 engine config，app 无需改（实现后确认） |
| 衍生 entry unload/统计 | admin `/v1/models` 列表区分 base/adapter entry |

## 测试计划 (Rule 9)

`tests/unit/test_lora_hotswap.py`:
- `test_per_request_adapter_switch`: 两 adapter → 两独立 engine
- `test_adapter_engine_reuse`: 重复请求 0 reload（mock load 计数）
- `test_lru_evict_adapter_entry`: 内存压力淘汰衍生 entry，base 不动
- `test_base_isolation`: base 与 adapter engine KV cache/状态独立
- `test_invalid_adapter_path`: 404/400 友好报错
- `test_profile_and_request_adapter_coexist`: per-profile 与请求级 adapter 共存
- `test_combo_key_stability`: 同 adapter 不同 base → 不同 entry；同 base 同 adapter → 复用

## 实施顺序 (Rule 10 — checkpoint)

1. EngineEntry 字段 + get_engine adapter_path 参数 + 衍生 entry 创建 → 单测衍生 entry 逻辑
2. api/models.py adapters 字段 + 路由传参 → curl 验证两 adapter
3. LRU 淘汰 + max_adapter_engines 上限 → 内存测试
4. 校验 + 错误处理
5. 文档 + CI 验证
