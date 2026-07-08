# #431 Per-Request Spec Routing — 分阶段实现计划

分支: `feat/per-request-spec-routing` (从 main 创建)

## 目标 + 验证标准 (Rule 4)

把 `SpecAutoRouter`（已实现，纯决策函数）从 boot-time-only 接到 **per-request 决策点**：每个 request 进入 pure-decode 时，按 `prompt_token_count` + 上次 acceptance rate + boot 已加载方法集，在**已加载**的 spec 方法间选 active 方法，dispatch 链按 active 过滤。

成功标准:
1. boot 单方法（最常见，suffix-only）时：router 仍选 suffix，行为 = 现状（no-op 正确，零回归）
2. operator 同时 `--enable-dflash --suffix-decoding` 加载两方法：长 prompt（>=4096 tok）路由 dflash，短 prompt 路由 suffix，per-request 切换不 reload
3. acceptance 滞回：当前方法 acceptance < 0.20 放弃并排除，>= 0.40 保留（已有 `SpecDecodeState` 机制复用）
4. 现有 spec 全套测试（ngram/dflash/dspark/mtp）零回归
5. CI lint（black+ruff）+ test 全绿
6. E2E: Qwen3.5-9B-4bit `--spec-decode auto` boot -> suffix；长 prompt 仍 suffix（单方法 no-op）；日志打印 `spec-route:` 决策

## 核心设计 (渐进式 — 不强制多方法 resident)

### 关键约束（确认 memory 判断）
DFlash/DSpark 需 drafter model（boot 加载），MTP 需 converted ckpt。**per-request 跨方法切换不 reload** 的前提是方法已 resident。强制多方法 resident = 内存爆炸 + 多套 state + 高回归（spec corruption 历史痛点）。

### 渐进式解决
`available` 方法集 = **boot 已加载的**（runtime 非空的方法）。router 只在 available 集内决策：
- boot 单方法（suffix-only，最常见）-> router 选 suffix（no-op，零回归）
- boot 多方法（operator 显式 `--enable-dflash --suffix-decoding`）-> router 在两者间 per-request 选
- 未加载的方法 router 不推荐（`available_methods()` 已过滤 `config_enabled`，扩展过滤 `runtime loaded`）

**不引入 lazy draft load**（首次切换秒级延迟不可接受），**不强制多方法 resident**。价值随 operator 加载方法数线性增长，零方法数风险。

### 注入点
`sched_step.py::_try_spec_decode` L504 已有 `request = self.running.get(request_id)`，`request.num_prompt_tokens` = router 的 `prompt_token_count` 信号。dispatch 链 L516-548 从硬 `if runtime is not None` 改 `if runtime is not None and method == active`。

## 分阶段切片 (Rule 10 checkpoint)

### Slice 1: per-request active 方法选择器（纯函数 + 单测，零 wiring）
- 新增 `speculative/per_request_route.py`：`select_active_method(signals: RouteSignals, loaded: dict[str,bool], recent_accept: float|None, current: str|None) -> str|None`
  - 包装 `SpecAutoRouter.decide`，但 `available` = `{m for m in loaded if loaded[m]}`
  - 返回 None = 无 spec（所有方法未加载/未启用）
  - 纯函数，无副作用，完全单测
- 单测 `tests/unit/test_per_request_route.py`：覆盖单方法 no-op、多方法 long-doc->dflash、滞回、available 为空

**Checkpoint**: 单测全绿，未碰 scheduler 代码，零回归风险。

### Slice 2: scheduler 接入 active 选择（wiring，核心改动）
- `sched_init.py`: 加 `self._spec_active_method: str | None = None`（per-request，每 request 重算）+ `self._spec_recent_accept: dict[str,float]`（method->上次 acceptance，跨 request 滞回）
- `_try_spec_decode`（sched_step.py L504 后）:
  - 算 `loaded = {"suffix": _ngram_spec_state is not None, "ddtree": _dflash_runtime is not None, "dspark": _dspark_runtime is not None, "mtp": enable_mtp...}`
  - 调 `select_active_method`，写 `self._spec_active_method`
  - dispatch 链每个 `if runtime is not None` 加 `and self._spec_active_method == METHOD_X`
- acceptance 反馈: `record_accepted` 路径更新 `self._spec_recent_accept[method]`

**Checkpoint**: boot 单方法时 active 恒 = 该方法（no-op）；多方法时 per-request 切。现有 spec 测试全绿（active 选择不改变单方法行为）。

### Slice 3: `--spec-decode auto` per-request 模式（CLI + 集成）
- `cli_serve.py` auto 分支（L1960）: 保留 boot-time 解析作 fallback，但加 `--spec-route per-request` flag（或 `auto` 升级为 per-request 当多方法加载时）
- auto 时**不再 boot 强制单选**：若 operator 同时给了 drafter paths，加载多方法，per-request router 接管
- 集成测试: 多方法加载 + 长/短 prompt 路由验证

**Checkpoint**: E2E Qwen3.5-9B-4bit 验证；CI 全绿。

## 风险 + 缓解 (Rule 1/12)
- **spec corruption 回归**: dispatch 链改动可能破坏 KV cache 一致性。缓解: Slice 2 单方法 no-op 严格保持，多方法切换只在 request 边界（`on_new_request`）发生，不在 mid-decode 切。每 slice 跑现有 spec 全套测试。
- **per-request 切换 mid-request**: 一个 request 内 active 方法固定（在 `_try_spec_decode` 首次进入该 request 时定，缓存到 request 结束）。避免 mid-decode 切换导致 KV 状态混乱。
- **acceptance 信号冷启动**: 首个 request 无 recent_accept -> router 用默认（current method 或 suffix），不阻塞。

## 不做 (Rule 2)
- 不做 lazy draft load（延迟不可接受）
- 不做 multi-method KV state 隔离重构（每方法已有独立 state 字段，复用）
- 不做 mid-request 方法切换（固定 per-request）
- 不改 macOS app（per-request routing 是 scheduler 层，app settings 不涉及）

## 文件清单
- 新增: `fusion_mlx/speculative/per_request_route.py`（~80 行）
- 新增: `tests/unit/test_per_request_route.py`（~120 行）
- 改: `fusion_mlx/scheduler/sched_init.py`（+2 字段）
- 改: `fusion_mlx/scheduler/sched_step.py`（dispatch 链 + active 选择，~20 行）
- 改: `fusion_mlx/scheduler/spec_decode.py`（acceptance 反馈写 recent_accept，~5 行）
- 改: `fusion_mlx/cli_serve.py`（auto per-request 模式，~15 行）
- 文档: 本文件 + README spec 章节

## 执行顺序
Slice 1 -> checkpoint(单测) -> Slice 2 -> checkpoint(spec 全套测试) -> Slice 3 -> checkpoint(E2E+CI) -> 更新 memory + README
