# Fusion-MLX 审计报告 — P2/P3 + 风险矩阵 + 待查项

> **范围**：P0/P1 已提 GitHub issue（#80-#84），本报告盖 P2/P3 中长期建议 + 全量风险矩阵 + 本次未深读的"待查"项。
> **审计来源**：`FUSION_MLX_AUDIT_REPORT_2026-07-12_full.md`（HEAD `11ec79f`）
> **日期**：2026-07-12

---

## 一、P2 建议（本月，8 条）

### P2-1 GUI 兼容路由（`routes/`）鉴权统一

**现状**：`fusion_mlx/routes/`（10699 行旧版路由）与 `fusion_mlx/api/`（16380 行新版）并存。`/v1/manager/*`、`/v1/discover/*` 等 GUI 兼容路由无显式 `Depends(require_admin)` 或 API key 鉴权。

**已有 issue**：#71（router/ vs routes/ 双路由职责边界不清）、#75（require_admin 允许 query param 传 key 泄露）

**建议**：
1. 优先撤 `routes/`，全迁 `api/`，统一鉴权层
2. 迁不动则在 `server.py` 装配时统一挂 `Depends`，避"旧路由无鉴权"盲区
3. CORS `allow_origins`（#52 已提 issue）同步收紧

---

### P2-2 admin/auth.py 去全局状态

**现状**：`_active_sessions: dict` 和 `_api_key: str` 模块级全局。单 worker uvicorn 无问题，但 `--workers N` 下各 worker 独立副本，session 不共享。

**已有 issue**：#73（_active_sessions 无 LRU 过期清扫）、#50（server.py 30 个模块级全局变量双轨）

**建议**：
1. 改 `class AdminAuth` 实例，支持多 worker 共享（Redis session store 或 JWT 无状态 token）
2. 部署文档明确禁多 worker，或改 stateless token 避共享
3. `_active_sessions` 加 LRU + 定期清扫协程（#73 已盖）

---

### P2-3 定义性能 SLO 基线

**现状**：`bench/tier_runner.py` + `community_bench/` 跑 benchmark，但**无 declared SLO**（如 "P50 TTFT < 200ms / P99 < 2s"）。无 SLO 则性能 regression 无客观判据。

**建议**：
1. 至少为 text-to-text LLM 推理定义 SLO：P50/P99 TTFT（time to first token）+ TPS（tokens per second）
2. bench/ 跑出基线后落 `docs/performance_slo.md`
3. CI 加 performance regression gate（超基线 -10% 告警）

---

### P2-4 5 个 git+hash 依赖做镜像

**现状**：`pyproject.toml` 5 个依赖用 `git+hash` 锁定：
- `mlx-lm @ git+...@ed1fca4`
- `mlx-embeddings @ git+...@32981fa`
- `mlx-vlm @ git+...@f96138e`
- `dflash-mlx @ git+...@1ba6713`
- `mlx-audio @ git+...@5175326`

**风险**：上游 repo force-push 或删则安装即失败，无 fallback。

**建议**：
1. fork 到自己 GitHub（`dahai80/mlx-lm-fork` 等）保 commit hash 永久可访问
2. 或用 `--index-url` 指可控镜像（如自建 PyPI 或 GitHub Packages）
3. `.uv` override 也同步改指 fork

---

### P2-5 telemetry consent 落盘路径权限审计

**现状**：`telemetry/state.py` 的 consent 状态落盘，路径未在本次深读中确认（若落 `~/.fusion-mlx/` 多用户系统有权限风险）。

**建议**：
1. 确认 consent 落盘路径（`state.py` 的 `get_consent_state` / `set_consent`）
2. 若落用户目录，加 `0o600` 权限；若落项目目录，文档明确多用户隔离方案

---

### P2-6 `/metrics` Prometheus 端点确认

**现状**：`cache/observability.py` + `cache/stats.py` 有统计，但未确认是否暴露给 `/metrics` Prometheus 端点。

**建议**：
1. 若无，加 `/metrics` 端点吐 Prometheus 格式（TTFT / TPS / cache hit rate / queue depth）
2. 若有，确认 redact（§7 DfX）已覆盖 metrics labels 的 PII 风险

---

### P2-7 SSD cache 写入节流

**现状**：`cache/paged_ssd_cache.py` 三级缓存的 SSD 层，Apple Silicon SSD 快但**写入量影响 SSD 寿命**。

**建议**：
1. 查 `paged_ssd_cache.py` 是否做写入量节流（LRU 淘汰时批量删 vs 逐个删）
2. 加 SSD 写入量 metric，长跑下监控累积写入

---

### P2-8 CORS `allow_origins` 收紧

**现状**：#52 已提 issue——默认 `allow_origins=['*']` + methods/headers='*'，单机 UX 友好但公网部署暴露 API。

**建议**：
1. 默认改 `["http://localhost:3000"]`（仅本机 GUI）
2. 公网部署需显式 `--cors-origins` 配置
3. 与 P2-1 鉴权统一一起做

---

## 二、P3 建议（技术债，择期，5 条）

### P3-1 patches/ vendored 代码加 VERSION.lock

**现状**：`patches/` 18484 行 vendored 代码（mlx_vlm vendor、mlx_lm_mtp、deepseek_v4、glm_moe_dsa），ruff 已 exclude 但**无版本锁**。

**已有 issue**：#67（patches/vendor 内嵌完整 mlx_vlm 副本 ~18k 行，与上游 drift 无同步机制）

**建议**：
1. `patches/` 加 `VERSION.lock` 文件记录上游 commit hash + 日期 + fork 源 URL
2. CI 定期 check 上游变动，drift 时告警
3. 与 #67 合并治理

---

### P3-2 96 个 noqa 定期复审

**现状**：96 个 ruff noqa，每个是"明知违规则故豁"，但累积是技术债信号。

**建议**：
1. 季度审一次，分类：可消除的（改代码）/ 不可消除的（保留并补注释说明）
2. `pyproject.toml` 的 per-file-ignores 已有注释说明，保持纪律

---

### P3-3 测试债偿频率提高

**现状**：最近 4 个 commit（`69ce62a`/`1b9ee04`/`447aecb`/`11ec79f`）集中偿还 82+136 个 quarantine test，说明债累积过久。

**建议**：
1. 每 PR 强制偿一定比例 quarantine 测试（如 10%）
2. 避免再累积到 200+ 才集中偿
3. CI 加 "quarantine test count" metric，超阈值告警

---

### P3-4 fusion_gui vs apps/fusion-mac 定位明确

**现状**：双 GUI 并存——`fusion_gui/`（Python 11 文件）和 `apps/fusion-mac/`（Swift 原生），定位不清。

**已有 issue**：#63（fusion_gui 与 fusion_mlx 双顶层包耦合，循环/可选依赖边界不清）

**建议**：
1. README 明确 `apps/fusion-mac` 是一等客户端，`fusion_gui` 是兼容老版的 legacy 层
2. 或反之，明确 fusion_gui 的退出时间表
3. 避免双轨维护负担

---

### P3-5 `fusion_mlx/tests/` 与顶层 `tests/` 分工文档化

**现状**：双测试树——`fusion_mlx/tests/`（模块内自测）和顶层 `tests/`（集成测，323258 行），无明确分工文档。

**建议**：
1. `docs/testing.md` 明确：`fusion_mlx/tests/` = 单元自测（模块作者维护），`tests/` = 集成测（QA 维护）
2. CI 分别跑两树，单元测每 PR、集成测每 merge

---

## 三、全量风险矩阵（16 项）

| # | 风险 | 概率 | 影响 | 等级 | 状态 | 对应 issue |
|---|---|---|---|---|---|---|
| R1 | scheduler abort 路径资源泄漏（Metal/generator） | 已发生多次 | crash / 显存耗尽 | **高** | **已提** | #81 |
| R2 | 782 处裸 `except Exception` 吞错误 | 高 | silent failure / 安全泄露 | **高** | **已提** | #82 |
| R3 | build artifact 入 git（1.8M 行） | 确定 | 仓库膨胀 / 不同步 | **高** | **已提** | #80 |
| R4 | shutdown bounded-wait 挂死 | 已修但缺回归 | 重启挂死 | **中高** | **已提** | #84 |
| R5 | 5 个 git+hash 依赖锁失效 | 低 | 安装即失败 | **中高** | P2-4 | — |
| R6 | async remove vs engine step 竞态 | 已修但缺回归 | 错误输出 / crash | **中** | **已提** | #84 |
| R7 | admin 全局状态多 worker 下 session 不共享 | 部署依赖 | session 失效 | **中** | P2-2 | #73 #50 |
| R8 | GUI 兼容路由无鉴权 | 确定 | 未权访问 | **中** | P2-1 | #71 #75 |
| R9 | 23 处 create_task 未 await | 中 | silent fizzle / 资源泄漏 | **中** | **已提** | #83 |
| R10 | 无性能 SLO 基线 | 确定 | regression 难判 | **中** | P2-3 | — |
| R11 | patches/ vendored 代码无版本锁 | 低 | 上游变不兼容 | **低** | P3-1 | #67 |
| R12 | 测试债累积过久 | 已发生 | CI 长期黄 | **低** | P3-3 | — |
| R13 | CORS allow_origins=['*'] | 确定 | 公网暴露 API | **中** | P2-8 | #52 |
| R14 | telemetry consent 落盘权限 | 待查 | 多用户权限冲突 | **低** | P2-5 | — |
| R15 | /metrics Prometheus 端点缺失 | 待查 | 监控盲区 | **低** | P2-6 | — |
| R16 | SSD cache 写入量影响寿命 | 待查 | SSD 寿命缩短 | **低** | P2-7 | — |

---

## 四、待查项（本次未深读，需后续审计补完）

| # | 项 | 位置 | 审计方法 |
|---|---|---|---|
| W1 | CORS `allow_origins` 具体值 | `server.py` CORSMiddleware 装配 | grep `allow_origins` |
| W2 | `shell=True` 在 67 处 subprocess 中的用量 | `fusion_mlx/` 全仓 | grep `shell=True` 逐处 |
| W3 | telemetry consent 落盘路径权限 | `telemetry/state.py` | 读 `get_consent_state` / `set_consent` |
| W4 | `/metrics` Prometheus 端点是否存在 | `server.py` / `routes/` | grep `metrics` / `prometheus` |
| W5 | SSD cache 写入节流 | `cache/paged_ssd_cache.py` | 读淘汰逻辑 |
| W6 | 782 处 except Exception 分类 | `fusion_mlx/` 全仓 | grep 分类 pass/return/log |
| W7 | 23 处 create_task 生命周期 | `fusion_mlx/` 全仓 | 逐处审 task vs request 生命周期 |
| W8 | `_tempfile_safe.py` 的 61 处 tempfile 用例 | `fusion_mlx/` 全仓 | 确认 mkstemp vs NamedTemporaryFile |

---

## 五、GitHub Issue 汇总

### 本轮新提（5 个）

| # | 标题 | 等级 |
|---|---|---|
| [#80](https://github.com/dahai80/fusion-mlx/issues/80) | [P0][BUILD] apps/fusion-mac/build/ 含 1.8M 行编译产物入 git | P0 |
| [#81](https://github.com/dahai80/fusion-mlx/issues/81) | [P0][STABILITY] scheduler abort 路径资源清理全审 | P0 |
| [#82](https://github.com/dahai80/fusion-mlx/issues/82) | [P1][CODE] 782 处裸 except Exception 需分类治理 | P1 |
| [#83](https://github.com/dahai80/fusion-mlx/issues/83) | [P1][CODE] 23 处 asyncio.create_task 未 await | P1 |
| [#84](https://github.com/dahai80/fusion-mlx/issues/84) | [P1][TEST] scheduler 最近稳定性修复缺回归测试 | P1 |

### 已有相关 issue（避重，未补评论）

| # | 标题 | 与本轮发现关系 |
|---|---|---|
| #71 | router/ vs routes/ 双路由并存 | P2-1 的上位 |
| #73 | admin/auth._active_sessions 无 LRU | P2-2 的子集 |
| #75 | require_admin 允许 query param 传 key | P2-1 鉴权相关 |
| #50 | server.py 30 个模块级全局变量 | P2-2 的上位 |
| #52 | CORS allow_origins=['*'] | P2-8 已盖 |
| #67 | patches/vendor 内嵌 mlx_vlm 副本无同步 | P3-1 已盖 |
| #63 | fusion_gui 与 fusion_mlx 双顶层包耦合 | P3-4 已盖 |

---

## 六、总结

| 类别 | 数量 | 去向 |
|---|---|---|
| P0（立即） | 2 | **GitHub issue #80 #81** |
| P1（本周） | 3 | **GitHub issue #82 #83 #84** |
| P2（本月） | 8 | **本报告 §一** |
| P3（择期） | 5 | **本报告 §二** |
| 风险矩阵 | 16 项 | **本报告 §三** |
| 待查项 | 8 项 | **本报告 §四** |

**P0/P1 已全部提 issue**，P2/P3 + 风险矩阵 + 待查项在本报告完整记录。待查项（W1-W8）需后续审计深读补完。
