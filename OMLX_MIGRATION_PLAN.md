# omlx → fusion-mlx 特性迁移计划

> **状态：已完成 (2026-07-07)** — 所有 P0–P4 项已迁移或按下方「不迁移的特性」显式跳过。
> 验证依据：`fusion_mlx/logging_config.py`（#375 结构化日志，已接入 `server.py`）、
> `fusion_mlx/cli_lifecycle.py` + `fusion_mlx/launch/cli.py`（#376 CLI lifecycle/launch，
> 已接入 `cli.py` 的 start/stop/restart/launch 分发）、`fusion_mlx/parsers/`（output_parser
> 三 session 补全）、`fusion_mlx/models/embedding*`、`fusion_mlx/api/markitdown*`、
> `fusion_mlx/model_settings.py`、`fusion_mlx/admin/oq_manager.py`。本文档保留为历史规划记录。

## 原则

1. **不改变 fusion-mlx 架构** — 保持独立 engine 类、scheduler/ 子包拆分、routes/ 子包拆分
2. **不暴力合并** — 理解原理后适配 fusion-mlx 风格（4 缩进、无 docstring、有日志）
3. **不改变性能目标** — 每步迁移后跑测试确认不退化
4. **增量迁移** — 每个特性独立可测试，不搞大爆炸合并

## 现状对比

| 维度 | omlx | fusion-mlx | 差距 |
|------|------|------------|------|
| adapter/ | harmony(495) + gemma4(499) + output_parser(661) | adapter/ 是空 stub；parsers/ 有 harmony(442) + gemma4(471) + output_parser(197) | output_parser 缺 3 个 parser session（MiniMaxM3, Cohere2Moe, _Normalizer） |
| eval/ | 19 个 benchmark 数据集 (2492 行) | 空 stub | 全部缺失 |
| api/markitdown | 867 + 263 行 | 无 | 文件上传→markdown 转换 |
| models/embedding | 738 行 (MLXEmbeddingModel) | 无 | 嵌入模型加载/推理 |
| models/qwen2_embedding | 329 行 | 无 | Qwen2 嵌入专用 |
| models/xlm_roberta | 509 行 | 无 | XLM-RoBERTa 嵌入 |
| models/base_model | 107 行 | 无 | 基础模型工具 |
| models/llm | 360 行 | 无 | GenerationOutput 等 |
| model_settings | 1192 行 (ModelSettings + ModelSettingsManager) | settings.py 仅 125 行 | 缺 per-model 持久配置 + profiles |
| logging_config | 283 行 (RequestContextFilter, JSON formatter, file rotation) | 无 | 缺结构化日志 |
| admin/oq_manager | 677 行 (OQManager 任务管理) | admin/oq.py 259 行 (简单路由) | 缺后台量化任务管理 |
| admin/hf_uploader | 546 行 | admin/hf_upload.py 存在但需对比 | HF 上传功能 |
| admin/ms_downloader | 1102 行 | admin/ms_download.py 存在但需对比 | ModelScope 下载 |
| admin/vendor_deps | 246 行 | 无 | 依赖管理 |
| admin/build_css | 87 行 | 无 | CSS 构建 |
| integrations/codex_app | 61 行 | 无 | Codex App 桌面集成 |
| CLI lifecycle | launch_command + lifecycle_command (start/stop/restart) | 无 | 缺 brew services/app 控制 |
| engine/vlm | 3723 行 | 1103 行 | VLM 引擎深度差距大 |
| engine/dflash | 1596 行 | 有 (pool/engine/dflash.py + engine/dflash.py) | 已有 |
| scheduler | 10170 行 (单体) | 拆分为 25+ 子模块 | 架构不同但功能对等 |

## 迁移优先级

### P0: 修复现有 bug（不引入新代码）

1. **scheduler/core.py adapter import bug** — `from .adapter.output_parser import` 应改为 `from ..parsers.output_parser import`（当前 ImportError 被 try/except 吞掉，`HAS_OUTPUT_PARSER` 永远 False）

### P1: 高价值、低风险、独立模块

2. **eval/ benchmark 套件** (2492 行)
   - 19 个数据集评估器：arc, bbq, cmmlu, gsm8k, hellaswag, humaneval, jmmlu, kmmlu, livecodebench, mathqa, mbpp, mmlu, mmlu_pro, safetybench, truthfulqa, winogrande
   - 无内部依赖（只依赖 asyncio + httpx/requests 下载数据集）
   - 直接复制 + 适配 import 路径
   - 替换 fusion_mlx/eval/__init__.py stub

3. **logging_config.py** (283 行)
   - RequestContextFilter, AdminStatsAccessFilter, ColoredFormatter, JsonFormatter
   - RequestLogContext (contextvars)
   - configure_logging(), configure_file_logging()
   - 无外部依赖（标准库 only）
   - fusion-mlx 当前用 uvicorn 默认日志，缺 request_id 跟踪

### P2: 中等价值、需要适配

4. **models/embedding.py** (738 行) — MLXEmbeddingModel
   - 统一嵌入模型加载/推理接口
   - 支持 native/custom 加载、mx.compile、Qwen2/XLM-RoBERTa 专用路径
   - 依赖：mlx.core, mlx.utils, models/qwen2_embedding, models/xlm_roberta
   - 需要先迁移 qwen2_embedding.py (329) + xlm_roberta.py (509) + base_model.py (107) + mlx_embeddings_compat.py (75)

5. **api/markitdown.py** (867 行) + markitdown_pdf_fallback.py (263 行)
   - 文件上传→markdown 转换（PDF, DOCX, etc.）
   - 依赖：markitdown 包（pip install markitdown）
   - 需要添加到 server 路由

6. **adapter/output_parser.py 补全** (661→197 差 464 行)
   - 缺失：_MiniMaxM3ProtocolNormalizer, MiniMaxM3OutputParserSession, Cohere2MoeOutputParserSession
   - 需要适配 fusion-mlx 的 parsers/ 路径而非 adapter/ 路径
   - 同时修复 adapter/ stub → re-export from parsers/

### P3: 较大改动、需仔细适配

7. **model_settings.py** (1192 行) — per-model 持久配置
   - ModelSettings dataclass + ModelSettingsManager
   - 与 fusion-mlx settings.py (125 行) 的 Settings dataclass 合并
   - 需要设计：是在 Settings.model_settings dict 中嵌入，还是独立模块
   - 影响面：admin/profile.py, admin/models_route.py, engine_pool

8. **admin/oq_manager.py** (677 行) — 量化任务管理
   - OQManager 后台任务队列
   - 当前 fusion-mlx admin/oq.py (259 行) 是简单路由
   - 需要适配 fusion-mlx 的 oq.py 量化器接口

9. **CLI lifecycle commands** (launch_command + lifecycle_command)
   - start/stop/restart via brew services 或 macOS app 控制套接字
   - launch 命令（codex, claude, copilot 等）
   - 需要适配 fusion-mlx CLI 结构

### P4: 低优先级

10. **integrations/codex_app.py** (61 行) — Codex App 桌面集成
11. **admin/vendor_deps.py** (246 行) + **admin/build_css.py** (87 行) — 构建工具
12. **models/llm.py** (360 行) — GenerationOutput 等（fusion-mlx 可能在别处已有）

## 实施步骤

### Step 1: Fix bug + eval/ + logging_config (P0+P1)

```
1a. Fix scheduler/core.py import: .adapter.output_parser → ..parsers.output_parser
1b. Migrate eval/ (19 files, 2492 lines) — 直接复制 + 适配 import
1c. Migrate logging_config.py (283 lines) — 直接复制
1d. Run tests, verify green
```

### Step 2: models/embedding suite (P2)

```
2a. Migrate models/base_model.py (107 lines)
2b. Migrate models/llm.py (360 lines) — 或确认 fusion-mlx 已有等价
2c. Migrate models/mlx_embeddings_compat.py (75 lines)
2d. Migrate models/qwen2_embedding.py (329 lines)
2e. Migrate models/xlm_roberta.py (509 lines)
2f. Migrate models/embedding.py (738 lines) — MLXEmbeddingModel
2g. Run tests, verify green
```

### Step 3: markitdown + output_parser 补全 (P2)

```
3a. Migrate api/markitdown.py (867 lines) + markitdown_pdf_fallback.py (263 lines)
3b. 补全 parsers/output_parser.py 缺失的 3 个 parser session
3c. 修复 adapter/ stub → re-export from parsers/
3d. Run tests, verify green
```

### Step 4: model_settings + oq_manager (P3)

```
4a. 设计 model_settings 与 Settings 的合并方案
4b. 实现 model_settings.py (1192 lines)
4c. 增强 admin/oq.py → oq_manager.py (677 lines)
4d. Run tests, verify green
```

### Step 5: CLI lifecycle + 低优先级 (P3+P4)

```
5a. 添加 CLI launch_command + lifecycle_command
5b. 添加 integrations/codex_app.py
5c. 添加 admin/vendor_deps.py + build_css.py
5d. Run tests, verify green
```

## 不迁移的特性

- **omlx scheduler.py 单体** — fusion-mlx 已拆分为 25+ 子模块，架构更优
- **omlx server.py 单体** — fusion-mlx 已拆分为 routes/ + api/ 子包
- **omlx engine/vlm.py 3723 行** — 需要单独评估，太大不宜整体迁移
- **omlx process_memory_enforcer.py** — fusion-mlx 已有 pool/memory_enforcer.py (1390 行)，功能对等
- **omlx patches/** — 已在之前迁移过
- **omlx oq.py** — 已在之前迁移过
