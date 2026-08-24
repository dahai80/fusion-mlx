# 公开 API 稳定层与 import guard

> 关联 issue: #613（建稳定层 `public_api.py`）, #620（补全 `__all__`）,
> #624（暴露 4 个下游依赖符号）, #615（CI import guard）。

## 背景

`fusion_mlx` 内部模块（`engine_core` / `pool` / `scheduler` / `dispatch`
/ `engines.*` / `video.*` / `model_registry` 等）随重构会变路径、变签名。
下游项目（fusion-comfyui 等）若直接 `from fusion_mlx.engine_core import ...`，
每次内部重构都可能静默断裂。

公开稳定层 `fusion_mlx/public_api.py` 解决此问题：它 re-export 一组承诺稳定
的符号，下游统一用 `from fusion_mlx.public_api import X`，内部重构不影响下游。

## 稳定层 `public_api.__all__`

`fusion_mlx/public_api.py` 的 `__all__` 列出全部对外承诺稳定的符号（当前 20 个）：

- 引擎类：`TTSEngine` / `STTEngine` / `STSEngine` / `EmbeddingEngine` /
  `RerankerEngine` / `ImageGenEngine` / `VideoGenEngine` / `VLMBatchedEngine`
- 引擎池：`EnginePool`（sequential offload 核心依赖）
- 配置与注册：`get_config` / `get_registry` / `list_available_models` /
  `ServerConfig` / `MemoryConfig` / `MemoryTier`
- 视频 pipeline：`LipsyncPipelineMLX` / `MuseTalkPipeline` / `PuLIDPipeline`
- 服务入口：`Server` / `create_app` / `__version__`

下游新增对外依赖时，优先把它提升进 `public_api.__all__`，而非让下游深入内部模块。

## CI import guard (#615)

`scripts/check_public_api_boundary.py` 扫描下游源码，检测
`from fusion_mlx.<内部模块> import` 越界导入。

### 边界定义

模块**路径**是边界，不是符号：

- `from fusion_mlx.public_api import X` → OK（公开入口）
- `from fusion_mlx import X` → OK（包顶层）
- `from fusion_mlx.<任何子模块> import X` → 越界（WARN），即使 `X` 也在
  `public_api.__all__` 里。因为路径才是会随重构断裂的东西。

### 白名单

`scripts/public_api_whitelist.txt` 列出迁移窗口内 grandfathered 的现有下游
导入（每行 `module:Symbol`，`*` 通配符号）。这些是 fusion-comfyui 当前真实
在用的 14 对，guard 对它们放行；**新增**越界导入不在白名单 → 触发 CI warning。

### 运行

CI job `public-api-boundary`（`.github/workflows/ci.yml`）在 PR 时浅克隆
fusion-comfyui 下游并运行 guard，warn-only（不带 `--fail-on-warn`）——
迁移窗口内只警示不阻断，等下游完成 `public_api` 迁移后再收紧为 fail。

本地手测：

```bash
# 扫单个文件 / 目录
python scripts/check_public_api_boundary.py --root tests/

# 扫下游 + 白名单
python scripts/check_public_api_boundary.py \
    --downstream /path/to/fusion-comfyui \
    --whitelist scripts/public_api_whitelist.txt

# 收紧：任何越界即 fail（迁移窗口结束后启用）
python scripts/check_public_api_boundary.py --downstream ... --fail-on-warn
```

### 迁移目标

白名单中已在 `public_api.__all__` 的符号（`EnginePool` /
`ImageGenEngine` / `VideoGenEngine` / `TTSEngine` / `VLMBatchedEngine` /
`get_registry` / `list_available_models` / `LipsyncPipelineMLX` /
`MuseTalkPipeline` / `PuLIDPipeline` / `get_config`）下游可立即迁移到
`from fusion_mlx.public_api import X`，迁移后删对应白名单行。

`_torch_stub.install` 是内部基础设施细节，暂无公开面，留在白名单直到出现
公开 hook 或下游不再需要。
