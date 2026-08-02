# 架构合规整改计划

> 审计日期: 2026-08-02
> 关联 Issue: #309
> 违规等级: P0（架构性违规，必须立即整改）
> 合规评级: C

## 层级定位

**一、底层算力基座** — Mac原生离线推理核心引擎

核心职责：提供 OpenAI 兼容 HTTP API 的本地 MLX 推理服务，仅此一项。

## 违规项与整改

| # | 违规项 | 整改方案 | 目标去向 | 截止 |
|---|--------|----------|----------|------|
| 1 | admin/ Web管理面板 | 整体目录迁出 | fusion-studio 或独立 fusion-admin | P0-S1 |
| 2 | gui_compat/ macOS App | 整体目录迁出 | 桌面应用层 | P0-S1 |
| 3 | eval/ 评测框架 | 迁至 fusion-bench，本项目只提供 /v1/completions 推理接口 | fusion-bench | P0-S1 |
| 4 | training/service.py | 迁至 fusion-trainer，本项目只暴露训练底层 Python API | fusion-trainer | P0-S1 |
| 5 | cluster/mdns.py | 迁至 fusion-multi-node | fusion-multi-node | P0-S2 |
| 6 | agents/ Agent适配器 | 删除，Agent 编排属上层 | - | P0-S2 |
| 7 | gradio 依赖 | 移除，pyproject.toml 删除 chat 可选依赖中的 gradio | - | P0-S2 |

## 整改阶段

### P0-S1（立即执行）
- [ ] admin/ 迁出
- [ ] gui_compat/ 迁出
- [ ] eval/ 迁出
- [ ] training/ 迁出

### P0-S2（S1 完成后）
- [ ] cluster/ 迁出
- [ ] agents/ 删除
- [ ] gradio 依赖移除

## 合规标准

整改完成后，fusion-mlx 应只包含：
- MLX 推理引擎核心代码
- OpenAI 兼容 HTTP API 服务
- 模型加载/卸载/管理 API
- 配置与日志

任何 UI、评测、训练、集群、Agent 相关代码均不应存在。
