---
doc_id: fusion-mlx/troubleshooting
product: fusion-mlx
class: core
entry: service
doc_type: troubleshooting
title: { zh: "fusion-mlx 故障排查", en: "fusion-mlx troubleshooting" }
audience: operator
status: draft
version: 0.10.52
updated: 2026-11-07
owner: @dahai
source_of_truth:
  - README.md
  - start.sh
  - docs/config-reference.md
---

# fusion-mlx 故障排查（Troubleshooting）

> 按症状索引。先跑 `./start.sh doctor`，多数环境问题它会直接给出结论。

## 启动类

### 端口 11434 被占用

- 症状：`start.sh start` 失败，日志出现 `address already in use`
- 诊断：`./start.sh status` 看是否本服务残留实例；`lsof -i :11434` 找占用进程
- 解法：本服务残留用 `./start.sh stop`；其他进程占用则迁移该进程，或换端口并同步 `~/fusion/architecture/port-registry.yaml`

### start.sh 反复重启

- 症状：日志里同一条崩溃栈循环出现
- 诊断：`tail -100 ~/.fusion-mlx/logs/server.log*`；带时间戳的 `.prev` 副本保留上一次崩溃证据，不要只看最新的
- 解法：按栈顶错误定位；依赖损坏时重建 venv（根目录 `scripts/sync-deps.sh --rebuild-venv`）

## 鉴权类

### Authentication required / invalid_api_key

- 症状：任何请求返回 `{"error":{"message":"Authentication required.",...}}`
- 诊断：请求没带 `Authorization: Bearer <key>`，或 key 与 `~/.fusion-mlx/settings.json` 的 `auth.api_key` 不一致
- 解法：客户端带正确 key；开发临时放开用环境变量 `FUSION_ALLOW_ANONYMOUS=true`（仅限本机调试，勿用于对外服务）

## 模型类

### `Model '<name>' not found`

- 症状：chat 请求 404，报错列出全部可用模型
- 诊断：模型未下载，或 repo id 写错（注意 `-` 与 `--` 的差别，如 `mlx-community/Qwen3.5-4B-MLX-4bit`）
- 解法：`fusion-mlx serve <repo-id>` 下载并加载；下载走 `hf-mirror.com` 镜像，镜像配置见 `docs/config-reference.md` 的 Mirror Configuration 节

### 模型加载慢或下载中断

- 症状：首次 serve 长时间无响应，或 HF 下载超时
- 诊断：模型缓存 `~/.fusion-mlx/models`；网络到 huggingface.co 不通时必走镜像
- 解法：确认镜像变量生效（`HF_MIRROR=https://hf-mirror.com`）；大模型分次下载可续传

## 性能类

### 并发上不去 / 长任务阻塞短请求

- 症状：一个长生成任务拖慢所有请求
- 诊断：执行器池配置过小或未分型；`/metrics` 看队列深度
- 解法：按 `docs/config-reference.md` 的 Typed Executor Pools 与 Scheduler Settings 调整

### 内存压力 / 进程被杀

- 症状：加载大模型后系统卡顿或 OOM
- 诊断：内存分级（Memory Tiers）档位与实机统一内存不匹配
- 解法：按 `docs/config-reference.md` 的 Memory Tiers 降档；TurboQuant KV 与 Paged Cache 可降 KV 占用

## 观测类

### 健康检查返回 ready=false

- 症状：`/health` 返回 `{"status":"healthy","ready":false}` 或模型未加载
- 诊断：服务在启动/加载中；看 `loaded_models` 是否为空、`started_at` 是否刚变化
- 解法：等待加载完成；若长期不就绪，结合 `server.log` 中的加载错误处理

## 反馈

同一症状出现 3 次以上的，应把根因与解法补充到本文件并附 issue 链接：
https://github.com/dahai80/fusion-mlx/issues
