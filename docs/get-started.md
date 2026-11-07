---
doc_id: fusion-mlx/get-started
product: fusion-mlx
class: core
entry: service
doc_type: get-started
title: { zh: "fusion-mlx 快速上手", en: "fusion-mlx quick start" }
audience: engineer
status: draft
version: 0.10.52
updated: 2026-11-07
owner: @dahai
source_of_truth:
  - README.md
  - start.sh
---

# fusion-mlx 快速上手（Get Started）

> 目标：5–10 分钟让推理服务在本机跑起来并完成第一次调用。

## 1. 前置条件

- Apple Silicon Mac，macOS 14+
- Python 3.12（仓库统一 `.venv`；见根目录 `scripts/sync-deps.sh`）
- 无必须预启动的其他 fusion 服务（fusion-mlx 是底层）

## 2. 安装

```bash
# 方式 A：从 PyPI（slim 默认，仅文本模态）
pip install fusion-mlx

# 方式 B：仓库开发态
pip install -e .
```

确认点：

```bash
fusion-mlx --version
# 期望输出：版本号（如 0.10.52）
```

需要图像/视频/TTS 时按 `docs/config-reference.md` 的 Install Extras 节追加 extras。

## 3. 启动

```bash
./start.sh start        # 仓库内；PyPI 用户用: fusion-mlx serve <model>
```

确认点：

```bash
./start.sh status
# 期望：● Running  PID=...  ADDR=127.0.0.1
# Health: {"status":"healthy","ready":true,...}
curl -s localhost:11434/health
# 期望：{"status":"healthy","ready":true,...,"version":"..."}
```

- 监听端口：`11434`（已在 `~/fusion/architecture/port-registry.yaml` 注册）
- 日志：`~/.fusion-mlx/logs/`
- 停止：`./start.sh stop`

## 4. 第一次调用

API key 在 `~/.fusion-mlx/settings.json` 的 `auth.api_key`（服务默认要求鉴权）。

```bash
KEY=$(python3 -c "import json;print(json.load(open('$HOME/.fusion-mlx/settings.json'))['auth']['api_key'])")
curl -s localhost:11434/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $KEY" \
  -d '{"model":"<已加载或已知存在的模型repo id>","messages":[{"role":"user","content":"say ok"}],"max_tokens":8}'
```

期望输出（节选）：

```json
{"choices":[{"message":{"content":"..."}}],"model":"...","usage":{...}}
```

模型名以 `/health` 的 `loaded_models` 或报错信息里列出的可用模型为准；
OpenAI SDK 用法：`OpenAI(base_url="http://localhost:11434/v1", api_key=<本机key>)`。

## 5. 下一步

- 完整能力与性能说明：[README](../README.md)
- 全部配置项：[config-reference](config-reference.md)
- CLI 子命令：[cli-reference](cli-reference.md)
- 架构：[architecture](architecture.md)

## 6. 常见问题（最多 3 条）

```yaml
troubleshooting:
  - q:
      zh: 请求返回 Authentication required / invalid_api_key
      en: Authentication required / invalid_api_key
    a:
      zh: 未带 Authorization 头或 key 不对。key 取自 ~/.fusion-mlx/settings.json 的 auth.api_key；匿名访问仅开发场景用 FUSION_ALLOW_ANONYMOUS=true。
      en: Missing Authorization header or wrong key. The key is auth.api_key in ~/.fusion-mlx/settings.json; anonymous access is dev-only via FUSION_ALLOW_ANONYMOUS=true.
  - q:
      zh: Model '<name>' not found
      en: Model '<name>' not found
    a:
      zh: 模型未下载或名字不对。报错会列出全部可用模型；下载新模型用 fusion-mlx serve <repo-id>（走 hf-mirror.com 镜像）。
      en: Model not downloaded or wrong id. The error lists all available models; download a new one with fusion-mlx serve <repo-id> (via the hf-mirror.com mirror).
  - q:
      zh: 启动报端口被占用
      en: Start fails with the port already in use
    a:
      zh: 11434 已有进程。./start.sh status 查现有实例；确认非本服务后用 ./start.sh stop 或换端口（需同步 port-registry）。
      en: Port 11434 is taken. Check the running instance with ./start.sh status; stop it or change the port (and sync port-registry).
```

## 提交检查清单

- [x] 每条命令本机跑过，输出真实（start.sh status、/health、/v1/chat/completions 均实测）
- [x] 每步有确认点
- [x] 端口 11434 在 port-registry 注册
- [x] zh/en 成对（命令除外）
- [x] 全篇 ≤ 120 行
