---
doc_id: fusion-mlx/product-page
product: fusion-mlx
class: core
entry: service
doc_type: product-page
title: { zh: "fusion-mlx 产品页", en: "fusion-mlx product page" }
audience: evaluator
status: draft
version: 0.10.52
updated: 2026-11-07
owner: "@dahai"
source_of_truth:
  - README.md
  - start.sh
  - fusion_mlx/
---

# fusion-mlx

```yaml
slug: mlx
name: fusion-mlx
tagline:
  zh: Apple Silicon 本地推理引擎，OpenAI 兼容 API，一核九端全部流量的模型执行层
  en: Local inference engine for Apple Silicon with an OpenAI-compatible API; the model execution layer for the whole fusion ecosystem
cta:
  zh: 五分钟在本机起一个 OpenAI 兼容端点
  en: Stand up an OpenAI-compatible endpoint on your machine in five minutes
repo: https://github.com/dahai80/fusion-mlx
```

## core 边界

```yaml
core:
  serves: 全部 fusion-* Python 应用 / fusion-cli / fusion-studio，经 HTTP 调用，不直连库
  depends_on: 无（推理栈本体，仅依赖 MLX / PyPI 生态）
  consumed_via: OpenAI 兼容 HTTP API（:11434），另有 Admin API 与 Prometheus /metrics
  replaceable: 通信协议兼容 OpenAI；上游换 vLLM/ollama 需重写调度与量化层
```

## capabilities

```yaml
capabilities:
  - icon: ⚡
    name:
      zh: 推理服务
      en: Inference serving
    tag:
      zh: 应用方只发 OpenAI 格式请求，不接触 MLX 细节
      en: Callers send OpenAI-format requests and never touch MLX internals
    points:
      - zh: /v1/chat/completions、/v1/completions、/v1/embeddings、Anthropic 兼容端点
        en: /v1/chat/completions, /v1/completions, /v1/embeddings, plus an Anthropic-compatible endpoint
      - zh: 并发批量解码与 chunked prefill，由调度器统一排队
        en: Concurrent batched decoding and chunked prefill, queued by the scheduler
      - zh: 约束：API key 默认必填，匿名访问需显式开 FUSION_ALLOW_ANONYMOUS
        en: Constraint: an API key is required by default; anonymous access needs FUSION_ALLOW_ANONYMOUS
    viz:
      - ["OpenAI 兼容", "chat / completions / embeddings"]
  - icon: 🚀
    name:
      zh: 投机解码
      en: Speculative decoding
    tag:
      zh: 按模型自动选路径，应用方不改一行代码
      en: Path auto-selected per model; callers change no code
    points:
      - zh: DFlash / DSpark / MTP / VLM-MTP 多路径，auto-router 按模型能力选择
        en: DFlash, DSpark, MTP and VLM-MTP paths, chosen by an auto-router per model
      - zh: 约束：仅部分模型有 draft 权重，无 draft 时回退普通解码
        en: Constraint: only some models ship draft weights; without one it falls back to standard decoding
    viz: []
  - icon: 🧰
    name:
      zh: 工具与多模态
      en: Tools and multimodality
    tag:
      zh: agent 调用方获得统一 tool_call 协议
      en: Agent callers get one unified tool_call protocol
    points:
      - zh: tool calling 含 Gemma/Hermes/MiniMax 多模板回退解析
        en: Tool calling with Gemma/Hermes/MiniMax template fallback parsers
      - zh: 图像生成/编辑、视频输入、TTS 走同一服务进程（模态 extras 按需装）
        en: Image generation/editing, video input and TTS run in the same process (modality extras installed on demand)
      - zh: 约束：多模态 extras 显著增大安装体积，slim 安装只含文本
        en: Constraint: multimodal extras grow the install substantially; the slim install is text-only
    viz: []
  - icon: 🖥
    name:
      zh: 运维面
      en: Operations surface
    tag:
      zh: start.sh 一个入口管生命周期
      en: One start.sh entry manages the lifecycle
    points:
      - zh: start/stop/restart/status/log/doctor/tune 子命令
        en: start/stop/restart/status/log/doctor/tune subcommands
      - zh: Prometheus /metrics、/health 就绪态、drain 优雅切流
        en: Prometheus /metrics, /health readiness, and drain for graceful failover
      - zh: 约束：单机单实例设计，多机由 fusion-multi-node 负责
        en: Constraint: designed as a single instance per machine; multi-node is fusion-multi-node's job
    viz: []
```

## workflow

```yaml
workflow:
  - name:
      zh: 安装
      en: Install
    action:
      zh: pip 安装（slim 默认或按模态选 extras）
      en: pip install (slim default or per-modality extras)
    input:
      zh: Apple Silicon Mac，Python 3.12
      en: Apple Silicon Mac, Python 3.12
    output:
      zh: fusion-mlx CLI 可用
      en: fusion-mlx CLI available
  - name:
      zh: 启动
      en: Start
    action:
      zh: ./start.sh start 或 fusion-mlx serve <model>
      en: ./start.sh start or fusion-mlx serve <model>
    input:
      zh: 模型别名或 HuggingFace repo id
      en: Model alias or HuggingFace repo id
    output:
      zh: :11434 就绪，/health 返回 ready
      en: :11434 ready, /health reports ready
  - name:
      zh: 调用
      en: Call
    action:
      zh: 应用以 OpenAI SDK 指向 localhost:11434
      en: Applications point the OpenAI SDK at localhost:11434
    input:
      zh: API key（~/.fusion-mlx/settings.json）
      en: API key (from ~/.fusion-mlx/settings.json)
    output:
      zh: 标准 chat.completion 响应
      en: Standard chat.completion response
  - name:
      zh: 运维
      en: Operate
    action:
      zh: status 看负载，doctor 体检，drain 切流
      en: status for load, doctor for health checks, drain for failover
    input:
      zh: 无
      en: none
    output:
      zh: 服务持续可用或安全下线
      en: Continued availability or a safe shutdown
```

## useCases

```yaml
useCases:
  - name:
      zh: 领域应用接本地模型
      en: Domain apps on local models
    headline:
      zh: 应用改一个 base_url 就从云端切到本机
      en: An app switches from cloud to local by changing one base_url
    body:
      zh: 应用原来持 OpenAI key 调 api.openai.com；改为指向 localhost:11434 并填本机 API key，请求/响应结构不变。差别：数据不出机、无网络延迟、无按 token 计费。
      en: The app previously called api.openai.com with an OpenAI key; it now points at localhost:11434 with the local API key, and the request/response shape is unchanged. Difference: data stays on the machine, no network latency, no per-token billing.
  - name:
      zh: agent 框架的工具调用
      en: Tool calling for agent frameworks
    headline:
      zh: agent 循环拿到统一 tool_call 而不用适配各家模板
      en: Agent loops get one tool_call format instead of adapting per-template parsers
    body:
      zh: agent 框架原来为每个模型家族写解析分支；现在服务端在单一收口点把 Gemma/Hermes/MiniMax 等输出归一为 tool_use。差别：新增模型家族时应用零改动。
      en: The framework previously kept per-family parsing branches; the server now normalizes Gemma/Hermes/MiniMax-style output into tool_use at a single chokepoint. Difference: new model families require zero app changes.
```

## techDepth

```yaml
techDepth:
  - category:
      zh: 调度
      en: Scheduling
    items:
      - name: { zh: 内存分级, en: Memory tiers }
        desc: { zh: 按可用统一内存分档决定并发与驻留策略, en: Concurrency and residency policies are tiered by available unified memory }
        steps:
          - zh: 配置见 docs/config-reference.md 的 Memory Tiers 节
            en: See the Memory Tiers section in docs/config-reference.md
      - name: { zh: 类型化执行器池, en: Typed executor pools }
        desc: { zh: 按任务类型划分线程池，避免长任务阻塞短请求, en: Thread pools are partitioned per task type so long jobs do not block short requests }
        steps:
          - zh: 配置见 docs/config-reference.md 的 Typed Executor Pools 节
            en: See the Typed Executor Pools section in docs/config-reference.md
```

## 提交检查清单

- [x] 所有字段 zh/en 成对
- [x] 数字均有出处（端口/路径/端点来自 start.sh 与实测 /health、/v1/chat/completions）
- [x] 每条能力含一条约束或限制
- [x] 无性能数字声明（未在本机做正式 bench，标注留白）
