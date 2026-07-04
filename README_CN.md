<div align="center">

# fusion-mlx

**Apple Silicon 统一本地模型推理服务**

Ollama / vLLM 的直接替代 —— 基于 MLX 原生运行在 Metal 上

[![Version](https://img.shields.io/badge/v0.4.1-blue.svg)](https://github.com/dahai80/fusion-mlx/releases)
[![Python](https://img.shields.io/badge/Python-3.11+-3776AB.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-Apache--2.0-green.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/Tests-1200+-success.svg)](tests/)

[English](README.md) | 中文

[快速开始](#快速开始) · [下载 App](https://github.com/dahai80/fusion-mlx/releases) · [性能基准](https://bench.dpdns.org/) · [文档](docs/)

</div>

---

## 为什么选择 fusion-mlx？

| | fusion-mlx | omlx | Ollama |
|---|---|---|---|
| Continuous batching | ✅ | ✅ | ❌ |
| 2× 并发吞吐量 | ✅ 36 vs 17.9 tok/s | 基准线 | — |
| TurboQuant KV (4-bit) | ✅ | ✅ (高级配置) | ❌ |
| Speculative decoding | ✅ 4 种方法 | ❌ | ❌ |
| OpenAI + Anthropic API | ✅ 双协议 | ✅ 双协议 | ✅ 双协议 |
| VLM + Paged KV cache | ✅ | ❌ | ❌ |
| 40+ 量化格式 | ✅ | ~15 | ~10 |
| macOS 原生应用 | ✅ SwiftUI | ✅ | ✅ |
| 8 种推理引擎 | ✅ | 2 | 2 |
| Admin 管理面板 | ✅ | ✅ | ❌ |

**性能基准** (Qwen3.6-27B, Apple M2 Ultra 137GB):

| 量化方式 | 模型体积 | bpw | 解码速度 | vs mxfp8 | vs mixed_3_4 |
|---|---|---|---|---|---|
| mxfp8 | 26 GB | 8.0 | 18.5 tok/s | 基准线 | — |
| mxfp4 | 13 GB | 4.0 | 32.3 tok/s | **+75%** | — |
| mixed_4_6 | 15 GB | 4.85 | 29.0 tok/s | **+57%** | — |
| mixed_3_4 | 12 GB | 3.68 | 36.2 tok/s | **+96%** | 基准线 |
| mixed_2_6 | 10 GB | 3.25 | 39.3 tok/s | **+112%** | +9% |
| mixed_2_4 | 9.3 GB | 2.95 | 42.8 tok/s | **+131%** | +18% |
| quant2 | 8.5 GB | 2.72 | 45.1 tok/s | **+144%** | +25% |
| quant2-g128 | 7.8 GB | 2.46 | 48.2 tok/s | **+161%** | +33% |
| quant2-all | 7.5 GB | 2.37 | 48.5 tok/s | **+162%** | **+34%** |
| quant2-flat | 7.1 GB | 2.25 | 49.4 tok/s | **+167%** | +36%* |

*\*quant2-flat: 极限速度，但 2-bit embedding 会损失质量。推荐使用 quant2-all 获得最佳质量/速度平衡。*

核心优化：quant2/quant2_128/quant2_flat 超激进 2-bit 量化方案、混合精度量化（降低内存带宽）、greedy decode 快速路径（argmax 跳过 logsumexp）、融合 QKV/gate 投影、融合 decode sampler、async_eval 双缓冲、GatedDeltaNet 线性注意力快速路径、StreamingJSONEncoder、B=1 快速路径。

## 特性一览

- **8 种推理引擎** — LLM、VLM、Embedding、Reranker、STT、TTS、STS、ImageGen (Flux 2)
- **OpenAI + Anthropic 双协议** — 一个服务同时支持两套 API，完全兼容
- **Continuous batching** — 类 vLLM 调度器，支持 chunked prefill、抢占式调度、优先级队列
- **Speculative decoding** — SuffixDecoding、DFlash、MTP、VLM MTP（2–5× 加速生成）
- **TurboQuant KV** — 4-bit KV cache 量化，内存访问量降低约 4 倍
- **40+ 量化格式** — GGUF (Q2_K → Q8_0)、Imatrix (IQ1_M → IQ4_XS)、TurboQuant (TQ1_0/TQ2_0)、MLX (mxfp4/mxfp8/6bit/4bit/8bit/F16/BF16/F32)
- **Paged KV cache** — SSD 冷数据层、block-aware prefix caching、COW 共享
- **Fused sampler** — 跳过 logsumexp、消除 GPU 同步、批量采样
- **SmartRouter** — 阶段感知路由，基于性能基准的后端选择，EMA 平滑
- **优先级调度** — REALTIME / BATCH / BACKGROUND 队列，配合 Metal command queue 优先级
- **4 级内存守护** — safe / balanced / aggressive / custom 硬限制，无死锁驱逐策略
- **多模型并发** — EnginePool 支持 LRU 驱逐、模型锁定（pinning）和 TTL
- **MCP 工具支持** — 通过 API 列出、发现和执行 MCP 工具
- **Admin 管理面板** — 模型管理、在线对话、HuggingFace 下载、在线量化
- **macOS 原生应用** — SwiftUI 菜单栏、自动更新、基准测试、模型管理
- **8 种集成** — Claude Code、OpenClaw、ComfyUI、Copilot、Codex、OpenCode、Pi、Hermes

## 快速开始

```bash
# 安装
pip install fusion-mlx

# 启动服务
fusion-mlx serve --model-dir ~/.cache/huggingface

# 对话
curl http://localhost:8000/v1/chat/completions \
   -H "Content-Type: application/json" \
   -d '{
     "model": "Qwen3-4B-Q4_K_M",
     "messages": [{"role": "user", "content": "2+2等于几？"}],
     "max_tokens": 64
   }'
```

OpenAI Python 客户端：

```python
from openai import OpenAI
client = OpenAI(base_url="http://localhost:8000/v1", api_key="local")
resp = client.chat.completions.create(
    model="Qwen3-4B-Q4_K_M",
    messages=[{"role": "user", "content": "2+2等于几？"}],
    max_tokens=64,
)
print(resp.choices[0].message.content)
```

Anthropic API：

```python
import anthropic
client = anthropic.Anthropic(base_url="http://localhost:8000/v1", api_key="local")
resp = client.messages.create(
    model="Qwen3-4B-Q4_K_M",
    max_tokens=64,
    messages=[{"role": "user", "content": "2+2等于几？"}],
)
print(resp.content[0].text)
```

## 安装

### pip 安装（推荐）

```bash
pip install fusion-mlx
```

### 从源码安装

```bash
git clone https://github.com/dahai80/fusion-mlx.git
cd fusion-mlx
pip install -e .
```

### macOS App

从 [GitHub Releases](https://github.com/dahai80/fusion-mlx/releases) 下载原生 SwiftUI 应用，支持：
- 一键启动模型和服务控制
- 量化模式切换：**oQ Online**（基于灵敏度）/ **MLX Recipe**（预调优方案）
- 吞吐量和精度基准测试
- 自动更新
- 模型管理和下载
- 菜单栏实时服务状态

## 支持的模型

| 类型 | 引擎 | 示例模型 |
|------|------|----------|
| LLM | `BatchedEngine` | Qwen、Llama、Mistral、Gemma、DeepSeek、Kimi |
| VLM | `VLMBatchedEngine` | Qwen2-VL、LLaVA、InternVL |
| Embedding | `EmbeddingEngine` | BGE、E5、GTE |
| Reranker | `RerankerEngine` | Cohere、Jina rerankers |
| STT | `STTEngine` | Whisper、VibeVoice-ASR |
| TTS | `TTSEngine` | Kokoro、VibeVoice |
| ImageGen | `ImageGenEngine` | Flux 2 |

## 量化格式

| 类别 | 格式 |
|------|------|
| GGUF/GGML | Q2_K, Q3_K_S/M/L, Q4_0, Q4_1, Q4_K_S/M, Q5_0, Q5_1, Q5_K_S/M, Q6_K, Q8_0, Q8_K |
| Imatrix | IQ1_M, IQ2_S, IQ2_XS, IQ2_XXS, IQ3_M, IQ3_S, IQ4_NL, IQ4_XS |
| TurboQuant | TQ1_0, TQ2_0 |
| MLX 原生 | mxfp4, mxfp8, 6bit (ParoQuant), 4bit, 8bit, F16, BF16, F32 |
| MLX 量化方案 | mixed_3_4, mixed_2_6, mixed_2_4, mixed_3_6, mixed_4_6, quant2_all, quant2, quant2_128, quant2_flat（见下方） |

### 量化方案（Quantization Recipes）

MLX 量化方案提供预调优的混合精度计划，可最大化 Apple Silicon 解码速度。两种模式均输出标准 mlx-lm safetensors，兼容任何 MLX 运行时。

macOS 应用提供模式切换：

- **oQ Online** — 基于灵敏度的逐层量化（原始模式）
- **MLX Recipe** — 预调优量化方案，底层调用 `mlx_lm.convert --quant-recipe <name>`

| 方案 | 标签 | BPW | 相对 mxfp8 加速 | 类别 |
|------|------|-----|-----------------|------|
| mixed_3_4 | Mixed 3/4-bit | 3.68 | +96% | 推荐 |
| mixed_2_6 | Mixed 2/6-bit | 3.25 | +112% | 推荐 |
| mixed_2_4 | Mixed 2/4-bit | 2.95 | +131% | 激进 |
| mixed_3_6 | Mixed 3/6-bit | 4.0 | +75% | 均衡 |
| mixed_4_6 | Mixed 4/6-bit | 4.85 | +57% | 保守 |
| quant2_all | quant2-all | 2.37 | +162% | 推荐 |
| quant2 | quant2 | 2.72 | +144% | 激进 |
| quant2_128 | quant2-g128 | 2.46 | +161% | 激进 |
| quant2_flat | quant2-flat | 2.25 | +167% | 实验性 |
| mxfp4 | MLX FP4 | 4.0 | +75% | 保守 |
| mxfp8 | MLX FP8 | 8.0 | 基准线 | 保守 |

**推荐**：`mixed_3_4` 或 `quant2_all` 获得最佳质量/速度平衡。**保守**：`mixed_4_6` 或 `mxfp4` 适用于质量优先场景。**激进**：`mixed_2_4` 或 `quant2` 适用于受限内存下追求极限速度。

## API 兼容性

| API | 端点 | 状态 |
|-----|------|------|
| OpenAI Chat | `/v1/chat/completions`, `/v1/models` | ✅ 完全兼容 |
| OpenAI Legacy | `/v1/completions` | ✅ 支持 |
| Anthropic Messages | `/v1/messages`, `/v1/count_tokens` | ✅ 完全兼容 |
| Audio | `/v1/audio/transcriptions`, `/v1/audio/speech` | ✅ 支持 |
| Images | `/v1/images/generate` | ✅ 支持 (Flux 2) |
| Embeddings | `/v1/embeddings` | ✅ 支持 |
| MCP | `/v1/mcp/tools`, `/v1/mcp/servers`, `/v1/mcp/execute` | ✅ 支持 |
| OpenClaw Agent | `/v1/openclaw/agent/*` | ✅ 会话、多轮、工具调用、SSE 流式 |

## 模型别名

```bash
fusion-mlx serve --model claude-4.6-sonnet   # → Qwen3.5-27B-Claude-4.6-Opus-Distilled-MLX-6bit
fusion-mlx serve --model gpt-4o               # → Qwen3-32B-A3B-Think-2512-MLX
```

## 集成

```bash
# Claude Code — 用 fusion-mlx 作为本地 Anthropic API
fusion-mlx launch claude

# OpenClaw — 批量 Agent 处理
fusion-mlx launch openclaw --model Qwen3-4B

# ComfyUI — Flux 2 图像生成
fusion-mlx launch comfyui

# GitHub Copilot
fusion-mlx launch copilot
```

## Admin 管理面板

访问 `http://localhost:8000/admin`：

- **模型管理** — 动态加载/卸载/锁定模型，ParoQuant 兼容检测
- **在线对话** — 实时对话界面，可测试任何模型
- **下载** — HuggingFace / ModelScope 模型下载，带进度追踪
- **量化** — 在线量化 (oQ) 流水线
- **基准测试** — 吞吐量和精度评测
- **监控** — 实时内存、性能和请求指标
- **设置** — 全局/单模型配置、子 API key 管理

## 性能

Apple M5 Max (128 GB RAM) 基准测试：

| 模型 | 量化 | PP (tok/s) | TG (tok/s) | TTFT (ms) |
|-------|-------|-----------|-----------|-----------|
| Qwen3.6-27B | mxfp8 | 264 | 29.8 | ~1000 |
| Qwen2.5-3B | Q4_K_M | 580 | 32 | ~500 |

并发吞吐量 (Qwen3.6-27B-mxfp8, 4 个并发请求)：

| 指标 | fusion-mlx | omlx |
|---|---|---|
| 聚合 TG | 36.0 tok/s | 17.9 tok/s |
| 单请求 TG | ~9 tok/s | ~9 tok/s |

欢迎提交你的基准测试结果到 [bench.dpdns.org](https://bench.dpdns.org/)。

## 项目结构

```
fusion-mlx/
├── fusion_mlx/
│    ├── api/             # OpenAI、Anthropic、Audio、Images、MCP、OpenClaw 路由
│    ├── cache/           # PagedCache、PagedSSDCache、PrefixCache
│    ├── engines/         # 8 种引擎 (LLM、VLM、Embedding 等)
│    ├── integrations/    # Claude Code、OpenClaw、ComfyUI、Copilot、Codex 等
│    ├── parsers/         # 工具调用解析器 (Gemma、Harmony、Hermes 等)
│    ├── pool/            # EnginePool、MemoryEnforcer、ModelDiscovery、PriorityScheduler
│    ├── router/          # RequestRouter、CloudRouter、SmartRouter
│    ├── scheduler/       # 25 模块调度器 (admission、batching、cache、step 等)
│    ├── speculative/     # SuffixDecoding、DFlash、MTP、VLM MTP
│    └── admin/           # 管理面板路由、基准测试、下载、设置
├── apps/fusion-mac/      # SwiftUI macOS 应用 (~80 个 Swift 文件)
├── docs/                 # API 参考、架构、CLI 指南、配置
├── examples/             # 12 个可运行的代码示例
├── tests/                # 1200+ 测试 (单元、GUI、集成、性能)
└── downstream/           # omlx 和 Rapid-MLX 分支的同步脚本
```

## 示例

| # | 示例 | 说明 |
|---|------|------|
| 01 | `basic-chat.py` | 简单非流式对话 |
| 02 | `streaming-chat.py` | SSE 流式响应 |
| 03 | `anthropic-api.py` | Anthropic Messages API |
| 04 | `tool-calling.py` | JSON Schema 函数调用 |
| 05 | `multi-model.py` | 多模型并发请求 |
| 06 | `image-generation.py` | Flux 2 图像生成 |
| 07 | `speech-to-text.py` | Whisper STT 语音识别 |
| 08 | `text-to-speech.py` | Kokoro TTS 语音合成 |
| 09 | `mcp-tools.py` | MCP 工具发现与执行 |
| 10 | `python-sdk.py` | OpenAI Python 客户端集成 |
| 11 | `comfyui-workflow.py` | ComfyUI 工作流执行 |
| 12 | `openclaw-agent.py` | OpenClaw Agent 协议 |

## 文档

- [API Reference](docs/api-reference.md) — 所有端点及请求/响应示例（英文）
- [Architecture](docs/architecture.md) — EnginePool、Scheduler（25 模块）、Cache 层、SmartRouter（英文）
- [CLI Reference](docs/cli-reference.md) — 所有命令和参数（英文）
- [Configuration](docs/configuration.md) — 内存分级、调度器设置、TurboQuant、别名、执行器线程池（英文）
- [架构详解](docs/architecture_CN.md) — 架构文档中文版
- [配置指南](docs/configuration_CN.md) — 配置文档中文版

## 许可证

Apache-2.0

## 致谢

- [MLX](https://github.com/ml-explore/mlx) 和 [mlx-lm](https://github.com/ml-explore/mlx-lm) — Apple 出品
- [mlx-vlm](https://github.com/Blaizzy/mlx-vlm) — Apple Silicon 上的视觉语言模型推理
- [vllm-mlx](https://github.com/waybarrios/vllm-mlx) — oMLX 起源于 vllm-mlx v0.1.0
- [omlx](https://github.com/jundot/omlx) — Continuous batching 和分层 KV 缓存
- [Rapid-MLX](https://github.com/raullenchai/Rapid-MLX) — Speculative decoding、多模态、云路由
- [dflash-mlx](https://github.com/bstnxbt/dflash-mlx) — Block diffusion speculative decoding
- [mlx-embeddings](https://github.com/Blaizzy/mlx-embeddings) — Embedding 模型支持
- [venvstacks](https://venvstacks.lmstudio.ai) — macOS 应用的可移植 Python 环境层
