<div align="center">

# fusion-mlx

**Apple Silicon 统一本地模型推理服务**

Ollama / vLLM 的直接替代 -- 基于 MLX 原生运行在 Metal 上

[![Version](https://img.shields.io/badge/v0.4.8-blue.svg)](https://github.com/dahai80/fusion-mlx/releases)
[![Python](https://img.shields.io/badge/Python-3.11+-3776AB.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-Apache--2.0-green.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/Tests-1200+-success.svg)](tests/)

[English](README.md) | 中文

[快速开始](#快速开始) · [下载 App](https://github.com/dahai80/fusion-mlx/releases) · [性能基准](https://bench.dpdns.org/) · [文档](docs/)

</div>

---

## 为什么选择 fusion-mlx？

**性能基准** (Qwen3.6-27B, Apple M2 Ultra 137GB):

| 量化方式 | 模型体积 | bpw | 解码速度 | vs mxfp8 | vs mixed_3_4 |
|---|---|---|---|---|---|
| mxfp8 | 26 GB | 8.0 | 18.5 tok/s | 基准线 | - |
| mxfp4 | 13 GB | 4.0 | 32.3 tok/s | **+75%** | - |
| mixed_4_6 | 15 GB | 4.85 | 29.0 tok/s | **+57%** | - |
| mixed_3_4 | 12 GB | 3.68 | 36.2 tok/s | **+96%** | 基准线 |
| mixed_2_6 | 10 GB | 3.25 | 39.3 tok/s | **+112%** | +9% |
| mixed_2_4 | 9.3 GB | 2.95 | 42.8 tok/s | **+131%** | +18% |
| quant2 | 8.5 GB | 2.72 | 45.1 tok/s | **+144%** | +25% |
| quant2-g128 | 7.8 GB | 2.46 | 48.2 tok/s | **+161%** | +33% |
| quant2-all | 7.5 GB | 2.37 | 48.5 tok/s | **+162%** | +34% |
| quant2-flat | 7.1 GB | 2.25 | 49.4 tok/s | **+167%** | +36%* |

*\*quant2-flat: 极限速度，但 2-bit embedding 会损失质量。推荐使用 quant2-all 获得最佳质量/速度平衡。*

核心优化：quant2/quant2_128/quant2_flat 超激进 2-bit 量化方案、混合精度量化（降低内存带宽）、greedy decode 快速路径（argmax 跳过 logsumexp）、融合 QKV/gate 投影、融合 decode sampler、async_eval 双缓冲、GatedDeltaNet 线性注意力快速路径、StreamingJSONEncoder、B=1 快速路径。

## 特性一览

- **9 种推理引擎** - LLM、VLM、Embedding、Reranker、STT、TTS、STS、ImageGen (Flux 2)、VideoGen (LTX-2、Wan2、SkyReels-V3)
- **OpenAI + Anthropic 双协议** - 一个服务同时支持两套 API，完全兼容
- **Continuous batching** - 类 vLLM 调度器，支持 chunked prefill、抢占式调度、优先级队列
- **Speculative decoding** - SuffixDecoding、DFlash、DSpark、MTP、VLM MTP（2–5× 加速生成）
- **TurboQuant KV** - 4-bit KV cache 量化，内存访问量降低约 4 倍
- **40+ 量化格式** - GGUF (Q2_K -> Q8_0)、Imatrix (IQ1_M -> IQ4_XS)、TurboQuant (TQ1_0/TQ2_0)、MLX (mxfp4/mxfp8/6bit/4bit/8bit/F16/BF16/F32)
- **Paged KV cache** - SSD 冷数据层、block-aware prefix caching、COW 共享
- **Fused sampler** - 跳过 logsumexp、消除 GPU 同步、批量采样
- **SmartRouter** - 阶段感知路由，基于性能基准的后端选择，EMA 平滑
- **优先级调度** - REALTIME / BATCH / BACKGROUND 队列，配合 Metal command queue 优先级
- **4 级内存守护** - safe / balanced / aggressive / custom 硬限制，无死锁驱逐策略
- **多模型并发** - EnginePool 支持 LRU 驱逐、模型锁定（pinning）和 TTL
- **MCP 工具支持** - 通过 API 列出、发现和执行 MCP 工具
- **Admin 管理面板** - 模型管理、在线对话、HuggingFace 下载、在线量化
- **macOS 原生应用** - SwiftUI 菜单栏、自动更新、基准测试、模型管理、**硬件感知设置向导**
- **SkyReels-V3 视频生成** - 最强开源视频生成模型纯 MLX 移植，R2V/V2V/A2V 三大分支全部真实权重端到端跑通，M5 Max 专属 dFlash 注意力 + NF4 量化，19B 模型 720P 常驻内存 ≤ 14GB
- **PyTorch -> MLX 全模型转换器** - `convert_skyreels_v3.py` 一键转换 SkyReels-V3 三分支 (DiT + T5 + VAE + CLIP + audio) PyTorch 权重到 MLX safetensors，支持 bfloat16/float16/float32 + NF4 量化，分 shard 增量写盘避统一内存冲高

### 高级特性推荐

首次启动 macOS 应用时，**6 步 Welcome 向导**会自动检测你的 Mac 硬件并推荐最优配置：

| 使用场景 | 推荐模型（可多选下载） | DFlash | DSpark | TurboQuant | 最大上下文 |
|----------|----------------------|--------|--------|------------|-----------|
| 🤖 Agent (OpenClaw) | DeepSeek-V4-Flash, Qwen3.6-27B | ✅ | ❌ | ✅ (≥64GB) | 65K |
| 💻 编程 | Qwen3.5-9B, DeepSeek-Coder-V2 | ❌ | ✅ | ✅ (≥64GB) | 131K |
| 💬 聊天 | Qwen3.5-9B, Gemma-4-31B | ❌ | ❌ | ✅ (≥64GB) | 32K |

推荐基于实时硬件检测（CPU 核心、统一内存、GPU 带宽、磁盘空间）。所有设置可手动修改，超范围值会显示校验警告。

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
| VideoGen | `VideoGenEngine` | LTX-2、Wan2、SkyReels-V3（纯 MLX 移植） |

## 量化格式

| 类别 | 格式 |
|------|------|
| GGUF/GGML | Q2_K, Q3_K_S/M/L, Q4_0, Q4_1, Q4_K_S/M, Q5_0, Q5_1, Q5_K_S/M, Q6_K, Q8_0, Q8_K |
| Imatrix | IQ1_M, IQ2_S, IQ2_XS, IQ2_XXS, IQ3_M, IQ3_S, IQ4_NL, IQ4_XS |
| TurboQuant | TQ1_0, TQ2_0 |
| MLX 原生 | mxfp4, mxfp8, 6bit (ParoQuant), 4bit, 8bit, F16, BF16, F32 |
| MLX 量化方案 | mixed_3_4, mixed_2_6, mixed_2_4, mixed_3_6, mixed_4_6, quant2_all, quant2, quant2_128, quant2_flat（见下方） |
| NVFP4 (只读) | NVFP4 (E2M1 + E4M3 block scale) - NVIDIA 4-bit checkpoint 加载时反量化为 bf16 (#179) |

> **NVFP4** 是格式兼容桥，非速度路径：NVIDIA NVFP4 权重 (4-bit E2M1, 每字节 2 个, 带 E4M3 block scale) 在 `safetensors` 加载时被检测并反量化为 bf16，使外部量化的 NVFP4 DiT checkpoint 无需单独转换即可运行。4-bit 存储优势在推理时不保留。检测保守 (uint8 权重 + 同级 block-scale, 每 16 元素 1 scale)，对非 NVFP4 checkpoint 静默 no-op。

### 量化方案（Quantization Recipes）

MLX 量化方案提供预调优的混合精度计划，可最大化 Apple Silicon 解码速度。两种模式均输出标准 mlx-lm safetensors，兼容任何 MLX 运行时。

macOS 应用提供模式切换：

- **oQ Online** - 基于灵敏度的逐层量化（原始模式）
- **MLX Recipe** - 预调优量化方案，底层调用 `mlx_lm.convert --quant-recipe <name>`

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

### 模型转换（convert）

使用 `convert` 命令把任意 HuggingFace 模型转换为 MLX（可选量化），接受模型别名或完整 HF 仓库：

```bash
fusion-mlx convert qwen3.5-9b --quant-bits 4 -o ./qwen3.5-9b-4bit
fusion-mlx convert mlx-community/Qwen3.5-9B --quant-bits 8 --upload-repo me/my-repo
```

这是保存到磁盘的**权重**量化，与 TurboQuant KV cache 压缩（`--kv-cache-turboquant`，运行时参数）不同。详见 [CLI 参考](docs/cli-reference.md)。

## API 兼容性

| API | 端点 | 状态 |
|-----|------|------|
| OpenAI Chat | `/v1/chat/completions`, `/v1/models` | ✅ 完全兼容 |
| OpenAI Legacy | `/v1/completions` | ✅ 支持 |
| Anthropic Messages | `/v1/messages`, `/v1/count_tokens` | ✅ 完全兼容 |
| Audio | `/v1/audio/transcriptions`, `/v1/audio/speech` | ✅ 支持 |
| Images | `/v1/images/generate` | ✅ 支持 (Flux 2) |
| Videos | `/v1/videos/generate` | ✅ 支持 (LTX-2、Wan2、SkyReels-V3；纯 MLX 移植) |
| Embeddings | `/v1/embeddings` | ✅ 支持 |
| MCP | `/v1/mcp/tools`, `/v1/mcp/servers`, `/v1/mcp/execute` | ✅ 支持 |
| OpenClaw Agent | `/v1/openclaw/agent/*` | ✅ 会话、多轮、工具调用、SSE 流式 |
| Agent Graph | `/v1/agents/graphs`, `/v1/agents/run` | ✅ CRUD + 导出 + 运行 (内存态) |
| Base Info | `/v1/base` | ✅ MLX 运行时能力检测 |
| Convert / Quantize | `/v1/convert`, `/v1/quantize` (+ `.../jobs/{id}`) | ✅ 异步 HF->MLX 转换 + 权重量化 |

## 模型别名

```bash
fusion-mlx serve --model claude-4.6-sonnet   # -> Qwen3.5-27B-Claude-4.6-Opus-Distilled-MLX-6bit
fusion-mlx serve --model gpt-4o               # -> Qwen3-32B-A3B-Think-2512-MLX
```

## 集成

```bash
# Claude Code - 用 fusion-mlx 作为本地 Anthropic API
fusion-mlx launch claude

# OpenClaw - 批量 Agent 处理
fusion-mlx launch openclaw --model Qwen3-4B

# ComfyUI - Flux 2 图像生成
fusion-mlx launch comfyui

# GitHub Copilot
fusion-mlx launch copilot
```

## Pipeline Stage API 与步进回调 (Fusion-ComfyUI)

针对需要逐阶段控制生成流水线（而非单次 `generate()` 调用）的 ComfyUI 类集成，图像与视频引擎提供流式 stage API 加每步进度回调。

### Stage API (#170)

`ImageGenEngine` 与 `VideoGenEngine` 暴露成对的 load / run / unload 方法，宿主可独立持有 text encoder、DiT、VAE 并在阶段间释放显存（`gc.collect()` + `mx.metal.clear_cache()` + 活跃内存日志）：

| 阶段 | Load | Run | Unload |
|---|---|---|---|
| Text encoder | `load_text_encoder()` | `encode_text(prompt) -> {"embed","text_ids"}` | `unload_text_encoder()` |
| DiT | `load_dit()` | `denoise(latent, pos_embed, neg_embed, steps, cfg, seed[, num_frames])` | `unload_dit()` |
| VAE | `load_vae()` | `decode(latent)` / `decode_tiled(latent, tile_size=256)` | `unload_vae()` |

latent 以 unpacked `(batch, c, h, w)` `mx.array` 在所有阶段间流转（匹配 mflux `prepare_latents` 输出与 `decode_packed_latents` 输入；`h`/`w` 由数组 shape 推导，无需额外尺寸参数）。

> **MLX stream 约束**：latent/embed 必须是 engine 原生 -- 由 `encode_text` 或另一个在单 image-executor 线程运行 (`max_workers=1`, `_init_mlx_thread`) 的 stage 创建。调用方线程创建的数组在每步 `mx.eval` 时触发 `RuntimeError: There is no Stream(gpu, 0) in current thread`。阶段间流转保持原生是因为 executor 单线程。

`unload_*` 把子模块引用置 `None`；mflux 在 `__init__` 加载所有阶段，故重载单个已卸载阶段需重新实例化引擎（load 方法会抛 `RuntimeError` 并给出此指引）。

视频后端继承 stage API 的 `NotImplementedError` 默认实现 (issue #170 phase 2)；`LegacyLTXBackend` 与 `Wan2Backend` 接入真实逐步 denoise，`LTX2Backend` / `SkyReelsBackend` 接受但仅打日志。

### 步进回调 (#171)

`generate()` (图像) 与 `VideoGenEngine.generate()` 接受 `on_step: Callable[[int, int], Awaitable[None]] | None`，在每个 denoise 步后以 `on_step(step, total_steps)` 触发。异步回调经 `asyncio.run_coroutine_threadsafe` 桥接到同步 mflux denoise 循环（fire-and-forget；错误仅记录日志，绝不阻塞生成）。图像在 `flux.callbacks` 上用真实逐步订阅器；视频经 `VideoGenParams.on_step` 接入。

### 模型注册枚举 (#172)

`fusion_mlx/model_registry.py` 的 `list_available_models()` 现以累加方式返回全部可发现模型（已注册 + 已发现），宿主无需单独 discovery 调用即可枚举模型。

## Admin 管理面板

访问 `http://localhost:8000/admin`：

- **模型管理** - 动态加载/卸载/锁定模型，ParoQuant 兼容检测
- **在线对话** - 实时对话界面，可测试任何模型
- **下载** - HuggingFace / ModelScope 模型下载，带进度追踪
- **量化** - 在线量化 (oQ) 流水线
- **基准测试** - 吞吐量和精度评测
- **监控** - 实时内存、性能和请求指标
- **设置** - 全局/单模型配置、子 API key 管理

## 性能

Apple M5 Max (128 GB RAM, 40 GPU 核心)，MLX 0.32.0.dev - 2026-07-04。
单流解码，Qwen3.6-27B-mxfp8（100 tokens，5 次预热）：

| 引擎 | TG 均值 (tok/s) | 中位数 | 标准差 | CV | 步长 (ms) |
|---|---|---|---|---|---|
| fusion-mlx | 18.46 | 18.52 | 0.18 | 1.0% | 54.17 |
| omlx | 18.49 | 18.53 | 0.18 | 1.0% | 54.09 |

比值 0.998 - 完全持平。对 GatedDeltaNet 混合模型自动关闭投机解码以保持输出连贯。

预填充吞吐量 (tok/s)：

| Prompt tokens | 64 | 128 | 256 | 512 | 1024 | 2048 |
|---|---|---|---|---|---|---|
| tok/s | 421 | 657 | 733 | 669 | 692 | 722 |

批量解码，fusion-mlx（聚合 / 单请求 tok/s）：

| Batch size | 1 | 2 | 4 |
|---|---|---|---|
| 聚合 TG | 18.09 | 17.75 | 16.61 |
| 单请求 TG | 18.09 | 8.87 | 4.15 |

> 早期 README 数据（TG 29.8 tok/s、并发 36.0 tok/s）是在开启投机解码时测得的，对该混合循环模型的输出造成了损坏。以上数据为连贯输出（自动关闭投机解码）下的真实可用吞吐量。M5 Max 上 27B mxfp8 的连贯上限约为 18.5 tok/s。

欢迎提交你的基准测试结果到 [bench.dpdns.org](https://bench.dpdns.org/)。

### Video Generation (SkyReels-V3)

SkyReels-V3 (R2V / V2V / A2V) 纯 MLX 移植，真实权重端到端跑通（完整 40 层 DiT 前向，非骨架 stub）。Apple M5 Max (128 GB, 40 GPU 核心)，MLX 0.32.0，2026-07-18，bfloat16，5 frames 256P latent:

| Branch | Model | 权重体积 | 加载 (s) | DiT fwd (s/step) | Metal 峰值 (GB) | FPS/step | 状态 |
|---|---|---|---|---|---|---|---|
| R2V | Reference-to-Video 14B | 28.6 GB (`transformer/`) | 6.84 | **0.092** | 75.3 | **54.3** | ✅ 跑通 |
| V2V | Video Extension 14B | 75 GB (14+6+1 shards) | 3.11 | **0.329** | 82.7 | **15.2** | ✅ 跑通 (mx.compile 融合 3.3×) |
| A2V | Talking Avatar 19B | 123 GB (18+6+1+1+1 shards) | 3.16 | **0.328** | 24.8 | **3.0** | ✅ 跑通 (audio_cross_attn+norm_x 重构 + kv_linear 转置 + mx.compile, 18×加速) |

**真 T5/VAE 端到端** (A2V-19B, 非 stub `text_emb`, 真实 UMT5Encoder.encode_text + WanVAE.decode, 5 frames 128×128 latent, bf16, 2026-07-18):

| 阶段 | 耗时 | 输出 shape | 说明 |
|---|---|---|---|
| T5 encode_text | 3.05 s | (1, 14, 4096) | 真前向, token_embedding 零占比 0% |
| DiT 30 步去噪 | 8.43 s (281 ms/step) | (1, 16, 5, 16, 16) | mx.compile 融合, Metal 峰值 24.8 GB |
| VAE decode | 0.30 s | (1, 3, 20, 128, 128) | 真前向, 零占比 0% (出真非零画面) |
| **端到端总耗** | **11.78 s** | 5 帧 128×128 | **0.42 FPS** (首版真端到端) |

**T5/VAE 修复关键** (解除真端到端阻塞的 4 个 bug):

| Bug | 位置 | 修复 |
|---|---|---|
| T5 加载路径漏 `t5/` 子目录 | weights:346 | 加 `t5_dir = model_path / "t5"` 分支 |
| `T5Encoder` 命名失配 (`embed_tokens`/`final_layer_norm`/`block`) | t5_encoder:204 | 改 `token_embedding`/`norm`/`blocks` 对齐真实权重 |
| `WanVAE` list 属性不入 MLX `_children` 致 `parameters()` 丢子层 | weights:181 | 加 `_inject_list_child_weights` 手动递归注入 |
| `nn.Conv2d` 权重布局 `(out,in,kh,kw)` vs MLX 期望 `(out,kh,kw,in)` | weights:233 | `_inject` 内 `transpose(0,2,3,1)` 自动转置 |

**MLX 全模型转换产物** (PyTorch -> MLX safetensors, `convert_skyreels_v3.py`):

| 分支 | DiT shards | T5 shards | VAE | CLIP | audio | 总体积 |
|---|---|---|---|---|---|---|
| R2V-14B-MLX | 7 (24 GB) | - | - | - | - | 24 GB |
| V2V-14B-MLX | 14 (53 GB) | 6 (21 GB) | 484 MB | - | - | 75 GB |
| A2V-19B-MLX | 18 (97 GB) | 6 (21 GB) | 484 MB | 4.4 GB | 218 tensors | 123 GB |

**关键修复** (解除 SkyReels-V3 真实权重端到端阻塞的 9 个 bug):

| Bug | 位置 | 修复 |
|---|---|---|
| `load_pytorch_state_dict` 漏子目录扫 | convert:104 | 加 `rglob` 递归扫 transformer/vae/text_encoder |
| `_load_safetensors_dir` numpy 不认 bf16 | convert:131 | 改 `framework=pt` + torch.float32 upcast |
| `_write_sharded_safetensors` mlx bf16 buffer 错 | convert:424 | try/except 捕获后 upcast float32 |
| `total_size` 求和漏 bf16 fix | convert:449 | `_safe_nbytes` helper |
| `mx.power(theta, -arange/dim)` Invalid Dtype | common:140 | 改 `mx.exp(-k*mx.log(theta))` |
| `grid_sizes` pre-patch 尺寸错位 | common:211 | ~~加 `patch_scale` 反推真实 seq_len~~ (patch_scale 本身是 #144 根因, 已移除; 见下 #144) |
| `context` 用错 dim 而非 text_dim | bench_skyreels:340 | 改 `branch_cfg.text_dim` |
| `noise_pred = zeros_like` 跳过 DiT | bench_skyreels:366 | 去兜底, 真实前向失败立即抛错 |
| `rope_apply` 广播错位 (padded 长度) | common:242 | 用 `seq_len` 而非全序列 `s` 广播 |

> 历史骨架压测 (1983/1151/906 FPS) 是假象 - `bench_skyreels.py:366` 的 `noise_pred = mx.zeros_like(latent_input)` 跳过了整个 DiT 前向, 只测了空循环开销。上表 0.110/2.862 s/step 才是真实推理速度。

**#139 权重加载修复** (2026-07-20, 解除 R2V 真实完整权重加载阻塞):

issue #122-#138 的修复策略经审视均为**症状追逐** (在随机初始化的 DiT 上修维度错误). 真正根因及修复:

| 根因 | 位置 | 修复 |
|---|---|---|
| `_load_safetensors_dir(MODEL_DIR)` 优先命中根目录 7 个**不完整分片** (315 keys, 仅 block 0-10, 无 index.json), 致 ~97% DiT 参数停留 init -> 前向 NaN/shape 错 | `weights.py:load_dit_weights` | 检测 `transformer/` 子目录存在时显式重定向 (diffusers 约定: DiT 权重位于 `transformer/diffusion_pytorch_model.safetensors`, 1095 keys, 28.6GB, 完整 40 层) |
| `_encode_context` 返回 `dim=5120` 而非 `text_dim=4096`, 致 `text_embedding.0` (Linear 4096->5120) 输入维度错配 | `pipelines/__init__.py` | context 维度改为 `text_dim=4096`, DiT `__call__` 内 `text_embedding` 做投影 |
| #137 `fp8_matmul` (out,in)/(in,out) 自动检测对方阵权重 (如 5120x5120) 误判 | `fp8_linear.py` | 移除自动检测, 权重恒 (out,in) 恒转置 (与 `nn.Linear` 一致) |
| #138 `FP8Linear` bias 截断 (掩盖 #137 的胶布) | `fp8_linear.py` | 移除截断, bias 恒 (out_features,) 形状必然匹配 |

修复后: **1055/1377** 模型参数命中源权重 (修复前 71/1377), 前向产出**有限值** (shape `(1,16,2,16,16)`, finite=True), 140 测试全绿 (49 skyreels + 20 fp8 回归 + 71 mfa).

> **#140 已知限制 (非阻塞)**: checkpoint 缺失 200 个 img-branch cross-attn 权重 (`k_img/v_img/norm_k_img`), 参考图引导分支停留随机初始化. config 声明 `cross_attn_type: i2v_cross_attn` (MLX 正确实例化 img-branch), 但源 checkpoint 的 `attn2` 仅含文本 cross-attn (无 `to_k_img/to_v_img`). 属上游 checkpoint 缺口, 待 Skywork 确认. 其余 121 个未覆盖参数 (`norm1`/`norm3`/`head.norm`, weight=ones=identity) + 1 `freqs` (rope buffer) 均良性, 不影响前向正确性.

**#144 R2V reshape 崩溃修复** (2026-07-20, 解除 R2V CFG 采样阻塞):

issue #142 修复 (FP8Linear `.weight`/`compute_dtype`) 解除 DiT 前向后, R2V CFG 采样暴露 `[reshape] Cannot reshape array of size 1612800 into shape (3080,1,523)`. 真根因及修复:

| 根因 | 位置 | 修复 |
|---|---|---|
| `rope_apply` 旧实现用 `patch_scale = total_seq/total_grid` 混用单样本 `total_seq=x.shape[1]` 与全 batch `total_grid=sum(f*h*w)`, CFG `b=2` 时 patch_scale 偏 1/b -> seq_len 算错 -> reshape 崩 | `common.py:rope_apply` | 对齐 `wan2/rope.py`: 每样本独立 `seq_len=f*h*w`, 截取 `x[i,:seq_len]`; 移除 patch_scale; grid_sizes 长度须等于 B (wan2 约定) |
| Pipeline `latent_h=height//16` 错 (VAE `patch_size=(1,8,8)` 下采样 8x, 应 //8); `grid_sizes` 传 pre-patch 网格 (DiT `_unpatchify`/`rope_apply` 期望 patch 后) | `pipelines/__init__.py` (r2v/v2v/a2v 三处) | `latent_h=height//8`; 用 `self.dit.patch_size` 算 post-patch `grid_sizes` |
| CFG `latent_input=concat([latents]*2)` 使 B=2, 但 `grid_sizes`/`seq_lens` 只传 1 条 -> `_unpatchify` 仅处理 1 样本 -> 输出 B=1 -> `perform_guidance` 半切得 B=0 | `pipelines/__init__.py:_denoise_sample` + a2v 循环 | CFG 时 `grid_sizes`/`seq_lens` 扩为 B 条 (`list(grid_sizes)*2`), 对齐 wan2 `grid_sizes=[gs]*batch_size` 约定 |

修复后: 真实 14B R2V 端到端跑通 (DiT 前向 + UniPC 采样 + VAE 解码), 输出 `(1,3,28,720,1280)`; 55 skyreels + 46 video_backends 测试全绿.

**#148 视频生成后端超时修复** (2026-07-20, 解除 720p 30 步长任务被 600s 超时杀死的运维阻塞):

#146 (每步 `mx.eval` 打断惰性图累积) 解除 R2V 推理 hang 后, 720p 30 步完整跑需 ~1hr (`~115s/步 × 30 + VAE`, NF4/FP4 量化下). 但 5 个视频后端 (skyreels R2V/V2V/A2V + wan2 + ltx2) 的 `generate()` 均硬编码 `asyncio.wait_for(..., timeout=600.0)`, 10 分钟天花板在生成中途抛 `TimeoutError` 杀死任务 (ThreadPoolExecutor 任务不可中途取消, 后台线程继续跑但结果丢失).

| 根因 | 位置 | 修复 |
|---|---|---|
| 5 处 `timeout=600.0` 硬编码 << 720p 30 步 ~1hr 实际耗时 | `engine_core.py` + `skyreels.py`(×3) + `wan2.py` + `ltx2.py` | 新增 `get_video_gen_timeout()` 助手 (env `FUSION_VIDEO_GEN_TIMEOUT`, 默认 7200s/2hr, 覆盖 720p 30 步 + VAE 余量); 5 处统一调用; 非法/≤0 值回退默认并告警 |

进度可见性 (#146 已落地): 每步 `denoise: start/step=i/N/done` + `vae decode: start/done` 日志, 服务端可观测生成进度. 默认 7200s 覆盖绝大多数配置; 超长任务可 `FUSION_VIDEO_GEN_TIMEOUT=14400` 等覆盖. (完整异步任务模型 + 客户端进度 API 为独立未来增强, 非本次范围.)

修复后: 真实 14B R2V 256×256 3 步端到端跑通 (`get_video_gen_timeout()=7200.0s`, 25.5s 完成 << 上限, mp4 ftyp 合法, 每步日志齐全); 53 video_backends 测试全绿 (含 7 个超时助手新测试).

**#149 SkyReels-V3 推理进度日志补全** (2026-07-20, 解除服务端生成期间无任何进度日志的可见性阻塞):

#148 解除 7200s 超时后, 720p 30 步完整跑 ~1hr 期间服务端日志仍静默: 后端 `_generate_r2v/v2v/a2v` 无请求级日志, 首次调用 `_load_models` (加载 DiT~14B + VAE + UMT5) 完成前数分钟无输出, 且每步日志 (`denoise: step=i/N t=...`, #146 落地) 仅在 ~115s 前向完成后才打印 -> 两行日志间 ~115s 静默, 误判卡死 (即 #149 "服务器日志无推理进度日志").

| 根因 | 位置 | 修复 |
|---|---|---|
| (1) 后端无请求级日志 (2) 首次加载模型静默 (3) 步级日志仅在前向后打印 | `skyreels.py` (后端) + `pipelines/__init__.py` (R2V/V2V + A2V) | (1) `video gen: start/done branch=r2v\|v2v\|a2v` + `Loading pipeline <Class> (DiT/VAE/T5, 首次可能数分钟)` 前置于 eager 加载 + `Created pipeline`; (2) `denoise: step=i/N starting t=...` 前置于每步 DiT 前向 (R2V/V2V 共享循环 + A2V 内联循环), 步开始即报, 不再等前向完成 |

修复后: 真实 14B R2V 256×256 3 步端到端跑通, `denoise: start` 与 `denoise: step=1/3 starting` **同秒打印** (前向前即报进度), 步间不再静默, VAE 解码 + `R2V generated` 日志齐全, mp4 合法, 14.6s 完成; 57 video_backends 测试全绿 (含 4 个进度日志新测试).

**#154 性能调优 Tier 1 + T2-1** (2026-07-20, SkyReels-V3 R2V 性能可见性 + 降速手段):

#149 修复进度可见性后, 720p 30 步 ~1hr 的系统慢速本身仍是问题. Tier 1 旨在让 xfuser 步级注意力策略生效; T1-3 证明 xfuser 在 mx.compile 下**根本无法生效**:

| 层 | 结论 | 证据 |
|---|---|---|
| T1-1 可见性 | `_log_optimizer_status` 打印 `should_compile/has_xfuser/step_strategy_modules/m5_applied/dit_blocks/branch/steps` + xfuser 被 compile 旁路时告警 | 真实 14B 运行 emit 告警 |
| T1-2 warmup | `warmup()` (256×256/9f/1step 去噪) 预编译 + 权重 Metal 常驻, `FUSION_SKYREELS_WARMUP` 门控 (默认 "1"), 非致命 | `skyreels.py` 建管道后调用 |
| **T1-3 根因 (xfuser 运行时 no-op)** | xfuser `attach_to_model` 注入 80 个 `MLXFastAttention` 模块到 DiT `_fast_attn`, 但 `mx.compile` 在 `SkyReelsDiT.__init__` (attach **前**) 已 trace, `_fast_attn=None` + `if False` 被 **bake 进固化 trace** -> 运行时 `fa_calls=0` (no-op). `mx.disable_compile()` (MLX 0.32) 是**全局切换函数返 None** (非 context manager, `with mx.disable_compile():` 抛 TypeError), 无法 un-bake 已固化 trace. 114s/step = compiled-no-xfuser DiT 前向**真实成本** (非每步 recompile). xfuser + mx.compile **根本不兼容**: attach 前 compile 则 bake fa=None; attach 后 compile 则 `step_idx` (Python int + `steps_method[step_idx]` 列表索引 + `method.has()` 控制流) 入 trace -> 每步 recompile (抵消编译收益) | monkey-patch `MLXFastAttention.__call__` 计数器 `fa_calls=0`; scheme-6 `disable_compile` 包裹 TypeError 后回退 |

**T2-1 降速手段 (env 覆盖采样步数)**: `FUSION_SKYREELS_STEPS` 覆盖 `num_inference_steps` (默认 30), 720p 30->20 步 ~ **-33% wall-clock** (UniPC `solver_order=2` 历史预测保稳). 在管道 init 时 (`_setup_optimizers` 前) 生效, 使 xfuser `step_methods` 长度与实际步数一致. 非法/≤0/非 int -> 告警并保留默认.

```bash
# 720p 默认 30 步 ~1hr; 降到 20 步 ~40min
FUSION_SKYREELS_STEPS=20 fusion-mlx serve --model SkyReels-V3-R2V-14B-MLX
# 关闭 warmup (默认开)
FUSION_SKYREELS_WARMUP=0 fusion-mlx serve --model SkyReels-V3-R2V-14B-MLX
```

**C DiT w8a16 权重量化 (load-level)**: `FUSION_SKYREELS_QUANT=w8a16` 在 `_load_models` 经 `load_all_weights(quantization=...)` -> `load_dit_weights` -> `nn.quantize(dit, bits=8, group_size=64)` 将 DiT 全部 `nn.Linear` 转 `QuantizedLinear` (8-bit 权重 + bf16 激活 = w8a16), 源 bf16 权重按量化格式加载. DiT=74% 瓶颈, 8-bit 权重显存减半 + int8 matmul 加速, 预期 ~2x. compile 安全: `mx.compile` 懒编译, 首次前向 (warmup) 在量化后构建 trace, 直接见 `QuantizedLinear` 结构 (非 stale). `nn.quantize` 递归 DiT `self.blocks` list (MLX nn.Module 收录 list-of-Module 子层). 也支持 `w4`/`nf4` (4-bit). 默认 off (全精度 bf16).

```bash
# w8a16 (8-bit 权重 + bf16 激活)
FUSION_SKYREELS_QUANT=w8a16 fusion-mlx serve --model SkyReels-V3-R2V-14B-MLX
# w4 / nf4 (4-bit 权重, 显存更省)
FUSION_SKYREELS_QUANT=w4 fusion-mlx serve --model SkyReels-V3-R2V-14B-MLX
# 关闭 (默认, 全精度 bf16)
FUSION_SKYREELS_QUANT=off fusion-mlx serve --model SkyReels-V3-R2V-14B-MLX
```

**B1 动态 CFG (sampler-level, compile-friendly)**: 早期步跑 cond+uncond (b=2, CFG 引导塑造结构), 晚期步仅跑 cond (b=1, guidance=1.0). 晚期步 DiT 前向算力减半; mx.compile 按 input shape 缓存 (b=1/b=2 各编译一次, 非逐步 recompile), 不触碰 DiT 固化 trace. 参考 Wang et al. "Faster Diffusion" (IAR 2024) late-step CFG reduction. `_cfg_keep_steps(n_steps)` 返回跑 b=2 的步数 = `int(n_steps * keep_ratio)`; b=1 步跳过 `perform_guidance` (noise_pred 已是 cond, scheduler 见 [1,...] 与 b=2 合并后一致).

```bash
# 默认开 (前 60% 步跑 CFG, 后 40% cond-only)
FUSION_SKYREELS_DYNAMIC_CFG=1 FUSION_SKYREELS_CFG_KEEP_RATIO=0.6 fusion-mlx serve --model SkyReels-V3-R2V-14B-MLX
# 关闭 (全部 b=2, 保持旧行为)
FUSION_SKYREELS_DYNAMIC_CFG=0 fusion-mlx serve --model SkyReels-V3-R2V-14B-MLX
```

> **T1-3 结论**: 不要再尝试在 mx.compile 下让 xfuser 生效 (已证根本不兼容). 降速用 `FUSION_SKYREELS_STEPS` 减步 (T2-1). 后续 T2-2 (DiT w8a16 权重量化, 复用 `/v1/convert`, DiT=74% 瓶颈 ~2x 加速) + T2-3 (CFG halve 研究) 为独立 follow-up.

修复后: 真实 14B R2V `FUSION_SKYREELS_STEPS=3` 生效 (`cfg.num_inference_steps=3`), xfuser 旁路告警 emit, `fa_calls=0` (no-op 确认), 生成 OK shape=(1,3,28,256,256), VERIFY_OK; 128 单元测试全绿.

### Radix 文本编码缓存 (#178)

多镜头短剧流水线中, 同一 prompt 跨镜头重复编码 (UMT5-XXL 24 层 4096 维, 单次编码数百 ms~秒级). `UMT5Encoder.encode_text` 接入 `DiffusionRadixCache` (radix 树 + LRU 字节预算 + pin/unpin), 相同 `prompt+max_length` 二次命中走零拷贝指针复用 (返回同一 `mx.array` 引用), 文本编码延迟 -> ~0ms.

- 缓存键: `f"umt5:{max_length}:{sha256(prompt)[:16]}"`, per-encoder 实例 (模型重载自动失效, 无陈旧 embedding).
- 零拷贝: `mx.array` 不可变, 命中直接返回缓存引用, 无内存拷贝.
- stub 模式不缓存 (避免零张量污染).
- LRU 字节预算默认 512MB (UMT5-XXL `[1,512,4096]` bf16 ~4MB/条, ~128 条).
- env `FUSION_DIFFUSION_TEXT_CACHE` (默认 `"1"` 开, `"0"` 关).

```bash
# 默认开
fusion-mlx serve --model SkyReels-V3-R2V-14B-MLX
# 关闭 (每次重新编码, 调试用)
FUSION_DIFFUSION_TEXT_CACHE=0 fusion-mlx serve --model SkyReels-V3-R2V-14B-MLX
```

> **范围**: phase-1 = UMT5Encoder 全键缓存 (相同 prompt -> 0ms). Phase-2 (本变更) = CLIPTextEncoder 接入 + admin 统计端点. 仍延期: 视频时序 latent 复用 (高风险, #177 负面先例). token 级前缀 KV 共享对 T5/UMT5 **语义无效** - T5 是双向编码器 (位置 i 的隐状态依赖完整序列), 前缀隐状态复用会破坏输出 (不同于因果 decoder LLM), 全键缓存才是正确做法.

**Phase-2 补充:**

- **CLIP 编码器接入**: `CLIPTextEncoder.encode_text` (Flux/SD 路径) 同样缓存 - 键 `f"clip:{max_length}:{sha256(text)[:16]}"` (列表输入用 `NUL` 连接). 命中在 `_ensure_loaded()` 之前返回, 重复 prompt 根本不加载 CLIP 模型 (价值超过去掉单次 forward). stub 模式不缓存.
- **admin 统计端点**: `GET /v1/cache/stats` (admin 鉴权) 通过模块级 `weakref` 注册表聚合所有存活缓存. 响应: `{"cache_type": "diffusion_text_encoding", "caches": [{name, hits, misses, evictions, insertions, leaf_count, total_bytes, max_bytes, hit_rate}, ...], "totals": {cache_count, hits, misses, evictions, insertions, total_bytes, hit_rate}}`. 已卸载编码器的缓存自动移除 (weakref). 报告的是扩散文本编码缓存, 非 LLM KV/prefix cache.

### Speculative Denoise (#177)

扩散模型版的 speculative decoding: 草稿 DiT (层剪枝, 跑前 M/N 个 transformer block + 共享 head) 顺序预测 K=3-5 步未来速度场, 完整 DiT 单次 batched forward 验证 K 步 (per-element timestep, Wan2/SkyReels DiT 原生支持 `t.ndim==1`), 接受最长一致前缀, 分歧处用完整速度场补一步 (bonus step, 永不卡住). 目标 14B 上 2-3x 加速.

- 草稿协同加载: `LayerPrunedDraft(dit, n_blocks=M)` 复用同一份权重, 无需独立 draft checkpoint (MLX 量化非速度路径, 见 #166; 暂无 1B/3B SkyReels draft).
- 验证: K 个 latent 在 K 个不同 timestep 上单次前向 (批 per-element timestep embedding), 接受 `||v_draft - v_full|| / ||v_full|| < epsilon` 的最长前缀.
- 1 阶 Euler 推测环 (UniPC 2 阶 corrector 需上一步 full 输出, 推测模式旁路).
- env: `FUSION_SPECULATIVE_DENOISE` (默认 `"0"` 关), `FUSION_SPEC_K` (默认 4), `FUSION_SPEC_EPSILON` (默认 0.1), `FUSION_SPEC_DRAFT_BLOCKS` (phase-2 已接线, 默认 `num_layers//4`).

```bash
# phase-1: 模块 + API + 合成 DiT 单元测试 (env-gated, 不改生产 denoise 环)
# phase-2: R2V DiT forward_partial 接线 + 真 14B 接受率 sweep (负向结论, 见下)
# #186 item 3: V2V/A2V pipeline 接线 (V2V 复用 base spec 路径; A2V 自有 override, audio+text 约定)
# 默认关, 不影响现有 SkyReels-V3 生成路径
fusion-mlx serve --model SkyReels-V3-R2V-14B-MLX
```

> **范围 (phase-1 + phase-2)**: phase-1 落地推测去噪算法 + 草稿协同加载 API (`DraftDiTMixin.forward_partial` / `LayerPrunedDraft`) + 合成 DiT 单元测试, env-gated, **零生产风险**. phase-2 落地 R2V DiT `forward_partial` 接线 + 真 14B 接受率 sweep. **phase-2 实测结论 (负向)**: 层剪枝 draft 在安全 epsilon(0.1) 下接受率 0% (保留 25%-75% blocks), 仅保留 95% blocks 时出现接受但 draft 成本≈full 无加速且质量劣化(maxdiff 0.097); 放宽 epsilon 到 0.5 无效 (draft 速度场误差远超 0.5). #177 假设在 MLX SkyReels-V3 14B 上证伪: DiT 速度场需完整深度, 不可由子网络预测 (异于 LLM token 预测). 机制正确 (全拒绝时 spec==baseline Euler, 误差 7e-4), 保持落地 (env-gated 默认关, 零生产风险) 作为未来蒸馏小 draft 的基础设施. **#186 item 3 (V2V/A2V pipeline 接线)**: V2V DiT 与 R2V 同为单 context 前向约定, 复用 base spec 路径 (无生产改动); A2V DiT 前向签名不同 (audio + text embeds), 走自有 `_denoise_sample_speculative` override + base branch guard. phase-3: fusion-comfyUI Stage API 接入. 详见 `fusion_mlx/video/skyreels_v3/SPECULATIVE_DENOISE.md`.

## 项目结构

```
fusion-mlx/
├── fusion_mlx/
│    ├── api/             # OpenAI、Anthropic、Audio、Images、Videos、MCP、OpenClaw 路由
│    ├── cache/           # PagedCache、PagedSSDCache、PrefixCache
│    ├── custom_kernels/  # MFA、TurboQuant、KV cache、xfuser attention
│    ├── engines/         # 9 种引擎 (LLM、VLM、Embedding 等)
│    ├── integrations/    # Claude Code、OpenClaw、ComfyUI、Copilot、Codex 等
│    ├── parsers/         # 工具调用解析器 (Gemma、Harmony、Hermes 等)
│    ├── pool/            # EnginePool、MemoryEnforcer、ModelDiscovery、PriorityScheduler
│    ├── router/          # RequestRouter、CloudRouter、SmartRouter
│    ├── scheduler/       # 25 模块调度器 (admission、batching、cache、step 等)
│    ├── speculative/     # SuffixDecoding、DFlash、DSpark、MTP、VLM MTP
│    ├── video/           # 纯 MLX 视频生成移植 (LTX-2、Wan2、SkyReels-V3)
│    └── admin/           # 管理面板路由、基准测试、下载、设置
├── apps/fusion-mac/      # SwiftUI macOS 应用 (~80 个 Swift 文件)
├── docs/                 # API 参考、架构、CLI 指南、配置
├── examples/             # 12 个可运行的代码示例
├── tests/                # 1200+ 测试 (单元、GUI、集成、性能)
└── downstream/           # omlx 和 Rapid-MLX 分支的同步脚本
```

## DSpark 投机解码 (vendored from dspark-metal, 2026-07-22)

DSpark = DeepSeek DeepSpec 块级投机解码, 针对纯文本 Qwen3 系列. 与 token 级 spec decode 不同, DSpark 用目标模型第 7 层隐藏状态训练轻量 draft (block7), 在线拒绝采样保证 lossless. fusion-mlx 将上游 `stefanopineda/dspark-metal` (MIT) **vendor 进** `fusion_mlx/speculative/dspark/engine/`, 不再依赖 pip 包 - 上游仓库已停滞 20+ 天无活动, fusion-mlx 独立演进.

- 引擎: `fusion_mlx/speculative/dspark/engine/` (13 模块 + LICENSE + NOTICE 上游归属).
- 边界: `runtime.py` 用 `from .engine import DSparkGenerator` 加载本地 vendored 引擎; `eligibility.have_runtime()` 探测 vendored 路径, 始终可用 (无需 `pip install dspark-metal`).
- VLM 扩展 (PR#2): `Qwen3VLTargetAdapter` (Direction B 原生多模态) 将 DSpark 扩展到 mlx-vlm 目标, ctx_taps 仅作用于 text position, 懒加载 mlx_vlm. 22 个 weight-free 测试见 `tests/unit/test_dspark_vlm_adapter.py`.
- 尺寸绑定: draft = 目标 block 7, 故 `dspark_qwen3_{4b,8b,14b}_block7` 须配同尺寸 Qwen3-{4B,8B,14B} (bf16/8bit+; 4-bit 被 eligibility gate 拒绝).
- convert: `python -m fusion_mlx.speculative.dspark.engine.convert <source> --target <target> -o <outdir>` (勿传 `--reuse-target-embeddings`).

> **E2E 状态**: vendoring (Phase 1+2) 已落地, 40 个 dspark 测试通过 (1 skipped), arch-handler 静态去风险 (`load_draft_model` 自构 `DSparkDraftModel`, 不走 mlx-lm registry). 真实模型 E2E (convert + load_runtime + generate) 因本地无匹配 Qwen3-4B/8B/14B 目标, 草稿+目标需经 hf-mirror 下载, 暂缓为后续验证项.

## 示例

| # | 示例 | 说明 |
|---|------|------|
| 01 | `basic-chat.py` | 简单非流式对话 |
| 02 | `streaming-chat.py` | SSE 流式响应 |
| 03 | `anthropic-api.py` | Anthropic Messages API |
| 04 | `tool-caling.py` | JSON Schema 函数调用 |
| 05 | `multi-model.py` | 多模型并发请求 |
| 06 | `image-generation.py` | Flux 2 图像生成 |
| 07 | `speech-to-text.py` | Whisper STT 语音识别 |
| 08 | `text-to-speech.py` | Kokoro TTS 语音合成 |
| 09 | `mcp-tools.py` | MCP 工具发现与执行 |
| 10 | `python-sdk.py` | OpenAI Python 客户端集成 |
| 11 | `comfyui-workflow.py` | ComfyUI 工作流执行 |
| 12 | `openclaw-agent.py` | OpenClaw Agent 协议 |

## 文档

- [API Reference](docs/api-reference.md) - 所有端点及请求/响应示例（英文）
- [Architecture](docs/architecture.md) - EnginePool、Scheduler（25 模块）、Cache 层、SmartRouter（英文）
- [CLI Reference](docs/cli-reference.md) - 所有命令和参数（英文）
- [Configuration](docs/configuration.md) - 内存分级、调度器设置、TurboQuant、别名、执行器线程池（英文）
- [Speculative Decoding](docs/speculative-decoding.md) - Suffix/DFlash/DSpark/MTP/VLM-MTP 五种投机解码方法、选型指南、auto-router（英文）
- [Video Input](docs/video-input.md) - VLM 视频输入：`video_url` API、帧抽取、Qwen 原生视频路径、限制（英文）
- [FR Differentiation](docs/FR_DIFFERENTIATION.md) - fusion-mlx 在投机解码/TurboQuant/调度方面差异化能力的核实分析（英文）
- [架构详解](docs/architecture_CN.md) - 架构文档中文版
- [配置指南](docs/configuration_CN.md) - 配置文档中文版

## whichllm 集成

macOS 应用的 **Welcome 向导**使用 [whichllm](https://github.com/Andyyyy64/whichllm) 进行硬件感知的模型推荐。whichllm 自动检测 Mac 的 GPU、CPU、RAM 和磁盘，然后从 HuggingFace 上排名最适合你系统的本地 LLM。

**集成特性：**
- **硬件检测** - Apple Silicon 芯片类型、统一内存、GPU 带宽、CPU 核心、可用磁盘（通过 `system_profiler`/`sysctl`）
- **模型推荐** - 按质量评分、速度 (tok/s)、VRAM 适配度和基准证据排名的 Top 模型
- **使用场景优化** - 针对 Agent / 编程 / 聊天 三种场景给出不同推荐
- **镜像源选择** - HuggingFace、HF Mirror 或 ModelScope（中国大陆用户免 VPN）
- **优雅降级** - whichllm 未安装时自动回退到 `ProcessInfo` + `sysctl`（零 Python 依赖）

**桥接架构：**
```
Swift App -> WhichLLMService -> PythonRuntime -> whichllm_bridge.py -> whichllm
            ↓ (回退)
       ProcessInfo + sysctl (零 Python 依赖)
```

## Flux 2 Klein 切换 (mx.compile denoise 加速, 2026-07-20)

`ImageGenEngine` 从 Flux1 切到 `Flux2Klein` (mflux 0.18.0)。Flux2Klein 在
`flux2_klein.py:281` 用 `mx.compile(predict)` 包裹 denoise，Flux1 无此编译。
编译 warmup 后首步 2.98s -> 稳定 1.56s/step (1.9x 加速)。

### 性能数据 (M5 Max / FLUX.2-klein-base-4B bf16 / 1024x1024)

| 步数 | 总时间 | s/step |
|---|---|---|
| 4 | 6.8s | 1.59 |
| 8 | 13.6s | 1.70 |

首次含模型加载 8.5s (9.6G lazy load)。

### Serve 步骤

mflux Flux2 repo 是 diffusers 格式 (`model_index.json`)，**无** mflux 的
`configuration.json` task manifest，discovery 会误判为 llm -> `BatchedEngine`
加载失败。需手动补 manifest:

```bash
# 下载 (FLUX.2-klein-base-4B 非 gated; 9b/4b/kv 均 gated 需 HF approval)
HF_ENDPOINT=https://hf-mirror.com hf download black-forest-labs/FLUX.2-klein-base-4B \
  --local-dir ~/.fusion-mlx/models/FLUX.2-klein-base-4B

# 补 task manifest (discovery 识别为 image)
echo '{"task":"text-to-image"}' > ~/.fusion-mlx/models/FLUX.2-klein-base-4B/configuration.json

# serve (必须 --model-dir discovery 路径; --model 单模型路径用 BatchedEngine 不支持 image)
fusion-mlx serve --model-dir ~/.fusion-mlx/models --port 11434

# 生成
curl -s http://127.0.0.1:11434/v1/images/generate \
  -H 'Content-Type: application/json' \
  -d '{"model":"FLUX.2-klein-base-4B","prompt":"a cat","width":1024,"height":1024,"steps":4,"n":1}'
```

`_infer_flux2_config` 按路径名解析变体: `base+4b` -> `flux2_klein_base_4b`,
`base+9b` -> `flux2_klein_base_9b`, `4b` -> `flux2_klein_4b`, `kv+9b` ->
`flux2_klein_9b_kv`, 默认 `flux2_klein_9b`。`negative_prompt` 降级为 warning
(Flux2Klein.generate_image 无此参数)。

### Flux2Klein 权重量化 (FUSION_FLUX_QUANT, 内存优化非速度)

`ImageGenEngine.__init__` 读取 `FUSION_FLUX_QUANT` env -> `mflux.Flux2Klein(quantize=...)`。
取值: `w8a16`/`w8`/`int8`/`8` -> 8-bit, `w4`/`nf4`/`int4`/`4` -> 4-bit,
`off`/`0`/`none`/`bf16`/空 -> bf16 (默认)。大小写不敏感。

**⚠️ 实测结论 (M5 Max / FLUX.2-klein-base-4B / 1024x1024 / 4 step)**:

| 模式 | 总时间 | s/step |
|---|---|---|
| bf16 | 6.81s | 1.70 |
| w8a16 | 8.20s | 2.05 |

w8a16 **慢 20%**。4B 模型在 bf16 下已完全装入统一内存, int8 反量化开销超过
带宽收益, 且 `mx.compile` 已优化 bf16 路径。**量化不是 Flux2Klein 的速度优化**,
仅作**内存优化** (9B 模型权重 ~18G -> ~9G, 适配 16G Mac)。

## Flux-1.lite-8B-MLX 深度优化 (2026-07-19)

### 性能数据 (M5 Max 128GB / MLX 0.32 / Q4)

| 指标 | 原基线 | block 整编译融合 | mlx-mfa Metal attn 对接 | 真上限 |
|---|---|---|---|---|
| step/s (512×512×4步) | 1.83 | 1.96 | **1.88** | 1.88-2.03 |
| Metal 峰值 | 10.8 GB | 10.6 GB | 10.5 GB | 10.5 GB |
| 256×256 真上限 | - | - | - | 4.62 step/s |

bench.dpdns.org 上传记录: id 27 (1.97), id 30 (1.96), id 31 (1.88), id 32 (1.88 mlx-mfa Metal attn)

### 落地优化项

1. **block 整编译融合** (`joint_transformer_block.py` + `single_transformer_block.py`)
   - 加 `_compiled_call = mx.compile(self._call_raw)` 封装整块编译, 融合 AdaLN+attn+FFN 子模块为单编译单元
   - 消跨 `nn.Module` 子调用断融合, `__call__` 入口走 `_compiled_call` 编译版本
   - `to_out` list -> `to_out_0` 命名属性 (MLX nn.Module 不收录 list 属性) + `flux_weight_mapping.py` 补 `to_out.0` -> `to_out_0` 映射

2. **mlx-mfa Metal Flash Attention 内核对接** (`attention_utils.py::compute_attention`)
   - 用 `mlx_mfa.flash_attention` 替 `mx.fast.scaled_dot_product_attention`, 走 M5 Neural Accelerator 优 Tile
   - `has_nax: True` ✅ Metal 内核真触, head_dim=128 在 mlx_mfa 支持范围
   - 实测对接成功但收益持平 (1.88 vs 1.88 step/s), 因单替 SDPA 不覆盖 RoPE + QKV 投影主瓶颈

3. **Fused QKV+RoPE+attn 单算子图融合** (搁置)
   - 写 `_fused_qkv_rope_attn` 融合函数 + `JointAttention` 接入 + `mx.compile` 封装
   - 实测 Q4 量化权重是 `(out, in/8)` 压缩布局, 手动 `mx.matmul/mx.addmm` 破 `quantized_matmul` 封装报 ValueError
   - 回滚保 `nn.Linear.__call__` 走 `quantized_matmul`, 整 block `mx.compile` 已融合

### mlx-mfa 预编译路径 (避 PyPI wheel build 耗时不可控)

```bash
# 本地源 + scikit-build-core + nanobind 触发 CMake build 生成 _ext.so
pip install scikit-build-core nanobind
pip install -e /path/to/mlx_mfa-2.61.0/ --no-build-isolation
# 验证: has_nax() True = Metal 内核真触, False = fallback SDPA
python3 -c "from mlx_mfa import has_nax; print(has_nax())"
```

### 瓶颈诊断结论

- **256 vs 512 比值 2.48×** (理论算力 4×) -> 混合瓶颈 (带宽+算力双优)
- **transformer 80%** 主瓶颈 / encode_prompt 10% / VAE 10%
- **schnell 天生不支持 CFG** (`supports_guidance=False`), `guidance=4.0` 是无效参数, 单分支已是最优
- **shape 抖动代价 21.4%**: 同 512×512 连续稳态 1.90 step/s, 不同尺寸交替降至 1.56 step/s
- **真上限已厘清**: 512×512 在 M5 Max Q4 + mlx_mfa Metal attn + 双层编译融合后 1.88 step/s 是硬件+Q4 量化+算子栈三元约束下的合理上限

### 关键经验沉淀

1. **MLX Q4 量化权重不可手动 matmul**: 权重是 `(out, in/8)` 压缩布局, 必走 `nn.Linear.__call__` 内部 `quantized_matmul`. 所有"手写 Fused 单内核"方案对 Q4 量化模型破封装不可行
2. **60+ block 整体 mx.compile 劣化是通用规律**: 算子图按 N× 累积触 Metal Command Buffer 雾溅, 双层编译 (block 整编译 + transformer 循环编译) 是最优路径
3. **mlx-mfa 预编译路径**: 本地源 + scikit-build-core + nanobind + `pip install -e --no-build-isolation` 成功触发 CMake build 生成 `_ext.so`, 避 PyPI wheel build 耗时不可控

## 许可证

Apache-2.0

## 致谢

- [MLX](https://github.com/ml-explore/mlx) 和 [mlx-lm](https://github.com/ml-explore/mlx-lm) - Apple 出品
- [mlx-vlm](https://github.com/Blaizzy/mlx-vlm) - Apple Silicon 上的视觉语言模型推理
- [vllm-mlx](https://github.com/waybarrios/vllm-mlx) - oMLX 起源于 vllm-mlx v0.1.0
- [omlx](https://github.com/jundot/omlx) - Continuous batching 和分层 KV 缓存
- [Rapid-MLX](https://github.com/raullenchai/Rapid-MLX) - Speculative decoding、多模态、云路由
- [dflash-mlx](https://github.com/bstnxbt/dflash-mlx) - Block diffusion speculative decoding
- [DeepSpec (DSpark)](https://github.com/deepseek-ai/DeepSpec) - 无损块级投机解码
- [mlx-embeddings](https://github.com/Blaizzy/mlx-embeddings) - Embedding 模型支持
- [venvstacks](https://venvstacks.lmstudio.ai) - macOS 应用的可移植 Python 环境层
