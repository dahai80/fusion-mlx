# SkyReels-V3 → fusion-mlx 迁移改造清单

> 目标：把 `~/video/SkyReels-V3` 三大视频生成主干（R2V 14B / V2V 14B / A2V 19B）移植到 `~/claude-home/fusion-mlx` 的 MLX/MPS 栈，复用底座 dFlash + mlx-mfa + xfuser attention + TurboQuant KV，在 M5 Max 上拿到速度+显存+效果三重质变。
>
> 底座关键路径（已核对存在）：
> - 注意力调度：`fusion_mlx/custom_kernels/mfa_bridge.py` → `flash_attention()`，自动分派到 mlx-mfa Metal 内核或 `mx.fast.scaled_dot_product_attention` 兜底
> - xfuser 快速注意力策略：`fusion_mlx/custom_kernels/xfuser_attention.py`（FULL / RESIDUAL_WINDOW / SPARSE / OUTPUT_SHARE / CFG_SHARE）
> - MFA 内核封装：`fusion_mlx/custom_kernels/mfa/attention.py` + `dispatch_policy.py`（AttentionBackend: STEEL / STEEL_DSPLIT / NAX）
> - 已有视频 DiT 端口参考：`fusion_mlx/video/wan2/`（WanAttentionBlock 纯 MLX 版）、`fusion_mlx/video/ltx2/`（LTX-2 纯 MLX 端口）
> - KV Cache 与量化：`fusion_mlx/custom_kernels/kv_cache.py`、`turboquant.py`、`fp8_linear.py`、`quantize.py`
> - 引擎池与调度：`fusion_mlx/engine_pool.py`、`scheduler/`、`runtime/`

---

## 阶段 1：前置环境与基础工具层（P0，最先改造）

### 1.1 全局设备判断与 MLX Stream 封装
- **删除项**：所有 `.to("cuda")`、`.cuda()`、`torch.cuda.is_available()`、`torch.cuda.nvtx.range_push/pop`
- **新增项**：`fusion_mlx/video/skyreels_v3/_device.py`
  - `get_stream()` → 返回 `mx.new_stream(mx.gpu)`，M5 默认 GPU stream
  - `is_m5()` → 读取 `mx.get_device_info()` 判断 M5 Max，开启分级 Tile
- **编译开关**：`mx.compile(func, inputs=[...])` 全量开启，M5 上对 DiT Block 套 `mx.compile` 拿融合算子收益
- **dtype 映射**：
  | PyTorch | MLX | 备注 |
  |---|---|---|
  | torch.float16 | mx.float16 | VAE 默认 |
  | torch.bfloat16 | mx.bfloat16 | DiT 主干 |
  | torch.float32 | mx.float32 | modulation/AdaLN 保精度 |
  | torch.float64 | mx.float32 | rope 计算（原版 fp64，MLX 无 fp64 用 fp32 替代，需数值核对）|

### 1.2 权重加载转换器
- **新文件**：`fusion_mlx/video/skyreels_v3/convert_skyreels_v3.py`
  - 入口：`convert_skyreels_v3(pt_state_dict_dir, mlx_out_dir, model_type)`，model_type ∈ {r2v_14b, v2v_14b, a2v_19b}
  - 分层映射（按 SkyReels-V3 模块结构核对）：
    | SkyReels 模块 | 源文件 | 映射要点 |
    |---|---|---|
    | PatchEmbed Conv3d | `modules/transformer.py` patch_embedding | Conv3d 权重转置 `[out,in,t,h,w]→[out,in,t*h*w]` 适配 MLX Conv3d |
    | Timestep Embed | `transformer.py` time_embedding | Linear 权重转置 `.T` |
    | WanAttentionBlock × N | `transformer.py` blocks | Q/K/V/O Linear + modulation(1,6,dim) + norm1/norm2 + ffn |
    | CrossAttention | `transformer.py` WAN_CROSSATTENTION_CLASSES | i2v 分支额外 k_img/v_img/norm_k_img |
    | Head | `transformer.py` Head 类 | modulation(1,2,dim) + head Linear |
    | freqs (rope) | `transformer.py` rope_params | 缓冲区，转 mx.array 常量 |
  - 视频时序维度单独处理：`num_frame_list` / `grid_size_list` / `context_window_size` 三参数链路保留
  - 增量加载：`.safetensors` 分块写，M5 统一内存下单次载入 < 4GB
  - 原生支持 FP4/FP8 权重加载，承接 TurboQuant 量化（见阶段 6）

### 1.3 基础工具函数替换
| PyTorch 接口 | MLX 替代 | 底座已有封装 |
|---|---|---|
| `torch.nn.functional.layer_norm` | `mx.nn.LayerNorm` | — |
| `torch.nn.GELU(approximate="tanh")` | `mx.nn.GELUApproximate` | ltx2/wan2 已用 |
| `nn.GELU()` (exact) | `mx.nn.GELU` | — |
| `F.scaled_dot_product_attention` | `mfa_bridge.flash_attention()` | ✅ 底座核心 |
| `torch.view_as_complex/real` (rope) | `mx.complex64` 算子 + reshape | ltx2/wan2 rope.py 已实现 |
| `torch.nn.Conv3d` | `mx.nn.Conv3d` (MLX 支持) 或拆为 Conv2d×T | 需核对 MLX 版本 |
| `torch.nn.Conv2d` | `mx.nn.Conv2d` | ✅ |
| `torch.split` / `torch.cat` | `mx.split` / `mx.concatenate` | — |
| `torch.rsqrt` | `mx.rsqrt` | — |
| `einops.rearrange/repeat` | `mx.reshape` + `mx.transpose` + `mx.broadcast` | — |

---

## 阶段 2：注意力模块改造（P0 核心，对接 dFlash / mlx-mfa）

### 2.1 底座可用成果（已核对）
- `mfa_bridge.flash_attention(q, k, v, *, scale, mask, causal, window_size, softcap, return_lse)` —— 统一入口
- 自动分派：`dispatch_policy.select_backend()` 按头维度/dtype/序列长度/是否 decode 选 STEEL / STEEL_DSPLIT / NAX / MLX-SDPA 兜底
- 支持滑动窗口：`window_size` 参数；支持因果/非因果掩码
- `xfuser_attention.MLXFastAttention` —— 步级注意力策略（FULL / RESIDUAL_WINDOW / SPARSE / CFG_SHARE / OUTPUT_SHARE），用于采样步间复用与 CFG 共享

### 2.2 空间 Self-Attention（单帧内部，非因果）
- **源**：`modules/attention.py` `attention()` → `flash_attn_varlen_func()` 或 `scaled_dot_product_attention()`
- **目标**：`fusion_mlx/video/skyreels_v3/spatial_attention.py`
  - 删除 `flash_attn_interface` / `flash_attn` CUDA 分支
  - 调用 `mfa_bridge.flash_attention(q, k, v, causal=False, window_size=ws, scale=1/sqrt(head_dim))`
  - 入参对齐：SkyReels 空间注意力非因果，`window_size=(-1,-1)` 全局
  - GQA 分组（SkyReels-V3 用 num_heads 多于 num_kv_heads）参数原样迁移，q/k/v reshape 严格对齐 `[B, L, N, D]`
- **M5 专属 Tile 分块**：`mfa/dispatch_policy.py` 已读 `mx.get_device_info()`，按 L2 缓存动态选 block 尺寸，无需改

### 2.3 时序 Temporal-Attention（多帧联动，视频防闪烁关键）
- **源**：`transformer_a2v.py` / `reference_to_video/transformer.py` 中时序分支
- **目标**：`fusion_mlx/video/skyreels_v3/temporal_attention.py`
  - 启用滑动窗口 SW-FA：`window_size=(W, W)`，控制上下文降低显存
  - 时序序列长度偏大，强制走 mlx-mfa STEEL 内核（短序列才退 MLX-SDPA，底座 `select_backend` 已做）
  - GQA 分组参数原样迁移，多分组查询逻辑严格对齐原版权重，避免画面崩坏

### 2.4 CrossAttention（文本 Prompt / 参考图引导）
- **源**：`transformer.py` `WanT2VCrossAttention` / `WanI2VCrossAttention`
- **目标**：`fusion_mlx/video/skyreels_v3/cross_attention.py`
  - 文本引导：`mfa_bridge.flash_attention(q, k, v, causal=False)`，关闭因果掩码
  - i2v 参考图引导：`WanI2VCrossAttention` 中 `context_img = context[:, :257]` 拆分逻辑保留，img_x 与 text_x 相加
  - 短序列兜底：`select_backend` 在 seq_len_q 短时自动退 MLX-SDPA

### 2.5 xfuser 步级策略接入（采样加速，M5 关键点）
- 在 SkyReels 采样循环中，每步传 `step_idx` 给 `MLXFastAttention`
- 配置 `steps_method`：前期 FULL_ATTN 保画质，后期 RESIDUAL_WINDOW_ATTN / SPARSE_ATTN 提速
- CFG_SHARE：无条件/条件分支共享 K/V 计算，减半注意力开销
- OUTPUT_SHARE：相邻步输出复用，跳过部分步计算

---

## 阶段 3：DiT 主干 Block 全量迁移（P1，模型主体）

### 3.1 已有底座参考（已核对）
- `fusion_mlx/video/wan2/transformer.py` —— WanAttentionBlock 纯 MLX 版，含 modulation(1,6,dim) + self_attn + cross_attn + ffn，是 SkyReels R2V/V2V 的直接蓝本
- `fusion_mlx/video/wan2/attention.py` —— WanSelfAttention / WanCrossAttention / WanLayerNorm / WanRMSNorm 纯 MLX 实现
- `fusion_mlx/video/wan2/rope.py` —— rope_params / rope_precompute_cos_sin 已实现

### 3.2 基础层替换清单
| 组件 | 源 | 目标 MLX | 要点 |
|---|---|---|---|
| PatchEmbed Conv3d | `transformer.py` patch_embedding | `mx.nn.Conv3d` 或拆解 | latent 隐空间尺寸适配，`patch_size=(1,2,2)` |
| AdaLN-Zero modulation | `WanAttentionBlock.modulation` | `mx.array (1,6,dim) float32` | wan2 已实现，复刻前置 shift/scale/gate 逻辑，参数不可改 |
| WanLayerNorm | `transformer.py` | `mx.nn.LayerNorm` | wan2 attention.py 已封装 |
| WanRMSNorm (qk_norm) | `transformer.py` | 自实现 `x * rsqrt(mean(x^2)+eps) * weight` | wan2 已实现 |
| QKV Linear 投影 | `nn.Linear` × 3 | `mx.nn.Linear` × 3 + `mx.compile` 融合 | 超大维度投影开启算子融合 |
| FFN (Linear+GELU+Linear) | `transformer.py` ffn | `mx.nn.Linear` + `mx.nn.GELUApproximate` + `mx.nn.Linear` | 开启 MLX 算子融合，消除中间临时张量 |
| Head (输出投影) | `transformer.py` Head 类 | wan2 已有 Head MLX 版 | modulation(1,2,dim) 保留 |

### 3.3 三大业务分支配置（不能混用）
| 分支 | 主干规模 | 注意力组合 | SkyReels 源模块 |
|---|---|---|---|
| R2V（参考图→视频）| 14B | 空间 Self-Attn + 参考图 Cross-Attn | `modules/reference_to_video/transformer.py` SkyReelsA2WanI2v3DModel |
| V2V（视频续写）| 14B | 扩大时序窗口，时序注意力占比更高 | `modules/transformer.py` WanModel + context_window_size 参数链路 |
| A2V（音频数字人）| 19B | 新增音频 Embedding 分支 + 音频编码器 | `modules/transformer_a2v.py` + `wav2vec2.py` + `xlm_roberta.py` |

### 3.4 新建目录结构
```
fusion_mlx/video/skyreels_v3/
├── __init__.py
├── _device.py                    # MLX Stream + M5 判断
├── convert_skyreels_v3.py        # 权重转换
├── config.py                     # 三大分支配置注册
├── attention.py                  # 空间注意力（调用 mfa_bridge）
├── spatial_attention.py          # 单帧空间分支
├── temporal_attention.py         # 多帧时序分支（SW-FA）
├── cross_attention.py            # 文本/参考图交叉注意力
├── transformer.py                # WanAttentionBlock 纯 MLX 端口
├── transformer_a2v.py            # A2V 数字人主干端口
├── transformer_r2v.py            # R2V 参考图主干端口
├── transformer_v2v.py            # V2V 视频续写主干端口
├── rope.py                       # rope_params / rope_apply MLX 版
├── patch_embed.py                # Conv3d PatchEmbed
├── head.py                       # 输出投影 Head
├── vae.py                        # VAE 解码器端口
├── clip.py                       # CLIP 文本编码器端口
├── t5.py                         # UMT5 文本编码器端口（复用 wan2/t5_encoder.py）
├── wav2vec2.py                   # 音频编码器端口（A2V 专用）
├── xlm_roberta.py                # XLM-RoBERTa 端口（A2V 专用）
├── scheduler/
│   └── fm_solvers_unipc.py       # Flow-Matching 采样器端口
├── pipelines/
│   ├── reference_to_video.py     # R2V pipeline 端口
│   ├── single_shot_extension.py  # V2V 单镜头续写 pipeline 端口
│   ├── shot_switching_extension.py # V2V 镜头切换 pipeline 端口
│   └── talking_avatar.py         # A2V 数字人 pipeline 端口
└── kv_cache.py                   # SkyReels 专用 KV 缓存（空间/时序双池）
```

---

## 阶段 4：KV Cache 缓存架构重构（P1，长视频决定性模块）

### 4.1 底座可用成果（已核对）
- `fusion_mlx/custom_kernels/kv_cache.py` —— KV Cache 基础封装
- `fusion_mlx/custom_kernels/turboquant.py` + `turboquant_fused.metal` —— 4-bit KV 量化，4× 内存带宽节省
- `fusion_mlx/custom_kernels/fp8_linear.py` —— FP8 线性层支持
- `fusion_mlx/custom_kernels/quantize.py` —— 量化工具
- `fusion_mlx/kv_cache_dtype.py` —— KV 缓存 dtype 调度

### 4.2 SkyReels 专用 KV 缓存改造
- **存储格式**：摒弃 PyTorch 列表式连续张量，改用 MLX 惰性数组托管，全程常驻 Metal 显存，杜绝 CPU/GPU 反复拷贝
- **双缓存池**：分别维护空间 KV 缓存（单帧内部）与时序 KV 缓存（多帧联动），视频续写时可复用前置帧缓存
- **滑动窗口淘汰**：实现 SW-KV 淘汰策略，超长序列自动淘汰老旧 KV，解决 720P 30s+ 长视频 OOM
- **预分配**：根据目标输出帧数提前预分配数组，避免迭代生成中动态扩容带来的性能抖动（M5 收益显著）
- **TurboQuant 接入**：长视频 KV 强制 4-bit 量化，显存占用再降 4×
- **核心价值**：原版 PyTorch MPS 生成 30s 视频显存持续膨胀，MLX 重构缓存后显存占用可下降 35%~55%

### 4.3 KV Cache 接口设计
```python
# fusion_mlx/video/skyreels_v3/kv_cache.py
class SkyReelsKVCache:
    def __init__(self, num_layers, num_heads, head_dim, max_frames, h, w, dtype=mx.bfloat16):
        # 预分配空间 KV + 时序 KV 双池
        ...
    def append_spatial(self, layer_idx, k, v): ...
    def append_temporal(self, layer_idx, k, v): ...
    def get_spatial(self, layer_idx): ...
    def get_temporal(self, layer_idx, window_size): ...  # SW 淘汰
    def quantize(self, bits=4): ...  # 接入 TurboQuant
```

---

## 阶段 5：采样器、调度器与后置流水线（P2）

### 5.1 采样器迁移
- **源**：`skyreels_v3/scheduler/fm_solvers_unipc.py` —— SkyReels 自研改进版 Flow-Matching 采样（非 DDIM）
- **目标**：`fusion_mlx/video/skyreels_v3/scheduler/fm_solvers_unipc.py`
  - 采样迭代计算全部由 Torch 迁 MLX 数组运算
  - 循环采样函数套 `mx.compile` 全局编译
  - **原版采样系数、时间步 schedule 参数完全保留原值，不可修改**，防止画风失真、主体漂移
  - UniPC 多步法逻辑严格对齐

### 5.2 文本编码器（UMT5 / CLIP）
- **UMT5**：复用底座 `fusion_mlx/video/t5_encoder.py`（已存在 `T5Encoder` / `T5EncoderConfig` / `load_t5_encoder`）
  - SkyReels 用 `UMT5EncoderModel`，需核对 wan2 t5_encoder 是否兼容 UMT5 变体，若不兼容则新增 `t5_umt5.py`
- **CLIP**：`modules/clip.py` 两种方案：
  - 方案 1（最优）：完整迁移 CLIP 到 MLX，全局推理链路闭环
  - 方案 2（兜底）：短时 Prompt 用 CPU 预处理，大批量生成采用 MLX 版本
- 底座已有 `fusion_mlx/mlx_clip/` 和 `fusion_mlx/mlx-embeddings/`，可直接复用

### 5.3 视频后置流水线
- 去噪输出 latent → VAE 解码器重构画面
- VAE 是重度卷积网络（`modules/vae.py` AutoencoderKLWan），全部卷积算子交由 MPS 执行
- 帧拼接、编码导出逻辑保留 Python 上层，张量计算全部 MLX 托管
- 复用底座 `fusion_mlx/video/wan2/vae.py` / `vae22.py` 已有 VAE 端口（需核对 SkyReels VAE 与 Wan VAE 结构差异）

---

## 阶段 6：M5 Max 专属专项优化（P0.5，fusion-mlx 独有竞争力）

### 6.1 利用 M5 GPU 内置 Neural Accelerator
- 在大矩阵乘法、QKV 投影层指定计算设备调度，分流通用 GPU 算力
- 通过 `mx.set_device()` 或 Metal command buffer 优先级调度

### 6.2 分级 Tile 自适应
- dFlash（mlx-mfa）自动读取设备 L2 缓存大小，动态调整注意力分块尺寸
- `mfa/dispatch_policy.py` 已按 `mx.get_device_info()` 分派，M5 专属 STEEL_DSPLIT 配置需新增

### 6.3 多级量化方案内置
- FP8 推理：`custom_kernels/fp8_linear.py` 已支持
- NF4 权重加载：`custom_kernels/quantize.py` 扩展
- 目标：19B 模型 720P 视频常驻内存压缩至 14GB 左右

### 6.4 异步数据流
- MLX Stream 异步加载下一帧输入，和当前迭代推理并行执行
- 底座 `fusion_mlx/engine_core.py` async_eval 双缓冲已有基础，迁移到视频帧加载

---

## 阶段 7：三大 Pipeline 端口

### 7.1 R2V（参考图→视频，14B-720P）
- **源**：`skyreels_v3/pipelines/reference_to_video_pipeline.py` WanSkyReelsA2WanT2VPipeline
- **目标**：`fusion_mlx/video/skyreels_v3/pipelines/reference_to_video.py`
- 要点：1~4 张参考图 + 文本 Prompt → 5s 720p 24fps 视频；SkyReelsA2WanI2v3DModel 主干端口

### 7.2 V2V（视频续写，14B-720P）
- **源**：`skyreels_v3/pipelines/single_shot_extension_pipeline.py` + `shot_switching_extension_pipeline.py`
- **目标**：`fusion_mlx/video/skyreels_v3/pipelines/single_shot_extension.py` + `shot_switching_extension.py`
- 要点：5s→30s 单镜头续写 + 镜头切换（Cut-In/Cut-Out/Shot-Reverse）；context_window_size 参数链路

### 7.3 A2V（音频数字人，19B-720P）
- **源**：`skyreels_v3/pipelines/talking_avatar_pipeline.py`
- **目标**：`fusion_mlx/video/skyreels_v3/pipelines/talking_avatar.py`
- 要点：音频驱动数字人说话；新增 wav2vec2 + xlm_roberta 音频编码器端口

---

## 阶段 8：调度器与 config 注册

### 8.1 config 注册
- 在 `fusion_mlx/video/__init__.py` 注册 SkyReels-V3 三大分支
- 配置文件格式对齐底座 `model-config.json`

### 8.2 generate_video 入口端口
- **源**：`SkyReels-V3/generate_video.py`（argparse + task_type 调度）
- **目标**：在 `fusion_mlx/video/skyreels_v3/__init__.py` 提供 `generate_video(task_type, ...)` 入口
- 复用底座 `fusion_mlx/cli.py` 命令注册

---

## 整体落地排期（基于现有 fusion-mlx 代码底座）

| 周期 | 工作内容 | 预期成果 |
|---|---|---|
| 第 1~3 天 | 环境适配 + 权重转换器 + dFlash 对接双分支注意力，完成短帧（16 帧）跑通 | 基础链路可用，输出低分辨率视频 |
| 第 4~10 天 | DiT 主干（WanAttentionBlock × 32）、AdaLN-Zero、FFN 全量迁移 + KV 缓存重构 | 720P 30 帧稳定生成 |
| 第 11~18 天 | 采样器（fm_solvers_unipc）+ VAE + CLIP/UMT5 迁移，M5 专属缓存与算子调优 | 速度相比原生 PyTorch MPS 提升 2.7~3.3 倍 |
| 第 19~21 天 | 三大 Pipeline（R2V/V2V/A2V）端口 + 压测、修复时序闪烁问题，合并进 fusion-mlx 仓库 | 正式 Release 版本 |

---

## 底座接口契约（迁移时严格遵守）

### 注意力调用
```python
from fusion_mlx.custom_kernels.mfa_bridge import flash_attention
out = flash_attention(q, k, v, scale=scale, mask=mask, causal=False, window_size=ws)
```

### xfuser 步级策略
```python
from fusion_mlx.custom_kernels.xfuser_attention import MLXFastAttention, FastAttnMethod
fa = MLXFastAttention(window_size=ws)
fa.set_methods(steps_method=[FastAttnMethod.FULL_ATTN, ...])
out = fa(q, k, v, step_idx=step, scale=scale, mask=mask)
```

### KV Cache 量化
```python
from fusion_mlx.custom_kernels.turboquant import turboquant_kv
# 4-bit KV 量化，长视频强制启用
```

### 编译开关
```python
# DiT Block 全量编译
dit_block = mx.compile(dit_block, inputs=[x, e, grid_sizes, freqs, context])
# 采样循环编译
sample_step = mx.compile(sample_step)
```

---

## 风险点与数值核对

1. **rope fp64→fp32 降精度**：原版 `rope_apply` 用 `torch.float64` 计算，MLX 无 fp64，用 fp32 替代，需在首帧数值层面核对（输出 MSE < 1e-5）
2. **GQA 分组维度对齐**：SkyReels-V3 多分组查询逻辑，reshape 时 `[B, L, N, D]` 与 `[B, L, Nk, D]` 的 N/Nk 比例严格对齐原版
3. **AdaLN modulation 精度**：原版 `amp.autocast(dtype=torch.float32)` 保精度，MLX 端 modulation 数组强制 `astype(mx.float32)`，wan2 已实现可复用
4. **时序注意力窗口边界**：SW-FA 窗口边界帧 KV 淘汰逻辑需与原版对齐，否则视频首尾帧闪烁
5. **Conv3d MLX 支持**：需核对当前 MLX 版本是否原生支持 `mx.nn.Conv3d`，若不支持则拆为 Conv2d × T 或用 Conv1d 沿时间轴
6. **diffusers 依赖剥离**：SkyReels-V3 源码重度依赖 `diffusers`（ModelMixin / ConfigMixin / DiffusionPipeline），MLX 端需用 `mlx.nn.Module` 重写，不保留 diffusers 依赖

---

## 验收标准

- [ ] R2V 14B：1~4 张参考图 + Prompt → 5s 720p 24fps 视频，画面与原版 CUDA 输出 PSNR > 35dB
- [ ] V2V 14B：5s 输入视频 → 30s 续写，时序连贯无闪烁
- [ ] A2V 19B：音频 + 参考图 → 数字人说话视频，口型同步
- [ ] M5 Max 速度：相比原生 PyTorch MPS 提升 2.7~3.3 倍
- [ ] M5 Max 显存：19B 720P 视频常驻内存 ≤ 14GB（NF4 量化）
- [ ] KV 缓存：30s 长视频显存占用相比原版下降 35%~55%
- [ ] 合并进 fusion-mlx 仓库，通过底座测试套件
