# Flash Attention 纯 MPS/MLX 迁移方案 — 整合版

> **目标**: 将 `flash_attn` 目录下的 Metal Flash Attention 代码（metal-flash-attention, 
> metal-flash-sdpa, mps-flash-attention, mlx_mfa-2.61.0）的加速能力完整迁移到 fusion-mlx， 
> 并让 xfuser 的注意力加速策略在 Apple Silicon 上原生运行。
>
> **日期**: 2026-07-13 | **状态**: 方案设计

---

## 目录

1. [现状分析](#1-现状分析)
2. [代码资产盘点](#2-代码资产盘点)
3. [迁移架构设计](#3-迁移架构设计)
4. [Phase A: MLX 原生 Metal Kernel 集成](#4-phase-a-mlx-原生-metal-kernel-集成)
5. [Phase B: xfuser 注意力策略移植](#5-phase-b-xfuser-注意力策略移植)
6. [Phase C: TurboQuant & 稀疏注意力](#6-phase-c-turboquant--稀疏注意力)
7. [Phase D: 视频生成模型加速](#7-phase-d-视频生成模型加速)
8. [Phase E: 编译 & 分发流水线](#8-phase-e-编译--分发流水线)
9. [风险与缓解](#9-风险与缓解)
10. [验收标准](#10-验收标准)

---

## 1. 现状分析

### 1.1 现有代码资产

| 目录 | 语言 | 定位 | 状态 |
|------|------|------|------|
| `metal-flash-attention/` | Swift + Metal | ccv 系 Swift 封装，生成 MSL 源码 | 上游参考 |
| `metal-flash-sdpa/` | C++ + Metal + Python | PyTorch MPS 扩展，ccv_nnc_mfa 桥接 | 迁移源 |
| `mps-flash-attention/` | C++/ObjC++ + Swift + Python | PyTorch MPS + Swift MFABridge dylib | 迁移源 |
| `mlx_mfa-2.61.0/` | C++ + Metal + Python (MLX) | **MLX 原生 MFA** — 最完整实现 | **核心迁移源** |
| `xfuser-0.4.5/` | Python | 扩散模型分布式注意力加速 | 适配目标 |
| `xDiT/` | Python | Diffusion Transformer 推理 | 空仓库 |

### 1.2 fusion-mlx 当前注意力能力

| 模块 | 能力 | 实现方式 |
|------|------|----------|
| `mx.fast.scaled_dot_product_attention` | MLX 内置 SDPA | MLX 框架层 Metal 实现 |
| `turboquant_attention.py` | 量化 KV Cache 注意力 | 重写 `mlx_lm` 的 SDPA |
| `pflash.py` | 长 Prompt 压缩 | CPU 端统计压缩，非 GPU 加速 |
| `video/wan2/attention.py` | WAN2 自注意力 | `mx.fast.scaled_dot_product_attention` |
| `video/ltx2/attention.py` | LTX-2 注意力 | `mx.fast.scaled_dot_product_attention` |
| `patches/llama4_attention.py` | Llama 4 注意力缩放 | 补丁层 |
| `patches/minimax_m3_sparse_attention.py` | MiniMax M3 稀疏注意力 | 补丁层 |

### 1.3 差距分析

**fusion-mlx 缺失的能力** (相对于 `mlx_mfa-2.61.0`):

- ❌ STEEL 前向/反向 Metal 内核（分块 Tile 式 Flash Attention）
- ❌ v6 NAX 内核（Apple Neural Engine 加速）
- ❌ SAGE 稀疏注意力
- ❌ GNA 门控注意力
- ❌ Paged KV Cache 注意力（Metal kernel 级）
- ❌ Varlen（变长序列）注意力
- ❌ FP8/INT8/NF4 量化注意力
- ❌ 滑动窗口注意力
- ❌ Split-K 校准
- ❌ 自动热加载预先编译的 metallib

**xfuser 适配的缺失**:

- ❌ `flash_attn` 不可用时的回退层
- ❌ 窗口注意力残差（Window Residual）无 Metal 加速
- ❌ 无 `mx.fast.scaled_dot_product_attention` 优化路径

---

## 2. 代码资产盘点

### 2.1 mlx_mfa-2.61.0 核心文件清单

```
csrc/
├── mfa_attention.cpp                      # MFAttention Primitive — 前向/反向入口
├── mfa_steel_fwd.cpp                      # STEEL FWD v1 实现
├── mfa_steel_fwd_v2.cpp                   # STEEL FWD v2 (改进分块)
├── mfa_steel_fwd_v3.cpp                   # STEEL FWD v3 (小 D 优化)
├── mfa_steel_fwd_v6_nax.cpp               # v6 NAX 前向
├── mfa_steel_bwd.cpp                      # STEEL 反向
├── mfa_gna_fwd.cpp                        # GNA 前向
├── mfa_sage_fwd.cpp                       # SAGE 稀疏前向
├── mfa_steel_paged_varlen_fwd.cpp         # Paged Varlen 前向
├── mfa_steel_paged_varlen_tq_fwd.cpp      # Paged Varlen + TurboQuant 前向
├── mfa_sparse_attention.cpp               # 稀疏注意力
├── mfa_shader_gen.cpp                     # MSL 源码生成器
├── mfa_quantize.cpp                       # 量化内核
├── mfa_conv_nax.cpp                       # NAX 卷积
├── shader_cache.hpp / shader_cache.mm     # Metal 着色器缓存
├── v6_nax_detect.hpp / v6_nax_detect.mm   # NAX 硬件检测
├── bindings.cpp                           # pybind11 绑定
├── async_v2_kernel.metal                  # 异步 v2 Metal 内核
mlx_mfa/
├── attention.py                           # 主注意力 API (8892行)
├── dispatch_policy.py                     # 分派策略
├── inference.py                           # 推理 API
├── kv_cache.py                            # KV Cache 实现
├── lcsa_nax.py                            # NAX 稀疏注意力
├── masks.py                               # 掩码工具
├── quantize.py                            # 量化
├── compile_metallib.py                    # metallib 编译
├── runtime.py                             # 运行时
├── turboquant.py                          # TurboQuant
├── topk_stream.py                         # Top-K 流式
├── tq_decode.py                           # TurboQuant 解码
└── integrations/                          # 模型集成
```

### 2.2 xfuser 核心注意力文件

```
xfuser/core/fast_attention/
├── attn_layer.py          # xFuserFastAttention — 窗口残差策略
├── fast_attn_state.py     # 状态管理
└── utils.py               # 校准/方法选择
```

### 2.3 metal-flash-sdpa 核心文件

```
csrc/mfa/
├── ccv_nnc_mfa_attention.cpp    # 注意力入口 (832行)
├── ccv_nnc_mfa.cpp              # 上下文管理
├── ccv_nnc_mfa.hpp              # 头文件
├── Metal.cpp                    # Metal 设备管理
├── v2/AttentionKernel.cpp       # 注意力内核 (3321行)
├── v2/AttentionKernel.hpp       # 内核头文件
├── v2/AttentionDescriptor.cpp   # 描述符
├── v2/CodeWriter.cpp            # MSL 代码生成
├── v2/GEMMHeaders.cpp           # GEMM 头
├── v2/NAAttentionKernel.cpp     # NAX 注意力内核
├── v2/NAMatMulKernel.cpp        # NAX 矩阵乘法
├── v2/ShaderCache.hpp           # 着色器缓存
└── v2/CastKernel.cpp            # 类型转换内核
```

---

## 3. 迁移架构设计

### 3.1 整体架构

```
┌─────────────────────────────────────────────────────────────────────┐
│                      fusion-mlx 用户层 API                           │
│  flash_attention() / flash_attention_kvcache() / sage_attention()   │
├─────────────────────────────────────────────────────────────────────┤
│                       Python 调度层                                  │
│  dispatch_policy.py — 自动选择 STEEL / NAX / SDPA fallback          │
├──────────────────┬──────────────────┬───────────────────────────────┤
│  MLX 原生路径     │  PyTorch MPS 路径  │  xfuser 兼容路径            │
│  (mx.fast +       │  (torch.mps +     │  (mx.fast SDPA +            │
│   mx.custom_call) │   Metal dylib)    │   窗口残差策略)              │
├──────────────────┴──────────────────┴───────────────────────────────┤
│                   Metal Shader Layer (MSL)                          │
│  STEEL fwd/bwd  │  v6 NAX  │  SAGE  │  GNA  │  TurboQuant          │
├─────────────────────────────────────────────────────────────────────┤
│                       Apple GPU (MPS)                               │
└─────────────────────────────────────────────────────────────────────┘
```

### 3.2 分阶段迁移策略

| Phase | 内容 | 依赖 | 预计工作量 |
|-------|------|------|-----------|
| **A** | MLX 原生 Metal Kernel 集成 | `mlx_mfa` 源码 | 3-4 周 |
| **B** | xfuser 注意力策略移植 | Phase A 完成 | 2-3 周 |
| **C** | TurboQuant & 稀疏注意力 | Phase A 完成 | 2-3 周 |
| **D** | 视频生成模型加速 | Phase A + C | 1-2 周 |
| **E** | 编译 & 分发流水线 | 全部完成 | 1 周 |

---

## 4. Phase A: MLX 原生 Metal Kernel 集成

### 4.1 核心策略

**不重新发明轮子** — 直接利用 `mlx_mfa-2.61.0` 的 C++ 源码作为静态库/源码集成，而非从头编写 Metal 内核。

### 4.2 技术方案

#### 方案 A1: 源码集成 (推荐)

将 `mlx_mfa-2.61.0/csrc/` 编译为静态库，通过 `mlx.core.custom_call` 注册为 custom primitive。

```
fusion_mlx/
├── custom_kernels/
│   ├── __init__.py
│   ├── mfa/                          # 从 mlx_mfa-2.61.0 移植
│   │   ├── CMakeLists.txt
│   │   ├── csrc/
│   │   │   ├── mfa_attention.cpp
│   │   │   ├── mfa_steel_fwd.cpp
│   │   │   ├── mfa_steel_bwd.cpp
│   │   │   ├── mfa_steel_fwd_v2.cpp
│   │   │   ├── mfa_steel_fwd_v3.cpp
│   │   │   ├── mfa_steel_fwd_v6_nax.cpp
│   │   │   ├── mfa_gna_fwd.cpp
│   │   │   ├── mfa_sage_fwd.cpp
│   │   │   ├── mfa_steel_paged_varlen_fwd.cpp
│   │   │   ├── mfa_steel_paged_varlen_tq_fwd.cpp
│   │   │   ├── mfa_sparse_attention.cpp
│   │   │   ├── mfa_shader_gen.cpp
│   │   │   ├── mfa_quantize.cpp
│   │   │   ├── mfa_conv_nax.cpp
│   │   │   ├── mfa_env.hpp
│   │   │   ├── shader_cache.hpp / .mm
│   │   │   ├── v6_nax_detect.hpp / .mm
│   │   │   └── bindings.cpp
│   │   ├── mlx_mfa/                  # Python 层 (精简)
│   │   │   ├── attention.py
│   │   │   ├── dispatch_policy.py
│   │   │   ├── kv_cache.py
│   │   │   ├── masks.py
│   │   │   ├── quantize.py
│   │   │   └── turboquant.py
│   │   └── precompiled/              # 预编译 metallib
│   │       └── *.metallib
│   └── phase_c/                      # 已有
```

#### 方案 A2: submodule + Python 桥接 (轻量)

将 `mlx_mfa-2.61.0` 作为 git submodule，通过 `mlx_mfa` 的 pip 包直接导入。

**推荐理由**: `mlx_mfa` 已经是一个完整的 pip 包，包含 `setup.py` 和编译脚本。直接依赖它比重新集成更省力。

### 4.3 推荐方案: 混合路径

**短期（1-2周）**: 将 `mlx_mfa` 作为可选依赖 (extras_require)

```python
# pyproject.toml
[project.optional-dependencies]
mfa = ["mlx-mfa>=2.61.0"]
```

```python
# fusion_mlx/custom_kernels/mfa_bridge.py
"""统一 MFA 桥接层 — 适配 mlx_mfa 到 fusion-mlx 的 API。"""

import logging
from typing import Optional

logger = logging.getLogger(__name__)

_HAS_MFA = False
try:
    import mlx_mfa
    _HAS_MFA = True
except ImportError:
    pass


def flash_attention(
    q, k, v,
    scale: Optional[float] = None,
    mask=None,
    causal: bool = False,
    **kwargs
):
    """统一 Flash Attention 入口 — 优先使用 MFA，回退到 mx.fast.sdpa。"""
    if _HAS_MFA:
        return mlx_mfa.flash_attention(q, k, v, scale=scale, mask=mask, causal=causal)
    # fallback
    import mlx.core as mx
    if scale is None:
        scale = q.shape[-1] ** -0.5
    return mx.fast.scaled_dot_product_attention(q, k, v, scale=scale, mask=mask)
```

**长期（3-4周）**: 提取 `mlx_mfa` 的 C++ Metal 内核源码，直接集成到 fusion-mlx 的 `custom_kernels/mfa/` 目录下，通过 `mlx.core.custom_call` 注册 custom primitive，形成不受外部依赖绑架的独立实现。

### 4.4 关键集成点

#### 4.4.1 STEEL 前向内核注册

```cpp
// custom_kernels/mfa/bindings.cpp (从 mlx_mfa-2.61.0/csrc/bindings.cpp 移植)
#include <mlx/core.h>
#include "mfa_attention.hpp"

void register_mfa_primitives() {
  mlx::core::metal::register_primitive("mfa_attention_forward", 
    [](mlx::core::Stream stream, const std::vector<mlx::core::array>& inputs) {
      // 从 inputs 提取参数
      // 调用 MFAttention::eval_gpu
    });
}
```

#### 4.4.2 MSL 着色器编译

```python
# fusion_mlx/custom_kernels/mfa/compile_metallib.py
"""从 mlx_mfa/compile_metallib.py 移植。"""
from mlx_mfa.compile_metallib import compile_metallib  # 短期直接复用

# 长期: 独立实现
def compile_mfa_metallib(output_path: str) -> None:
    """编译 MFA MSL 源码为 metallib。"""
    # 使用 Metal 编译器 (xcrun metal)
    # 生成预编译 metallib 存入 precompiled/
```

#### 4.4.3 分派策略

```python
# fusion_mlx/custom_kernels/mfa/dispatch_policy.py
"""从 mlx_mfa/dispatch_policy.py 移植决策逻辑。"""

# 核心决策树:
# 1. NAX 可用 (M3+/M4+)? → v6 NAX 内核
# 2. head_dim ∈ {64, 128, 256}? → STEEL 内核
# 3. 量化 K/V? → TurboQuant 内核
# 4. 大 head_dim (512)? → D-split STEEL
# 5. 否则 → mx.fast.scaled_dot_product_attention
```

### 4.5 实现步骤

```
Step 1: 将 mlx_mfa 添加为可选依赖，编写 mfa_bridge.py 适配层
Step 2: 验证基本 flash_attention 调用路径
Step 3: 提取 C++ 绑定层，通过 custom_call 注册 custom primitive
Step 4: 实现 dispatch_policy 自动选择最佳内核
Step 5: 添加 KV Cache 注意力 (flash_attention_kvcache)
Step 6: 添加 Varlen 支持
Step 7: 集成测试 & 基准测试
```

---

## 5. Phase B: xfuser 注意力策略移植

### 5.1 xfuser 核心机制

xfuser 的注意力加速围绕 **DiTFastAttn** 论文设计，核心思想：

1. **窗口注意力残差 (Window Residual)**: 在少数 full-attention step 计算残差，中间 step 只算窗口注意力+残差
2. **CFG Share**: 无条件分支的注意力输出直接复用到有条件分支
3. **Output Share**: 直接复用上一步的输出
4. **校准 (Calibration)**: 用 COCO 数据集自动选择每层的压缩策略

### 5.2 移植方案

```python
# fusion_mlx/custom_kernels/xfuser_attention.py
"""xfuser 注意力策略的 MLX 原生移植。"""

import mlx.core as mx
from enum import Flag, auto
from typing import Optional

class FastAttnMethod(Flag):
    FULL_ATTN = auto()
    RESIDUAL_WINDOW_ATTN = auto()
    OUTPUT_SHARE = auto()
    CFG_SHARE = auto()

class MLXFastAttention:
    """MLX 版 xFuserFastAttention — 使用 mx.fast.scaled_dot_product_attention 或 MFA。"""
    
    def __init__(self, window_size: int = -1):
        self.window_size = [window_size, window_size]
        self.steps_method: list[FastAttnMethod] = []
        self.cached_output: Optional[mx.array] = None
        self.cached_residual: Optional[mx.array] = None
    
    def __call__(self, q, k, v, method: FastAttnMethod, scale: float, 
                 need_residual: bool = False):
        """执行注意力计算。"""
        # 使用 MFA 或 mx.fast.sdpa
        if method == FastAttnMethod.FULL_ATTN:
            out = self._full_attention(q, k, v, scale)
            if need_residual:
                window_out = self._window_attention(q, k, v, scale)
                self.cached_residual = out - window_out
            return out
        elif method == FastAttnMethod.RESIDUAL_WINDOW_ATTN:
            window_out = self._window_attention(q, k, v, scale)
            return window_out + self.cached_residual
        # ... 其他策略
    
    def _full_attention(self, q, k, v, scale):
        if _HAS_MFA:
            return mlx_mfa.flash_attention(q, k, v, scale=scale)
        return mx.fast.scaled_dot_product_attention(q, k, v, scale=scale)
    
    def _window_attention(self, q, k, v, scale):
        window_mask = self._make_window_mask(q, k)
        return mx.fast.scaled_dot_product_attention(q, k, v, scale=scale, mask=window_mask)
```

### 5.3 模型注入点

| 模型 | 注入位置 | 修改方式 |
|------|----------|----------|
| **Stable Diffusion 3** | `transformer_sd3.py` | 替换 AttentionProcessor |
| **Flux** | `transformer_flux.py` | 替换 JointAttention |
| **PixArt-α** | `pixart_transformer_2d.py` | 替换 Attention |
| **HunyuanDiT** | `hunyuan_transformer_2d.py` | 替换 Attention |
| **CogVideoX** | `cogvideox_transformer_3d.py` | 替换 3D Attention |
| **WAN2** | `video/wan2/transformer.py` | 替换 WanSelfAttention |
| **LTX-2** | `video/ltx2/transformer.py` | 替换 Attention |

### 5.4 校准流水线

```python
# fusion_mlx/custom_kernels/xfuser_attention.py (续)

def calibrate_attention_strategy(
    pipeline,
    calib_prompts: list[str],
    n_steps: int,
    threshold: float = 0.1,
) -> list[list[FastAttnMethod]]:
    """自动校准每层每步的注意力策略 (移植自 xfuser/utils.py)。"""
    # 1. 注册 forward pre-hook
    # 2. 对每个 step 尝试不同策略
    # 3. 计算压缩损失 (compression_loss)
    # 4. 选择损失 < 阈值的策略
    # 5. 返回策略矩阵
    pass
```

### 5.5 实现步骤

```
Step 1: 实现 MLXFastAttention 核心类 (窗口注意力 + 残差)
Step 2: 移植校准流水线 (calibrate_attention_strategy)
Step 3: 注入 SD3 / Flux 等主流扩散模型
Step 4: 注入 WAN2 / LTX-2 视频模型
Step 5: 基准测试 & 调优
```

---

## 6. Phase C: TurboQuant & 稀疏注意力

### 6.1 TurboQuant 移植

fusion-mlx 已有 `turboquant_attention.py` 补丁层，但缺少 Metal 级量化内核。

**移植内容**:

```python
# fusion_mlx/custom_kernels/mfa/turboquant.py
"""从 mlx_mfa/turboquant.py 移植。"""

# 核心能力:
# 1. FP8_E4M3 / FP8_E5M2 量化注意力
# 2. INT8 量化注意力
# 3. NF4 量化注意力
# 4. Paged + TurboQuant 注意力
# 5. TurboQuant + RoPE 融合

# 内核接口:
# mfa_steel_paged_varlen_tq_forward(pool_k, pool_v, ...)
# mfa_quantize_kv_cache(k, v, quant_type) -> (k_quant, v_quant, scales)
```

### 6.2 稀疏注意力移植

```python
# fusion_mlx/custom_kernels/mfa/sparse_attention.py
"""从 mlx_mfa/lcsa_nax.py + attention.py 中稀疏相关部分移植。"""

# 核心能力:
# 1. LCSA (Local Context Sparse Attention) — NAX 加速
# 2. Top-K 稀疏注意力
# 3. Block sparse attention (block_mask)
# 4. 混合稀疏注意力 (v6nax_sparse_hybrid)
```

### 6.3 实现步骤

```
Step 1: 移植 TurboQuant 前向内核 (复用 mfa_steel_paged_varlen_tq_fwd.cpp)
Step 2: 集成到 turborquant_attention.py 补丁层
Step 3: 移植稀疏注意力内核
Step 4: 添加 NAX 检测 & 路由
Step 5: 集成测试
```

---

## 7. Phase D: 视频生成模型加速

### 7.1 当前状态

| 模型 | 注意力实现 | 瓶颈 |
|------|-----------|------|
| **WAN2** | `mx.fast.scaled_dot_product_attention` | 无滑动窗口 Metal 加速 |
| **LTX-2** | `mx.fast.scaled_dot_product_attention` | 无 3D 注意力优化 |

### 7.2 优化方案

```python
# fusion_mlx/video/wan2/attention.py (增强)
from fusion_mlx.custom_kernels.mfa_bridge import flash_attention as mfa_attn

class WanSelfAttention(nn.Module):
    def __call__(self, x, seq_lens, grid_sizes, freqs, ...):
        # ... QKV 投影 + RoPE ...
        
        # 使用 MFA 加速 (如果可用)
        if _HAS_MFA:
            out = mfa_attn(q, k, v, scale=self.scale, mask=attn_mask)
        else:
            out = mx.fast.scaled_dot_product_attention(
                q, k, v, scale=self.scale, mask=attn_mask
            )
```

### 7.3 实现步骤

```
Step 1: 注入 MFA 桥接到 WAN2 注意力
Step 2: 注入 MFA 桥接到 LTX-2 注意力
Step 3: 添加滑动窗口注意力支持 (WAN2 需要)
Step 4: 基准测试 (tokens/sec vs 当前实现)
```

---

## 8. Phase E: 编译 & 分发流水线

### 8.1 预编译 metallib

```python
# scripts/build_mfa_metallibs.sh
#!/bin/bash
# 编译 MFA Metal 着色器为预编译 metallib
# 从 mlx_mfa/scripts/build_metallibs.sh 移植

xcrun -sdk macosx metal \
    -c csrc/mfa_steel_fwd.metal \
    -o mfa_steel_fwd.air \
    -O2

xcrun -sdk macosx metallib \
    mfa_steel_fwd.air \
    -o fusion_mlx/custom_kernels/mfa/precompiled/mfa_steel_fwd.metallib
```

### 8.2 分发策略

```python
# pyproject.toml
[project.optional-dependencies]
mfa = ["mlx-mfa>=2.61.0"]  # 短期: pip 依赖
# 长期: 内置轮子包含预编译 metallib

[tool.setuptools.package_data]
"fusion_mlx.custom_kernels.mfa.precompiled" = ["*.metallib"]
```

---

## 9. 风险与缓解

| 风险 | 影响 | 缓解措施 |
|------|------|----------|
| `mlx_mfa` 依赖 MLX 特定版本 | 版本冲突 | 设置 `mlx>=0.20.0` 约束；使用 `pip check` |
| MSL 着色器编译失败 | 内核不可用 | 优雅降级到 `mx.fast.sdpa` |
| Apple GPU 架构差异 (M1 vs M4) | 性能次优 | 运行时检测 GPU 架构，选择不同分块参数 |
| NAX 不可用 (M1/M2) | 无法使用 v6 内核 | 自动回退到 STEEL 内核 |
| xfuser 校准耗时 | 启动慢 | 缓存校准结果到 JSON 文件；支持预加载 |
| 预编译 metallib 二进制大小 | 包体积增大 | 仅包含常用配置 (head_dim=64/128) |

---

## 10. 验收标准

### 10.1 功能验收

| 验收项 | 标准 | 验证方式 |
|--------|------|----------|
| Flash Attention 前向 | 与 `mx.fast.sdpa` 数值一致 | 测试 `test_attention.py` |
| Flash Attention 反向 | 梯度与参考实现一致 | 测试 `test_backward.py` |
| KV Cache 注意力 | 缓存追加/查询正确 | 测试 `test_kvcache.py` |
| 滑动窗口注意力 | 窗口外信息被屏蔽 | 测试 `test_sliding_window.py` |
| 量化注意力 | FP8/INT8/NF4 结果正确 | 测试 `test_quantized.py` |
| xfuser 策略 | 窗口残差计算正确 | 像素级比对 |

### 10.2 性能验收

| 模型 | 场景 | 加速目标 |
|------|------|---------|
| LLM prefll (4K tokens) | 纯 MLX | ≥ 1.5× vs `mx.fast.sdpa` |
| LLM decode (单 token) | 纯 MLX | ≥ 1.2× |
| Diffusion (SD3, 50步) | xfuser 策略 | ≥ 2.0× (窗口残差) |
| WAN2 视频生成 | 纯 MLX | ≥ 1.3× |

### 10.3 集成测试

```
tests/
├── custom_kernels/
│   ├── test_mfa_bridge.py          # MFA 桥接层测试
│   ├── test_dispatch_policy.py     # 分派策略测试
│   ├── test_xfuser_attention.py    # xfuser 策略测试
│   └── test_sparse_attention.py    # 稀疏注意力测试
├── video/
│   ├── test_wan2_attention.py      # WAN2 注意力增强测试
│   └── test_ltx2_attention.py      # LTX-2 注意力增强测试
└── integration/
    └── test_mfa_xfuser_e2e.py      # 端到端测试
```

---

## 附录 A: 关键文件映射

| 源文件 (flash_attn/) | 目标文件 (fusion-mlx/) | 说明 |
|----------------------|----------------------|------|
| `mlx_mfa-2.61.0/mlx_mfa/attention.py` | `custom_kernels/mfa/attention.py` | 主注意力 API |
| `mlx_mfa-2.61.0/mlx_mfa/dispatch_policy.py` | `custom_kernels/mfa/dispatch_policy.py` | 分派策略 |
| `mlx_mfa-2.61.0/mlx_mfa/kv_cache.py` | `custom_kernels/mfa/kv_cache.py` | KV Cache |
| `mlx_mfa-2.61.0/mlx_mfa/turboquant.py` | `custom_kernels/mfa/turboquant.py` | TurboQuant |
| `mlx_mfa-2.61.0/mlx_mfa/masks.py` | `custom_kernels/mfa/masks.py` | 掩码工具 |
| `mlx_mfa-2.61.0/mlx_mfa/quantize.py` | `custom_kernels/mfa/quantize.py` | 量化 |
| `mlx_mfa-2.61.0/csrc/mfa_steel_fwd.cpp` | `custom_kernels/mfa/csrc/mfa_steel_fwd.cpp` | STEEL 前向 |
| `mlx_mfa-2.61.0/csrc/mfa_steel_bwd.cpp` | `custom_kernels/mfa/csrc/mfa_steel_bwd.cpp` | STEEL 反向 |
| `mlx_mfa-2.61.0/csrc/mfa_steel_fwd_v6_nax.cpp` | `custom_kernels/mfa/csrc/mfa_steel_fwd_v6_nax.cpp` | v6 NAX 前向 |
| `mlx_mfa-2.61.0/csrc/bindings.cpp` | `custom_kernels/mfa/csrc/bindings.cpp` | pybind11 绑定 |
| `xfuser-0.4.5/xfuser/core/fast_attention/attn_layer.py` | `custom_kernels/xfuser_attention.py` | xfuser 策略 |
| `xfuser-0.4.5/xfuser/core/fast_attention/utils.py` | `custom_kernels/xfuser_attention.py` | 校准逻辑 |
| `metal-flash-sdpa/csrc/mfa/ccv_nnc_mfa_attention.cpp` | — | 参考实现 (不直接移植) |
| `mps-flash-attention/mps_flash_attn/__init__.py` | `patches/torch_mps_bridge.py` | PyTorch MPS 兼容 |

## 附录 B: 快速启动命令

```bash
# 1. 安装 mlx_mfa 依赖 (短期方案)
pip install mlx-mfa>=2.61.0

# 2. 验证基本调用
python -c "
from fusion_mlx.custom_kernels.mfa_bridge import flash_attention
import mlx.core as mx
q = mx.random.uniform(shape=(1, 4, 1024, 128))
k = mx.random.uniform(shape=(1, 4, 1024, 128))
v = mx.random.uniform(shape=(1, 4, 1024, 128))
out = flash_attention(q, k, v, causal=True)
print('MFA available:', out is not None)
"

# 3. 编译 precompiled metallib (长期方案)
python -m fusion_mlx.custom_kernels.mfa.compile_metallib

# 4. 运行 xfuser 策略校准
python -c "
from fusion_mlx.custom_kernels.xfuser_attention import calibrate_attention_strategy
# 需要加载 diffusion pipeline
"

# 5. 运行完整测试
pytest tests/custom_kernels/
```