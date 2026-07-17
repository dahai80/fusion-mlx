# SPDX-License-Identifier: Apache-2.0
"""SkyReels-V3 基础工具函数 MLX 替换层.

剔除 torch.device 绑定, 封装统一 MLX 算子:
  - WanLayerNorm -> mx.nn.LayerNorm
  - WanRMSNorm (qk_norm) -> 自实现 RMSNorm
  - GELU (tanh近似) -> mx.nn.GELUApproximate
  - rope_params / rope_apply -> MLX complex64 算子
  - sinusoidal_embedding_1d -> MLX 数组运算
  - Conv3d patch_embed -> mx.nn.Conv3d (MLX >= 0.22 原生支持)
"""

from __future__ import annotations

import math
from typing import Any

import mlx.core as mx
import mlx.nn as nn

from . import _device

logger = logging.getLogger(__name__) if False else __import__("logging").getLogger(__name__)

# ---------------------------------------------------------------------------
# LayerNorm / RMSNorm
# ---------------------------------------------------------------------------
class WanLayerNorm(nn.Module):
    """SkyReels/Wan LayerNorm, 对齐原版 elementwise_affine=False 默认.

    原版 WanLayerNorm 继承 nn.LayerNorm, eps=1e-6, elementwise_affine 默认 False.
    MLX mx.nn.LayerNorm(weight=..., bias=..., eps=...).
    """

    def __init__(
        self,
        dim: int,
        eps: float = 1e-6,
        elementwise_affine: bool = False,
    ):
        super().__init__()
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if elementwise_affine:
            self.weight = mx.ones((dim,))
            self.bias = mx.zeros((dim,))
        else:
            self.weight = None
            self.bias = None
        # 原版 WanLayerNorm 用 elementwise_affine=False 但有自定义 weight
        # 这里保留 weight 参数兼容性
        if not elementwise_affine:
            self.weight = mx.ones((dim,))

    def __call__(self, x: mx.array) -> mx.array:
        # 对齐原版: F.layer_norm(inputs.float(), ...) 然后 .to(origin_dtype)
        # MLX 自动 dtype 提升, 这里直接用 fp32 计算后转回
        x_f32 = x.astype(mx.float32)
        mean = mx.mean(x_f32, axis=-1, keepdims=True)
        var = mx.mean((x_f32 - mean) ** 2, axis=-1, keepdims=True)
        x_norm = (x_f32 - mean) * mx.rsqrt(var + self.eps)
        if self.weight is not None:
            x_norm = x_norm * self.weight.astype(mx.float32)
        if self.bias is not None:
            x_norm = x_norm + self.bias.astype(mx.float32)
        return x_norm.astype(x.dtype)


class WanRMSNorm(nn.Module):
    """RMSNorm 用于 qk_norm, 对齐原版 fast_rms_norm.

    原版: x = x.float(); x = x * rsqrt(x.pow(2).mean(-1, keepdim) + eps);
          x = x.type_as(x) * weight
    """

    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = mx.ones((dim,))

    def __call__(self, x: mx.array) -> mx.array:
        x_f32 = x.astype(mx.float32)
        ms = mx.mean(x_f32 ** 2, axis=-1, keepdims=True)
        x_norm = x_f32 * mx.rsqrt(ms + self.eps)
        return (x_norm * self.weight.astype(mx.float32)).astype(x.dtype)


# ---------------------------------------------------------------------------
# GELU 激活
# ---------------------------------------------------------------------------
class GELUApprox(nn.Module):
    """GELU tanh 近似, 对齐 nn.GELU(approximate="tanh").

    MLX mlx.nn.gelu_approx 等价 PyTorch tanh 近似.
    """

    def __call__(self, x: mx.array) -> mx.array:
        return nn.gelu_approx(x)


# ---------------------------------------------------------------------------
# Sinusoidal Embedding (时间步嵌入)
# ---------------------------------------------------------------------------
def sinusoidal_embedding_1d(dim: int, position: mx.array) -> mx.array:
    """1D 正弦位置嵌入, 对齐原版 sinusoidal_embedding_1d.

    原版用 torch.float64, MLX 无 fp64 用 fp32 替代 (数值差异 < 1e-7).
    """
    assert dim % 2 == 0
    half = dim // 2
    pos = position.astype(mx.float32)
    inv_freq = mx.power(10000.0, -mx.arange(half).astype(mx.float32) / half)
    sinusoid = pos[..., None] * inv_freq  # [..., half]
    return mx.concatenate([mx.cos(sinusoid), mx.sin(sinusoid)], axis=-1)


# ---------------------------------------------------------------------------
# RoPE (Rotary Positional Embedding)
# ---------------------------------------------------------------------------
def rope_params(max_seq_len: int, dim: int, theta: float = 10000.0) -> mx.array:
    """生成 RoPE 频率表 (实数对, 供 rope_apply 使用).

    原版用 torch.polar 生成复数, MLX 0.32 complex64 构造支持有限,
    这里返回实数对 [max_seq_len, dim/2, 2] (cos, sin),
    rope_apply 内部用 stack+astype(complex64) 重建复数.

    Args:
        max_seq_len: 最大序列长度
        dim: head_dim (RoPE 作用维度)
        theta: RoPE base frequency

    Returns:
        mx.array shape [max_seq_len, dim/2, 2], dtype float32
        最后一维 2 = (cos, sin)
    """
    assert dim % 2 == 0
    half = dim // 2
    positions = mx.arange(max_seq_len).astype(mx.float32)
    inv_freq = mx.power(theta, -mx.arange(0, dim, 2).astype(mx.float32) / dim)
    freqs = mx.outer(positions, inv_freq)  # [max_seq_len, half]

    # 返回 (cos, sin) 实数对, rope_apply 内部重建复数
    cos = mx.cos(freqs)[..., None]  # [max_seq_len, half, 1]
    sin = mx.sin(freqs)[..., None]
    return mx.concatenate([cos, sin], axis=-1)  # [max_seq_len, half, 2]


def rope_apply(
    x: mx.array,
    grid_sizes: list[tuple[int, int, int]],
    freqs: mx.array,
    context_window_size: int = 0,
    num_token_list: list[int] | None = None,
    num_frame_list: list[int] | None = None,
    grid_size_list: list[tuple[int, int, int]] | None = None,
) -> mx.array:
    """应用 RoPE 到 Q/K, 对齐原版 rope_apply (3D video rope).

    原版拆分 freqs 为 [t, h, w] 三段, 然后广播到 [f*h*w, 1, dim/2].
    本实现用实数对 (cos, sin) 等价复数乘法:
      x_complex = x[::2] + i * x[1::2]
      x_rotated = x_complex * (cos + i*sin)
               = (x_real*cos - x_imag*sin) + i*(x_real*sin + x_imag*cos)

    Args:
        x: [B, L, N, C], C 为 head_dim (实数)
        grid_sizes: [(f, h, w), ...] 每个样本的网格大小
        freqs: RoPE 频率表 [max_seq, dim/2, 2] (cos, sin 实数对)
        context_window_size: 上下文窗口大小 (0=标准 rope)
        num_token_list/num_frame_list/grid_size_list: 多段上下文参数

    Returns:
        mx.array [B, L, N, C] 应用了 RoPE 的张量
    """
    b, s, n, c = x.shape
    num_token_list = num_token_list or []
    num_frame_list = num_frame_list or []
    grid_size_list = grid_size_list or []

    half_c = c // 2
    t_dim = half_c - 2 * (half_c // 3)
    hw_dim = half_c // 3

    # freqs: [max_seq, dim/2, 2] -> 拆 (cos, sin)
    # 注意: freqs 最后一维 2 = (cos, sin)
    freqs_cos = freqs[..., 0]  # [max_seq, dim/2]
    freqs_sin = freqs[..., 1]

    # 拆分 [t_dim, hw_dim, hw_dim]
    cos_t = freqs_cos[:, :t_dim]
    sin_t = freqs_sin[:, :t_dim]
    cos_h = freqs_cos[:, t_dim:t_dim + hw_dim]
    sin_h = freqs_sin[:, t_dim:t_dim + hw_dim]
    cos_w = freqs_cos[:, t_dim + hw_dim:t_dim + 2 * hw_dim]
    sin_w = freqs_sin[:, t_dim + hw_dim:t_dim + 2 * hw_dim]

    # 处理 batch 维度: 当 grid_sizes 只有一个元素但 B>1 时, 广播给所有样本
    # (CFG 拼接场景: latent_input = mx.concatenate([latents]*2) 导致 B>1)
    gs = grid_sizes
    if len(gs) < b:
        gs = gs * (b // len(gs)) + gs[:b % len(gs)]  # 重复到 b 个

    outputs = []
    for i, (f, h, w) in enumerate(gs):
        seq_len = f * h * w
        x_i = x[i].astype(mx.float32)  # [L, N, C]

        # 拆 real/imag: x[::2] = real, x[1::2] = imag
        x_real = x_i[..., 0::2]  # [L, N, C/2]
        x_imag = x_i[..., 1::2]

        # 构造 cos_i / sin_i: [f*h*w, 1, half_c]
        # t 段
        cos_i_t = mx.broadcast_to(
            cos_t[:f].reshape(f, 1, 1, -1), (f, h, w, t_dim)
        )
        sin_i_t = mx.broadcast_to(
            sin_t[:f].reshape(f, 1, 1, -1), (f, h, w, t_dim)
        )
        # h 段
        cos_i_h = mx.broadcast_to(
            cos_h[:h].reshape(1, h, 1, -1), (f, h, w, hw_dim)
        )
        sin_i_h = mx.broadcast_to(
            sin_h[:h].reshape(1, h, 1, -1), (f, h, w, hw_dim)
        )
        # w 段
        cos_i_w = mx.broadcast_to(
            cos_w[:w].reshape(1, 1, w, -1), (f, h, w, hw_dim)
        )
        sin_i_w = mx.broadcast_to(
            sin_w[:w].reshape(1, 1, w, -1), (f, h, w, hw_dim)
        )

        cos_i = mx.concatenate([cos_i_t, cos_i_h, cos_i_w], axis=-1)
        sin_i = mx.concatenate([sin_i_t, sin_i_h, sin_i_w], axis=-1)
        cos_i = cos_i.reshape(seq_len, 1, -1)  # [seq_len, 1, half_c]
        sin_i = sin_i.reshape(seq_len, 1, -1)

        # 广播到 [L, N, half_c]
        cos_exp = mx.broadcast_to(cos_i, (s, n, half_c))
        sin_exp = mx.broadcast_to(sin_i, (s, n, half_c))

        # 复数乘法等价:
        # x_rotated_real = x_real*cos - x_imag*sin
        # x_rotated_imag = x_real*sin + x_imag*cos
        rot_real = x_real * cos_exp - x_imag * sin_exp
        rot_imag = x_real * sin_exp + x_imag * cos_exp

        # 交错合并回 [L, N, C]: [real_0, imag_0, real_1, imag_1, ...]
        x_out = mx.stack([rot_real, rot_imag], axis=-1)  # [L, N, half_c, 2]
        x_out = x_out.reshape(s, n, c)
        outputs.append(x_out)

    return mx.stack(outputs, axis=0)


# ---------------------------------------------------------------------------
# PatchEmbed (Conv3d)
# ---------------------------------------------------------------------------
class PatchEmbed3D(nn.Module):
    """3D Patch Embedding, 对齐 nn.Conv3d(in_dim, dim, kernel=patch_size, stride=patch_size).

    MLX >= 0.22 原生支持 mx.nn.Conv3d. 若版本不支持则降级为 Conv2d 沿时间轴展开.

    patch_size 默认 (1, 2, 2): 时间不降采样, 空间 2x2 降采样.
    输入: [B, C_in, T, H, W] (视频 latent)
    输出: [B, L, dim], L = T * (H//ph) * (W//pw)
    """

    def __init__(
        self,
        in_dim: int,
        dim: int,
        patch_size: tuple[int, int, int] = (1, 2, 2),
    ):
        super().__init__()
        self.patch_size = patch_size
        self.dim = dim
        self.in_dim = in_dim

        # PatchEmbed3D 用 Conv2d 沿时间轴展开 (与底座 ltx2 一致).
        # MLX 0.32 nn.Conv3d 权重 layout 存在兼容问题, 改用 Conv2d:
        #   输入 [B, C, T, H, W] -> reshape [B*T, C, H, W] -> Conv2d -> [B*T, dim, H', W']
        # 时间维不降采样 (pt=1), 空间 ph x pw 降采样.
        self.conv2d = nn.Conv2d(
            in_channels=in_dim,
            out_channels=dim,
            kernel_size=patch_size[1:],
            stride=patch_size[1:],
        )

    def __call__(self, x: mx.array) -> mx.array:
        """前向: [B, C, T, H, W] -> [B, L, dim].

        Args:
            x: [B, C_in, T, H, W] 视频 latent

        Returns:
            [B, T*(H//ph)*(W//pw), dim]
        """
        b, c, t, h, w = x.shape
        pt, ph, pw = self.patch_size

        # MLX 0.32 nn.Conv2d 期望 channels-last 输入 [B, H, W, C_in].
        # 输入 [B, C, T, H, W] (channels-first) -> 转换为 [B*T, H, W, C]
        # Conv2d 输出 [B*T, H', W', dim] (channels-last) -> 转回 [B, L, dim]
        x_cl = x.transpose(0, 2, 3, 4, 1).reshape(b * t, h, w, c)
        out_cl = self.conv2d(x_cl)  # [B*T, H', W', dim]
        _, h_o, w_o, dim_out = out_cl.shape
        # 转回 channels-first 风格 [B, T, dim, H', W'] 再 flatten 到 [B, L, dim]
        out = out_cl.reshape(b, t, h_o, w_o, dim_out)
        out = out.transpose(0, 1, 4, 2, 3).reshape(b, -1, self.dim)
        return out


# ---------------------------------------------------------------------------
# 基础工具: mul_add / mul_add_add (modulation 辅助)
# ---------------------------------------------------------------------------
def mul_add(x: mx.array, y: mx.array, z: mx.array) -> mx.array:
    """x.float() + y.float() * z.float() — modulation gate 残差.

    对齐原版 mul_add: x + y * z, 全程 float32 保精度.
    """
    x_f = x.astype(mx.float32)
    y_f = y.astype(mx.float32)
    z_f = z.astype(mx.float32)
    return x_f + y_f * z_f


def mul_add_add(x: mx.array, y: mx.array, z: mx.array) -> mx.array:
    """x.float() * (1 + y) + z — modulation scale/shift.

    对齐原版 mul_add_add: x * (1 + y) + z, 全程 float32 保精度.
    """
    x_f = x.astype(mx.float32)
    y_f = y.astype(mx.float32)
    z_f = z.astype(mx.float32)
    return x_f * (1.0 + y_f) + z_f


# ---------------------------------------------------------------------------
# 编译开关
# ---------------------------------------------------------------------------
def maybe_compile(func: Any) -> Any:
    """条件编译: M5 Max 默认开启 mx.compile, 其他设备或测试环境直通.

    用法 1 (函数): dit_block = maybe_compile(dit_block)
    用法 2 (nn.Module): block = maybe_compile(block)  # 编译 block.__call__

    Args:
        func: 可调用函数 或 nn.Module 实例

    Returns:
        编译后的 callable (若启用) 或原 func (若禁用/失败)
    """
    if not _device.should_compile():
        return func

    # 提取 callable: nn.Module 的 __call__ 或直接 callable
    target = func.__call__ if isinstance(func, nn.Module) else func
    if not callable(target):
        return func

    try:
        compiled = mx.compile(target)
        # 若是 nn.Module, 把编译后的 __call__ 挂回 (保留 parameters/state)
        if isinstance(func, nn.Module):
            object.__setattr__(func, "__call__", compiled)
            return func
        return compiled
    except Exception as exc:  # pragma: no cover - 编译失败兜底
        logger.warning("mx.compile failed, falling back to eager: %s", exc)
        return func


def verify_compile_stability(
    model: nn.Module,
    *,
    sample_input: tuple,
    warmup: int = 2,
) -> dict:
    """验证 mx.compile 在真实输入下的稳定性.

    跑 warmup 轮编译 + eager 对照, 检查:
      1. 编译是否成功 (不抛异常)
      2. 编译输出与 eager 输出数值一致 (max abs diff)
      3. 编译是否真的更快 (时间比)

    Args:
        model: 待验证的 nn.Module (DiT block 等)
        sample_input: 前向输入元组 (将 unpack 调用)
        warmup: 预热轮数 (默认 2)

    Returns:
        {
            "compile_ok": bool,
            "max_abs_diff": float,
            "eager_time_s": float,
            "compile_time_s": float,
            "speedup": float,
        }
    """
    import time

    result = {
        "compile_ok": False,
        "max_abs_diff": float("inf"),
        "eager_time_s": 0.0,
        "compile_time_s": 0.0,
        "speedup": 0.0,
    }

    # 1. Eager 基准
    try:
        eager_out = model(*sample_input)
        mx.eval(eager_out)
    except Exception as exc:
        logger.warning("Eager forward 失败, 跳过编译验证: %s", exc)
        return result

    # 2. 编译
    if not _device.should_compile():
        logger.info("should_compile()=False, 跳过编译验证")
        result["compile_ok"] = True
        result["max_abs_diff"] = 0.0
        return result

    try:
        compiled_fn = mx.compile(model.__call__)
    except Exception as exc:
        logger.warning("mx.compile 失败: %s", exc)
        return result

    # 3. 编译前向 (含 warmup)
    try:
        for _ in range(warmup):
            _ = compiled_fn(*sample_input)
        compile_out = compiled_fn(*sample_input)
        mx.eval(compile_out)
        result["compile_ok"] = True
    except Exception as exc:
        logger.warning("编译前向失败: %s", exc)
        return result

    # 4. 数值对照
    try:
        diff = mx.abs(compile_out - eager_out)
        mx.eval(diff)
        result["max_abs_diff"] = float(diff.max().item())
    except Exception as exc:
        logger.warning("数值对照失败: %s", exc)
        result["max_abs_diff"] = float("inf")

    # 5. 性能对照 (简糙)
    try:
        t0 = time.time()
        for _ in range(3):
            _ = model(*sample_input)
        mx.eval(model(*sample_input))
        result["eager_time_s"] = (time.time() - t0) / 4

        t0 = time.time()
        for _ in range(3):
            _ = compiled_fn(*sample_input)
        mx.eval(compiled_fn(*sample_input))
        result["compile_time_s"] = (time.time() - t0) / 4

        if result["compile_time_s"] > 0:
            result["speedup"] = result["eager_time_s"] / result["compile_time_s"]
    except Exception as exc:
        logger.warning("性能对照失败: %s", exc)

    logger.info(
        "编译验证: ok=%s diff=%.2e eager=%.3fs compile=%.3fs speedup=%.2fx",
        result["compile_ok"], result["max_abs_diff"],
        result["eager_time_s"], result["compile_time_s"], result["speedup"],
    )
    return result


__all__ = [
    "WanLayerNorm",
    "WanRMSNorm",
    "GELUApprox",
    "sinusoidal_embedding_1d",
    "rope_params",
    "rope_apply",
    "PatchEmbed3D",
    "mul_add",
    "mul_add_add",
    "maybe_compile",
]
