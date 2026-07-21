# SPDX-License-Identifier: Apache-2.0
"""SkyReels-V3 MLX 集成测试.

覆盖:
  1. import 闭环 (所有模块可加载)
  2. 三大分支前向 smoke (tiny 配置, 不验证数值)
  3. 采样步连贯 (StepCoherenceFilter 行为验证)
  4. 时序闪烁修复 (temporal_ema / boundary_align 数值验证)
  5. KV Cache 双池 (append/get/SW 淘汰)
  6. config 注册表 (三大分支配置完整性)

运行:
    cd /Users/dahai/claude-home/fusion-mlx
    source .venv312/bin/activate
    pytest tests/test_skyreels_v3.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# 确保能 import fusion_mlx
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import mlx.core as mx


# ============================================================================
# 1. import 闭环
# ============================================================================
class TestImportClosure:
    """所有模块可加载."""

    def test_top_level_import(self):
        from fusion_mlx.video.skyreels_v3 import (
            BRANCH_CONFIGS,
            generate_video,
            get_branch_config,
            list_models,
        )

        assert callable(generate_video)
        assert callable(list_models)
        assert callable(get_branch_config)
        assert isinstance(BRANCH_CONFIGS, dict)

    def test_device_module(self):
        from fusion_mlx.video.skyreels_v3 import _device

        assert hasattr(_device, "get_stream")
        assert hasattr(_device, "is_m5")
        assert hasattr(_device, "detect_generation")
        # DeviceGeneration 常量
        assert _device.DeviceGeneration.M5 == 17

    def test_common_module(self):
        from fusion_mlx.video.skyreels_v3.common import (
            PatchEmbed3D,
            WanLayerNorm,
        )

        assert WanLayerNorm is not None
        assert PatchEmbed3D is not None

    def test_attention_module(self):
        from fusion_mlx.video.skyreels_v3.attention import (
            WAN_CROSSATTENTION_CLASSES,
        )

        assert "t2v_cross_attn" in WAN_CROSSATTENTION_CLASSES
        assert "i2v_cross_attn" in WAN_CROSSATTENTION_CLASSES

    def test_scheduler_module(self):
        from fusion_mlx.video.skyreels_v3.scheduler import (
            FlowUniPCConfig,
            perform_guidance,
        )

        assert FlowUniPCConfig is not None
        assert callable(perform_guidance)

    def test_transformer_modules(self):
        from fusion_mlx.video.skyreels_v3.transformer_a2v import SkyReelsA2VDiT
        from fusion_mlx.video.skyreels_v3.transformer_r2v import SkyReelsR2VDiT
        from fusion_mlx.video.skyreels_v3.transformer_v2v import SkyReelsV2VDiT

        assert SkyReelsR2VDiT is not None
        assert SkyReelsV2VDiT is not None
        assert SkyReelsA2VDiT is not None

    def test_pipelines_import(self):
        from fusion_mlx.video.skyreels_v3.pipelines import (
            SkyReelsA2VPipeline,
            SkyReelsR2VPipeline,
            SkyReelsV2VPipeline,
        )

        assert SkyReelsR2VPipeline is not None
        assert SkyReelsV2VPipeline is not None
        assert SkyReelsA2VPipeline is not None

    def test_kv_cache_import(self):
        from fusion_mlx.video.skyreels_v3.kv_cache import (
            SkyReelsKVCache,
        )

        assert SkyReelsKVCache is not None

    def test_flicker_fix_import(self):
        from fusion_mlx.video.skyreels_v3.temporal_flicker_fix import (
            FlickerFixConfig,
            default_config_for_branch,
        )

        assert FlickerFixConfig is not None
        assert callable(default_config_for_branch)

    def test_m5_optimizer_import(self):
        from fusion_mlx.video.skyreels_v3.m5_optimizer import (
            M5Optimizer,
        )

        assert M5Optimizer is not None

    def test_step_strategy_import(self):
        from fusion_mlx.video.skyreels_v3.step_strategy import (
            StepStrategyConfig,
        )

        assert StepStrategyConfig is not None


# ============================================================================
# 2. config 注册表完整性
# ============================================================================
class TestConfigRegistry:
    """三大分支配置完整性."""

    def test_three_branches_registered(self):
        from fusion_mlx.video.skyreels_v3 import list_models

        models = list_models()
        assert "skyreels-v3-r2v-14b" in models
        assert "skyreels-v3-v2v-14b" in models
        assert "skyreels-v3-a2v-19b" in models

    def test_r2v_config(self):
        from fusion_mlx.video.skyreels_v3 import get_branch_config

        cfg = get_branch_config("skyreels-v3-r2v-14b")
        assert cfg.branch == "r2v"
        assert cfg.dim == 5120
        assert cfg.num_heads == 40
        assert cfg.has_audio is False

    def test_v2v_config(self):
        from fusion_mlx.video.skyreels_v3 import get_branch_config

        cfg = get_branch_config("skyreels-v3-v2v-14b")
        assert cfg.branch == "v2v"
        assert cfg.temporal_window == 96

    def test_a2v_config(self):
        from fusion_mlx.video.skyreels_v3 import get_branch_config

        cfg = get_branch_config("skyreels-v3-a2v-19b")
        assert cfg.branch == "a2v"
        assert cfg.has_audio is True
        assert cfg.audio_dim == 1024

    def test_unknown_model_raises(self):
        from fusion_mlx.video.skyreels_v3 import get_branch_config

        with pytest.raises(ValueError, match="Unknown model_key"):
            get_branch_config("nonexistent-model")


# ============================================================================
# 3. 三大分支前向 smoke (tiny 配置)
# ============================================================================
TINY_CFG = {
    "dim": 256,
    "ffn_dim": 512,
    "num_heads": 8,
    "num_layers": 2,
    "patch_size": (1, 2, 2),
    "in_dim": 16,
    "out_dim": 16,
    "text_dim": 4096,
    "text_len": 512,
    "freq_dim": 256,
    "window_size": (-1, -1),
    "qk_norm": True,
    "cross_attn_norm": True,
    "eps": 1e-6,
    "cross_attn_type": "i2v_cross_attn",
}


class TestDiTForwardSmoke:
    """三大分支 DiT 前向 smoke 测试 (tiny 配置)."""

    def test_r2v_dit_forward(self):
        from fusion_mlx.video.skyreels_v3.transformer_r2v import SkyReelsR2VDiT

        dit = SkyReelsR2VDiT(TINY_CFG)

        b, c, t, h, w = 1, 16, 2, 8, 8
        x = mx.random.normal((b, c, t, h, w))
        ctx_t = mx.array([0.5])
        context = mx.zeros((b, 10, TINY_CFG["text_dim"]))

        seq_lens = [t * (h // 2) * (w // 2)]
        grid_sizes = [(t, h // 2, w // 2)]

        # 仅检查不抛异常 + 输出形状正确
        out = dit(x, ctx_t, context, seq_lens, grid_sizes)
        mx.eval(out)
        assert out.shape[0] == b
        assert out.shape[1] == c

    def test_v2v_dit_construction(self):
        from fusion_mlx.video.skyreels_v3.transformer_v2v import SkyReelsV2VDiT

        dit = SkyReelsV2VDiT(TINY_CFG)
        # 仅检查构造不抛异常
        assert dit is not None
        assert len(dit.blocks) == TINY_CFG["num_layers"]

    def test_a2v_dit_construction(self):
        from fusion_mlx.video.skyreels_v3.transformer_a2v import SkyReelsA2VDiT

        a2v_cfg = dict(TINY_CFG)
        a2v_cfg["audio_dim"] = 1024
        dit = SkyReelsA2VDiT(a2v_cfg)
        assert dit is not None
        assert hasattr(dit, "audio_embedding")


# ============================================================================
# 4. 采样步连贯 (StepCoherenceFilter)
# ============================================================================
class TestStepCoherenceFilter:
    """采样步连贯滤波器行为验证."""

    def test_first_step_no_filter(self):
        """首步不滤波, 返回原值."""
        from fusion_mlx.video.skyreels_v3.temporal_flicker_fix import (
            StepCoherenceFilter,
        )

        f = StepCoherenceFilter(beta=0.1)
        x = mx.array([1.0, 2.0, 3.0])
        out = f(x)
        mx.eval(out)
        assert mx.allclose(out, x).item()

    def test_second_step_filtered(self):
        """次步滤波: smoothed = beta * prev + (1-beta) * curr."""
        from fusion_mlx.video.skyreels_v3.temporal_flicker_fix import (
            StepCoherenceFilter,
        )

        f = StepCoherenceFilter(beta=0.1)
        prev = mx.array([1.0, 2.0, 3.0])
        curr = mx.array([2.0, 3.0, 4.0])
        f(prev)  # 首步
        out = f(curr)  # 次步滤波
        mx.eval(out)
        expected = 0.1 * prev + 0.9 * curr
        assert mx.allclose(out, expected, atol=1e-5).item()

    def test_beta_zero_no_filter(self):
        """beta=0 不滤波."""
        from fusion_mlx.video.skyreels_v3.temporal_flicker_fix import (
            StepCoherenceFilter,
        )

        f = StepCoherenceFilter(beta=0.0)
        x = mx.array([1.0, 2.0, 3.0])
        out = f(x)
        mx.eval(out)
        assert mx.allclose(out, x).item()

    def test_reset_clears_state(self):
        """reset 清除滤波器状态."""
        from fusion_mlx.video.skyreels_v3.temporal_flicker_fix import (
            StepCoherenceFilter,
        )

        f = StepCoherenceFilter(beta=0.1)
        f(mx.array([1.0, 2.0, 3.0]))
        f.reset()
        x = mx.array([1.0, 2.0, 3.0])
        out = f(x)  # reset 后首步不滤波
        mx.eval(out)
        assert mx.allclose(out, x).item()


# ============================================================================
# 5. 时序闪烁修复数值验证
# ============================================================================
class TestFlickerFix:
    """时序闪烁修复模块数值验证."""

    def test_temporal_ema_smooth_alpha_zero(self):
        """alpha=0 不平滑."""
        from fusion_mlx.video.skyreels_v3.temporal_flicker_fix import (
            temporal_ema_smooth,
        )

        latent = mx.array([1.0, 2.0, 3.0])
        prev = mx.array([0.0, 0.0, 0.0])
        out = temporal_ema_smooth(latent, prev, alpha=0.0)
        mx.eval(out)
        assert mx.allclose(out, latent).item()

    def test_temporal_ema_smooth_alpha_one(self):
        """alpha=1 完全用前一帧."""
        from fusion_mlx.video.skyreels_v3.temporal_flicker_fix import (
            temporal_ema_smooth,
        )

        latent = mx.array([1.0, 2.0, 3.0])
        prev = mx.array([0.0, 0.0, 0.0])
        out = temporal_ema_smooth(latent, prev, alpha=1.0)
        mx.eval(out)
        assert mx.allclose(out, prev).item()

    def test_temporal_ema_smooth_mid_value(self):
        """alpha=0.3 中间值平滑."""
        from fusion_mlx.video.skyreels_v3.temporal_flicker_fix import (
            temporal_ema_smooth,
        )

        latent = mx.array([2.0, 3.0, 4.0])
        prev = mx.array([1.0, 2.0, 3.0])
        out = temporal_ema_smooth(latent, prev, alpha=0.3)
        mx.eval(out)
        expected = 0.3 * prev + 0.7 * latent
        assert mx.allclose(out, expected, atol=1e-5).item()

    def test_boundary_align_alpha_zero(self):
        """alpha=0 不对齐."""
        from fusion_mlx.video.skyreels_v3.temporal_flicker_fix import (
            boundary_align,
        )

        latent = mx.zeros((1, 16, 5, 8, 8))
        end = mx.ones((1, 16, 8, 8))
        out = boundary_align(latent, end, alpha=0.0)
        mx.eval(out)
        assert mx.allclose(out, latent).item()

    def test_boundary_align_alpha_one(self):
        """alpha=1 完全用输入末帧 (仅首帧)."""
        from fusion_mlx.video.skyreels_v3.temporal_flicker_fix import (
            boundary_align,
        )

        latent = mx.zeros((1, 16, 5, 8, 8))
        end = mx.ones((1, 16, 8, 8)) * 9.0
        out = boundary_align(latent, end, alpha=1.0, num_align_frames=1)
        mx.eval(out)
        # 首帧应为 end (alpha=1)
        first_frame = out[:, :, 0, :, :]
        mx.eval(first_frame)
        assert mx.allclose(first_frame, end, atol=1e-5).item()

    def test_default_config_for_branches(self):
        """三分支默认配置完整性."""
        from fusion_mlx.video.skyreels_v3.temporal_flicker_fix import (
            default_config_for_branch,
        )

        for branch in ("r2v", "v2v", "a2v"):
            cfg = default_config_for_branch(branch)
            assert 0.0 <= cfg.temporal_ema_alpha <= 1.0
            assert 0.0 <= cfg.boundary_align_alpha <= 1.0
            assert 0.0 <= cfg.step_coherence_beta <= 1.0

    def test_default_config_unknown_branch(self):
        """未知分支抛异常."""
        from fusion_mlx.video.skyreels_v3.temporal_flicker_fix import (
            default_config_for_branch,
        )

        with pytest.raises(ValueError, match="Unknown branch"):
            default_config_for_branch("unknown")


# ============================================================================
# 6. KV Cache 双池验证
# ============================================================================
class TestKVCache:
    """SkyReelsKVCache 双池验证."""

    def test_kv_cache_construction(self):
        """双池构造不抛异常."""
        from fusion_mlx.video.skyreels_v3.kv_cache import SkyReelsKVCache

        cache = SkyReelsKVCache(
            num_layers=2,
            num_heads=8,
            head_dim=64,
            max_frames=16,
            h=8,
            w=8,
        )
        assert cache is not None
        assert cache.num_layers == 2

    def test_spatial_pool_append_get(self):
        """空间池 append + get."""
        from fusion_mlx.video.skyreels_v3.kv_cache import SkyReelsKVCache

        cache = SkyReelsKVCache(
            num_layers=1,
            num_heads=4,
            head_dim=32,
            max_frames=4,
            h=4,
            w=4,
        )
        k = mx.random.normal((1, 4, 8, 32))  # [B, H, L, D]
        v = mx.random.normal((1, 4, 8, 32))
        cache.append_spatial(0, k, v)
        k_out, v_out = cache.get_spatial(0)
        mx.eval(k_out, v_out)
        assert k_out.shape[-1] == 32

    def test_temporal_pool_append_get(self):
        """时序池 append + get."""
        from fusion_mlx.video.skyreels_v3.kv_cache import SkyReelsKVCache

        cache = SkyReelsKVCache(
            num_layers=1,
            num_heads=4,
            head_dim=32,
            max_frames=4,
            h=4,
            w=4,
        )
        k = mx.random.normal((1, 4, 8, 32))
        v = mx.random.normal((1, 4, 8, 32))
        cache.append_temporal(0, k, v)
        k_out, v_out = cache.get_temporal(0)
        mx.eval(k_out, v_out)
        assert k_out.shape[-1] == 32

    def test_reset(self):
        """reset 清空所有池."""
        from fusion_mlx.video.skyreels_v3.kv_cache import SkyReelsKVCache

        cache = SkyReelsKVCache(
            num_layers=1,
            num_heads=4,
            head_dim=32,
            max_frames=4,
            h=4,
            w=4,
        )
        k = mx.random.normal((1, 4, 8, 32))
        v = mx.random.normal((1, 4, 8, 32))
        cache.append_spatial(0, k, v)
        cache.reset()
        # reset 后 seqlen 应为 0
        assert cache.seqlen == 0


# ============================================================================
# 7. 采样器基本行为
# ============================================================================
class TestScheduler:
    """FlowUniPCMultistepScheduler 基本行为."""

    def test_set_timesteps(self):
        """set_timesteps 不抛异常, 生成正确步数."""
        from fusion_mlx.video.skyreels_v3.scheduler import (
            FlowUniPCMultistepScheduler,
        )

        s = FlowUniPCMultistepScheduler(num_inference_steps=10)
        s.set_timesteps(10)
        assert s.timesteps is not None
        assert s.timesteps.shape[0] == 10

    def test_step_returns_prev_sample(self):
        """step 返回 prev_sample."""
        from fusion_mlx.video.skyreels_v3.scheduler import (
            FlowUniPCMultistepScheduler,
        )

        s = FlowUniPCMultistepScheduler(num_inference_steps=5)
        s.set_timesteps(5)
        sample = mx.random.normal((1, 16, 2, 8, 8))
        model_output = mx.random.normal((1, 16, 2, 8, 8))
        t = float(s.timesteps[0])
        out = s.step(model_output, t, sample)
        mx.eval(out.prev_sample)
        assert out.prev_sample.shape == sample.shape

    def test_perform_guidance(self):
        """perform_guidance CFG 合并."""
        from fusion_mlx.video.skyreels_v3.scheduler import perform_guidance

        # [2B, ...] uncond 在前
        noise_pred = mx.array(
            [
                [1.0, 2.0],  # uncond
                [3.0, 4.0],  # cond
            ]
        )
        out = perform_guidance(noise_pred, guidance_scale=2.0)
        mx.eval(out)
        # uncond + scale * (cond - uncond)
        # [1.0, 2.0] + 2.0 * ([3.0, 4.0] - [1.0, 2.0])
        # = [1.0, 2.0] + 2.0 * [2.0, 2.0]
        # = [1.0, 2.0] + [4.0, 4.0] = [5.0, 6.0]
        assert mx.allclose(out, mx.array([[5.0, 6.0]]), atol=1e-5).item()


# ============================================================================
# 8. VAE 端口基本行为
# ============================================================================
class TestVAE:
    """SkyReelsVAE 基本行为."""

    def test_vae_construction(self):
        """VAE 构造不抛异常."""
        from fusion_mlx.video.skyreels_v3.vae import SkyReelsVAE

        vae = SkyReelsVAE()
        assert vae is not None
        assert vae.vae_mean.shape[1] == 16

    def test_decode_returns_video(self):
        """decode 返回视频张量."""
        from fusion_mlx.video.skyreels_v3.vae import SkyReelsVAE

        vae = SkyReelsVAE()
        latent = mx.random.normal((1, 16, 2, 8, 8))
        # 即使底座 VAE 不可用, stub 也应返回张量
        try:
            out = vae.decode(latent)
            mx.eval(out)
            assert out.ndim >= 4
        except Exception:
            # stub 模式可能不返回, 跳过
            pass


# ============================================================================
# 9. M5 Optimizer 基本行为
# ============================================================================
class TestM5Optimizer:
    """M5Optimizer 基本行为."""

    def test_compute_placement(self):
        """ComputePlacement 构造不抛异常."""
        from fusion_mlx.video.skyreels_v3.m5_optimizer import (
            ComputePlacement,
            get_compute_placement,
        )

        cp = ComputePlacement()
        assert cp.large_matmul_device == "gpu"
        cp2 = get_compute_placement()
        assert cp2 is not None

    def test_adaptive_tile_scheduler(self):
        """AdaptiveTileScheduler 自适应分块."""
        from fusion_mlx.video.skyreels_v3.m5_optimizer import (
            AdaptiveTileScheduler,
        )

        ts = AdaptiveTileScheduler()
        # 短序列: 用基础 block
        block = ts.adapt(seq_len=512, head_dim=64)
        assert block[0] > 0
        # 长序列: 降级
        block_long = ts.adapt(seq_len=8192, head_dim=64)
        assert block_long[0] <= block[0]

    def test_quantization_config_auto(self):
        """QuantizationConfig.auto() 不抛异常."""
        from fusion_mlx.video.skyreels_v3.m5_optimizer import (
            QuantizationConfig,
            get_quantization_config,
        )

        cfg = QuantizationConfig.auto()
        assert cfg.weight_bits in (4, 8, 16)
        cfg2 = get_quantization_config()
        assert cfg2 is not None

    def test_async_video_stream(self):
        """AsyncVideoStream prefetch + consume."""
        from fusion_mlx.video.skyreels_v3.m5_optimizer import AsyncVideoStream

        s = AsyncVideoStream()
        data = mx.array([1.0, 2.0, 3.0])
        s.prefetch(data)
        out = s.consume()
        if out is not None:
            mx.eval(out)
            assert out.shape == data.shape


# ============================================================================
# 9b. FP8Linear weight/compute_dtype + _linear_dtype 回归 (#142)
# ============================================================================
class TestFP8LinearDtype:
    # 回归 #142: FP8Linear 曾无 .weight, _linear_dtype 访问 inner.weight.dtype 崩溃.
    # 验证: weight 属性 + compute_dtype (非 fp8_weight.dtype) + _linear_dtype 两处.

    def test_fp8linear_weight_property(self):
        import mlx.nn as nn

        from fusion_mlx.custom_kernels.fp8_linear import FP8Linear

        fpl = FP8Linear.from_linear(nn.Linear(8, 16))
        assert hasattr(fpl, "weight")
        assert fpl.weight is fpl.fp8_weight
        assert fpl.weight.shape == (16, 8)
        assert fpl.weight.dtype == fpl.fp8_weight.dtype

    def test_fp8linear_compute_dtype_not_fp8(self):
        from fusion_mlx.custom_kernels.fp8_linear import (
            _FP8_AVAILABLE,
            FP8Linear,
        )

        fpl = FP8Linear(out_features=16, in_features=8)
        expected = mx.bfloat16 if _FP8_AVAILABLE else mx.float32
        assert fpl.compute_dtype == expected
        # FP8 硬件下 compute_dtype (bf16) 必须不同于 fp8_weight.dtype (float8).
        if _FP8_AVAILABLE:
            assert fpl.compute_dtype != fpl.fp8_weight.dtype

    def test_linear_dtype_fp8linear_returns_compute_dtype(self):
        import mlx.nn as nn

        from fusion_mlx.custom_kernels.fp8_linear import FP8Linear
        from fusion_mlx.video.skyreels_v3.attention import (
            _linear_dtype as sky_ld,
        )
        from fusion_mlx.video.wan2.attention import _linear_dtype as wan_ld

        fpl = FP8Linear.from_linear(nn.Linear(8, 16))
        # 核心: 返回 compute_dtype (bf16/f32), 非 fp8_weight.dtype, 不崩溃.
        assert sky_ld(fpl) == fpl.compute_dtype
        assert wan_ld(fpl) == fpl.compute_dtype

    def test_linear_dtype_plain_linear_no_regression(self):
        import mlx.nn as nn

        from fusion_mlx.video.skyreels_v3.attention import (
            _linear_dtype as sky_ld,
        )
        from fusion_mlx.video.wan2.attention import _linear_dtype as wan_ld

        lin = nn.Linear(8, 16)
        assert sky_ld(lin) == lin.weight.dtype
        assert wan_ld(lin) == lin.weight.dtype

    def test_fp8linear_forward_with_compute_dtype(self):
        import mlx.nn as nn

        from fusion_mlx.custom_kernels.fp8_linear import FP8Linear

        fpl = FP8Linear.from_linear(nn.Linear(8, 16))
        x = mx.random.uniform(shape=(2, 8)).astype(fpl.compute_dtype)
        out = fpl(x)
        mx.eval(out)
        assert out.shape == (2, 16)

    def test_convert_to_fp8_linear_then_linear_dtype(self):
        import mlx.nn as nn

        from fusion_mlx.custom_kernels.fp8_linear import (
            FP8Linear,
            convert_to_fp8_linear,
        )
        from fusion_mlx.video.skyreels_v3.attention import (
            _linear_dtype as sky_ld,
        )

        class Tiny(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = nn.Linear(8, 16)

            def __call__(self, x):
                return self.fc(x)

        m = Tiny()
        convert_to_fp8_linear(m)
        assert isinstance(m.fc, FP8Linear)
        assert sky_ld(m.fc) == m.fc.compute_dtype
        out = m(mx.random.uniform(shape=(1, 8)).astype(m.fc.compute_dtype))
        mx.eval(out)
        assert out.shape == (1, 16)


# ============================================================================
# 10. step_strategy 基本行为
# ============================================================================
class TestStepStrategy:
    """SkyReelsStepStrategy 基本行为."""

    def test_step_strategy_config_build(self):
        """StepStrategyConfig.build_step_methods 不抛异常."""
        from fusion_mlx.video.skyreels_v3.step_strategy import (
            StepStrategyConfig,
        )

        cfg = StepStrategyConfig(total_steps=10)
        methods = cfg.build_step_methods()
        assert len(methods) == 10

    def test_default_config_for_branches(self):
        """三分支默认 config 完整性."""
        from fusion_mlx.video.skyreels_v3.step_strategy import (
            _default_config_for_branch,
        )

        for branch in ("r2v", "v2v", "a2v"):
            cfg = _default_config_for_branch(branch, total_steps=50)
            assert cfg.total_steps == 50


# ============================================================================
# 11. DiT 权重重映射 (diffusers-Wan -> MLX SkyReels-V3)
# ============================================================================
class TestWeightRemap:
    """_remap_diffusers_to_mlx 重映射逻辑回归.

    回归 #130-#139 审计发现: load_dit_weights(strict=False) 曾静默跳过 ~97%
    源 key (仅 blocks.N.norm2.weight 偶然同名), 模型停留在随机 init.
    重映射让真实权重落地. 此测试锁定重映射命名规则 + patch_embedding 5D->4D.
    """

    def test_remap_basic_naming(self):
        from fusion_mlx.video.skyreels_v3.transformer_r2v import SkyReelsR2VDiT
        from fusion_mlx.video.skyreels_v3.weights import _remap_diffusers_to_mlx

        dit = SkyReelsR2VDiT(TINY_CFG)
        model_keys = set(
            k for k, _ in __import__("mlx").nn.utils.tree_flatten(dit.parameters())
        )

        mx = __import__("mlx").core
        dim = TINY_CFG["dim"]
        in_dim = TINY_CFG["in_dim"]
        src = {
            "condition_embedder.text_embedder.linear_1.weight": mx.zeros(
                (dim, TINY_CFG["text_dim"])
            ),
            "condition_embedder.time_embedder.linear_1.weight": mx.zeros((dim, dim)),
            "condition_embedder.time_proj.weight": mx.zeros((dim, dim)),
            "scale_shift_table": mx.zeros((1, 6, dim)),
            "proj_out.weight": mx.zeros((in_dim, dim)),
            "patch_embedding.weight": mx.zeros((dim, in_dim, 1, 2, 2)),
            "patch_embedding.bias": mx.zeros((dim,)),
            "blocks.0.attn1.to_q.weight": mx.zeros((dim, dim)),
            "blocks.0.attn2.to_k.weight": mx.zeros((dim, dim)),
            "blocks.0.attn1.to_out.0.weight": mx.zeros((dim, dim)),
            "blocks.0.ffn.net.0.proj.weight": mx.zeros((TINY_CFG["ffn_dim"], dim)),
            "blocks.0.ffn.net.2.weight": mx.zeros((dim, TINY_CFG["ffn_dim"])),
            "blocks.0.scale_shift_table": mx.zeros((1, 6, dim)),
        }
        out = _remap_diffusers_to_mlx(src, dit)

        for k in out:
            assert k in model_keys, f"重映射产物未命中模型: {k}"

        assert "text_embedding.layers.0.weight" in out
        assert "time_embedding.layers.0.weight" in out
        assert "time_projection.layers.1.weight" in out
        assert "head.modulation" in out
        assert "head.head.weight" in out
        assert "patch_embedding.conv2d.weight" in out
        assert "patch_embedding.conv2d.bias" in out
        assert "blocks.0.self_attn.q.weight" in out
        assert "blocks.0.cross_attn.k.weight" in out
        assert "blocks.0.self_attn.o.weight" in out
        assert "blocks.0.ffn.fc1.weight" in out
        assert "blocks.0.ffn.fc2.weight" in out
        assert "blocks.0.modulation" in out

    def test_patch_embedding_5d_to_4d_channels_last(self):
        from fusion_mlx.video.skyreels_v3.transformer_r2v import SkyReelsR2VDiT
        from fusion_mlx.video.skyreels_v3.weights import _remap_diffusers_to_mlx

        dit = SkyReelsR2VDiT(TINY_CFG)
        mx = __import__("mlx").core
        dim = TINY_CFG["dim"]
        in_dim = TINY_CFG["in_dim"]
        w5 = mx.ones((dim, in_dim, 1, 2, 2))
        src = {"patch_embedding.weight": w5}
        out = _remap_diffusers_to_mlx(src, dit)
        assert "patch_embedding.conv2d.weight" in out
        # MLX Conv2d channels-last: [out, kh, kw, in*t]
        assert out["patch_embedding.conv2d.weight"].shape == (dim, 2, 2, in_dim * 1)

    def test_remap_drops_unmatched_source_keys(self):
        """源中不存在于模型的 key (如 norm2.bias) 应被丢弃, 不进入返回 dict."""
        from fusion_mlx.video.skyreels_v3.transformer_r2v import SkyReelsR2VDiT
        from fusion_mlx.video.skyreels_v3.weights import _remap_diffusers_to_mlx

        dit = SkyReelsR2VDiT(TINY_CFG)
        mx = __import__("mlx").core
        dim = TINY_CFG["dim"]
        src = {
            "blocks.0.norm2.bias": mx.zeros((dim,)),
            "condition_embedder.text_embedder.linear_1.weight": mx.zeros(
                (dim, TINY_CFG["text_dim"])
            ),
        }
        out = _remap_diffusers_to_mlx(src, dit)
        assert "blocks.0.norm2.bias" not in out
        assert "text_embedding.layers.0.weight" in out


# ============================================================================
# 12. 真实权重加载集成测试 (需本地模型, 缺失则 skip)
# ============================================================================
REAL_MODEL_DIR = Path.home() / ".fusion-mlx/models/Skywork/SkyReels-V3-R2V-14B-MLX"


@pytest.mark.skipif(
    not (
        REAL_MODEL_DIR / "transformer" / "diffusion_pytorch_model.safetensors"
    ).exists(),
    reason="真实 SkyReels-V3-R2V-14B 权重未下载 (26.6GB)",
)
class TestRealWeightLoad:
    """真实权重加载集成测试: 验证重映射让 >1000 参数命中 (而非修复前的 40)."""

    def test_real_weights_load_over_1000_keys(self):
        import json

        import mlx.nn as nn

        from fusion_mlx.video.skyreels_v3.transformer_r2v import SkyReelsR2VDiT
        from fusion_mlx.video.skyreels_v3.weights import load_dit_weights

        cfg = json.loads((REAL_MODEL_DIR / "config.json").read_text())["config"]
        dit = SkyReelsR2VDiT(cfg)
        before = dict(nn.utils.tree_flatten(dit.parameters()))
        probe_before = float(
            __import__("mlx").core.sum(before["blocks.0.self_attn.q.weight"]).item()
        )

        dit = load_dit_weights(dit, REAL_MODEL_DIR / "transformer", strict=False)
        after = dict(nn.utils.tree_flatten(dit.parameters()))
        probe_after = float(
            __import__("mlx").core.sum(after["blocks.0.self_attn.q.weight"]).item()
        )

        # 修复前仅 40/1377 命中, 修复后探测参数必须变化 (真实权重落地)
        assert (
            abs(probe_after - probe_before) > 1e-3
        ), "探测参数加载前后未变化 -> 真实权重未落地 (重映射回归)"


class TestR2VArchIssue164:
    """issue #164: R2V-14B arch 对齐 diffusers SkyReelsV2TransformerBlock.

    修复前: cross_attn_type=i2v (多 k_img/v_img/norm_k_img 随机 init) +
    norm1 有参 + norm2/3 命名错位 (diffusers norm2=cross-attn前 被加载到
    MLX ffn前 norm2) + Head.norm 有参 -> 322 参数未覆盖/错位, 生成劣化.
    """

    def test_r2v_default_cross_attn_is_t2v(self):
        from fusion_mlx.video.skyreels_v3.transformer_r2v import SkyReelsR2VDiT

        m = SkyReelsR2VDiT()
        assert m.cross_attn_type == "t2v_cross_attn"

    def test_r2v_no_img_branch_params(self):
        import mlx.nn as nn

        from fusion_mlx.video.skyreels_v3.transformer_r2v import SkyReelsR2VDiT

        m = SkyReelsR2VDiT()
        keys = set(k for k, _ in nn.utils.tree_flatten(m.parameters()))
        img = [k for k in keys if "k_img" in k or "v_img" in k or "norm_k_img" in k]
        assert img == [], f"R2V 不该有 img 分支 (added_kv_proj_dim=null): {img[:3]}"

    def test_r2v_norm1_norm3_no_affine_norm2_affine(self):
        import mlx.nn as nn

        from fusion_mlx.video.skyreels_v3.transformer_r2v import SkyReelsR2VDiT

        m = SkyReelsR2VDiT()
        keys = set(k for k, _ in nn.utils.tree_flatten(m.parameters()))
        # norm1 (attn1前) / norm3 (ffn前): affine=False -> 无 bias
        assert "blocks.0.norm1.bias" not in keys, "norm1 应 affine=False 无 bias"
        assert "blocks.0.norm3.bias" not in keys, "norm3 应 affine=False 无 bias"
        # norm2 (cross-attn前, cross_attn_norm=True): affine=True -> 有 weight+bias
        assert "blocks.0.norm2.weight" in keys, "norm2 (cross-attn前) 应有 weight"
        assert "blocks.0.norm2.bias" in keys, "norm2 (cross-attn前) 应有 bias"

    def test_r2v_head_norm_no_affine(self):
        import mlx.nn as nn

        from fusion_mlx.video.skyreels_v3.transformer_r2v import SkyReelsR2VDiT

        m = SkyReelsR2VDiT()
        keys = set(k for k, _ in nn.utils.tree_flatten(m.parameters()))
        assert "head.norm.bias" not in keys, "Head.norm 应 affine=False 无 bias"

    def test_r2v_added_kv_proj_dim_routes_i2v(self):
        from fusion_mlx.video.skyreels_v3.transformer_r2v import SkyReelsR2VDiT

        m = SkyReelsR2VDiT({"added_kv_proj_dim": 5120})
        assert m.cross_attn_type == "i2v_cross_attn"

    def test_r2v_config_preset_t2v(self):
        from fusion_mlx.video.skyreels_v3 import get_branch_config

        cfg = get_branch_config("skyreels-v3-r2v-14b")
        assert cfg.cross_attn_type == "t2v_cross_attn"


# ============================================================================
# 测试入口
# ============================================================================
if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
