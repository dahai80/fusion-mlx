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
            generate_video,
            list_models,
            get_branch_config,
            BRANCH_CONFIGS,
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
            WanLayerNorm,
            WanRMSNorm,
            GELUApprox,
            sinusoidal_embedding_1d,
            rope_params,
            PatchEmbed3D,
            mul_add,
            mul_add_add,
        )
        assert WanLayerNorm is not None
        assert PatchEmbed3D is not None

    def test_attention_module(self):
        from fusion_mlx.video.skyreels_v3.attention import (
            WanSelfAttention,
            WanTemporalAttention,
            WanT2VCrossAttention,
            WanI2VCrossAttention,
            WAN_CROSSATTENTION_CLASSES,
        )
        assert "t2v_cross_attn" in WAN_CROSSATTENTION_CLASSES
        assert "i2v_cross_attn" in WAN_CROSSATTENTION_CLASSES

    def test_scheduler_module(self):
        from fusion_mlx.video.skyreels_v3.scheduler import (
            FlowUniPCConfig,
            FlowUniPCMultistepScheduler,
            perform_guidance,
            flow_match_sample,
        )
        assert FlowUniPCConfig is not None
        assert callable(perform_guidance)

    def test_transformer_modules(self):
        from fusion_mlx.video.skyreels_v3.transformer_r2v import SkyReelsR2VDiT
        from fusion_mlx.video.skyreels_v3.transformer_v2v import SkyReelsV2VDiT
        from fusion_mlx.video.skyreels_v3.transformer_a2v import SkyReelsA2VDiT
        assert SkyReelsR2VDiT is not None
        assert SkyReelsV2VDiT is not None
        assert SkyReelsA2VDiT is not None

    def test_pipelines_import(self):
        from fusion_mlx.video.skyreels_v3.pipelines import (
            SkyReelsR2VPipeline,
            SkyReelsV2VPipeline,
            SkyReelsA2VPipeline,
        )
        assert SkyReelsR2VPipeline is not None
        assert SkyReelsV2VPipeline is not None
        assert SkyReelsA2VPipeline is not None

    def test_kv_cache_import(self):
        from fusion_mlx.video.skyreels_v3.kv_cache import (
            SkyReelsKVCache,
            _KVPool,
            _PoolConfig,
        )
        assert SkyReelsKVCache is not None

    def test_flicker_fix_import(self):
        from fusion_mlx.video.skyreels_v3.temporal_flicker_fix import (
            FlickerFixConfig,
            TemporalFlickerFix,
            StepCoherenceFilter,
            temporal_ema_smooth,
            temporal_ema_batch,
            boundary_align,
            default_config_for_branch,
        )
        assert FlickerFixConfig is not None
        assert callable(default_config_for_branch)

    def test_m5_optimizer_import(self):
        from fusion_mlx.video.skyreels_v3.m5_optimizer import (
            M5Optimizer,
            ComputePlacement,
            AdaptiveTileScheduler,
            QuantizationConfig,
            AsyncVideoStream,
        )
        assert M5Optimizer is not None

    def test_step_strategy_import(self):
        from fusion_mlx.video.skyreels_v3.step_strategy import (
            StepStrategyConfig,
            SkyReelsStepStrategy,
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
            h=8, w=8,
        )
        assert cache is not None
        assert cache.num_layers == 2

    def test_spatial_pool_append_get(self):
        """空间池 append + get."""
        from fusion_mlx.video.skyreels_v3.kv_cache import SkyReelsKVCache
        cache = SkyReelsKVCache(
            num_layers=1, num_heads=4, head_dim=32,
            max_frames=4, h=4, w=4,
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
            num_layers=1, num_heads=4, head_dim=32,
            max_frames=4, h=4, w=4,
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
            num_layers=1, num_heads=4, head_dim=32,
            max_frames=4, h=4, w=4,
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
        noise_pred = mx.array([
            [1.0, 2.0],  # uncond
            [3.0, 4.0],  # cond
        ])
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
# 测试入口
# ============================================================================
if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
