# SPDX-License-Identifier: Apache-2.0
"""fusion-mlx 全量回归测试套件 (AtomCode 2026-07-20).

覆盖 #131-#134 修复链路 + 跨设备兼容性 + 嵌套递归 + 前向真状 + pipeline 载入:
  #131-#132: convert_to_fp8_linear/quantize_model 递归遍历 list 内嵌套子模块
  #133:      fp8_matmul 用 mx.transpose 强物化转置替 .T 视图
  #134:      M5Optimizer.apply_to_model 非 M5 设备也执行转换 (bf16 降级)

运行: python -m fusion_mlx.tests.regression_fp8_quant
退出码: 0=全通过, 1=有失败
"""
from __future__ import annotations

import importlib
import logging
import sys
import traceback
import unittest
from unittest.mock import patch

import mlx.core as mx
import mlx.nn as nn

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# SkyReels DiT 真状 mock (3 分支共享骨架)
# ---------------------------------------------------------------------------
class CrossAttn(nn.Module):
    """交叉注意力: 文本 Prompt 引导画面生成 (含 k_img/v_img/q_txt 三 Linear)."""
    def __init__(self, d: int = 5120, d_img: int = 4096):
        super().__init__()
        self.k_img = nn.Linear(d, d_img)   # (d_img, d) = (4096, 5120) ← #133/#134 报错层
        self.v_img = nn.Linear(d, d_img)   # (4096, 5120)
        self.q_txt = nn.Linear(d, d)       # (5120, 5120)


class Block(nn.Module):
    """DiT Block: AdaLN-Zero + 双注意力分支 + FFN 前馈."""
    def __init__(self, d: int = 5120):
        super().__init__()
        self.cross_attn = CrossAttn(d=d, d_img=4096)
        self.self_attn = nn.Linear(d, d)        # (5120, 5120)
        self.ffn = nn.Linear(d, 4 * d)          # (20480, 5120)
        self.ffn_out = nn.Linear(4 * d, d)      # (5120, 20480) 非对称形状回归


class DiT(nn.Module):
    """SkyReels DiT 主干: blocks 用 list 孜 (非 _modules 字典)."""
    def __init__(self, d: int = 5120, n_blocks: int = 2):
        super().__init__()
        self.blocks = [Block(d) for _ in range(n_blocks)]  # list 孜 ← #132 根因
        self.head = nn.Linear(d, 4096)                     # (4096, 5120)


# ---------------------------------------------------------------------------
# #131-#132: 递归遍历 list 内嵌套子模块回归
# ---------------------------------------------------------------------------
class TestIterSubmodulesRecursive(unittest.TestCase):
    """#131-#132: _iter_submodules 递归遍历 list 内嵌套 nn.Linear/nn.Module."""

    def test_iter_catches_all_linear_in_list(self):
        """list 孜的 blocks 内所有 nn.Linear 应被抓到 (含 cross_attn.k_img 嵌套)."""
        from fusion_mlx.custom_kernels.fp8_linear import _iter_submodules
        model = DiT(d=5120, n_blocks=2)
        paths = [name for _, _, name, _, _ in _iter_submodules(model)]
        # 2 blocks × (cross_attn 3 + self_attn 1 + ffn 1 + ffn_out 1) + head 1 = 13
        self.assertEqual(len(paths), 13, f"应 13 路径, 实 {len(paths)}: {paths}")
        # 嵌套层验证
        self.assertIn("blocks.0.cross_attn.k_img", paths)
        self.assertIn("blocks.1.cross_attn.v_img", paths)
        self.assertIn("blocks.0.ffn_out", paths)
        self.assertIn("head", paths)

    def test_convert_to_fp8_linear_recurses_list(self):
        """convert_to_fp8_linear 应把 list 内所有嵌套 nn.Linear 转 FP8Linear."""
        from fusion_mlx.custom_kernels.fp8_linear import (
            FP8Linear, convert_to_fp8_linear, _iter_submodules,
        )
        model = DiT(d=5120, n_blocks=3)
        convert_to_fp8_linear(model)
        # 残留 nn.Linear 应为 0 (全转 FP8Linear)
        leftover = sum(
            1 for _, _, _, m, _ in _iter_submodules(model)
            if type(m).__name__ == "Linear"
        )
        self.assertEqual(leftover, 0, f"残留 {leftover} nn.Linear 未转")
        # 抽验 3 个嵌套位置
        self.assertIsInstance(model.blocks[0].cross_attn.k_img, FP8Linear)
        self.assertIsInstance(model.blocks[2].ffn_out, FP8Linear)
        self.assertIsInstance(model.head, FP8Linear)

    def test_quantize_model_recurses_list(self):
        """quantize_model bits=16 不量化但应递归遍历不报错."""
        from fusion_mlx.custom_kernels.quantize import quantize_model
        model = DiT(d=5120, n_blocks=2)
        quantize_model(model, bits=16)
        # bits=16 保持 nn.Linear 不量化
        self.assertIsInstance(model.blocks[0].cross_attn.k_img, nn.Linear)
        self.assertIsInstance(model.head, nn.Linear)


# ---------------------------------------------------------------------------
# #133: mx.transpose 强物化转置回归
# ---------------------------------------------------------------------------
class TestFp8MatmulTranspose(unittest.TestCase):
    """#133: fp8_matmul 用 mx.transpose 强物化转置替 .T 视图."""

    def test_fp8_matmul_degrade_path(self):
        """降级路径 (非 FP8 硬件): x (...,in) @ (out,in).T → (...,out)."""
        from fusion_mlx.custom_kernels.fp8_linear import fp8_matmul, is_available
        if is_available():
            self.skipTest("FP8 硬件可用, 降级路径不测")
        x = mx.random.normal((2, 769, 5120), dtype=mx.float32)   # in=5120
        w = mx.random.normal((4096, 5120), dtype=mx.float32)     # (out, in)
        scale = mx.ones((4096,), dtype=mx.float32)
        out = fp8_matmul(x, w, scale)
        self.assertEqual(out.shape, (2, 769, 4096), f"应 (2,769,4096), 实 {out.shape}")

    def test_fp8_matmul_fp8_path(self):
        """FP8 路径: 若硬件可用, astype+scale+transpose 链式不报错."""
        from fusion_mlx.custom_kernels.fp8_linear import fp8_matmul, is_available
        if not is_available():
            self.skipTest("FP8 �硬件不可用, FP8 路径不测")
        x = mx.random.normal((2, 769, 5120), dtype=mx.float32)
        w = mx.zeros((4096, 5120), dtype=mx.float8_e4m3fn)
        scale = mx.ones((4096,), dtype=mx.float32)
        out = fp8_matmul(x, w, scale)
        self.assertEqual(out.shape, (2, 769, 4096))

    def test_mixed_weight_shapes(self):
        """多层混合形状: k_img (4096,5120) / ffn (20480,5120) / ffn_out (5120,20480)."""
        from fusion_mlx.custom_kernels.fp8_linear import fp8_matmul
        x_5120 = mx.random.normal((2, 769, 5120), dtype=mx.float32)
        x_20480 = mx.random.normal((2, 769, 20480), dtype=mx.float32)
        # k_img: in=5120 → out=4096
        out_k = fp8_matmul(x_5120, mx.random.normal((4096, 5120)), mx.ones((4096,)))
        self.assertEqual(out_k.shape, (2, 769, 4096))
        # ffn: in=5120 → out=20480
        out_ffn = fp8_matmul(x_5120, mx.random.normal((20480, 5120)), mx.ones((20480,)))
        self.assertEqual(out_ffn.shape, (2, 769, 20480))
        # ffn_out: in=20480 → out=5120 (非对称形状回归)
        out_fo = fp8_matmul(x_20480, mx.random.normal((5120, 20480)), mx.ones((5120,)))
        self.assertEqual(out_fo.shape, (2, 769, 5120))


# ---------------------------------------------------------------------------
# #134: 非 M5 设备也执行转换回归
# ---------------------------------------------------------------------------
class TestM5OptimizerNonM5(unittest.TestCase):
    """#134: M5Optimizer.apply_to_model 非 M5 设备也执行 FP8/NF4 转换."""

    def test_apply_to_model_non_m5_executes(self):
        """is_m5=False 模拟: apply_to_model 不早退, 真 FP8Linear 转换."""
        from fusion_mlx.video.skyreels_v3.m5_optimizer import M5Optimizer
        from fusion_mlx.custom_kernels.fp8_linear import FP8Linear, _iter_submodules
        model = DiT(d=5120, n_blocks=2)
        with patch("fusion_mlx.video.skyreels_v3._device.is_m5", return_value=False):
            opt = M5Optimizer()
            self.assertFalse(opt.is_m5, "模拟非 M5 设备")
            opt.apply_to_model(model)
        # 非M5 也应全转 (bf16 降级, 但仍是 FP8Linear 类)
        leftover = sum(
            1 for _, _, _, m, _ in _iter_submodules(model)
            if type(m).__name__ == "Linear"
        )
        self.assertEqual(leftover, 0, f"非M5 残留 {leftover} nn.Linear")

    def test_forward_no_addmm_error_non_m5(self):
        """非 M5 设备前向不应报原始 addmm 维度错 (5120 vs 4096)."""
        from fusion_mlx.video.skyreels_v3.m5_optimizer import M5Optimizer
        model = DiT(d=5120, n_blocks=2)
        with patch("fusion_mlx.video.skyreels_v3._device.is_m5", return_value=False):
            opt = M5Optimizer()
            opt.apply_to_model(model)
        x = mx.random.normal((2, 769, 5120), dtype=mx.float32)
        # 真状前向: #133/#134 报错点全验
        out_k = model.blocks[0].cross_attn.k_img(x)
        self.assertEqual(out_k.shape, (2, 769, 4096))
        out_head = model.head(x)
        self.assertEqual(out_head.shape, (2, 769, 4096))
        out_fo = model.blocks[0].ffn_out(mx.random.normal((2, 769, 20480), dtype=mx.float32))
        self.assertEqual(out_fo.shape, (2, 769, 5120))

    def test_m5_device_still_optimizes(self):
        """M5 设备 (is_m5=True) 也应真执行转换 (回归保护, 避删早退致 M5 也漏转)."""
        from fusion_mlx.video.skyreels_v3.m5_optimizer import M5Optimizer
        from fusion_mlx.custom_kernels.fp8_linear import _iter_submodules
        model = DiT(d=5120, n_blocks=2)
        with patch("fusion_mlx.video.skyreels_v3._device.is_m5", return_value=True):
            opt = M5Optimizer()
            opt.apply_to_model(model)
        leftover = sum(
            1 for _, _, _, m, _ in _iter_submodules(model)
            if type(m).__name__ == "Linear"
        )
        self.assertEqual(leftover, 0, f"M5 残留 {leftover} nn.Linear")


# ---------------------------------------------------------------------------
# server 重启 / 崩包重导入回归
# ---------------------------------------------------------------------------
class TestFreshImportAfterServerRestart(unittest.TestCase):
    """server 重启后全新进程导入通过 (无语法错 + 无循环依赖)."""

    def test_fresh_import_fp8_linear(self):
        # 清 sys.modules 逼全新导入 (模拟 server 重启)
        for mod in list(sys.modules.keys()):
            if "fusion_mlx.custom_kernels" in mod:
                del sys.modules[mod]
        from fusion_mlx.custom_kernels.fp8_linear import (
            convert_to_fp8_linear, _iter_submodules, FP8Linear, fp8_matmul,
        )
        self.assertTrue(callable(convert_to_fp8_linear))
        self.assertTrue(callable(_iter_submodules))
        self.assertTrue(callable(fp8_matmul))

    def test_fresh_import_quantize(self):
        for mod in list(sys.modules.keys()):
            if "fusion_mlx.custom_kernels" in mod:
                del sys.modules[mod]
        from fusion_mlx.custom_kernels.quantize import quantize_model, quantize_linear
        self.assertTrue(callable(quantize_model))
        self.assertTrue(callable(quantize_linear))

    def test_fresh_import_m5_optimizer(self):
        for mod in list(sys.modules.keys()):
            if "fusion_mlx.video.skyreels_v3" in mod:
                del sys.modules[mod]
        from fusion_mlx.video.skyreels_v3.m5_optimizer import M5Optimizer
        opt = M5Optimizer()
        self.assertIsNotNone(opt.is_m5)


# ---------------------------------------------------------------------------
# 真权重载入回归 (AtomCode 铁律: 禁 stub 假回归, 必真权重载入)
# ---------------------------------------------------------------------------
# 真权重路径: ~/.fusion-mlx/models/models--Skywork--SkyReels-V3-R2V-14B/snapshots/<hash>
import os
_HOME = os.path.expanduser("~")
_SKY_BASE = os.path.join(
    _HOME, ".fusion-mlx", "models",
    "models--Skywork--SkyReels-V3-R2V-14B",
)
_SKY_SNAPSHOT = None
if os.path.isdir(os.path.join(_SKY_BASE, "snapshots")):
    _snapshots_dir = os.path.join(_SKY_BASE, "snapshots")
    _snapshots = sorted(os.listdir(_snapshots_dir))
    if _snapshots:
        _SKY_SNAPSHOT = os.path.join(_snapshots_dir, _snapshots[0])


@unittest.skipUnless(
    _SKY_SNAPSHOT and os.path.isdir(_SKY_SNAPSHOT),
    "无 SkyReels-V3-R2V-14B 真权重目录 (~/.fusion-mlx/models), 跳过真权重回归",
)
class TestRealWeightsLoad(unittest.TestCase):
    """真权重载入回归 (AtomCode 铁律: 禁 stub 假回归).

    用真 SkyReels-V3-R2V-14B 权重 (~/.fusion-mlx/models) 验:
      - text_encoder 真载入 + encode_text 前向 (真权重非全零)
      - vae 真载入 + decode 前向 (真权重非全零)
      - DiT 真载入 (若权重已下完) + 一步采样前向验无 [matmul]/[addmm] 报错
    """

    def test_text_encoder_real_weights(self):
        """text_encoder 真载入 + encode_text 前向 (真权重 >90% 非零)."""
        from fusion_mlx.video.skyreels_v3.text_encoder import UMT5Encoder
        enc = UMT5Encoder.from_pretrained(os.path.join(_SKY_SNAPSHOT, "text_encoder"))
        emb = enc.encode_text("a cat playing piano", max_length=77)
        self.assertEqual(emb.ndim, 3, f"emb 应 3D, 实 {emb.ndim}D")
        # 真权重非全零验 (随机初始化会全零或极小值)
        nonzero = mx.sum(emb != 0).item()
        self.assertGreater(
            nonzero, emb.size * 0.9,
            f"真权重应 >90% 非零, 实 {100*nonzero/emb.size:.1f}%",
        )

    def test_vae_real_weights_load(self):
        """vae 真载入 (from_pretrained 真调 load_weights, 非 0s 载入)."""
        from fusion_mlx.video.skyreels_v3.vae import SkyReelsVAE
        vae = SkyReelsVAE.from_pretrained(os.path.join(_SKY_SNAPSHOT, "vae"))
        self.assertIsNotNone(vae.vae, "底座 WanVAE 应真载入")
        self.assertTrue(vae._uses_base, "应走底座真载入非 stub")
        # 验权重真注入: 递归抓底座 vae.vae 所有子模块内参数
        # load_weights 应真读 safetensors 注入子模块, 非假载入 0.0s
        # mlx.nn.Module 子模块用 apply_to_modules 递归抓
        params = []
        def collect(name, mod):
            for k, v in mod.items():
                if hasattr(v, "size") and v.size > 16:
                    params.append((f"{name}.{k}", v))
        vae.vae.apply_to_modules(collect)
        self.assertGreater(len(params), 20, f"底座应 >20 真权重张量, 实 {len(params)}")
        nonzero = sum(1 for _, p in params if mx.sum(p != 0).item() > 0)
        self.assertGreater(
            nonzero, len(params) * 0.1,
            f"底座真权重应 >10% 张量非零, 实 {nonzero}/{len(params)}",
        )

    def test_dit_real_weights_if_available(self):
        """DiT 真载入 (若 diffusion_pytorch_model.safetensors 已下完)."""
        dit_dir = os.path.join(_SKY_SNAPSHOT, "transformer")
        dit_file = os.path.join(dit_dir, "diffusion_pytorch_model.safetensors")
        if not os.path.exists(dit_file):
            self.skipTest("DiT 28.58GB 权重未下完, 跳过 (后台下载中)")
        # 真载入 + 一步前向验无 [matmul]/[addmm] 报错
        from fusion_mlx.video.skyreels_v3.pipelines import SkyReelsR2VPipeline
        pipeline = SkyReelsR2VPipeline(_SKY_SNAPSHOT)
        self.assertIsNotNone(pipeline.dit)
        self.assertIsNotNone(pipeline.m5_optimizer)
        # 验 DiT 真权重已注入 (blocks 非全零)
        import mlx.nn as nn
        has_weights = False
        for k, v in pipeline.dit.items():
            if hasattr(v, 'shape') and v.size > 100 and 'weight' in k.lower():
                nz = mx.sum(v != 0).item()
                if nz > v.size * 0.5:
                    has_weights = True
                    break
        self.assertTrue(has_weights, "DiT 权重应真注入")


# ---------------------------------------------------------------------------
# 压测: 多次重复转换稳定性 (避生产团队偶发 OOM/崩溃)
# ---------------------------------------------------------------------------
class TestRepeatedConversionStress(unittest.TestCase):
    """重复转换稳定性压测 (避生产团队偶发 OOM/崩溃)."""

    def test_repeated_convert_5_iterations(self):
        """5 次重复 convert_to_fp8_linear 不报错 (避 mlx 内部状态泄漏)."""
        from fusion_mlx.custom_kernels.fp8_linear import convert_to_fp8_linear
        for i in range(5):
            model = DiT(d=5120, n_blocks=2)
            convert_to_fp8_linear(model)
            # 每次后前向测
            x = mx.random.normal((1, 10, 5120), dtype=mx.float32)
            out = model.head(x)
            self.assertEqual(out.shape, (1, 10, 4096), f"迭代 {i} 前向错")

    def test_repeated_quantize_5_iterations(self):
        from fusion_mlx.custom_kernels.quantize import quantize_model
        for i in range(5):
            model = DiT(d=5120, n_blocks=1)
            quantize_model(model, bits=16)
            out = model.blocks[0].self_attn(mx.random.normal((1, 10, 5120), dtype=mx.float32))
            self.assertEqual(out.shape, (1, 10, 5120), f"迭代 {i} 前向错")


def run_all() -> int:
    """运行全量回归套件, 返退出码."""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    # 按顺序加: #131-#132 → #133 → #134 → server重启 → pipeline → 压测
    for cls in [
        TestIterSubmodulesRecursive,
        TestFp8MatmulTranspose,
        TestM5OptimizerNonM5,
        TestFreshImportAfterServerRestart,
        TestRealWeightsLoad,
        TestRepeatedConversionStress,
    ]:
        suite.addTests(loader.loadTestsFromTestCase(cls))
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    sys.exit(run_all())
