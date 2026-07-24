# SPDX-License-Identifier: Apache-2.0
"""E2E tests for video adapters with real model weights.

These tests require downloaded models and are skipped if models are not found.
Run with: HF_ENDPOINT=https://hf-mirror.com FUSION_E2E_ADAPTERS=1 pytest tests/e2e/test_adapter_real_weights.py -v

Model locations:
  - CLIP: ~/.fusion-mlx/models/laion/CLIP-ViT-H-14-laion2B-s32B-b79K/model.safetensors
  - AnimateDiff: ~/.fusion-mlx/models/guoyww/AnimateDiff/mm_sdxl_v10_beta.ckpt
  - ControlNet: ~/.fusion-mlx/models/Kwai-Kolors/Kolors-ControlNet-Canny/diffusion_pytorch_model.safetensors

Key findings:
  - AnimateDiff/ControlNet upstream weights are SD-UNet (dim=320/640/1280),
    incompatible with SkyReels DiT (dim=5120). Zero-init is correct behavior.
  - IP-Adapter CLIP vision encoder loads real weights successfully.
  - IP-Adapter projection weights require a dedicated IP-Adapter checkpoint (not yet available).
"""

import os
import tempfile

import pytest

import mlx.core as mx
import numpy as np

pytestmark = pytest.mark.skipif(
    os.environ.get("FUSION_E2E_ADAPTERS") != "1",
    reason="Set FUSION_E2E_ADAPTERS=1 to run real-weight adapter tests",
)

CLIP_PATH = os.path.expanduser(
    "~/.fusion-mlx/models/laion/CLIP-ViT-H-14-laion2B-s32B-b79K/model.safetensors"
)


class TestIPAdapterRealWeights:
    def test_clip_weight_remap(self):
        from fusion_mlx.video.adapters.ip_adapter import remap_clip_weights

        if not os.path.exists(CLIP_PATH):
            pytest.skip(f"CLIP model not found: {CLIP_PATH}")

        raw = mx.load(CLIP_PATH)
        remapped = remap_clip_weights(raw)
        assert len(remapped) > 0, "remap_clip_weights returned empty dict"
        block_ids = set()
        for k in remapped:
            if k.startswith("block_"):
                block_ids.add(int(k.split(".")[0].split("_")[1]))
        assert len(block_ids) == 32, f"Expected 32 blocks, got {len(block_ids)}"

    def test_clip_forward_pass(self):
        from fusion_mlx.video.adapters.ip_adapter import (
            IPAdapterClipVisionEncoder,
            remap_clip_weights,
        )

        if not os.path.exists(CLIP_PATH):
            pytest.skip(f"CLIP model not found: {CLIP_PATH}")

        raw = mx.load(CLIP_PATH)
        remapped = remap_clip_weights(raw)
        clip = IPAdapterClipVisionEncoder(dim=1280)
        clip.load_weights(list(remapped.items()))

        dummy = mx.random.normal((1, 224, 224, 3)).astype(mx.float32)
        out = clip(dummy)
        mx.eval(out)
        assert out.shape == (1, 257, 1280)
        assert not np.isnan(np.array(out)).any()

    def test_ip_adapter_modify_context(self):
        from fusion_mlx.video.adapters import create_adapter

        if not os.path.exists(CLIP_PATH):
            pytest.skip(f"CLIP model not found: {CLIP_PATH}")

        clip_dir = os.path.dirname(CLIP_PATH)
        adapter = create_adapter("ip_adapter", scale=1.0, config={"text_dim": 4096})
        adapter.load(clip_dir)

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            from PIL import Image

            img = Image.fromarray(
                np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
            )
            img.save(f.name)
            text_ctx = mx.random.normal((1, 512, 4096)).astype(mx.float32)
            augmented = adapter.modify_context(text_ctx, image=f.name)
            mx.eval(augmented)
            assert augmented.shape == (1, 769, 4096)
            os.unlink(f.name)

        adapter.unload()


class TestAnimateDiffZeroInit:
    def test_zero_init_identity(self):
        from fusion_mlx.video.adapters import create_adapter

        adapter = create_adapter(
            "animatediff",
            scale=0.5,
            config={"dim": 5120, "num_heads": 40, "num_layers": 3},
        )
        adapter.load()
        x = mx.random.normal((1, 100, 5120)).astype(mx.float32)
        out = adapter._modules[0](x, 5)
        mx.eval(out)
        assert out.shape == x.shape
        assert np.abs(np.array(out)).max() < 1e-4

    def test_inject_remove(self):
        from fusion_mlx.video.adapters import create_adapter

        adapter = create_adapter(
            "animatediff",
            scale=0.5,
            config={"dim": 5120, "num_heads": 40, "num_layers": 2},
        )
        adapter.load()

        class FakeBlock:
            pass

        class FakeDiT:
            _num_blocks = 2

        dit = FakeDiT()
        for i in range(2):
            setattr(dit, f"block_{i}", FakeBlock())

        adapter.inject(dit)
        for i in range(2):
            assert hasattr(getattr(dit, f"block_{i}"), "motion_module")
        adapter.remove(dit)
        for i in range(2):
            assert not hasattr(getattr(dit, f"block_{i}"), "motion_module")
        adapter.unload()


class TestControlNetForward:
    def test_compute_residuals(self):
        from fusion_mlx.video.adapters import create_adapter

        adapter = create_adapter(
            "controlnet",
            scale=1.0,
            config={"dim": 5120, "text_dim": 4096, "num_layers": 2},
        )
        adapter.load()

        control = mx.random.normal((1, 3, 90, 160)).astype(mx.float32)
        t = mx.array([0.5])
        ctx = mx.random.normal((1, 769, 4096)).astype(mx.float32)

        residuals = adapter.compute_residuals(control, t, ctx)
        assert residuals is not None
        assert len(residuals) == 3
        for r in residuals:
            mx.eval(r)
            assert r.ndim == 3
        adapter.unload()
