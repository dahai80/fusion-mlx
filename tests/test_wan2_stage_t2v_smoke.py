import asyncio
import os
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest

# Real-model smoke for the #652 staged T2V path through the PUBLIC
# VideoGenEngine (the only surface downstream fusion-comfyui reaches via
# fusion_mlx.public_api). Runs the full staged chain (load_text_encoder ->
# encode_text -> unload_text_encoder -> load_dit -> denoise(control=None) ->
# unload_dit) against real pure-T2V 14B weights, 1 step, fixed seed. Guards the
# MLX thread/stream hazard (#630): denoise runs on get_executor("video") and
# returns an mx.eval'd 5D latent that must stay finite. Pure T2V (control=None)
# is bit-identical to the pre-#652 staged path.
#
# Gated behind FUSION_MLX_REAL_MODEL_TESTS + a fully-installed pure-T2V model
# (t5_encoder + vae present). The default Wan2.1-T2V-14B dir on this host is a
# partial diffusers-shard download (DiT only, no t5/vae) so the test skips;
# point FUSION_652_WAN2_MODEL at a complete T2V dir to run it. The i2v
# checkpoint "Wan2.1-14B" has in_dim 32->36 (mask+video channel-concat) so its
# patch_embedding rejects a 16-channel pure-noise latent — NOT a staged-path
# bug, just the wrong checkpoint for control=None. Per-variant I2V/VACE/camera
# bit-exact vs-monolith acceptance (the heavier #652 surface) is a gated
# post-PR follow-up tracked in PR #694. The load-bearing public-API plumbing
# regression guard is tests/unit/test_pipeline_stage_api.py (always runs).
REAL_MODEL_DIR = Path(
    os.environ.get(
        "FUSION_652_WAN2_MODEL",
        str(Path.home() / ".fusion-mlx/models/Wan2.1-T2V-14B"),
    )
)

pytestmark = pytest.mark.real_model


def _skip_unless_real_model():
    if not os.environ.get("FUSION_MLX_REAL_MODEL_TESTS"):
        pytest.skip(
            "set FUSION_MLX_REAL_MODEL_TESTS=1 to run real-model wan2 stage smoke"
        )
    if not REAL_MODEL_DIR.exists():
        pytest.skip(f"wan2 model not installed at {REAL_MODEL_DIR}")
    # Some on-disk dirs are partial (diffusers-sharded DiT only, no t5/vae).
    # Skip cleanly instead of raising RuntimeError mid-load.
    if not (REAL_MODEL_DIR / "t5_encoder.safetensors").exists():
        pytest.skip(
            f"t5_encoder.safetensors missing at {REAL_MODEL_DIR} (partial model)"
        )


# Wan2.1-T2V-14B: vae_stride (4,8,8), vae_z_dim 16, patch_size (1,2,2) (all
# from WanModelConfig dataclass defaults — config.json omits them). 17 frames
# -> t_latent 5, 480x832 -> 60x104. Empty zeros latent drives shape inference
# in denoise() (matches FusionKSampler.create_empty_latent contract).
def _empty_latent_14b(num_frames: int = 17, height: int = 480, width: int = 832):
    t_latent = (num_frames - 1) // 4 + 1
    h_latent = height // 8
    w_latent = width // 8
    return mx.zeros((1, 16, t_latent, h_latent, w_latent))


def test_staged_t2v_denoise_smoke():
    _skip_unless_real_model()
    from fusion_mlx.public_api import VideoGenEngine

    async def _run():
        eng = VideoGenEngine(str(REAL_MODEL_DIR))
        await eng.start()
        await eng.load_text_encoder()
        emb = await eng.encode_text("a cat walking on a beach")
        assert "embed" in emb
        await eng.unload_text_encoder()
        await eng.load_dit()
        latent = await eng.denoise(
            latent=_empty_latent_14b(),
            pos_embed=emb["embed"],
            neg_embed=None,
            steps=1,
            cfg=1.0,
            seed=42,
            num_frames=17,
            control=None,
        )
        await eng.unload_dit()
        await eng.stop()
        return latent

    latent = asyncio.run(_run())
    arr = np.array(latent)
    assert latent.ndim == 5, f"staged denoise must return 5D latent, got {latent.ndim}"
    assert not np.any(np.isnan(arr)), "staged denoise produced NaN latent"
    assert not np.all(arr == 0), "staged denoise produced all-zero latent"
    print(f"\n#652 staged T2V smoke: out_shape={arr.shape} mean={arr.mean():.4f}")


def test_staged_t2v_denoise_is_seeded_reproducible():
    _skip_unless_real_model()
    from fusion_mlx.public_api import VideoGenEngine

    async def _run(seed):
        eng = VideoGenEngine(str(REAL_MODEL_DIR))
        await eng.start()
        await eng.load_text_encoder()
        emb = await eng.encode_text("a dog playing in snow")
        await eng.unload_text_encoder()
        await eng.load_dit()
        latent = await eng.denoise(
            latent=_empty_latent_14b(),
            pos_embed=emb["embed"],
            neg_embed=None,
            steps=1,
            cfg=1.0,
            seed=seed,
            num_frames=17,
            control=None,
        )
        await eng.unload_dit()
        await eng.stop()
        return np.array(latent)

    a = asyncio.run(_run(7))
    b = asyncio.run(_run(7))
    assert np.array_equal(
        a, b
    ), "same seed must produce identical staged denoise latent"
    print(f"\n#652 staged T2V seed reproducibility: equal={np.array_equal(a, b)}")
