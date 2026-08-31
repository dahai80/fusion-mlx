import asyncio
import os
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest

from fusion_mlx.public_api import VideoGenEngine

# Wan2.1-14B: full-encoder VAE (84 keys), 14B DiT. Surface A roundtrip + Surface C
# inpaint + the pure-T2V baseline for the Surface B diff. Override per-test via env.
WAN2_DIR = Path(
    os.environ.get(
        "FUSION_653_WAN2_MODEL",
        str(Path.home() / ".fusion-mlx/models/Wan2.1-14B"),
    )
)
# Surface B: TheDenk wan2.1-t2v-14b-controlnet-canny-v1 (adapter weights, loaded on
# top of the 14B DiT). Override via FUSION_653_CONTROLNET_DIR.
CONTROLNET_DIR = Path(
    os.environ.get(
        "FUSION_653_CONTROLNET_DIR",
        str(Path.home() / ".fusion-mlx/models/models--TheDenk--wan2.1-t2v-14b-controlnet-canny-v1"),
    )
)
# SkyReels V2: Wan2.2-TI2V-5B q8 (alias wan22-ti2v-5b). Surface D (SkyReels inpaint).
SKYREELS_DIR = Path(
    os.environ.get(
        "FUSION_653_SKYREELS_MODEL",
        str(Path.home() / ".fusion-mlx/models/wan22-ti2v-5b"),
    )
)

pytestmark = pytest.mark.real_model


def _skip_unless_real_model(model_dir: Path, label: str, need_files=(), require_t2v: bool = False):
    if not os.environ.get("FUSION_MLX_REAL_MODEL_TESTS"):
        pytest.skip(f"set FUSION_MLX_REAL_MODEL_TESTS=1 to run #653 real-model {label}")
    if not model_dir.exists():
        pytest.skip(f"{label} model not installed at {model_dir}")
    for f in need_files:
        if not (model_dir / f).exists():
            pytest.skip(f"{label} partial model: missing {f} at {model_dir}")
    # Surface B/C need a pure-T2V checkpoint (in_dim==vae_z_dim==16): the i2v
    # "Wan2.1-14B" dir has in_dim 36 (16 latent + 20 image channel-concat) so
    # its patch_embedding rejects a 16-channel pure-noise latent with an addmm
    # shape error. NOT a surface bug — wrong checkpoint for control=None. This
    # mirrors tests/test_wan2_stage_t2v_smoke.py (same host has no MLX-format
    # t2v-14B; Wan2.1-T2V-14B is diffusers-only). NEEDS_CONTEXT for a real
    # t2v-14B MLX model to exercise the ControlNet/inpaint e2e path.
    if require_t2v:
        import json

        cfg_path = model_dir / "config.json"
        if cfg_path.exists():
            try:
                with open(cfg_path) as f:
                    cfg = json.load(f)
                in_dim = cfg.get("in_dim")
                z_dim = cfg.get("vae_z_dim", 16)
                if in_dim is not None and int(in_dim) != int(z_dim):
                    pytest.skip(
                        f"{label} model {model_dir} is i2v (in_dim={in_dim}, "
                        f"vae_z_dim={z_dim}); pure-T2V surface needs in_dim==vae_z_dim "
                        f"(no MLX-format t2v-14B installed; see test_wan2_stage_t2v_smoke.py)"
                    )
            except (ValueError, OSError):
                pass


# ---- Surface A: SkyReels VAE encode roundtrip (new this task) ----

def _structured_pixels(t: int, h: int = 256, w: int = 256) -> mx.array:
    ys = mx.arange(h, dtype=mx.float32) / h
    xs = mx.arange(w, dtype=mx.float32) / w
    ts = mx.arange(t, dtype=mx.float32) / max(t - 1, 1)
    frame = (ys[:, None] + xs[None, :]) * 0.5
    frames = []
    for i in range(t):
        chan_r = frame + 0.1 * ts[i]
        chan_g = frame * 0.7
        chan_b = mx.broadcast_to(ts[i], (h, w))
        frames.append(mx.stack([chan_r, chan_g, chan_b], axis=-1))
    return mx.stack(frames, axis=0)[None]


def _pixel_corr(a: np.ndarray, b: np.ndarray) -> float:
    a = a.reshape(-1).astype(np.float64)
    b = b.reshape(-1).astype(np.float64)
    a = a - a.mean()
    b = b - b.mean()
    denom = np.sqrt((a * a).sum()) * np.sqrt((b * b).sum())
    if denom == 0:
        return 0.0
    return float((a * b).sum() / denom)


def test_skyreels_vae_encode_decode_roundtrip():
    # Surface A on SkyReels. SkyReelsVAE.encode (vae.py:212) [B,3,T,H,W] -> [B,16,T,H/8,W/8].
    # 5 frames = 1 streaming chunk (iter_=3 odd, same wan-style streaming). Guards the
    # numpy-bridge #630 invariant through the SkyReels encode path (Task 8).
    _skip_unless_real_model(SKYREELS_DIR, "skyreels")
    eng = VideoGenEngine(str(SKYREELS_DIR))

    async def _run():
        await eng.start()
        await eng.load_vae()
        pixels = _structured_pixels(5, 256, 256)
        lat = await eng.encode(pixels)
        assert lat.ndim == 5
        assert not np.any(np.isnan(np.array(lat))), "skyreels encode produced NaN"
        out = await eng.decode(lat)
        await eng.unload_vae()
        await eng.stop()
        return pixels, out

    pixels, out = asyncio.run(_run())
    arr = np.array(out)
    assert not np.any(np.isnan(arr)), "skyreels decode produced NaN"
    corr = _pixel_corr(np.array(pixels), arr)
    print(f"\n#653 Surface A skyreels roundtrip pixel_corr={corr:.4f} shape={arr.shape}")
    assert corr >= 0.9, f"skyreels VAE roundtrip corr {corr:.4f} < 0.9 (scrambled)"


# ---- Surface B: Wan2 ControlNet e2e (steers output vs pure-T2V) ----

def _empty_latent_14b(num_frames: int = 17, height: int = 480, width: int = 832):
    t_latent = (num_frames - 1) // 4 + 1
    h_latent = height // 8
    w_latent = width // 8
    return mx.zeros((1, 16, t_latent, h_latent, w_latent))


def test_wan2_controlnet_steers_output_vs_pure_t2v():
    # Surface B: encode_control(controlnet_image=...) returns a ControlState carrying
    # controlnet_adapter + controlnet_latent (Task 5/6); denoise(control=...) injects
    # per-step residuals into the Wan2 DiT block loop (R1). The guard: a canny-edge
    # control image MUST produce a latent that differs from the pure-T2V (control=None)
    # latent at the same seed. Bit-exactness of the residual math is verified by the
    # ControlNet library's own tests; this test proves the WIRING carries the adapter
    # all the way through the public engine surface. 1 step, fixed seed, small frame
    # count to keep it a smoke (not a quality test).
    _skip_unless_real_model(WAN2_DIR, "wan2-controlnet", need_files=("t5_encoder.safetensors",), require_t2v=True)
    _skip_unless_real_model(CONTROLNET_DIR, "controlnet-adapter")
    # Need a control image: synthesize a canny-ish edge frame to a temp png. The
    # adapter's encode_control reads the path. Use a deterministic gradient + threshold.
    import tempfile

    try:
        from PIL import Image
    except ImportError:
        pytest.skip("Pillow not installed; cannot synthesize control image")

    tmp = Path(tempfile.mkdtemp(prefix="fusion_653_cn_"))
    ctrl_path = tmp / "control.png"
    ys, xs = np.mgrid[0:480, 0:832]
    edge = ((xs % 64 < 4) | (ys % 64 < 4)).astype(np.uint8) * 255
    Image.fromarray(np.stack([edge, edge, edge], axis=-1)).save(ctrl_path)

    async def _run(use_cn: bool):
        eng = VideoGenEngine(str(WAN2_DIR))
        await eng.start()
        await eng.load_text_encoder()
        emb = await eng.encode_text("a city street, neon, night")
        await eng.unload_text_encoder()
        await eng.load_dit()
        await eng.load_vae_encoder()
        ctrl = None
        if use_cn:
            ctrl = await eng.encode_control(
                controlnet_image=str(ctrl_path),
                control_type="canny",
                controlnet_strength=1.0,
            )
            assert ctrl is not None, "encode_control returned None for controlnet_image"
        else:
            ctrl = await eng.encode_control()  # pure-T2V -> None
            assert ctrl is None
        latent = await eng.denoise(
            latent=_empty_latent_14b(),
            pos_embed=emb["embed"],
            neg_embed=None,
            steps=1,
            cfg=1.0,
            seed=42,
            num_frames=17,
            control=ctrl,
        )
        await eng.unload_vae_encoder()
        await eng.unload_dit()
        await eng.stop()
        return np.array(latent)

    pure = asyncio.run(_run(False))
    with_cn = asyncio.run(_run(True))
    # Cleanup temp
    import shutil
    shutil.rmtree(tmp, ignore_errors=True)
    assert not np.any(np.isnan(pure)), "pure-T2V latent has NaN"
    assert not np.any(np.isnan(with_cn)), "controlnet latent has NaN"
    diff = float(np.abs(with_cn - pure).mean())
    print(f"\n#653 Surface B controlnet vs pure-T2V mean|diff|={diff:.6f} shape={pure.shape}")
    assert diff > 0.0, "ControlNet produced bit-identical latent to pure-T2V (wiring broken)"


# ---- Surface C: Wan2 inpaint e2e (frozen region preserved) ----

def test_wan2_inpaint_preserves_frozen_region():
    # Surface C: denoise(..., inpaint_mask=, init_latent=) re-composites
    # mask*latents + (1-mask)*init after every sched.step (Task 3). The guard: where
    # mask=0 (frozen), the final latent MUST equal init_latent exactly — the denoiser
    # cannot have touched it. Where mask=1 (reactive), it MUST differ from init (the
    # denoiser did work). 1 step is enough: one re-composite proves the wiring; more
    # steps would test the scheduler, not the surface.
    _skip_unless_real_model(WAN2_DIR, "wan2-inpaint", need_files=("t5_encoder.safetensors",), require_t2v=True)
    latent = _empty_latent_14b()
    # init_latent = a DISTINCT structured latent (not zeros). mask=0 on the first
    # temporal slab (freeze frame 0), mask=1 on the rest (reactive).
    init = mx.array(np.linspace(0.5, 1.5, latent.size, dtype=np.float32).reshape(latent.shape))
    mask = mx.ones_like(latent)
    mask[:, :, 0, :, :] = 0.0  # freeze t_latent=0 slab
    mx.eval(init, mask)

    async def _run():
        eng = VideoGenEngine(str(WAN2_DIR))
        await eng.start()
        await eng.load_text_encoder()
        emb = await eng.encode_text("a horse galloping across a field")
        await eng.unload_text_encoder()
        await eng.load_dit()
        out = await eng.denoise(
            latent=latent,
            pos_embed=emb["embed"],
            neg_embed=None,
            steps=1,
            cfg=1.0,
            seed=42,
            num_frames=17,
            control=None,
            inpaint_mask=mask,
            init_latent=init,
        )
        await eng.unload_dit()
        await eng.stop()
        return np.array(out), np.array(init), np.array(mask)

    out, init_arr, mask_arr = asyncio.run(_run())
    assert not np.any(np.isnan(out)), "inpaint latent has NaN"
    frozen = out[mask_arr == 0]
    frozen_init = init_arr[mask_arr == 0]
    reactive = out[mask_arr == 1]
    reactive_init = init_arr[mask_arr == 1]
    print(
        f"\n#653 Surface C wan2 inpaint: frozen_equal={np.array_equal(frozen, frozen_init)} "
        f"reactive_diff_mean={float(np.abs(reactive - reactive_init).mean()):.6f}"
    )
    assert np.array_equal(frozen, frozen_init), (
        "frozen region (mask=0) was modified by denoise — apply_inpaint_mask not re-compositing"
    )
    assert float(np.abs(reactive - reactive_init).mean()) > 0.0, (
        "reactive region (mask=1) is bit-identical to init — denoiser did no work"
    )


# ---- Surface C: SkyReels inpaint e2e (frozen region preserved) ----

def test_skyreels_inpaint_preserves_frozen_region():
    # Surface C on SkyReels: same guard as the Wan2 test, SkyReels denoise loop.
    # SkyReels (Wan2.2-TI2V-5B) latents are z_dim=48, vae_stride=[4,16,16] — NOT
    # the 16-channel stride-8 shape _empty_latent_14b assumes (note 5: fix the
    # shape constant from the actual model config; do NOT relax the assertion).
    # Derive the latent shape from the model's config.json so init/mask match the
    # post-denoise latent bit-for-bit (apply_inpaint_mask requires exact shape
    # equality). 17 frames 480x832 -> (1, 48, 5, 30, 52) on wan22-ti2v-5b.
    _skip_unless_real_model(SKYREELS_DIR, "skyreels-inpaint")
    import json

    cfg_path = SKYREELS_DIR / "config.json"
    with open(cfg_path) as f:
        scfg = json.load(f)
    z_dim = int(scfg.get("vae_z_dim", 48))
    vae_stride = scfg.get("vae_stride", [4, 16, 16])
    num_frames, height, width = 17, 480, 832
    t_latent = (num_frames - 1) // int(vae_stride[0]) + 1
    h_latent = height // int(vae_stride[1])
    w_latent = width // int(vae_stride[2])
    latent = mx.zeros((1, z_dim, t_latent, h_latent, w_latent))
    # apply_inpaint_mask runs on the 4D squeezed latent (z_dim, t, h, w) inside
    # run_denoise (stage.py:533 squeezes the batch dim after sched.step), so
    # init_latent and inpaint_mask must be 4D to match latents.shape exactly.
    lat_4d = mx.zeros((z_dim, t_latent, h_latent, w_latent))
    init = mx.array(np.linspace(0.5, 1.5, lat_4d.size, dtype=np.float32).reshape(lat_4d.shape))
    mask = mx.ones_like(lat_4d)
    mask[:, 0, :, :] = 0.0  # freeze t_latent=0 slab (4D: z, t, h, w)
    mx.eval(init, mask)

    async def _run():
        eng = VideoGenEngine(str(SKYREELS_DIR))
        await eng.start()
        await eng.load_text_encoder()
        emb = await eng.encode_text("waves crashing on rocks")
        await eng.unload_text_encoder()
        await eng.load_dit()
        out = await eng.denoise(
            latent=latent,
            pos_embed=emb["embed"],
            neg_embed=None,
            steps=1,
            cfg=1.0,
            seed=42,
            num_frames=17,
            control=None,
            inpaint_mask=mask,
            init_latent=init,
        )
        await eng.unload_dit()
        await eng.stop()
        return np.array(out), np.array(init), np.array(mask)

    out, init_arr, mask_arr = asyncio.run(_run())
    assert not np.any(np.isnan(out)), "skyreels inpaint latent has NaN"
    # out is 5D (1, z, t, h, w); mask/init are 4D (z, t, h, w) matching the
    # internal apply_inpaint_mask shape. Squeeze the batch dim for the boolean
    # index so out4d[mask==0] aligns with the 4D mask.
    out4d = out.reshape(init_arr.shape)
    frozen = out4d[mask_arr == 0]
    frozen_init = init_arr[mask_arr == 0]
    reactive = out4d[mask_arr == 1]
    reactive_init = init_arr[mask_arr == 1]
    print(
        f"\n#653 Surface C skyreels inpaint: frozen_equal={np.array_equal(frozen, frozen_init)} "
        f"reactive_diff_mean={float(np.abs(reactive - reactive_init).mean()):.6f}"
    )
    assert np.array_equal(frozen, frozen_init), (
        "skyreels frozen region (mask=0) modified — apply_inpaint_mask not wired in SkyReels loop"
    )
    assert float(np.abs(reactive - reactive_init).mean()) > 0.0, (
        "skyreels reactive region (mask=1) bit-identical to init — denoiser did no work"
    )
