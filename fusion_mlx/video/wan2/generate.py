import argparse
import gc
import logging
import math
import os
import random
import time
from collections.abc import Callable
from pathlib import Path

import mlx.core as mx
import numpy as np
from tqdm import tqdm

logger = logging.getLogger(__name__)

from fusion_mlx.cache.latent_cache import (
    get_image_latent_cache,
    image_latent_key,
)

from .i2v_utils import build_i2v_mask, preprocess_image
from .postprocess import save_video
from .utils import (
    encode_text,
    load_t5_encoder,
    load_vae_decoder,
    load_vae_encoder,
    load_wan_model,
)

try:
    from fusion_mlx.custom_kernels.xfuser_attention import (
        fast_attn_step as _fa_step,
    )
except Exception:  # pragma: no cover - xfuser strategy optional
    from contextlib import nullcontext as _fa_step


class Colors:
    CYAN = "\033[96m"
    BLUE = "\033[94m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    MAGENTA = "\033[95m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    RESET = "\033[0m"


# Backward-compat alias (tests and external code may use the old name)
_build_i2v_mask = build_i2v_mask


def _load_video_frames(
    video_path: str, width: int, height: int, num_frames: int
) -> mx.array:
    import cv2

    cap = cv2.VideoCapture(video_path)
    frames = []
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    indices = np.linspace(0, max(total - 1, 0), num_frames, dtype=int)
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
        frames.append(frame)
    cap.release()
    while len(frames) < num_frames:
        frames.append(
            frames[-1] if frames else np.zeros((height, width, 3), dtype=np.uint8)
        )
    arr = np.stack(frames[:num_frames], axis=0)  # [T, H, W, 3]
    arr = arr.astype(np.float32) / 255.0 * 2.0 - 1.0
    return mx.array(arr.transpose(3, 0, 1, 2))  # [3, T, H, W]


def _load_mask_frames(
    mask_path: str, width: int, height: int, num_frames: int
) -> mx.array:
    import cv2

    cap = cv2.VideoCapture(mask_path)
    frames = []
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total > 1:
        indices = np.linspace(0, max(total - 1, 0), num_frames, dtype=int)
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_NEAREST)
            frames.append(frame)
    else:
        ret, frame = cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_NEAREST)
            frames = [frame] * num_frames
        else:
            frames = [np.zeros((height, width), dtype=np.uint8)] * num_frames
    cap.release()
    while len(frames) < num_frames:
        frames.append(
            frames[-1] if frames else np.zeros((height, width), dtype=np.uint8)
        )
    arr = np.stack(frames[:num_frames], axis=0)  # [T, H, W]
    arr = arr.astype(np.float32) / 255.0  # 0=black=conditioning, 1=white=generation
    return mx.array(arr)  # [T, H, W]


def _load_ref_image(image_path: str, width: int, height: int) -> mx.array:
    from PIL import Image

    img = Image.open(image_path).convert("RGB")
    scale = max(width / img.width, height / img.height)
    img = img.resize(
        (round(img.width * scale), round(img.height * scale)), Image.LANCZOS
    )
    x1, y1 = (img.width - width) // 2, (img.height - height) // 2
    img = img.crop((x1, y1, x1 + width, y1 + height))
    arr = np.array(img, dtype=np.float32) / 255.0 * 2.0 - 1.0
    return mx.array(arr.transpose(2, 0, 1))  # [3, H, W]


def _prepare_vace_control_latents(
    vae_encoder,
    control_video: mx.array,
    control_mask: mx.array,
    reference_images: list[mx.array] | None,
    vae_stride: tuple,
    h_latent: int,
    w_latent: int,
    t_latent: int,
    z_dim: int,
) -> mx.array:
    logger.info(
        "VACE control: video=%s mask=%s refs=%d",
        control_video.shape,
        control_mask.shape,
        len(reference_images or []),
    )
    S = vae_stride[1]  # spatial scale, e.g. 8
    num_frames = control_video.shape[1]

    # mask: [T, H, W] -> [1, T, H, W] for broadcasting with [3, T, H, W]
    mask_exp = control_mask[None, :, :, :]  # [1, T, H, W]
    mask_exp = mx.broadcast_to(mask_exp, control_video.shape)  # [3, T, H, W]

    # inactive = video * (1 - mask)  — conditioning region
    # reactive = video * mask       — generation region (will be replaced by denoising)
    inactive_video = control_video * (1.0 - mask_exp)
    reactive_video = control_video * mask_exp

    # Encode each through VAE encoder -> [1, z_dim, T_lat, H_lat, W_lat]
    inactive_lat = vae_encoder.encode(inactive_video[None])[
        0
    ]  # [z_dim, T_lat, H_lat, W_lat]
    mx.eval(inactive_lat)
    reactive_lat = vae_encoder.encode(reactive_video[None])[0]
    mx.eval(reactive_lat)

    # Video conditioning latents: cat along channel dim -> [32, T_lat, H_lat, W_lat]
    cond_latents = mx.concatenate([inactive_lat, reactive_lat], axis=0)

    # Reference images: each encoded, added as extra TIME frames with zeros padding
    if reference_images:
        ref_frames = []
        for ref_img in reference_images:
            # ref_img: [3, H, W] -> [3, 1, H, W]
            ref_video = ref_img[:, None, :, :]
            ref_lat = vae_encoder.encode(ref_video[None])[0]  # [z_dim, 1, H_lat, W_lat]
            mx.eval(ref_lat)
            ref_zeros = mx.zeros_like(ref_lat)  # [z_dim, 1, H_lat, W_lat]
            ref_frame = mx.concatenate(
                [ref_lat, ref_zeros], axis=0
            )  # [32, 1, H_lat, W_lat]
            ref_frames.append(ref_frame)
        # Prepend reference frames along TIME dimension
        ref_all = mx.concatenate(ref_frames, axis=1)  # [32, num_refs, H_lat, W_lat]
        cond_latents = mx.concatenate([ref_all, cond_latents], axis=1)

    # Prepare patch-level mask: [T, H, W] -> [S*S, T_lat, H_lat, W_lat]
    # Downsample mask to latent resolution
    mask_lat_h = h_latent
    mask_lat_w = w_latent

    # Reshape mask into patches: each SxS patch -> 1 token with S*S channels
    # mask: [T, H, W] -> [T, H_lat, S, W_lat, S] -> [T, H_lat, W_lat, S, S]
    mask_reshaped = control_mask.reshape(num_frames, mask_lat_h, S, mask_lat_w, S)
    mask_reshaped = mask_reshaped.transpose(0, 1, 3, 2, 4)  # [T, H_lat, W_lat, S, S]

    # Pad time to match t_latent (temporal compression)
    # VAE temporal stride = vae_stride[0], e.g. 4
    t_stride = vae_stride[0]
    if mask_reshaped.shape[0] < t_latent:
        pad_t = t_latent - mask_reshaped.shape[0]
        mask_reshaped = mx.concatenate(
            [
                mask_reshaped,
                mx.broadcast_to(
                    mask_reshaped[-1:], (pad_t, mask_lat_h, mask_lat_w, S, S)
                ),
            ],
            axis=0,
        )
    # Also need to handle temporal compression: replicate first frame t_stride times
    # like diffusers: mask with 1 at first position, repeat 4x, then reshape
    # Simplified: average over temporal patches -> [t_latent, H_lat, W_lat, S, S]
    mask_patches = []
    for t_idx in range(t_latent):
        start = t_idx * t_stride if t_idx > 0 else 0
        end = start + t_stride if t_idx > 0 else 1
        chunk = mask_reshaped[start:end]  # [t_stride, H_lat, W_lat, S, S] or [1, ...]
        # Take first frame of chunk (matches causal conv temporal behavior)
        mask_patches.append(chunk[0])  # [H_lat, W_lat, S, S]
    mask_patches = mx.stack(mask_patches, axis=0)  # [t_latent, H_lat, W_lat, S, S]

    # Flatten S*S into channels: [t_latent, H_lat, W_lat, S, S] -> [S*S, t_latent, H_lat, W_lat]
    mask_channels = mask_patches.reshape(t_latent, mask_lat_h, mask_lat_w, S * S)
    mask_channels = mask_channels.transpose(3, 0, 1, 2)  # [S*S, t_latent, H_lat, W_lat]

    # Adjust cond_latents time dimension if we added reference frames
    # cond_latents already has correct t_latent (with refs prepended)
    # mask_channels needs same T as cond_latents channel-0
    cond_t = cond_latents.shape[1]
    if mask_channels.shape[0] != S * S:
        logger.warning(
            "VACE mask channels mismatch: expected %d got %d",
            S * S,
            mask_channels.shape[0],
        )
    if mask_channels.shape[1] < cond_t:
        # Pad mask time dim to match conditioning latents
        pad_len = cond_t - mask_channels.shape[1]
        mask_channels = mx.concatenate(
            [
                mask_channels,
                mx.broadcast_to(
                    mask_channels[:, -1:], (S * S, pad_len, mask_lat_h, mask_lat_w)
                ),
            ],
            axis=1,
        )
    elif mask_channels.shape[1] > cond_t:
        mask_channels = mask_channels[:, :cond_t]

    # Final: cat([video_latents(32), mask(S*S=64)], dim=0) -> [96, T_lat, H_lat, W_lat]
    control = mx.concatenate([cond_latents, mask_channels], axis=0)
    mx.eval(control)

    logger.info("VACE control_hidden_states shape: %s", control.shape)
    return control


def _best_output_size(w, h, dw, dh, max_area):
    ratio = w / h
    ow = (max_area * ratio) ** 0.5
    oh = max_area / ow

    # Option 1: process width first
    ow1 = int(ow // dw * dw)
    oh1 = int(max_area / ow1 // dh * dh)
    ratio1 = ow1 / oh1

    # Option 2: process height first
    oh2 = int(oh // dh * dh)
    ow2 = int(max_area / oh2 // dw * dw)
    ratio2 = ow2 / oh2

    if max(ratio / ratio1, ratio1 / ratio) < max(ratio / ratio2, ratio2 / ratio):
        return ow1, oh1
    return ow2, oh2


# #500: above this self-attn seq the fused (B,H,seq,seq) matrix overflows
# Metal and yields all-NaN latents. Q-chunking (FUSION_WAN2_ATTN_CHUNK)
# sidesteps it by capping the per-op matrix to (B,H,chunk,seq).
WAN2_SAFE_SEQ = 16384


def _auto_enable_attn_chunk(seq_len: int) -> None:
    if seq_len <= WAN2_SAFE_SEQ:
        return
    if os.getenv("FUSION_WAN2_ATTN_CHUNK"):
        return
    os.environ["FUSION_WAN2_ATTN_CHUNK"] = "8192"
    logger.warning(
        "wan2 seq=%d exceeds safe threshold %d; auto-enabling "
        "FUSION_WAN2_ATTN_CHUNK=8192 to avoid self-attn NaN (#500)",
        seq_len,
        WAN2_SAFE_SEQ,
    )


def _resolve_model_file(model_dir: Path, flat_name: str, sub_dir: str) -> Path:
    flat = model_dir / flat_name
    if flat.exists():
        return flat
    sub = model_dir / sub_dir
    if sub.is_dir() or sub.is_symlink():
        safetensors = sorted(sub.glob("*.safetensors"))
        if safetensors:
            # Subdir with config.json (e.g. text_encoder/): return directory
            # for from_pretrained-style loading that reads config.json + safetensors.
            if (sub / "config.json").exists():
                return sub
            # Single-file subdir without config: return the file for mx.load compat.
            # Multi-file subdir: return directory for _load_safetensors.
            if len(safetensors) == 1:
                return safetensors[0]
            return sub
    return flat


def generate_video(
    model_dir: str,
    prompt: str,
    negative_prompt: str | None = None,
    image: str | None = None,
    width: int = 1280,
    height: int = 704,
    num_frames: int = 81,
    steps: int = None,
    guide_scale: str | float | tuple = None,
    shift: float = None,
    seed: int = -1,
    output_path: str = "output.mp4",
    output_format: str = "mp4",
    scheduler: str = "unipc",
    loras: list | None = None,
    loras_high: list | None = None,
    loras_low: list | None = None,
    tiling: str = "auto",
    no_compile: bool = False,
    precomputed_context: tuple | None = None,
    keep_t5: bool = False,
    trim_first_frames: int = 0,
    debug_latents: bool = False,
    on_step_sync: Callable[[int, int], None] | None = None,
    session_id: str | None = None,
    control_hidden_states: list | None = None,
    control_scales: list[float] | None = None,
    control_video: str | None = None,
    control_mask: str | None = None,
    reference_images: list[str] | None = None,
    camera_conditions: str | mx.array | None = None,
):
    import json

    from .config import WanModelConfig
    from .scheduler import (
        FlowDPMPP2MScheduler,
        FlowMatchEulerScheduler,
        FlowUniPCScheduler,
    )

    model_dir = Path(model_dir)

    # Load config from model dir if available, otherwise auto-detect
    config_path = model_dir / "config.json"
    quantization = None
    if config_path.exists():
        with open(config_path) as f:
            config_dict = json.load(f)
        # Extract quantization config (not a model config field)
        quantization = config_dict.pop("quantization", None)
        # Handle tuple fields stored as lists in JSON
        for key in ("patch_size", "vae_stride", "window_size", "sample_guide_scale"):
            if key in config_dict and isinstance(config_dict[key], list):
                config_dict[key] = tuple(config_dict[key])
        config = WanModelConfig(
            **{
                k: v
                for k, v in config_dict.items()
                if k in WanModelConfig.__dataclass_fields__
            }
        )
    else:
        # Auto-detect: dual model files → 2.2, single model → 2.1
        if (model_dir / "low_noise_model.safetensors").exists():
            config = WanModelConfig.wan22_t2v_14b()
        else:
            # Detect 1.3B vs 14B from weight shapes
            model_path = model_dir / "model.safetensors"
            if not model_path.exists() and (model_dir / "dit").is_dir():
                model_path = model_dir / "dit"
            if model_path.exists():
                from .utils import _load_safetensors

                probe = _load_safetensors(model_path)
                for k, v in probe.items():
                    if "patch_embedding_proj.weight" in k:
                        dim = v.shape[0]
                        if dim <= 2048:
                            config = WanModelConfig.wan21_t2v_1_3b()
                        else:
                            config = WanModelConfig.wan21_t2v_14b()
                        break
                else:
                    config = WanModelConfig.wan21_t2v_14b()
                del probe
            else:
                config = WanModelConfig.wan21_t2v_14b()

    is_dual = config.dual_model
    is_i2v = image is not None
    is_vace = config.model_type == "vace"
    has_camera = config.add_control_adapter

    # Validate config against actual weights (handles mismatched config.json)
    if not is_dual:
        model_path = model_dir / "model.safetensors"
        if not model_path.exists() and (model_dir / "dit").is_dir():
            model_path = model_dir / "dit"
        if model_path.exists():
            from .utils import _load_safetensors

            probe = _load_safetensors(model_path)
            for k, v in probe.items():
                if "patch_embedding_proj.weight" in k:
                    actual_dim = v.shape[0]
                    if actual_dim != config.dim:
                        print(
                            f"{Colors.YELLOW}  Config dim={config.dim} doesn't match weights dim={actual_dim}, auto-correcting...{Colors.RESET}"
                        )
                        if actual_dim <= 2048:
                            config = WanModelConfig.wan21_t2v_1_3b()
                        else:
                            config = WanModelConfig.wan21_t2v_14b()
                    break
            del probe

    # Auto-correct Wan2.2 VAE params from stale configs
    if config.in_dim == 48 and config.vae_z_dim != 48:
        print(
            f"{Colors.YELLOW}  Auto-correcting Wan2.2 VAE params (in_dim=48 but vae_z_dim={config.vae_z_dim}){Colors.RESET}"
        )
        config = WanModelConfig(
            **{
                **{
                    f.name: getattr(config, f.name)
                    for f in config.__dataclass_fields__.values()
                },
                "vae_z_dim": 48,
                "vae_stride": (4, 16, 16),
                "sample_fps": 24,
            }
        )

    # Re-derive in_dim from the DiT patch_embedding weight when config.json
    # is stale (e.g. Wan2.1-14B dir holds an i2v checkpoint with in_dim=36
    # but config.json says 32). A wrong in_dim makes the channel-concat y
    # tensor hold the wrong channel count -> addmm shape error at the first
    # DiT block. Issue #456.
    from .utils import correct_in_dim

    config = correct_in_dim(config, model_dir)

    # Apply defaults from config if not overridden
    if steps is None:
        steps = config.sample_steps
    if shift is None:
        shift = config.sample_shift
    if guide_scale is None:
        guide_scale = config.sample_guide_scale

    # Normalize guide_scale
    if isinstance(guide_scale, (int, float)):
        guide_scale = float(guide_scale)
    elif isinstance(guide_scale, str):
        parts = [float(x) for x in guide_scale.split(",")]
        guide_scale = tuple(parts) if len(parts) > 1 else parts[0]

    # Detect CFG-disabled mode (guide_scale=1.0 for all models → skip uncond pass for 2x speedup)
    if isinstance(guide_scale, tuple):
        cfg_disabled = all(gs <= 1.0 for gs in guide_scale)
    else:
        cfg_disabled = guide_scale <= 1.0

    # Validate frame count
    assert (num_frames - 1) % 4 == 0, f"num_frames must be 4n+1, got {num_frames}"

    gen_frames = num_frames
    if trim_first_frames > 0:
        gen_frames = num_frames + trim_first_frames * 4
        print(
            f"{Colors.DIM}  Trim: generating {gen_frames} frames, will discard first {trim_first_frames * 4}{Colors.RESET}"
        )

    version_str = f"Wan{config.model_version}"
    mode_str = "dual-model" if is_dual else "single-model"
    pipeline_str = "Image-to-Video" if is_i2v else "Text-to-Video"
    if is_vace:
        pipeline_str += " + VACE"
    if has_camera:
        pipeline_str += " + Camera"
    # Resolve negative prompt: explicit user value > config default
    # The official Wan2.2 uses a Chinese negative prompt (config.sample_neg_prompt)
    # that prevents oversaturation, artifacts, and comic look. We use it by default.
    # Text cleaning (_clean_text) normalizes fullwidth chars to match official tokenization.
    if negative_prompt is None:
        neg_prompt_resolved = config.sample_neg_prompt
    else:
        neg_prompt_resolved = negative_prompt
    print(f"{Colors.CYAN}{'=' * 60}")
    print(f"  {version_str} {pipeline_str} Generation (MLX, {mode_str})")
    print(f"{'=' * 60}{Colors.RESET}")
    print(f"{Colors.DIM}  Prompt: {prompt}")
    if is_i2v:
        print(f"  Image: {image}")
    if neg_prompt_resolved and neg_prompt_resolved.strip():
        neg_display = (
            neg_prompt_resolved[:60] + "..."
            if len(neg_prompt_resolved) > 60
            else neg_prompt_resolved
        )
        print(f"  Neg prompt: {neg_display}")
    print(f"  Size: {width}x{height}, Frames: {num_frames}")
    print(
        f"  Steps: {steps}, Guide: {guide_scale}, Shift: {shift}, Solver: {scheduler}"
    )
    if cfg_disabled:
        print("  CFG: disabled (guide_scale≤1 → B=1 fast path, 2x denoising speedup)")
    print(f"{Colors.RESET}")

    # Seed
    if seed < 0:
        seed = random.randint(0, 2**32 - 1)
    mx.random.seed(seed)
    np.random.seed(seed)
    print(f"{Colors.DIM}  Seed: {seed}{Colors.RESET}")

    # Align dimensions to patch_size * vae_stride (required for patchify)
    vae_stride = config.vae_stride
    patch_size = config.patch_size
    align_h = patch_size[1] * vae_stride[1]  # e.g. 2*16=32
    align_w = patch_size[2] * vae_stride[2]
    if height % align_h != 0 or width % align_w != 0:
        old_h, old_w = height, width
        height = (height // align_h) * align_h
        width = (width // align_w) * align_w
        if height == 0:
            height = align_h
        if width == 0:
            width = align_w
        print(
            f"{Colors.DIM}  Aligned {old_w}x{old_h} → {width}x{height} (must be divisible by {align_w}x{align_h}){Colors.RESET}"
        )

    # Enforce max_area constraint (model-specific resolution limit)
    if config.max_area > 0 and height * width > config.max_area:
        old_h, old_w = height, width
        width, height = _best_output_size(
            width, height, align_w, align_h, config.max_area
        )
        print(
            f"{Colors.YELLOW}  ⚠ Resolution {old_w}x{old_h} exceeds model's max area "
            f"({config.max_area:,}px). Adjusted → {width}x{height}{Colors.RESET}"
        )

    # Compute target latent shape
    z_dim = config.vae_z_dim
    t_latent = (gen_frames - 1) // vae_stride[0] + 1
    h_latent = height // vae_stride[1]
    w_latent = width // vae_stride[2]

    # VACE reference-only / reference+control: native Wan VACE prepends reference
    # frames to BOTH the control signal and the denoised latent so the VACE block
    # `ctrl + x` aligns. We extend the denoised latent by num_ref_lat frames and
    # trim them after decode. Reference frames are 1 latent frame each (VAE encodes
    # a single image to T_lat=1). Only count when this is a VACE run with refs.
    num_ref_lat = 0
    if is_vace and reference_images:
        num_ref_lat = len(reference_images)
        t_latent = t_latent + num_ref_lat
        logger.info(
            "VACE: extending denoised latent by %d reference frames -> t_latent=%d",
            num_ref_lat,
            t_latent,
        )
    target_shape = (z_dim, t_latent, h_latent, w_latent)
    # t_latent_gen = latent frames actually generated (excluding reference prefix)
    t_latent_gen = t_latent - num_ref_lat

    # Sequence length for transformer
    seq_len = math.ceil(
        (h_latent * w_latent) / (patch_size[1] * patch_size[2]) * t_latent
    )

    print(f"{Colors.DIM}  Latent shape: {target_shape}")
    print(f"  Sequence length: {seq_len}{Colors.RESET}")

    # #500: large self-attn seq overflows Metal on a single (B,H,seq,seq)
    # matrix -> all-NaN latents -> VAE decode zeros NaN -> static video.
    # Auto-enable Q-chunking when seq exceeds the safe threshold so users
    # never need to set FUSION_WAN2_ATTN_CHUNK by hand. A user-set env is
    # honored as-is (even 0 = force-off, on their own risk).
    _auto_enable_attn_chunk(seq_len)

    # Load T5 encoder
    t1 = time.time()

    if precomputed_context is not None:
        context = precomputed_context[0]
        if cfg_disabled:
            context_null = None
        else:
            context_null = precomputed_context[1]
        print(
            f"{Colors.DIM}  Using precomputed text embeddings (cfg_disabled={cfg_disabled}){Colors.RESET}"
        )
    else:
        print(f"\n{Colors.BLUE}Loading T5 encoder...{Colors.RESET}")
        t5_path = _resolve_model_file(
            model_dir, "t5_encoder.safetensors", "text_encoder"
        )
        t5_encoder = load_t5_encoder(t5_path, config)

        # Load tokenizer
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained("google/umt5-xxl")

        # Encode prompts
        print(f"{Colors.BLUE}Encoding text...{Colors.RESET}")
        context = encode_text(t5_encoder, tokenizer, prompt, config.text_len)
        if cfg_disabled:
            context_null = None
            mx.eval(context)
        else:
            context_null = encode_text(
                t5_encoder, tokenizer, neg_prompt_resolved, config.text_len
            )
            mx.eval(context, context_null)

        if not keep_t5:
            del t5_encoder
            gc.collect()
            mx.clear_cache()
    print(f"{Colors.DIM}  T5 encoding: {time.time() - t1:.1f}s{Colors.RESET}")

    # I2V: encode image to latent space
    z_img = None
    i2v_mask = None
    i2v_mask_tokens = None
    y_i2v = None
    is_i2v_channel_concat = is_i2v and config.model_type == "i2v"
    is_i2v_mask_blend = is_i2v and config.model_type != "i2v"
    if is_i2v:
        print(f"\n{Colors.BLUE}Encoding input image...{Colors.RESET}")
        t_img = time.time()

        vae_path = _resolve_model_file(model_dir, "vae.safetensors", "vae")

        # Phase-2: check session tail cache first (reuse previous shot's
        # denoised tail-frame latent as this shot's first-frame conditioning).
        from fusion_mlx.cache.latent_cache import get_session_tail

        _session_tail_reused = False
        if session_id is not None and is_i2v_mask_blend:
            tail_latent = get_session_tail(session_id, str(model_dir))
            if tail_latent is not None:
                z_img = tail_latent
                _session_tail_reused = True
                logger.info(
                    "session tail reused as first-frame latent (skip VAE encode)"
                )

        if _session_tail_reused:
            i2v_mask, i2v_mask_tokens = build_i2v_mask(target_shape, config.patch_size)
        elif is_i2v_channel_concat:
            # I2V-14B: encode full video (first frame = image, rest = zeros)
            # and construct y tensor with mask + encoded latents
            from PIL import Image

            img = Image.open(image).convert("RGB")
            scale = max(width / img.width, height / img.height)
            img = img.resize(
                (round(img.width * scale), round(img.height * scale)), Image.LANCZOS
            )
            x1, y1 = (img.width - width) // 2, (img.height - height) // 2
            img = img.crop((x1, y1, x1 + width, y1 + height))
            img_arr = mx.array(
                np.array(img, dtype=np.float32) / 255.0 * 2.0 - 1.0
            )  # [H, W, 3]
            img_chw = img_arr.transpose(2, 0, 1)  # [3, H, W]

            # Build video: first frame = image, rest = zeros -> [3, F, H, W]
            # Chunked encoding processes 1-frame + 4-frame chunks with temporal caching
            video = mx.concatenate(
                [
                    img_chw[:, None, :, :],
                    mx.zeros((3, num_frames - 1, height, width)),
                ],
                axis=1,
            )

            # Encode through Wan2.1 VAE -> [1, z_dim, T_lat, H_lat, W_lat]
            vae_enc = load_vae_encoder(vae_path, config)
            z_video = vae_enc.encode(video[None])  # [1, 16, T_lat, H_lat, W_lat]
            mx.eval(z_video)
            z_video = z_video[0]  # [16, T_lat, H_lat, W_lat]

            # Build mask: 1 for first frame, 0 for rest -> rearrange to [4, T_lat, H, W]
            msk = mx.ones((1, num_frames, h_latent, w_latent))
            msk = mx.concatenate(
                [msk[:, :1], mx.zeros((1, num_frames - 1, h_latent, w_latent))], axis=1
            )
            # Repeat first frame 4x, concat rest: [1, 4 + (F-1), H_lat, W_lat]
            msk = mx.concatenate(
                [
                    mx.repeat(msk[:, :1], 4, axis=1),
                    msk[:, 1:],
                ],
                axis=1,
            )
            # Reshape to [1, T_lat, 4, H_lat, W_lat] then transpose -> [4, T_lat, H_lat, W_lat]
            msk = msk.reshape(1, msk.shape[1] // 4, 4, h_latent, w_latent)
            msk = msk.transpose(0, 2, 1, 3, 4)[0]  # [4, T_lat, H_lat, W_lat]

            # Channel count mirrors upstream WAN21.concat_cond (model_base.py:1594):
            #   extra_channels = in_dim - vae_z_dim.
            # Wan2.2-14B (in_dim=36): extra=20 == z_dim(16)+4 -> y=[mask(4), video(16)]=20ch.
            # Wan2.1-14B / Fun-Camera-1.3B (in_dim=32): extra=16 == z_dim -> y=video only (16ch),
            #   the 4-ch mask path is NOT taken (extra != z_dim+4). Issue #456.
            extra_channels = config.in_dim - config.vae_z_dim
            if extra_channels == config.vae_z_dim + 4:
                y_i2v = mx.concatenate(
                    [msk, z_video], axis=0
                )  # [20, T_lat, H_lat, W_lat]
            else:
                y_i2v = z_video  # [16, T_lat, H_lat, W_lat]
                logger.info(
                    "i2v channel-concat: in_dim=%d extra=%d -> video-only y (%d ch, no mask)",
                    config.in_dim,
                    extra_channels,
                    y_i2v.shape[0],
                )
            mx.eval(y_i2v)

            del vae_enc, img_arr, img_chw, video, z_video, msk
        else:
            # TI2V-5B: encode single image, blend with noise via mask
            # UMA Radix Latent cache (#2 Phase-1): repeat I2V requests with
            # the same image+resolution reuse the cached VAE latent and skip
            # the VAE encoder load + forward entirely (zero-copy on UMA).
            img_tensor = preprocess_image(
                image, width, height
            )  # [1,3,1,H,W] channels-first
            is_wan22_vae = config.vae_z_dim == 48
            if is_wan22_vae:
                # Wan2.2 VAE _patchify expects channels-last [B,T,H,W,C];
                # preprocess_image returns channels-first [B,C,T,H,W] (Wan2.1 format).
                # Without this the encode crashes: "Cannot reshape array of size N
                # into shape (1,3,0,2,...)" (Hfull//patch=0). Mirrors the decode
                # path which transposes latents to channels-last before vae22.decode.
                img_tensor = img_tensor.transpose(0, 2, 3, 4, 1)  # [1,1,H,W,3]
                print(
                    f"{Colors.DIM}  wan2.2 VAE: img_tensor -> channels-last "
                    f"{tuple(img_tensor.shape)} for encode{Colors.RESET}"
                )
            mx.eval(img_tensor)

            latent_cache = get_image_latent_cache(str(model_dir))
            z_img = None
            cache_key = None
            if latent_cache is not None:
                cache_key = image_latent_key(
                    str(model_dir), image, height, width, mx.float32
                )
                z_img = latent_cache.get(cache_key)
                if z_img is not None:
                    print(
                        f"{Colors.DIM}  latent cache hit: "
                        f"{height}x{width} ({cache_key}){Colors.RESET}"
                    )
            if z_img is None:
                vae_enc = load_vae_encoder(vae_path, config)
                z_img = vae_enc.encode(img_tensor)
                # Wan2.1 VAE returns channels-first [B,C,T,H,W]; Wan2.2 VAE returns
                # channels-last [B,T,H,W,C] -> transpose to channels-first so the
                # mask_blend broadcast ([C,T,H,W]) and noise add work. (For Wan2.1
                # do NOT transpose: the old transpose(3,0,1,2) assumed channels-last
                # output and produced [H,C,T,W]=(64,16,1,64), breaking broadcast.)
                if is_wan22_vae:
                    z_img = z_img.transpose(0, 4, 1, 2, 3)  # [1,z_dim,1,H,W]
                mx.eval(z_img)
                if latent_cache is not None:
                    latent_cache.put(cache_key, z_img)
                    print(
                        f"{Colors.DIM}  latent cache miss+insert: "
                        f"{height}x{width} ({cache_key}){Colors.RESET}"
                    )
                del vae_enc
            # encode() returns channels-first [B, C, T, H, W] (Wan2.2 transposed
            # above); squeeze batch -> [C, T, H, W] = [z_dim, 1, H_lat, W_lat].
            z_img = z_img[0]  # [z_dim, 1, H_lat, W_lat]
            i2v_mask, i2v_mask_tokens = build_i2v_mask(target_shape, config.patch_size)

            del img_tensor

        gc.collect()
        mx.clear_cache()
        print(f"{Colors.DIM}  Image encoding: {time.time() - t_img:.1f}s{Colors.RESET}")

    # VACE: encode control video + mask -> control_hidden_states
    if is_vace and control_hidden_states is None:
        if control_video is None and reference_images:
            # Reference-only mode (no control_video, only reference_image).
            # Upstream ComfyUI WanVaceToVideo synthesizes a gray filler video
            # (torch.ones*0.5) with an all-white mask so the VACE control branch
            # still runs and the reference image is encoded/prepended. Without
            # this, reference_images would be dropped and the VACE DiT would run
            # as plain T2V with no reference guidance. Neutral gray in [-1,1] is
            # 0.0; all-white mask (1.0) means "generate full region".
            logger.info(
                "VACE: reference-only mode (no control_video) -> synthesizing "
                "gray filler video + all-white mask so reference_images apply"
            )
            video_frames = mx.zeros((3, gen_frames, height, width), dtype=mx.float32)
            mask_frames = mx.ones((gen_frames, height, width), dtype=mx.float32)
            mx.eval(video_frames, mask_frames)

            print(
                f"\n{Colors.BLUE}Encoding VACE reference-only control...{Colors.RESET}"
            )
            t_vace = time.time()

            vae_path = _resolve_model_file(model_dir, "vae.safetensors", "vae")
            vae_enc = load_vae_encoder(vae_path, config)

            ref_imgs = [_load_ref_image(p, width, height) for p in reference_images]
            mx.eval(*ref_imgs)

            control_hidden_states = [
                _prepare_vace_control_latents(
                    vae_encoder=vae_enc,
                    control_video=video_frames,
                    control_mask=mask_frames,
                    reference_images=ref_imgs,
                    vae_stride=config.vae_stride,
                    h_latent=h_latent,
                    w_latent=w_latent,
                    t_latent=t_latent_gen,
                    z_dim=config.vae_z_dim,
                )
            ]

            del vae_enc, video_frames, mask_frames, ref_imgs
            gc.collect()
            mx.clear_cache()
            print(
                f"{Colors.DIM}  VACE control encoding: {time.time() - t_vace:.1f}s{Colors.RESET}"
            )
        elif control_video is not None:
            print(f"\n{Colors.BLUE}Encoding VACE control video+mask...{Colors.RESET}")
            t_vace = time.time()

            vae_path = _resolve_model_file(model_dir, "vae.safetensors", "vae")
            vae_enc = load_vae_encoder(vae_path, config)

            video_frames = _load_video_frames(control_video, width, height, gen_frames)
            mx.eval(video_frames)

            if control_mask is not None:
                mask_frames = _load_mask_frames(control_mask, width, height, gen_frames)
            else:
                # Auto-generate all-white mask (full generation region)
                # white=1.0 means "generate this region", black=0.0 means "condition on it"
                logger.info(
                    "VACE: no control_mask provided, using all-white default (full generation)"
                )
                mask_frames = mx.ones((gen_frames, height, width), dtype=mx.float32)
            mx.eval(mask_frames)

            ref_imgs = None
            if reference_images:
                ref_imgs = [_load_ref_image(p, width, height) for p in reference_images]
                mx.eval(*ref_imgs)

            control_hidden_states = [
                _prepare_vace_control_latents(
                    vae_encoder=vae_enc,
                    control_video=video_frames,
                    control_mask=mask_frames,
                    reference_images=ref_imgs,
                    vae_stride=config.vae_stride,
                    h_latent=h_latent,
                    w_latent=w_latent,
                    t_latent=t_latent_gen,
                    z_dim=config.vae_z_dim,
                )
            ]

            del vae_enc, video_frames, mask_frames, ref_imgs
            gc.collect()
            mx.clear_cache()
            print(
                f"{Colors.DIM}  VACE control encoding: {time.time() - t_vace:.1f}s{Colors.RESET}"
            )
        else:
            logger.warning(
                "VACE model but no control_video and no reference_images — running without control"
            )

    # Camera: prepare y_camera for Fun-Camera models
    y_camera_arg = None
    if has_camera and camera_conditions is not None:
        # If camera_conditions is a file path, load it as video frames
        if isinstance(camera_conditions, str):
            print(f"\n{Colors.BLUE}Loading camera conditions video...{Colors.RESET}")
            cam_frames = _load_video_frames(
                camera_conditions, width, height, gen_frames
            )
            mx.eval(cam_frames)
            camera_conditions = cam_frames  # [3, T, H, W]
            logger.info(
                "Camera conditions loaded from file, shape: %s", camera_conditions.shape
            )
        # camera_conditions: [C_cam, F, H, W] -> expand to [1, C_cam, F, H, W] for batch
        if camera_conditions.ndim == 4:
            y_camera_arg = [camera_conditions[None]]
        else:
            y_camera_arg = [camera_conditions]
        logger.info("Camera conditions shape: %s", camera_conditions.shape)
    elif has_camera and camera_conditions is None:
        logger.warning(
            "Camera model but no camera_conditions provided — running without camera control"
        )

    # Load transformer models
    print(f"\n{Colors.BLUE}Loading transformer model(s)...{Colors.RESET}")
    if quantization:
        print(
            f"{Colors.DIM}  Using {quantization['bits']}-bit quantized weights (group_size={quantization['group_size']}){Colors.RESET}"
        )
    t2 = time.time()

    # Merge per-model LoRAs with shared LoRAs
    _loras_low = (loras or []) + (loras_low or []) or None
    _loras_high = (loras or []) + (loras_high or []) or None
    _loras_single = loras

    if is_dual:
        low_noise_path = model_dir / "low_noise_model.safetensors"
        high_noise_path = model_dir / "high_noise_model.safetensors"
        low_noise_model = load_wan_model(
            low_noise_path, config, quantization, loras=_loras_low
        )
        high_noise_model = load_wan_model(
            high_noise_path, config, quantization, loras=_loras_high
        )
    else:
        # Support both flat (model.safetensors) and diffusers (dit/ subdir) layouts
        dit_path = model_dir / "model.safetensors"
        if not dit_path.exists() and (model_dir / "dit").is_dir():
            dit_path = model_dir / "dit"
        single_model = load_wan_model(
            dit_path, config, quantization, loras=_loras_single
        )
    print(f"{Colors.DIM}  Models loaded: {time.time() - t2:.1f}s{Colors.RESET}")

    # Precompute text embeddings once (avoids redundant MLP in every step)
    # Each model has its own text_embedding weights, so dual models need separate embeddings
    if cfg_disabled:
        # No CFG: only compute cond embeddings (B=1 forward pass, 2x faster)
        if is_dual:
            context_emb_low = low_noise_model.embed_text([context])
            context_emb_high = high_noise_model.embed_text([context])
            mx.eval(context_emb_low, context_emb_high)
            context_cond_low = context_emb_low[0:1]
            context_cond_high = context_emb_high[0:1]
        else:
            context_emb = single_model.embed_text([context])
            mx.eval(context_emb)
            context_cond = context_emb[0:1]
    else:
        if is_dual:
            context_emb_low = low_noise_model.embed_text([context, context_null])
            context_emb_high = high_noise_model.embed_text([context, context_null])
            mx.eval(context_emb_low, context_emb_high)
            context_cfg_low = mx.concatenate(
                [context_emb_low[0:1], context_emb_low[1:2]], axis=0
            )
            context_cfg_high = mx.concatenate(
                [context_emb_high[0:1], context_emb_high[1:2]], axis=0
            )
        else:
            context_emb = single_model.embed_text([context, context_null])
            mx.eval(context_emb)
            context_cfg = mx.concatenate([context_emb[0:1], context_emb[1:2]], axis=0)

    # Precompute cross-attention K/V caches (constant across all steps)
    if cfg_disabled:
        if is_dual:
            cross_kv_low = low_noise_model.prepare_cross_kv(context_cond_low)
            cross_kv_high = high_noise_model.prepare_cross_kv(context_cond_high)
            mx.eval(cross_kv_low, cross_kv_high)
        else:
            cross_kv = single_model.prepare_cross_kv(context_cond)
            mx.eval(cross_kv)
    else:
        if is_dual:
            cross_kv_low = low_noise_model.prepare_cross_kv(context_cfg_low)
            cross_kv_high = high_noise_model.prepare_cross_kv(context_cfg_high)
            mx.eval(cross_kv_low, cross_kv_high)
        else:
            cross_kv = single_model.prepare_cross_kv(context_cfg)
            mx.eval(cross_kv)

    # Precompute RoPE frequencies (grid sizes are constant across all steps)
    f_grid = t_latent // patch_size[0]
    h_grid = h_latent // patch_size[1]
    w_grid = w_latent // patch_size[2]
    if cfg_disabled:
        rope_grid_sizes = [(f_grid, h_grid, w_grid)]
    else:
        rope_grid_sizes = [(f_grid, h_grid, w_grid), (f_grid, h_grid, w_grid)]
    if is_dual:
        rope_cos_sin_low = low_noise_model.prepare_rope(rope_grid_sizes)
        rope_cos_sin_high = high_noise_model.prepare_rope(rope_grid_sizes)
        mx.eval(rope_cos_sin_low, rope_cos_sin_high)
    else:
        rope_cos_sin = single_model.prepare_rope(rope_grid_sizes)
        mx.eval(rope_cos_sin)

    # Setup scheduler
    _schedulers = {
        "euler": FlowMatchEulerScheduler,
        "dpm++": FlowDPMPP2MScheduler,
        "unipc": FlowUniPCScheduler,
    }
    sched_cls = _schedulers.get(scheduler, FlowUniPCScheduler)
    sched = sched_cls(num_train_timesteps=config.num_train_timesteps)
    sched.set_timesteps(steps, shift=shift)

    # Generate initial noise
    noise = mx.random.normal(target_shape)

    # I2V initialization: TI2V-5B blends image with noise, I2V-14B uses pure noise
    if is_i2v_mask_blend:
        latents = (1.0 - i2v_mask) * z_img + i2v_mask * noise
    else:
        latents = noise

    # Boundary for model switching (dual model only)
    boundary = (config.boundary * config.num_train_timesteps) if is_dual else None

    # Diffusion loop
    print(f"\n{Colors.GREEN}Denoising ({steps} steps)...{Colors.RESET}")
    t3 = time.time()

    # Compile model forward for faster denoising
    if not no_compile:
        models_to_compile = (
            [high_noise_model, low_noise_model] if is_dual else [single_model]
        )
        for m in models_to_compile:
            m._compiled = mx.compile(m)

    # Pre-convert timesteps to Python list to avoid .item() sync each step
    timestep_list = sched.timesteps.tolist()

    for i, t in enumerate(tqdm(range(steps), desc="Diffusion")):
        timestep_val = timestep_list[i]

        # Select model, cached K/V, and precomputed RoPE
        if is_dual:
            if timestep_val >= boundary:
                model = high_noise_model
                kv = cross_kv_high
                rcs = rope_cos_sin_high
            else:
                model = low_noise_model
                kv = cross_kv_low
                rcs = rope_cos_sin_low
        else:
            model = single_model
            kv = cross_kv
            rcs = rope_cos_sin

        # Use compiled forward when available (faster after first trace)
        _call = getattr(model, "_compiled", model)

        if cfg_disabled:
            # No CFG: B=1 forward pass (2x faster than B=2 CFG batch)
            if is_i2v_mask_blend:
                t_tokens = i2v_mask_tokens * timestep_val
                pad_len = seq_len - t_tokens.shape[1]
                if pad_len > 0:
                    t_tokens = mx.concatenate(
                        [t_tokens, mx.full((1, pad_len), timestep_val)], axis=1
                    )
                t_batch = t_tokens  # [1, L]
            else:
                t_batch = mx.array([timestep_val])

            y_arg = [y_i2v] if is_i2v_channel_concat else None

            if is_dual:
                ctx = (
                    context_cond_high if timestep_val >= boundary else context_cond_low
                )
            else:
                ctx = context_cond
            with _fa_step(i):
                preds = _call(
                    [latents],
                    t=t_batch,
                    context=ctx,
                    seq_len=seq_len,
                    cross_kv_caches=kv,
                    y=y_arg,
                    rope_cos_sin=rcs,
                    control_hidden_states=control_hidden_states,
                    control_scales=control_scales,
                    y_camera=y_camera_arg,
                )
            noise_pred = preds[0]
            del preds
        else:
            # CFG: batch cond + uncond into single B=2 forward pass
            if is_dual:
                if isinstance(guide_scale, (int, float)):
                    gs = guide_scale
                else:
                    gs = guide_scale[1] if timestep_val >= boundary else guide_scale[0]
            else:
                gs = (
                    guide_scale
                    if isinstance(guide_scale, (int, float))
                    else guide_scale[0]
                )

            if is_i2v_mask_blend:
                t_tokens = i2v_mask_tokens * timestep_val
                pad_len = seq_len - t_tokens.shape[1]
                if pad_len > 0:
                    t_tokens = mx.concatenate(
                        [t_tokens, mx.full((1, pad_len), timestep_val)], axis=1
                    )
                t_batch = mx.concatenate([t_tokens, t_tokens], axis=0)
            else:
                t_batch = mx.array([timestep_val, timestep_val])

            y_arg = [y_i2v, y_i2v] if is_i2v_channel_concat else None

            ctx = (
                context_cfg
                if not is_dual
                else (context_cfg_high if timestep_val >= boundary else context_cfg_low)
            )
            with _fa_step(i):
                preds = _call(
                    [latents, latents],
                    t=t_batch,
                    context=ctx,
                    seq_len=seq_len,
                    cross_kv_caches=kv,
                    y=y_arg,
                    rope_cos_sin=rcs,
                    control_hidden_states=control_hidden_states,
                    control_scales=control_scales,
                    y_camera=y_camera_arg,
                )
            noise_pred_cond, noise_pred_uncond = preds[0], preds[1]
            noise_pred = noise_pred_uncond + gs * (noise_pred_cond - noise_pred_uncond)
            del noise_pred_cond, noise_pred_uncond, preds

        latents = sched.step(noise_pred[None], timestep_val, latents[None]).squeeze(0)

        # TI2V-5B: re-apply mask to keep first frame frozen
        if is_i2v_mask_blend:
            latents = (1.0 - i2v_mask) * z_img + i2v_mask * latents

        # Release temporaries before eval to free memory for graph execution
        del noise_pred
        mx.eval(latents)

        # Early NaN detection: break if latents go bad
        if i <= 2 or i == steps - 1:
            _lat_np = np.array(latents)
            _nan_c = int(np.isnan(_lat_np).sum())
            if _nan_c > 0:
                logger.warning(
                    "Denoise step %d/%d: %d NaN in latents — aborting",
                    i + 1,
                    steps,
                    _nan_c,
                )
                break
        if debug_latents:
            lat_np = np.array(latents)
            _nan_count = int(np.isnan(lat_np).sum())
            print(
                f"  Step {i}: mean={lat_np.mean():.4f} std={lat_np.std():.4f} "
                f"nan={_nan_count} zero_pct={100 * (lat_np == 0).sum() / lat_np.size:.1f}%"
            )
            if _nan_count > 0:
                print(f"  *** NaN DETECTED at step {i}! Aborting. ***")
                break
        if on_step_sync is not None:
            on_step_sync(i + 1, steps)

    print(f"{Colors.DIM}  Denoising: {time.time() - t3:.1f}s{Colors.RESET}")

    # Phase-2: capture tail-frame latent for multi-shot session reuse
    if session_id is not None:
        from fusion_mlx.cache.latent_cache import put_session_tail

        # Wan2 latents shape: [C, T, H, W] -> tail = [:, -1:, :, :]
        tail = latents[:, -1:, :, :]
        put_session_tail(session_id, str(model_dir), tail)

    # Diagnostic: per-temporal-position latent statistics
    if debug_latents:
        lat_np = np.array(latents)  # [C, T, H, W]
        n_t = lat_np.shape[1]
        print(
            f"\n{Colors.CYAN}  Latent diagnostics (shape {lat_np.shape}):{Colors.RESET}"
        )
        print(
            f"  {'Pos':>4s}  {'Mean':>8s}  {'Std':>8s}  {'Min':>8s}  {'Max':>8s}  {'AbsMean':>8s}"
        )
        for t_pos in range(min(n_t, 8)):
            frame = lat_np[:, t_pos, :, :]
            print(
                f"  {t_pos:4d}  {frame.mean():8.4f}  {frame.std():8.4f}  "
                f"{frame.min():8.4f}  {frame.max():8.4f}  {np.abs(frame).mean():8.4f}"
            )
        if n_t > 8:
            interior = lat_np[:, 4:, :, :]
            print(
                f"  {'4+':>4s}  {interior.mean():8.4f}  {interior.std():8.4f}  "
                f"{interior.min():8.4f}  {interior.max():8.4f}  {np.abs(interior).mean():8.4f}"
            )
        print()

    # Free transformer models and text embeddings
    if is_dual:
        del low_noise_model, high_noise_model, cross_kv_low, cross_kv_high
        if cfg_disabled:
            del context_cond_low, context_cond_high
        else:
            del context_cfg_low, context_cfg_high
    else:
        del single_model, cross_kv
        if cfg_disabled:
            del context_cond
        else:
            del context_cfg
    del model, kv, context
    if context_null is not None:
        del context_null
    gc.collect()
    mx.clear_cache()

    # Load VAE and decode
    print(f"\n{Colors.BLUE}Decoding with VAE...{Colors.RESET}")
    t4 = time.time()
    vae_path = _resolve_model_file(model_dir, "vae.safetensors", "vae")
    vae = load_vae_decoder(vae_path, config)

    is_wan22_vae = config.vae_z_dim == 48

    # VACE: drop the leading reference latent frames we prepended for ctrl+x
    # alignment. The denoised reference-prefix frames are not part of the output;
    # reference guidance was applied through the control branch. Mirrors native
    # TrimVideoLatent (s1[:, :, trim_amount:]).
    if num_ref_lat > 0:
        logger.info(
            "VACE: trimming %d leading reference latent frames before decode "
            "(latents T=%d -> %d)",
            num_ref_lat,
            latents.shape[1],
            latents.shape[1] - num_ref_lat,
        )
        latents = latents[:, num_ref_lat:, :]
        mx.eval(latents)

    # Temporal extend: prepend reflected latent frames to the VAE input so that
    # the CausalConv3d zero-padding artifacts fall on the prefix (which we crop).
    # This gives the first real frame a full temporal receptive field of real data.
    # Select tiling configuration
    from ..ltx2.video_vae.tiling import TilingConfig

    if tiling == "none":
        tiling_config = None
    elif tiling == "auto":
        tiling_config = TilingConfig.auto(height, width, num_frames)
    elif tiling == "default":
        tiling_config = TilingConfig.default()
    elif tiling == "aggressive":
        tiling_config = TilingConfig.aggressive()
    elif tiling == "conservative":
        tiling_config = TilingConfig.conservative()
    elif tiling == "spatial":
        tiling_config = TilingConfig.spatial_only()
    elif tiling == "temporal":
        tiling_config = TilingConfig.temporal_only()
    else:
        print(
            f"{Colors.YELLOW}  Unknown tiling mode '{tiling}', using auto{Colors.RESET}"
        )
        tiling_config = TilingConfig.auto(height, width, num_frames)

    if tiling_config is not None:
        spatial_info = (
            f"{tiling_config.spatial_config.tile_size_in_pixels}px"
            if tiling_config.spatial_config
            else "none"
        )
        temporal_info = (
            f"{tiling_config.temporal_config.tile_size_in_frames}f"
            if tiling_config.temporal_config
            else "none"
        )
        print(
            f"{Colors.DIM}  Tiling ({tiling}): spatial={spatial_info}, temporal={temporal_info}{Colors.RESET}"
        )

    if is_wan22_vae:
        from .vae22 import denormalize_latents

        # latents: [C, T, H, W] → [1, T, H, W, C] (channels-last for Wan2.2 VAE)
        z = latents.transpose(1, 2, 3, 0)[None]
        z = denormalize_latents(z)
        if tiling_config is not None:
            video = vae.decode_tiled(z, tiling_config)
        else:
            video = vae(z)
        mx.eval(video)
        print(f"{Colors.DIM}  VAE decode: {time.time() - t4:.1f}s{Colors.RESET}")

        video = np.array(video[0])  # [T', H', W', 3]
        video = (video + 1.0) / 2.0
        nan_count = int(np.isnan(video).sum())
        if nan_count > 0:
            # #500: fail visibly instead of silently zeroing NaN -> static
            # video. NaN here means the denoised latents already went bad
            # (overflow/freeze). Tell the user and abort, don't emit a
            # misleading "successful" static clip.
            raise RuntimeError(
                f"Wan2 VAE decode produced {nan_count} NaN pixels — "
                f"denoised latents are corrupt (self-attn overflow at this "
                f"resolution/frame-count). Reduce resolution or frames. "
                f"seq={seq_len} chunk={os.getenv('FUSION_WAN2_ATTN_CHUNK','0')}"
            )
        video = np.clip(video * 255.0, 0, 255).astype(np.uint8)
    else:
        if tiling_config is not None:
            video = vae.decode_tiled(latents[None], tiling_config)
        else:
            video = vae.decode(latents[None])
        mx.eval(video)
        print(f"{Colors.DIM}  VAE decode: {time.time() - t4:.1f}s{Colors.RESET}")

        video = np.array(video[0])  # [3, T', H, W]
        video = (video + 1.0) / 2.0
        nan_count = int(np.isnan(video).sum())
        if nan_count > 0:
            # #500: fail visibly (see wan22 branch above for rationale)
            raise RuntimeError(
                f"Wan2 VAE decode produced {nan_count} NaN pixels — "
                f"denoised latents are corrupt (self-attn overflow at this "
                f"resolution/frame-count). Reduce resolution or frames. "
                f"seq={seq_len} chunk={os.getenv('FUSION_WAN2_ATTN_CHUNK','0')}"
            )
        video = np.clip(video * 255.0, 0, 255).astype(np.uint8)
        video = video.transpose(1, 2, 3, 0)  # [T, H, W, 3]

    # Trim first N temporal chunks if requested (avoids first-frame artifacts)
    if trim_first_frames > 0:
        trim_pixels = trim_first_frames * 4
        video = video[trim_pixels:]
        print(
            f"{Colors.DIM}  Trimmed first {trim_pixels} frames ({video.shape[0]} remaining){Colors.RESET}"
        )

    if output_format == "raw":
        print(
            f"{Colors.DIM}  Raw output: returning {video.shape} uint8 frames{Colors.RESET}"
        )
        print(f"{Colors.DIM}  Total time: {time.time() - t1:.1f}s{Colors.RESET}")
        return video

    save_video(video, output_path, fps=config.sample_fps)
    print(f"\n{Colors.GREEN}✓ Video saved to {output_path}{Colors.RESET}")
    print(f"{Colors.DIM}  Total time: {time.time() - t1:.1f}s{Colors.RESET}")


def main():
    parser = argparse.ArgumentParser(description="Wan Text-to-Video Generation (MLX)")
    parser.add_argument(
        "--model-dir",
        type=str,
        required=True,
        help="Path to converted MLX model directory",
    )
    parser.add_argument("--prompt", type=str, required=True, help="Text prompt")
    parser.add_argument(
        "--image",
        type=str,
        default=None,
        help="Path to input image for I2V (omit for T2V mode)",
    )
    parser.add_argument(
        "--negative-prompt",
        type=str,
        default=None,
        help="Negative prompt for CFG (default: official Chinese prompt from config)",
    )
    parser.add_argument(
        "--no-negative-prompt",
        action="store_true",
        help="Disable negative prompt (use empty string instead of config default)",
    )
    parser.add_argument(
        "--width", type=int, default=1280, help="Video width (default: 1280)"
    )
    parser.add_argument(
        "--height",
        type=int,
        default=704,
        help="Video height (default: 704; 720p models use 704)",
    )
    parser.add_argument(
        "--num-frames", type=int, default=81, help="Number of frames (must be 4n+1)"
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=None,
        help="Number of diffusion steps (default: from config)",
    )
    parser.add_argument(
        "--guide-scale",
        type=str,
        default=None,
        help="Guidance scale: single float or low,high pair",
    )
    parser.add_argument(
        "--shift",
        type=float,
        default=None,
        help="Noise schedule shift (default: from config)",
    )
    parser.add_argument("--seed", type=int, default=-1, help="Random seed")
    parser.add_argument(
        "--output-path", type=str, default="output.mp4", help="Output video path"
    )
    parser.add_argument(
        "--scheduler",
        type=str,
        default="unipc",
        choices=["euler", "dpm++", "unipc"],
        help="Diffusion solver: euler (1st order), dpm++ (2nd order), unipc (2nd order PC, default/official)",
    )
    parser.add_argument(
        "--lora",
        nargs=2,
        action="append",
        metavar=("PATH", "STRENGTH"),
        help="Apply a LoRA to all models (repeatable). Format: --lora path.safetensors 0.8",
    )
    parser.add_argument(
        "--lora-high",
        nargs=2,
        action="append",
        metavar=("PATH", "STRENGTH"),
        help="Apply a LoRA to high-noise model only (dual-model, repeatable)",
    )
    parser.add_argument(
        "--lora-low",
        nargs=2,
        action="append",
        metavar=("PATH", "STRENGTH"),
        help="Apply a LoRA to low-noise model only (dual-model, repeatable)",
    )
    parser.add_argument(
        "--tiling",
        type=str,
        default="auto",
        choices=[
            "auto",
            "none",
            "default",
            "aggressive",
            "conservative",
            "spatial",
            "temporal",
        ],
        help="VAE tiling mode to reduce memory during decoding (default: auto)",
    )
    parser.add_argument(
        "--no-compile",
        action="store_true",
        help="Disable mx.compile on models (for debugging)",
    )
    parser.add_argument(
        "--trim-first-frames",
        type=int,
        default=0,
        metavar="N",
        help="Generate N extra temporal chunks (N×4 frames) and discard them from the start. "
        "Fixes first-frame color/lighting artifacts on 14B models. Try 1 first (4 frames). "
        "Default: 0 (disabled)",
    )
    parser.add_argument(
        "--debug-latents",
        action="store_true",
        help="Print per-temporal-position latent statistics after denoising (diagnostic)",
    )
    args = parser.parse_args()

    # Parse guide scale
    guide_scale = None
    if args.guide_scale is not None:
        parts = [float(x) for x in args.guide_scale.split(",")]
        guide_scale = tuple(parts) if len(parts) > 1 else parts[0]

    # Handle negative prompt: --no-negative-prompt forces empty, otherwise pass through
    neg_prompt = args.negative_prompt
    if args.no_negative_prompt:
        neg_prompt = ""

    # Parse LoRA configs: convert [path, strength_str] → (path, float)
    def _parse_lora_args(lora_list):
        if not lora_list:
            return None
        return [(path, float(strength)) for path, strength in lora_list]

    generate_video(
        model_dir=args.model_dir,
        prompt=args.prompt,
        negative_prompt=neg_prompt,
        image=args.image,
        width=args.width,
        height=args.height,
        num_frames=args.num_frames,
        steps=args.steps,
        guide_scale=guide_scale,
        shift=args.shift,
        seed=args.seed,
        output_path=args.output_path,
        scheduler=args.scheduler,
        loras=_parse_lora_args(args.lora),
        loras_high=_parse_lora_args(args.lora_high),
        loras_low=_parse_lora_args(args.lora_low),
        tiling=args.tiling,
        no_compile=args.no_compile,
        trim_first_frames=args.trim_first_frames,
        debug_latents=args.debug_latents,
    )


if __name__ == "__main__":
    main()
