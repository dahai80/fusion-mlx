# SPDX-License-Identifier: Apache-2.0
"""ControlNet adapter for video DiT (SkyReels-V3 / Wan2.1/Wan2.2).

Port of TheDenk's WanControlnet (dilated ControlNet for Wan video DiT):
  https://github.com/TheDenk/wan2.1-dilated-controlnet

Architecture (matches real WanControlnet weights):
  1. control_encoder: 3-stage Conv3d stack downscales control image
     (Canny/depth/HED) to latent-space resolution
  2. Encoded control concatenated with hidden_states along channel dim
  3. patch_embedding (shared Conv3d with main DiT) embeds concat
  4. condition_embedder: time + text projections (same as main DiT)
  5. WanTransformerBlock x num_layers (6 for 14B, same block as main DiT)
  6. controlnet_blocks: zero-init Linear per block -> per-block residuals
  7. Strided injection: residuals added to main DiT every `stride` blocks

Weight source: TheDenk/wan2.1-t2v-14b-controlnet-{canny,depth,hed}-v1
               TheDenk/wan2.2-ti2v-5b-controlnet-{canny,depth,hed}-v1
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any

import mlx.core as mx
import mlx.nn as nn

from . import VideoAdapter, register_adapter

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Control encoder: 3-stage Conv3d to compress control image to latent space
# ---------------------------------------------------------------------------


class ControlEncoderStage(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel: int, stride: int, padding: int):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=kernel, stride=stride, padding=padding)
        self.gn = nn.GroupNorm(2, out_ch)

    def __call__(self, x: mx.array) -> mx.array:
        x = self.conv(x)
        x = nn.gelu_approx(x)
        x = self.gn(x)
        return x


class ControlEncoder(nn.Module):
    """3-stage spatial+temporal compression of control image.

    Stage 1: Conv3d stride=(1,D,D) — spatial downscale
    Stage 2: Conv3d stride=(2,1,1) — temporal downscale
    Stage 3: Conv3d stride=(2,1,1) — temporal downscale

    Real WanControlnet uses Conv3d; we approximate with Conv2d per-frame
    + reshape since MLX lacks Conv3d. For single-frame control images
    (Canny/depth from a still), Conv2d is sufficient.
    """

    def __init__(self, in_channels: int = 3, downscale_coef: int = 8):
        super().__init__()
        start_ch = in_channels * (downscale_coef**2)
        mid_ch = start_ch // 2
        end_ch = start_ch // 4

        self.stage1 = ControlEncoderStage(
            in_channels, start_ch,
            kernel=downscale_coef + 1, stride=downscale_coef,
            padding=downscale_coef // 2,
        )
        self.stage2 = ControlEncoderStage(
            start_ch, mid_ch, kernel=3, stride=1, padding=1,
        )
        self.stage3 = ControlEncoderStage(
            mid_ch, end_ch, kernel=3, stride=1, padding=1,
        )

    def __call__(self, x: mx.array) -> mx.array:
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        return x


# ---------------------------------------------------------------------------
# WanControlnetBlock: same as main DiT WanTransformerBlock + zero-init output
# ---------------------------------------------------------------------------


class WanControlnetBlock(nn.Module):
    """One block of the ControlNet DiT — delegates to WanAttentionBlock.

    Reuses the same SkyReelsDiTBlock subclass as the main DiT, ensuring
    weight naming compatibility with TheDenk's WanControlnet safetensors.
    Uses t2v_cross_attn (no k_img/v_img) matching the real ControlNet weights.
    """

    def __init__(
        self,
        dim: int,
        ffn_dim: int,
        num_heads: int,
        qk_norm: bool = True,
        cross_attn_norm: bool = True,
        eps: float = 1e-6,
        text_dim: int | None = None,
    ):
        super().__init__()
        from ..skyreels_v3.blocks import WanAttentionBlock

        self.block = WanAttentionBlock(
            dim=dim,
            ffn_dim=ffn_dim,
            num_heads=num_heads,
            qk_norm=qk_norm,
            cross_attn_norm=cross_attn_norm,
            eps=eps,
            has_temporal=False,
            cross_attn_type="t2v_cross_attn",
            text_dim=text_dim,
        )

    def __call__(
        self,
        x: mx.array,
        e: mx.array,
        seq_lens: list,
        grid_sizes: list,
        freqs: mx.array,
        context: mx.array,
        context_lens: list | None = None,
    ) -> mx.array:
        return self.block(
            x, e, seq_lens, grid_sizes, freqs, context,
            context_lens=context_lens,
        )


# ---------------------------------------------------------------------------
# WanControlnet: real Wan2.1/Wan2.2 ControlNet DiT
# ---------------------------------------------------------------------------


class WanControlnet(nn.Module):
    """Wan2.1/Wan2.2 Dilated ControlNet (matches TheDenk weights).

    Key params from config.json:
      - num_attention_heads=12, attention_head_dim=128 -> inner_dim=1536
      - num_layers=6 (14B stride=4, 5B stride varies)
      - ffn_dim=8960
      - out_proj_dim=5120 (14B) or 3072 (5B)
      - downscale_coef=8 (14B) or 16 (5B)
      - vae_channels=16 (14B) or 48 (5B)
      - patch_size=[1,2,2]
      - freq_dim=256, text_dim=4096
    """

    def __init__(
        self,
        inner_dim: int = 1536,
        ffn_dim: int = 8960,
        num_attention_heads: int = 12,
        num_layers: int = 6,
        in_channels: int = 3,
        vae_channels: int = 16,
        text_dim: int = 4096,
        freq_dim: int = 256,
        out_proj_dim: int = 5120,
        patch_size: tuple = (1, 2, 2),
        downscale_coef: int = 8,
        cross_attn_norm: bool = True,
        qk_norm: bool = True,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.inner_dim = inner_dim
        self.num_layers = num_layers
        self.patch_size = patch_size
        self.vae_channels = vae_channels
        self.downscale_coef = downscale_coef
        self.freq_dim = freq_dim
        self.num_attention_heads = num_attention_heads

        start_ch = in_channels * (downscale_coef**2)
        end_ch = start_ch // 4

        self.control_encoder = ControlEncoder(in_channels, downscale_coef)

        self.patch_embedding = nn.Conv2d(
            vae_channels + end_ch, inner_dim,
            kernel_size=patch_size[1], stride=patch_size[1], padding=0,
        )

        self.text_embedding = nn.Sequential(
            nn.Linear(text_dim, inner_dim),
            nn.GELU(),
            nn.Linear(inner_dim, inner_dim),
        )

        self.time_embedding = nn.Sequential(
            nn.Linear(freq_dim, inner_dim),
            nn.SiLU(),
            nn.Linear(inner_dim, inner_dim),
        )
        self.time_projection = nn.Sequential(
            nn.SiLU(),
            nn.Linear(inner_dim, inner_dim * 6),
        )

        for i in range(num_layers):
            setattr(
                self, f"block_{i}",
                WanControlnetBlock(
                    inner_dim, ffn_dim, num_attention_heads,
                    qk_norm=qk_norm, cross_attn_norm=cross_attn_norm, eps=eps,
                    text_dim=text_dim,
                ),
            )

        for i in range(num_layers):
            proj = nn.Linear(inner_dim, out_proj_dim)
            proj.weight = mx.zeros_like(proj.weight)
            proj.bias = mx.zeros_like(proj.bias)
            setattr(self, f"controlnet_block_{i}", proj)

    def forward(
        self,
        hidden_states: mx.array,
        t: mx.array,
        context: mx.array,
        control_states: mx.array,
        seq_lens: list | None = None,
        grid_sizes: list | None = None,
    ) -> list[mx.array]:
        """Run WanControlnet forward, return per-block residuals.

        Args:
            hidden_states: [B, C_vae, H, W] main DiT latent
            t: [B] timestep
            context: [B, L, text_dim] text context
            control_states: [B, C_ctrl, H_ctrl, W_ctrl] control image
            seq_lens: sequence lengths per sample (for RoPE attention)
            grid_sizes: (F, H, W) grid per sample

        Returns:
            List of num_layers residuals, each [B, L_tokens, out_proj_dim]
        """
        from ..skyreels_v3.common import sinusoidal_embedding_1d

        # Ensure NHWC format for MLX Conv2d.
        # Detect NCHW: if shape[1] matches expected channel count, transpose.
        if control_states.ndim == 4:
            s = control_states.shape
            if s[1] == 3 or (s[1] < s[2] and s[1] < s[3]):
                control_states = control_states.transpose(0, 2, 3, 1)

        ctrl = self.control_encoder(control_states)

        if hidden_states.ndim == 4:
            s = hidden_states.shape
            if s[1] == self.vae_channels or (s[1] < s[2] and s[1] < s[3]):
                hidden_states = hidden_states.transpose(0, 2, 3, 1)

        B = hidden_states.shape[0]
        h, w = ctrl.shape[1], ctrl.shape[2]

        if hidden_states.shape[1] != h or hidden_states.shape[2] != w:
            hidden_states_resized = _spatial_downsample(hidden_states, h, w)
        else:
            hidden_states_resized = hidden_states

        concat = mx.concatenate([hidden_states_resized, ctrl], axis=-1)

        x = self.patch_embedding(concat)
        x = x.reshape(B, self.inner_dim, -1).transpose(0, 2, 1)

        L = x.shape[1]
        if seq_lens is None:
            seq_lens = [L] * B
        if grid_sizes is None:
            pt, ph, pw = self.patch_size
            grid_h = h // ph
            grid_w = w // pw
            grid_sizes = [(1, grid_h, grid_w)] * B

        ctx = self.text_embedding(context)
        t_emb = sinusoidal_embedding_1d(self.freq_dim, t)
        t_emb = self.time_embedding(t_emb)
        e = self.time_projection(t_emb)

        freqs = _compute_rope_freqs(self.inner_dim, self.num_attention_heads, grid_sizes, seq_lens)

        residuals = []
        for i in range(self.num_layers):
            block = getattr(self, f"block_{i}")
            x = block(x, e, seq_lens, grid_sizes, freqs, ctx)
            cn_block = getattr(self, f"controlnet_block_{i}")
            residuals.append(cn_block(x))

        return residuals


def _spatial_downsample(x: mx.array, target_h: int, target_w: int) -> mx.array:
    if x.ndim != 4:
        return x
    B, H, W, C = x.shape
    if H == target_h and W == target_w:
        return x
    step_h = max(1, H // target_h)
    step_w = max(1, W // target_w)
    return x[:, ::step_h, ::step_w, :]


def _compute_rope_freqs(
    dim: int,
    num_heads: int,
    grid_sizes: list,
    seq_lens: list,
) -> mx.array:
    """Compute RoPE frequency table for WanAttentionBlock.

    Reuses rope_params from common.py to produce [max_seq_len, half, 2] format.
    """
    from ..skyreels_v3.common import rope_params

    head_dim = dim // num_heads
    max_seq_len = max(seq_lens) if seq_lens else 1024
    return rope_params(max_seq_len, head_dim)


# ---------------------------------------------------------------------------
# Control image preprocessing utilities
# ---------------------------------------------------------------------------


def preprocess_canny(
    image_path: str, low_threshold: int = 100, high_threshold: int = 200
) -> mx.array:
    import numpy as np
    from PIL import Image

    try:
        import cv2

        img = Image.open(image_path).convert("RGB")
        img_np = np.array(img)
        edges = cv2.Canny(img_np, low_threshold, high_threshold)
        edges = np.stack([edges] * 3, axis=-1).astype(np.float32) / 255.0
        return mx.array(edges[np.newaxis].transpose(0, 3, 1, 2))
    except ImportError:
        logger.warning("OpenCV not installed, falling back to grayscale edge detection")
        img = Image.open(image_path).convert("L")
        from PIL import ImageFilter

        edges = img.filter(ImageFilter.FIND_EDGES)
        arr = np.array(edges).astype(np.float32) / 255.0
        arr = np.stack([arr] * 3, axis=-1)
        return mx.array(arr[np.newaxis].transpose(0, 3, 1, 2))


def preprocess_depth(image_path: str) -> mx.array:
    import numpy as np
    from PIL import Image

    img = Image.open(image_path).convert("RGB")
    arr = np.array(img).astype(np.float32) / 255.0
    arr = arr * 2.0 - 1.0
    return mx.array(arr[np.newaxis].transpose(0, 3, 1, 2))


def preprocess_control_image(
    image_path: str,
    control_type: str = "canny",
) -> mx.array:
    if control_type == "canny":
        return preprocess_canny(image_path)
    elif control_type == "depth":
        return preprocess_depth(image_path)
    else:
        import numpy as np
        from PIL import Image

        img = Image.open(image_path).convert("RGB")
        arr = np.array(img).astype(np.float32) / 255.0
        arr = arr * 2.0 - 1.0
        return mx.array(arr[np.newaxis].transpose(0, 3, 1, 2))


# ---------------------------------------------------------------------------
# Weight remapping (HF WanControlnet -> MLX)
# ---------------------------------------------------------------------------


def remap_wan_controlnet_weights(
    weights: dict,
    num_layers: int = 6,
) -> dict:
    """Remap WanControlnet weights from HF safetensors to MLX naming.

    HF naming (from TheDenk models):
      control_encoder.0.conv.weight, control_encoder.0.gn.weight, ...
      patch_embedding.weight/bias
      condition_embedder.time_embedder.* / text_embedder.*
      blocks.{N}.norm1/norm2/norm3/self_attn/cross_attn/ffn/modulation.*
      controlnet_blocks.{N}.weight/bias
    """
    remapped = {}
    for key, value in weights.items():
        clean = key
        for prefix in ("controlnet.", "model."):
            if clean.startswith(prefix):
                clean = clean[len(prefix):]

        # control_encoder -> control_encoder (Conv3d -> Conv2d conversion)
        # HF: control_encoder.{stage_idx}.{sublayer_idx} where:
        #   sublayer 0 = conv, sublayer 2 = GroupNorm
        # Our: control_encoder.stage{N}.conv / .gn
        if clean.startswith("control_encoder."):
            m = re.match(r"control_encoder\.(\d+)\.(\d+)\.(.*)", clean)
            if m:
                stage_idx = int(m.group(1))
                sub_idx = int(m.group(2))
                param = m.group(3)
                is_conv = sub_idx == 0
                if value.ndim == 5:
                    mid = value.shape[2] // 2
                    value = value[:, :, mid, :, :]  # [out, in, k_h, k_w] (PyTorch)
                # PyTorch Conv2d weight: (out_ch, in_ch, kH, kW)
                # MLX Conv2d weight: (out_ch, kH, kW, in_ch) — transpose
                if is_conv and value.ndim == 4 and param == "weight":
                    value = value.transpose(0, 2, 3, 1)
                stage_name = f"stage{stage_idx + 1}"
                layer_name = "conv" if is_conv else "gn"
                remapped[f"control_encoder.{stage_name}.{layer_name}.{param}"] = value
            else:
                remapped[clean] = value
            continue

        # patch_embedding -> patch_embedding (Conv3d -> Conv2d conversion)
        # HF weight is [out_ch, in_ch, k_t, k_h, k_w] (PyTorch Conv3d)
        # MLX Conv2d weight: [out_ch, k_h, k_w, in_ch] — squeeze k_t then transpose
        if clean.startswith("patch_embedding."):
            if value.ndim == 5 and value.shape[2] == 1:
                value = value.squeeze(2)  # [out, in, k_h, k_w] (PyTorch)
            elif value.ndim == 5:
                logger.warning(
                    "ControlNet: patch_embedding Conv3d kernel_t=%d != 1, using slice [0]",
                    value.shape[2],
                )
                value = value[:, :, 0, :, :]
            # PyTorch -> MLX: (O, I, kH, kW) -> (O, kH, kW, I)
            if value.ndim == 4 and clean.endswith(".weight"):
                value = value.transpose(0, 2, 3, 1)
            remapped[clean] = value
            continue

        # condition_embedder -> time/text embedding
        # MLX Sequential uses layers.{idx} naming:
        #   text_embedding: Sequential(Linear, GELU, Linear) -> layers.0, layers.2
        #   time_embedding: Sequential(Linear, SiLU, Linear) -> layers.0, layers.2
        #   time_projection: Sequential(SiLU, Linear) -> layers.1
        if clean.startswith("condition_embedder."):
            rest = clean[len("condition_embedder."):]
            if rest.startswith("time_embedder."):
                sub = rest[len("time_embedder."):]
                sub = sub.replace("linear_1.", "layers.0.").replace("linear_2.", "layers.2.")
                remapped[f"time_embedding.{sub}"] = value
            elif rest.startswith("time_proj."):
                sub = rest[len("time_proj."):]
                remapped[f"time_projection.layers.1.{sub}"] = value
            elif rest.startswith("text_embedder."):
                sub = rest[len("text_embedder."):]
                sub = sub.replace("linear_1.", "layers.0.").replace("linear_2.", "layers.2.")
                remapped[f"text_embedding.{sub}"] = value
            else:
                pass  # skip unused (image proj etc.)
            continue

        # blocks.{N}.* -> block_{N}.block.*
        # WanControlnetBlock wraps WanAttentionBlock as self.block
        m = re.match(r"blocks\.(\d+)\.(.*)", clean)
        if m:
            idx, rest = m.group(1), m.group(2)
            mapped = _remap_block_keys(rest)
            remapped[f"block_{idx}.block.{mapped}"] = value
            continue

        # controlnet_blocks.{N}.* -> controlnet_block_{N}.*
        m = re.match(r"controlnet_blocks\.(\d+)\.(.*)", clean)
        if m:
            idx, param = m.group(1), m.group(2)
            remapped[f"controlnet_block_{idx}.{param}"] = value
            continue

        # rope / norm_out / proj_out — skip (not needed for ControlNet forward)
        if clean.startswith(("rope.", "norm_out.", "proj_out.", "head.")):
            continue

        remapped[clean] = value

    return remapped


def _remap_block_keys(rest: str) -> str:
    """Remap HF diffusers WanTransformerBlock naming to our WanAttentionBlock.

    HF uses:
      attn1 = self-attention: to_q, to_k, to_v, to_out.0, norm_q, norm_k
      attn2 = cross-attention: same
      ffn.net.0.proj = fc1, ffn.net.2 = fc2
      scale_shift_table = modulation

    Our SkyReelsDiTBlock uses:
      self_attn.q, self_attn.k, self_attn.v, self_attn.o, self_attn.norm_q, self_attn.norm_k
      cross_attn.q, ... (same pattern)
      ffn.fc1, ffn.fc2
      modulation
    """
    mapping = {}
    for attn_src, attn_dst in [("attn1", "self_attn"), ("attn2", "cross_attn")]:
        for param_src, param_dst in [
            ("to_q.weight", "q.weight"),
            ("to_q.bias", "q.bias"),
            ("to_k.weight", "k.weight"),
            ("to_k.bias", "k.bias"),
            ("to_v.weight", "v.weight"),
            ("to_v.bias", "v.bias"),
            ("to_out.0.weight", "o.weight"),
            ("to_out.0.bias", "o.bias"),
            ("norm_q.weight", "norm_q.weight"),
            ("norm_k.weight", "norm_k.weight"),
        ]:
            mapping[f"{attn_src}.{param_src}"] = f"{attn_dst}.{param_dst}"

    mapping["ffn.net.0.proj.weight"] = "ffn.fc1.weight"
    mapping["ffn.net.0.proj.bias"] = "ffn.fc1.bias"
    mapping["ffn.net.2.weight"] = "ffn.fc2.weight"
    mapping["ffn.net.2.bias"] = "ffn.fc2.bias"
    mapping["norm2.weight"] = "norm2.weight"
    mapping["norm2.bias"] = "norm2.bias"
    mapping["scale_shift_table"] = "modulation"

    return mapping.get(rest, rest)


# Legacy remap for old SD-UNet ControlNet weights (kept for backward compat)
def remap_controlnet_weights(weights: dict, num_layers: int = 20) -> dict:
    return remap_wan_controlnet_weights(weights, num_layers)


# ---------------------------------------------------------------------------
# ControlNet: main adapter class
# ---------------------------------------------------------------------------


@register_adapter("controlnet")
class ControlNet(VideoAdapter):
    """ControlNet: structural guidance via parallel DiT + strided residual injection.

    Architecture matches TheDenk's WanControlnet:
      1. Control image (Canny/depth/HED) -> control_encoder -> latent-space features
      2. Control features concatenated with main DiT hidden states
      3. Patch embedding -> WanTransformerBlock x num_layers
      4. Zero-init output projection per block -> per-block residuals
      5. Strided injection: residuals added to main DiT every `stride` blocks

    Integration with main DiT (_run_blocks):
      - _run_blocks loops over main DiT blocks and adds controlnet_residuals[i//stride]
        when idx % stride == 0 and i//stride < len(residuals)
    """

    name = "controlnet"

    # Model configs for known ControlNet variants
    MODEL_CONFIGS = {
        "wan2.1-t2v-14b": {
            "inner_dim": 1536,
            "ffn_dim": 8960,
            "num_attention_heads": 12,
            "num_layers": 6,
            "out_proj_dim": 5120,
            "downscale_coef": 8,
            "vae_channels": 16,
            "stride": 4,
        },
        "wan2.2-ti2v-5b": {
            "inner_dim": 1536,
            "ffn_dim": 8960,
            "num_attention_heads": 12,
            "num_layers": 6,
            "out_proj_dim": 3072,
            "downscale_coef": 16,
            "vae_channels": 48,
            "stride": 4,
        },
    }

    DEFAULT_REPO_MAP = {
        "canny": "TheDenk/wan2.1-t2v-14b-controlnet-canny-v1",
        "depth": "TheDenk/wan2.1-t2v-14b-controlnet-depth-v1",
        "hed": "TheDenk/wan2.1-t2v-14b-controlnet-hed-v1",
    }

    def __init__(
        self,
        *,
        scale: float = 1.0,
        image: str | None = None,
        config: dict | None = None,
    ):
        self.scale = scale
        self.image = image
        self.config = config or {}
        self.control_type = self.config.get("control_type", "canny")
        self.stride = self.config.get("stride", 4)
        self.model_variant = self.config.get("model_variant", "wan2.1-t2v-14b")

        model_cfg = self.MODEL_CONFIGS.get(
            self.model_variant, self.MODEL_CONFIGS["wan2.1-t2v-14b"]
        )
        self._model_cfg = {**model_cfg, **self.config}

        self.dit: WanControlnet | None = None
        self._loaded = False
        self._residuals: list[mx.array] | None = None

    def load(self, model_path: str | None = None) -> None:
        if self._loaded:
            logger.debug("ControlNet: already loaded, skipping")
            return

        cfg = self._model_cfg
        self.dit = WanControlnet(
            inner_dim=cfg.get("inner_dim", 1536),
            ffn_dim=cfg.get("ffn_dim", 8960),
            num_attention_heads=cfg.get("num_attention_heads", 12),
            num_layers=cfg.get("num_layers", 6),
            in_channels=cfg.get("in_channels", 3),
            vae_channels=cfg.get("vae_channels", 16),
            text_dim=cfg.get("text_dim", 4096),
            freq_dim=cfg.get("freq_dim", 256),
            out_proj_dim=cfg.get("out_proj_dim", 5120),
            patch_size=tuple(cfg.get("patch_size", [1, 2, 2])),
            downscale_coef=cfg.get("downscale_coef", 8),
            cross_attn_norm=cfg.get("cross_attn_norm", True),
            qk_norm=cfg.get("qk_norm", True),
            eps=cfg.get("eps", 1e-6),
        )
        self.stride = cfg.get("stride", 4)

        logger.info(
            "ControlNet: created WanControlnet (inner_dim=%d ffn=%d layers=%d heads=%d out_proj=%d stride=%d)",
            cfg.get("inner_dim", 1536),
            cfg.get("ffn_dim", 8960),
            cfg.get("num_layers", 6),
            cfg.get("num_attention_heads", 12),
            cfg.get("out_proj_dim", 5120),
            self.stride,
        )

        loaded = False
        if model_path is not None:
            loaded = self._load_weights(model_path)
        if not loaded:
            loaded = self._load_from_hf_cache()
        if not loaded:
            logger.warning(
                "ControlNet: weights not loaded, random init (zero_init output ensures identity)"
            )

        self._loaded = True
        logger.info(
            "ControlNet: loaded (weights=%s scale=%.2f type=%s stride=%d)",
            loaded,
            self.scale,
            self.control_type,
            self.stride,
        )

    def _load_weights(self, model_path: str) -> bool:
        import glob

        path = Path(model_path)
        search_dirs = [path]
        for subdir in ("controlnet", "control_net", ""):
            candidate = path / subdir
            if candidate.is_dir():
                search_dirs.append(candidate)

        for search_dir in search_dirs:
            safetensors = sorted(glob.glob(str(search_dir / "*.safetensors")))
            if not safetensors:
                continue
            try:
                raw_weights = {}
                for wf in safetensors:
                    raw_weights.update(mx.load(wf))
                remapped = remap_wan_controlnet_weights(
                    raw_weights, self._model_cfg.get("num_layers", 6)
                )
                if remapped:
                    self.dit.load_weights(list(remapped.items()), strict=False)
                    logger.info(
                        "ControlNet: weights loaded from %s (%d params)",
                        search_dir,
                        len(remapped),
                    )
                    return True
            except Exception as exc:
                logger.warning(
                    "ControlNet: failed to load from %s: %s", search_dir, exc
                )
        return False

    def _load_from_hf_cache(self) -> bool:
        try:
            import os
            from huggingface_hub import hf_hub_download

            cn_repo = self.config.get(
                "controlnet_repo",
                self.DEFAULT_REPO_MAP.get(self.control_type, "TheDenk/wan2.1-t2v-14b-controlnet-canny-v1"),
            )
            logger.info("ControlNet: trying HF cache: %s", cn_repo)

            # Use hf-mirror.com for downloads
            old_endpoint = os.environ.get("HF_ENDPOINT")
            if not old_endpoint:
                os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

            sf = None
            for fname in ("diffusion_pytorch_model.safetensors", "model.safetensors"):
                try:
                    sf = hf_hub_download(cn_repo, fname)
                    break
                except Exception:
                    continue

            # Restore env
            if old_endpoint is None and "HF_ENDPOINT" in os.environ:
                del os.environ["HF_ENDPOINT"]

            if sf is None:
                logger.warning("ControlNet: no safetensors found for %s", cn_repo)
                return False
            raw = mx.load(sf)
            remapped = remap_wan_controlnet_weights(
                raw, self._model_cfg.get("num_layers", 6)
            )
            if remapped:
                self.dit.load_weights(list(remapped.items()), strict=False)
                logger.info(
                    "ControlNet: weights loaded from HF %s (%d params)",
                    cn_repo, len(remapped),
                )
                return True
        except ImportError:
            logger.debug(
                "ControlNet: huggingface_hub not installed, skipping HF download"
            )
        except Exception as exc:
            logger.warning("ControlNet: HF load failed: %s", exc)
        return False

    def unload(self) -> None:
        self.dit = None
        self._residuals = None
        self._loaded = False
        logger.info("ControlNet: unloaded")

    def encode_control(
        self,
        image_path: str | None = None,
        control_type: str | None = None,
    ) -> mx.array | None:
        path = image_path or self.image
        if path is None:
            logger.warning("ControlNet: no control image provided")
            return None

        ctype = control_type or self.control_type
        try:
            return preprocess_control_image(path, ctype)
        except Exception as exc:
            logger.error("ControlNet: control image preprocessing failed: %s", exc)
            return None

    def compute_residuals(
        self,
        hidden_states: mx.array,
        t: mx.array,
        context: mx.array,
        control_states: mx.array,
        seq_lens: list | None = None,
        grid_sizes: list | None = None,
    ) -> list[mx.array] | None:
        """Run WanControlnet forward to compute per-block residuals.

        Args:
            hidden_states: [B, C_vae, H, W] current denoising latent
            t: [B] timestep
            context: [B, L, text_dim] text context
            control_states: [B, C_ctrl, H_ctrl, W_ctrl] preprocessed control image
            seq_lens: sequence lengths per sample
            grid_sizes: (F, H, W) grid per sample

        Returns:
            List of num_layers residuals, each [B, L_tokens, out_proj_dim], or None
        """
        if not self._loaded:
            self.load()

        if self.dit is None:
            logger.warning("ControlNet: DiT not loaded, cannot compute residuals")
            return None

        try:
            residuals = self.dit.forward(
                hidden_states, t, context, control_states,
                seq_lens=seq_lens, grid_sizes=grid_sizes,
            )
            if self.scale != 1.0:
                residuals = [r * self.scale for r in residuals]
            self._residuals = residuals
            logger.debug(
                "ControlNet: computed %d residuals (stride=%d scale=%.2f)",
                len(residuals),
                self.stride,
                self.scale,
            )
            return residuals
        except Exception as exc:
            logger.error("ControlNet: residual computation failed: %s", exc, exc_info=True)
            return None

    def get_residuals(self) -> list[mx.array] | None:
        return self._residuals

    def modify_denoise_step(
        self,
        dit: Any,
        latents: mx.array,
        t: mx.array,
        context: mx.array,
        **kw: Any,
    ) -> mx.array:
        """Pre-compute ControlNet residuals for this denoising step.

        Returns:
            latents unchanged (residuals stored in self._residuals)
        """
        control_image = kw.get("control_image", self.image)
        if control_image is None:
            return latents

        control_latent = self.encode_control(
            control_image,
            control_type=kw.get("control_type", self.control_type),
        )
        if control_latent is None:
            return latents

        residuals = self.compute_residuals(
            latents, t, context, control_latent,
            seq_lens=kw.get("seq_lens"),
            grid_sizes=kw.get("grid_sizes"),
        )
        if residuals is None:
            logger.warning("ControlNet: residual computation failed, step unchanged")

        return latents


__all__ = [
    "ControlNet",
    "WanControlnet",
    "WanControlnetBlock",
    "ControlEncoder",
    "preprocess_control_image",
    "preprocess_canny",
    "preprocess_depth",
    "remap_wan_controlnet_weights",
]
