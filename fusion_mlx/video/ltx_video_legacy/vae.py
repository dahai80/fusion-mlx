# SPDX-License-Identifier: Apache-2.0
# Ported from LTX-Video 0.9.x (ltx_video/models/autoencoders/, MIT licensed).
# Pure-MLX reimplementation of the 0.9.x CausalVideoAutoencoder *decoder* path.
# Decoder-only: encode/quant path is not needed for diffusion sampling.
import glob
import json
import logging
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
from safetensors import safe_open

logger = logging.getLogger(__name__)

OURS_VAE_CONFIG = {
    "_class_name": "CausalVideoAutoencoder",
    "dims": 3,
    "in_channels": 3,
    "out_channels": 3,
    "latent_channels": 128,
    "blocks": [
        ["res_x", 4],
        ["compress_all", 1],
        ["res_x_y", 1],
        ["res_x", 3],
        ["compress_all", 1],
        ["res_x_y", 1],
        ["res_x", 3],
        ["compress_all", 1],
        ["res_x", 3],
        ["res_x", 4],
    ],
    "scaling_factor": 1.0,
    "norm_layer": "pixel_norm",
    "patch_size": 4,
    "latent_log_var": "uniform",
    "use_quant_conv": False,
    "causal_decoder": False,
}

DIFFUSERS_VAE_CONFIG = {
    "_class_name": "AutoencoderKLLTXVideo",
    "_diffusers_version": "0.32.0.dev0",
    "block_out_channels": [128, 256, 512, 512],
    "decoder_causal": False,
    "encoder_causal": True,
    "in_channels": 3,
    "latent_channels": 128,
    "layers_per_block": [4, 3, 3, 3, 4],
    "out_channels": 3,
    "patch_size": 4,
    "patch_size_t": 1,
    "resnet_norm_eps": 1e-06,
    "scaling_factor": 1.0,
    "spatio_temporal_scaling": [True, True, True, False],
}

# Verbatim from ltx_video/utils/diffusers_config_mapping.py (order matters:
# longer/more-specific prefixes precede shorter ones so replace() is unambiguous).
VAE_KEYS_RENAME_DICT = {
    "decoder.up_blocks.3.conv_in": "decoder.up_blocks.7",
    "decoder.up_blocks.3.upsamplers.0": "decoder.up_blocks.8",
    "decoder.up_blocks.3": "decoder.up_blocks.9",
    "decoder.up_blocks.2.upsamplers.0": "decoder.up_blocks.5",
    "decoder.up_blocks.2.conv_in": "decoder.up_blocks.4",
    "decoder.up_blocks.2": "decoder.up_blocks.6",
    "decoder.up_blocks.1.upsamplers.0": "decoder.up_blocks.2",
    "decoder.up_blocks.1": "decoder.up_blocks.3",
    "decoder.up_blocks.0": "decoder.up_blocks.1",
    "decoder.mid_block": "decoder.up_blocks.0",
    "encoder.down_blocks.3": "encoder.down_blocks.8",
    "encoder.down_blocks.2.downsamplers.0": "encoder.down_blocks.7",
    "encoder.down_blocks.2": "encoder.down_blocks.6",
    "encoder.down_blocks.1.downsamplers.0": "encoder.down_blocks.4",
    "encoder.down_blocks.1.conv_out": "encoder.down_blocks.5",
    "encoder.down_blocks.1": "encoder.down_blocks.3",
    "encoder.down_blocks.0.conv_out": "encoder.down_blocks.2",
    "encoder.down_blocks.0.downsamplers.0": "encoder.down_blocks.1",
    "encoder.down_blocks.0": "encoder.down_blocks.0",
    "encoder.mid_block": "encoder.down_blocks.9",
    "conv_shortcut.conv": "conv_shortcut",
    "resnets": "res_blocks",
    "norm3": "norm3.norm",
    "latents_mean": "per_channel_statistics.mean-of-means",
    "latents_std": "per_channel_statistics.std-of-means",
}


def _flatten_keys(tree, prefix=""):
    keys = []
    if isinstance(tree, dict):
        for k, v in tree.items():
            keys.extend(_flatten_keys(v, prefix + k + "."))
    elif isinstance(tree, list):
        for i, v in enumerate(tree):
            keys.extend(_flatten_keys(v, prefix + str(i) + "."))
    else:
        if prefix:
            keys.append(prefix[:-1])
    return keys


def _conv3d(x, weight, bias, kernel_size, causal):
    # x: NCDHW (B,C,D,H,W). MLX conv3d wants NDHWC + weight (out,kD,kH,kW,in).
    kt = kernel_size
    if kt > 1:
        if causal:
            pad = mx.repeat(x[:, :, :1, :, :], kt - 1, axis=2)
            x = mx.concatenate([pad, x], axis=2)
        else:
            n = (kt - 1) // 2
            fp = mx.repeat(x[:, :, :1, :, :], n, axis=2)
            lp = mx.repeat(x[:, :, -1:, :, :], n, axis=2)
            x = mx.concatenate([fp, x, lp], axis=2)
    x = x.transpose(0, 2, 3, 4, 1)
    y = mx.conv3d(
        x,
        weight,
        stride=(1, 1, 1),
        padding=(0, kernel_size // 2, kernel_size // 2),
        dilation=(1, 1, 1),
    )
    if bias is not None:
        y = y + bias
    y = y.transpose(0, 4, 1, 2, 3)
    return y


def _pixel_shuffle_nd(x, p1, p2, p3):
    # NCDHW (B, C*p1*p2*p3, D, H, W) -> (B, C, D*p1, H*p2, W*p3)
    B, Cp, D, H, W = x.shape
    Cc = Cp // (p1 * p2 * p3)
    x = x.reshape(B, Cc, p1, p2, p3, D, H, W)
    x = x.transpose(0, 1, 5, 2, 6, 3, 7, 4)
    x = x.reshape(B, Cc, D * p1, H * p2, W * p3)
    return x


class PixelNorm(nn.Module):
    def __init__(self, eps=1e-8):
        super().__init__()
        self.eps = eps

    def __call__(self, x):
        x32 = x.astype(mx.float32)
        out = x32 / mx.sqrt(mx.mean(x32 * x32, axis=1, keepdims=True) + self.eps)
        return out.astype(x.dtype)


class LayerNorm(nn.Module):
    # Channel-last nn.LayerNorm over the C axis of an NCDHW tensor; exposes a
    # .norm submodule so checkpoint keys are `*.norm3.norm.weight`.
    def __init__(self, dim, eps=1e-6, elementwise_affine=True):
        super().__init__()
        self.norm = nn.LayerNorm(dim, eps=eps, affine=elementwise_affine)

    def __call__(self, x):
        x = x.transpose(0, 2, 3, 4, 1)
        x = self.norm(x)
        x = x.transpose(0, 4, 1, 2, 3)
        return x


class CausalConv3d(nn.Module):
    # make_conv_nd(dims=3, causal=True) -> CausalConv3d with a nested .conv
    # submodule. Causal/non-causal is a forward-time flag (constructor causal
    # is ignored by the reference, which always passes causal=self.causal).
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super().__init__()
        self.kernel_size = kernel_size
        self.conv = nn.Conv3d(
            in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1
        )

    def __call__(self, x, causal=True):
        return _conv3d(x, self.conv.weight, self.conv.bias, self.kernel_size, causal)


class LinearND(nn.Module):
    # make_linear_nd(dims=3) -> plain 1x1x1 conv, weight at .weight (no .conv
    # nesting, matching the OURS-format checkpoint for conv_shortcut).
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.weight = mx.zeros((out_channels, 1, 1, 1, in_channels))
        self.bias = mx.zeros((out_channels,))

    def __call__(self, x):
        return _conv3d(x, self.weight, self.bias, 1, False)


class ResnetBlock3D(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        norm_layer="pixel_norm",
        eps=1e-6,
        spatial_padding_mode="zeros",
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.norm1 = _make_norm(in_channels, norm_layer, eps)
        self.non_linearity = nn.SiLU()
        self.conv1 = CausalConv3d(in_channels, out_channels, kernel_size=3)
        self.norm2 = _make_norm(out_channels, norm_layer, eps)
        self.conv2 = CausalConv3d(out_channels, out_channels, kernel_size=3)
        if in_channels != out_channels:
            self.conv_shortcut = LinearND(in_channels, out_channels)
            self.norm3 = LayerNorm(in_channels, eps=eps, elementwise_affine=True)
        else:
            self.conv_shortcut = None
            self.norm3 = None

    def __call__(self, input_tensor, causal=True):
        hidden = self.norm1(input_tensor)
        hidden = self.non_linearity(hidden)
        hidden = self.conv1(hidden, causal=causal)
        hidden = self.norm2(hidden)
        hidden = self.non_linearity(hidden)
        hidden = self.conv2(hidden, causal=causal)
        if self.norm3 is not None:
            shortcut = self.norm3(input_tensor)
            shortcut = self.conv_shortcut(shortcut)
        else:
            shortcut = input_tensor
        return shortcut + hidden


class UNetMidBlock3D(nn.Module):
    def __init__(
        self,
        in_channels,
        num_layers=1,
        norm_layer="pixel_norm",
        eps=1e-6,
        spatial_padding_mode="zeros",
    ):
        super().__init__()
        self.res_blocks = [
            ResnetBlock3D(
                in_channels=in_channels,
                out_channels=in_channels,
                norm_layer=norm_layer,
                eps=eps,
                spatial_padding_mode=spatial_padding_mode,
            )
            for _ in range(num_layers)
        ]
        self.attention_blocks = None

    def __call__(self, hidden_states, causal=True):
        for resnet in self.res_blocks:
            hidden_states = resnet(hidden_states, causal=causal)
        return hidden_states


class DepthToSpaceUpsample(nn.Module):
    def __init__(
        self,
        in_channels,
        stride=(2, 2, 2),
        residual=False,
        out_channels_reduction_factor=1,
        spatial_padding_mode="zeros",
    ):
        super().__init__()
        self.stride = stride
        self.residual = residual
        self.out_channels_reduction_factor = out_channels_reduction_factor
        out_channels = (
            int(stride[0])
            * int(stride[1])
            * int(stride[2])
            * in_channels
            // out_channels_reduction_factor
        )
        self.conv = CausalConv3d(in_channels, out_channels, kernel_size=3)
        self.p1, self.p2, self.p3 = int(stride[0]), int(stride[1]), int(stride[2])

    def __call__(self, x, causal=True):
        if self.residual:
            x_in = _pixel_shuffle_nd(x, self.p1, self.p2, self.p3)
            num_repeat = (
                self.p1 * self.p2 * self.p3
            ) // self.out_channels_reduction_factor
            x_in = mx.repeat(x_in, num_repeat, axis=1)
            if self.stride[0] == 2:
                x_in = x_in[:, :, 1:, :, :]
        x = self.conv(x, causal=causal)
        x = _pixel_shuffle_nd(x, self.p1, self.p2, self.p3)
        if self.stride[0] == 2:
            x = x[:, :, 1:, :, :]
        if self.residual:
            x = x + x_in
        return x


def _make_norm(channels, norm_layer, eps):
    if norm_layer == "pixel_norm":
        return PixelNorm(eps=1e-8)
    if norm_layer == "layer_norm":
        return LayerNorm(channels, eps=eps, elementwise_affine=True)
    raise ValueError(
        f"unsupported norm_layer={norm_layer} (only pixel_norm/layer_norm)"
    )


class VAEConfig:
    def __init__(
        self,
        dims=3,
        in_channels=3,
        out_channels=3,
        latent_channels=128,
        blocks=None,
        scaling_factor=1.0,
        norm_layer="pixel_norm",
        patch_size=4,
        latent_log_var="uniform",
        use_quant_conv=False,
        causal_decoder=False,
        base_channels=128,
    ):
        self.dims = dims
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.latent_channels = latent_channels
        self.blocks = blocks if blocks is not None else OURS_VAE_CONFIG["blocks"]
        self.scaling_factor = scaling_factor
        self.norm_layer = norm_layer
        self.patch_size = patch_size
        self.latent_log_var = latent_log_var
        self.use_quant_conv = use_quant_conv
        self.causal_decoder = causal_decoder
        self.base_channels = base_channels

    @classmethod
    def from_dict(cls, cfg):
        cls_name = cfg.get("_class_name", "")
        if cls_name == "AutoencoderKLLTXVideo":
            cfg = OURS_VAE_CONFIG
        return cls(
            dims=cfg.get("dims", 3),
            in_channels=cfg.get("in_channels", 3),
            out_channels=cfg.get("out_channels", 3),
            latent_channels=cfg.get("latent_channels", 128),
            blocks=cfg.get("blocks", OURS_VAE_CONFIG["blocks"]),
            scaling_factor=cfg.get("scaling_factor", 1.0),
            norm_layer=cfg.get("norm_layer", "pixel_norm"),
            patch_size=cfg.get("patch_size", 4),
            latent_log_var=cfg.get("latent_log_var", "uniform"),
            use_quant_conv=cfg.get("use_quant_conv", False),
            causal_decoder=cfg.get("causal_decoder", False),
            base_channels=cfg.get("decoder_base_channels", 128),
        )


class Decoder(nn.Module):
    def __init__(self, config: VAEConfig):
        super().__init__()
        self.patch_size = config.patch_size
        self.causal = config.causal_decoder
        self.blocks_desc = config.blocks
        in_channels = config.latent_channels
        out_channels = config.out_channels * config.patch_size**2
        base_channels = config.base_channels
        norm_layer = config.norm_layer
        blocks = config.blocks

        output_channel = base_channels
        for block_name, _ in list(reversed(blocks)):
            if block_name == "res_x_y":
                output_channel = output_channel * 2
            elif block_name.startswith("compress"):
                output_channel = output_channel * 1

        self.conv_in = CausalConv3d(in_channels, output_channel, kernel_size=3)
        self.up_blocks: list[nn.Module] = []

        for block_name, block_params in list(reversed(blocks)):
            input_channel = output_channel
            if isinstance(block_params, int):
                block_params = {"num_layers": block_params}
            if block_name == "res_x":
                block = UNetMidBlock3D(
                    in_channels=input_channel,
                    num_layers=block_params.get("num_layers", 1),
                    norm_layer=norm_layer,
                )
            elif block_name == "res_x_y":
                output_channel = output_channel // block_params.get("multiplier", 2)
                block = ResnetBlock3D(
                    in_channels=input_channel,
                    out_channels=output_channel,
                    norm_layer=norm_layer,
                )
            elif block_name == "compress_all":
                output_channel = output_channel // block_params.get("multiplier", 1)
                block = DepthToSpaceUpsample(
                    in_channels=input_channel,
                    stride=(2, 2, 2),
                    residual=block_params.get("residual", False),
                    out_channels_reduction_factor=block_params.get("multiplier", 1),
                )
            elif block_name == "compress_time":
                block = DepthToSpaceUpsample(
                    in_channels=input_channel,
                    stride=(2, 1, 1),
                    residual=block_params.get("residual", False),
                    out_channels_reduction_factor=block_params.get("multiplier", 1),
                )
            elif block_name == "compress_space":
                block = DepthToSpaceUpsample(
                    in_channels=input_channel,
                    stride=(1, 2, 2),
                    residual=block_params.get("residual", False),
                    out_channels_reduction_factor=block_params.get("multiplier", 1),
                )
            else:
                raise ValueError(f"unknown decoder block: {block_name}")
            self.up_blocks.append(block)

        self.conv_norm_out = _make_norm(output_channel, norm_layer, eps=1e-6)
        self.conv_act = nn.SiLU()
        self.conv_out = CausalConv3d(output_channel, out_channels, kernel_size=3)

    def __call__(self, sample, target_shape=None):
        assert target_shape is not None, "vae: target_shape must be provided"
        sample = self.conv_in(sample, causal=self.causal)
        for up_block in self.up_blocks:
            sample = up_block(sample, causal=self.causal)
        sample = self.conv_norm_out(sample)
        sample = self.conv_act(sample)
        sample = self.conv_out(sample, causal=self.causal)
        sample = _pixel_shuffle_nd(sample, 1, self.patch_size, self.patch_size)
        return sample


def _is_diffusers_format(decoder_keys):
    joined = "\n".join(decoder_keys)
    return "decoder.mid_block" in joined or ".resnets." in joined


def _map_vae_decoder_weights(raw):
    # Keep only decoder.* keys, rename diffusers->OURS if needed, strip prefix.
    decoder_raw = {k: v for k, v in raw.items() if k.startswith("decoder.")}
    if not decoder_raw:
        return {}
    if _is_diffusers_format(list(decoder_raw.keys())):
        renamed = {}
        for k, v in decoder_raw.items():
            nk = k
            for src, dst in VAE_KEYS_RENAME_DICT.items():
                if src in nk:
                    nk = nk.replace(src, dst)
            renamed[nk] = v
        decoder_raw = renamed
    out = {}
    for k, v in decoder_raw.items():
        nk = k[len("decoder.") :] if k.startswith("decoder.") else k
        if v.ndim == 5:
            # PyTorch (out,in,kD,kH,kW) -> MLX (out,kD,kH,kW,in)
            v = v.transpose(0, 2, 3, 4, 1)
        out[nk] = v
    return out


def _audit_weights(model, mapped):
    model_keys = set(_flatten_keys(model.parameters()))
    mapped_keys = set(mapped.keys())
    missing = len(model_keys - mapped_keys)
    unexpected = len(mapped_keys - model_keys)
    if unexpected:
        sample = sorted(mapped_keys - model_keys)[:5]
        logger.warning("vae: unexpected keys (first 5): %s", sample)
    return missing, unexpected


class LTVideoVAE(nn.Module):
    # Decoder-only wrapper. use_quant_conv=False and normalize_latent_channels=
    # False in the 0.9.x config collapse the decode path to decoder(z) directly.
    def __init__(self, config: VAEConfig):
        super().__init__()
        self.config = config
        self.scaling_factor = config.scaling_factor
        self.decoder = Decoder(config)
        self.mean_of_means: mx.array | None = None
        self.std_of_means: mx.array | None = None

    def decode(self, z, target_shape=None):
        if target_shape is None:
            target_shape = z.shape
        return self.decoder(z, target_shape=target_shape)

    def __call__(self, z, target_shape=None):
        return self.decode(z, target_shape=target_shape)

    @classmethod
    def from_pretrained(cls, model_path, dtype=mx.float32) -> "LTVideoVAE":
        t0 = time.time()
        model_path = Path(model_path)
        cfg_file = model_path / "config.json"
        if not cfg_file.exists():
            cfg_file = model_path / "vae" / "config.json"
        if cfg_file.exists():
            cfg = json.loads(cfg_file.read_text())
            config = VAEConfig.from_dict(cfg)
        else:
            logger.info("vae: no config.json, using OURS default config")
            config = VAEConfig.from_dict(OURS_VAE_CONFIG)

        model = cls(config)

        shards = sorted(
            glob.glob(str(model_path / "diffusion_pytorch_model*.safetensors"))
        )
        if not shards:
            shards = sorted(
                glob.glob(
                    str(model_path / "vae" / "diffusion_pytorch_model*.safetensors")
                )
            )
        if not shards:
            raise FileNotFoundError(f"vae: no safetensors in {model_path}")

        logger.info(
            "vae: load path=%s dims=%d latent_ch=%d patch=%d causal=%s shards=%d",
            model_path.name,
            config.dims,
            config.latent_channels,
            config.patch_size,
            config.causal_decoder,
            len(shards),
        )
        raw = {}
        for shard in shards:
            with safe_open(shard, framework="numpy") as f:
                for k in f.keys():  # noqa: SIM118
                    raw[k] = f.get_tensor(k)

        mapped = _map_vae_decoder_weights({k: mx.array(v) for k, v in raw.items()})
        # per-channel statistics are top-level; stash for the future pipeline.
        for stat_key, attr in (
            ("per_channel_statistics.mean-of-means", "mean_of_means"),
            ("per_channel_statistics.std-of-means", "std_of_means"),
            ("latents_mean", "mean_of_means"),
            ("latents_std", "std_of_means"),
        ):
            if stat_key in raw:
                val = mx.array(raw[stat_key]).astype(dtype)
                setattr(model, attr, val)

        pairs = [(k, v.astype(dtype)) for k, v in mapped.items()]
        n_params = sum(int(v.size) for _, v in pairs)
        model.decoder.load_weights(pairs, strict=False)
        mx.eval(model.parameters())
        missing, unexpected = _audit_weights(model.decoder, mapped)
        logger.info(
            "vae: ready params=%.2fM mapped=%d missing=%d unexpected=%d dt=%.2fs",
            n_params / 1e6,
            len(mapped),
            missing,
            unexpected,
            time.time() - t0,
        )
        if missing:
            logger.warning("vae: missing %d weight tensors (init defaults)", missing)
        return model
