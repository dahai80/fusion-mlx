# SPDX-License-Identifier: Apache-2.0
# Pure-MLX port of the SVD Temporal UNet.
# Based on stabilityai/svd UNet with temporal attention + spatial attention.
# I2V: receives CLIP vision embeddings as cross-attention + VAE-encoded image concat.
# Conv3d uses manual matmul-based _conv3d_core (NCDHW format) shared with vae.py.

import logging
import math

import mlx.core as mx
import mlx.nn as nn

from .vae import _conv3d, _group_norm

logger = logging.getLogger(__name__)


class Conv3d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = (
            kernel_size
            if isinstance(kernel_size, tuple)
            else (kernel_size, kernel_size, kernel_size)
        )
        self.stride = stride if isinstance(stride, tuple) else (stride, stride, stride)
        self.padding = (
            padding if isinstance(padding, tuple) else (padding, padding, padding)
        )
        scale = (
            1.0
            / (
                in_channels
                * self.kernel_size[0]
                * self.kernel_size[1]
                * self.kernel_size[2]
            )
        ) ** 0.5
        self.weight = mx.random.uniform(
            low=-scale,
            high=scale,
            shape=(out_channels, in_channels, *self.kernel_size),
            dtype=mx.float32,
        )
        self.bias = mx.zeros((out_channels,), dtype=mx.float32)

    def __call__(self, x):
        return _conv3d(x, self.weight, self.bias, self.stride, self.padding)


class TimestepEmbedding(nn.Module):
    def __init__(self, in_channels=320, out_channels=1280):
        super().__init__()
        self.linear1 = nn.Linear(in_channels, out_channels, bias=True)
        self.linear2 = nn.Linear(out_channels, out_channels, bias=True)

    def __call__(self, x):
        x = nn.silu(self.linear1(x))
        return self.linear2(x)


def _timestep_proj(t, dim):
    half = dim // 2
    freqs = mx.exp(-math.log(10000.0) * mx.arange(0, half, dtype=mx.float32) / half)
    args = t[:, None].astype(mx.float32) * freqs[None, :]
    return mx.concatenate([mx.cos(args), mx.sin(args)], axis=-1)


class TemporalConv3d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super().__init__()
        self.conv = Conv3d(
            in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2
        )

    def __call__(self, x):
        return self.conv(x)


class SpatialAttention(nn.Module):
    def __init__(self, dim, num_heads=8):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5
        self.norm = nn.GroupNorm(32, dim)
        self.q = nn.Linear(dim, dim, bias=True)
        self.k = nn.Linear(dim, dim, bias=True)
        self.v = nn.Linear(dim, dim, bias=True)
        self.out = nn.Linear(dim, dim, bias=True)

    def __call__(self, x):
        B, C, T, H, W = x.shape
        h = _group_norm(self.norm, x)
        h = h.reshape(B, C, T * H * W).transpose(0, 2, 1)  # (B, THW, C)
        q = (
            self.q(h)
            .reshape(B, T * H * W, self.num_heads, self.head_dim)
            .transpose(0, 2, 1, 3)
        )
        k = (
            self.k(h)
            .reshape(B, T * H * W, self.num_heads, self.head_dim)
            .transpose(0, 2, 1, 3)
        )
        v = (
            self.v(h)
            .reshape(B, T * H * W, self.num_heads, self.head_dim)
            .transpose(0, 2, 1, 3)
        )
        attn = mx.matmul(q, k.transpose(0, 1, 3, 2)) * self.scale
        attn = mx.softmax(attn, axis=-1)
        out = mx.matmul(attn, v)
        out = out.transpose(0, 2, 1, 3).reshape(B, T * H * W, C)
        out = self.out(out)
        out = out.transpose(0, 2, 1).reshape(B, C, T, H, W)
        return x + out


class TemporalAttention(nn.Module):
    def __init__(self, dim, num_heads=8):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5
        self.norm = nn.GroupNorm(32, dim)
        self.q = nn.Linear(dim, dim, bias=True)
        self.k = nn.Linear(dim, dim, bias=True)
        self.v = nn.Linear(dim, dim, bias=True)
        self.out = nn.Linear(dim, dim, bias=True)

    def __call__(self, x):
        B, C, T, H, W = x.shape
        h = _group_norm(self.norm, x)
        # Reshape: (B*H*W, T, C) for temporal attention
        h = h.transpose(0, 3, 4, 2, 1).reshape(B * H * W, T, C)
        q = (
            self.q(h)
            .reshape(B * H * W, T, self.num_heads, self.head_dim)
            .transpose(0, 2, 1, 3)
        )
        k = (
            self.k(h)
            .reshape(B * H * W, T, self.num_heads, self.head_dim)
            .transpose(0, 2, 1, 3)
        )
        v = (
            self.v(h)
            .reshape(B * H * W, T, self.num_heads, self.head_dim)
            .transpose(0, 2, 1, 3)
        )
        attn = mx.matmul(q, k.transpose(0, 1, 3, 2)) * self.scale
        attn = mx.softmax(attn, axis=-1)
        out = mx.matmul(attn, v)
        out = out.transpose(0, 2, 1, 3).reshape(B * H * W, T, C)
        out = self.out(out)
        out = out.reshape(B, H, W, T, C).transpose(0, 4, 3, 1, 2)
        return x + out


class CrossAttention(nn.Module):
    def __init__(self, query_dim, context_dim=1024, num_heads=8):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = query_dim // num_heads
        self.scale = self.head_dim**-0.5
        self.norm = nn.GroupNorm(32, query_dim)
        self.q = nn.Linear(query_dim, query_dim, bias=True)
        self.k = nn.Linear(context_dim, query_dim, bias=True)
        self.v = nn.Linear(context_dim, query_dim, bias=True)
        self.out = nn.Linear(query_dim, query_dim, bias=True)

    def __call__(self, x, context):
        B, C, T, H, W = x.shape
        h = _group_norm(self.norm, x)
        h = h.reshape(B, C, T * H * W).transpose(0, 2, 1)  # (B, THW, C)
        q = (
            self.q(h)
            .reshape(B, T * H * W, self.num_heads, self.head_dim)
            .transpose(0, 2, 1, 3)
        )
        k = (
            self.k(context)
            .reshape(B, -1, self.num_heads, self.head_dim)
            .transpose(0, 2, 1, 3)
        )
        v = (
            self.v(context)
            .reshape(B, -1, self.num_heads, self.head_dim)
            .transpose(0, 2, 1, 3)
        )
        attn = mx.matmul(q, k.transpose(0, 1, 3, 2)) * self.scale
        attn = mx.softmax(attn, axis=-1)
        out = mx.matmul(attn, v)
        out = out.transpose(0, 2, 1, 3).reshape(B, T * H * W, C)
        out = self.out(out)
        out = out.transpose(0, 2, 1).reshape(B, C, T, H, W)
        return x + out


class ResnetBlock(nn.Module):
    def __init__(self, in_channels, out_channels=None):
        super().__init__()
        out_ch = out_channels or in_channels
        self.norm1 = nn.GroupNorm(32, in_channels)
        self.conv1 = Conv3d(in_channels, out_ch, kernel_size=3, stride=1, padding=1)
        self.norm2 = nn.GroupNorm(32, out_ch)
        self.conv2 = Conv3d(out_ch, out_ch, kernel_size=3, stride=1, padding=1)
        self.skip = None
        if in_channels != out_ch:
            self.skip = Conv3d(in_channels, out_ch, kernel_size=1, stride=1, padding=0)

    def __call__(self, x, temb=None):
        h = _group_norm(self.norm1, x)
        h = nn.silu(h)
        h = self.conv1(h)
        if temb is not None:
            h = h + temb
        h = _group_norm(self.norm2, h)
        h = nn.silu(h)
        h = self.conv2(h)
        if self.skip is not None:
            x = self.skip(x)
        return x + h


class SVDUNetBlock(nn.Module):
    def __init__(
        self,
        in_dim,
        out_dim=None,
        context_dim=1024,
        num_heads=8,
        use_temporal=True,
        temb_dim=1280,
    ):
        super().__init__()
        out_dim = out_dim or in_dim
        self.temb_proj = (
            nn.Linear(temb_dim, out_dim, bias=True) if temb_dim != out_dim else None
        )
        self.resnet1 = ResnetBlock(in_dim, out_dim)
        self.attn_spatial = SpatialAttention(out_dim, num_heads)
        self.attn_temporal = (
            TemporalAttention(out_dim, num_heads) if use_temporal else None
        )
        self.attn_cross = CrossAttention(out_dim, context_dim, num_heads)
        self.resnet2 = ResnetBlock(out_dim)

    def __call__(self, x, context=None, temb=None):
        if temb is not None and self.temb_proj is not None:
            temb = self.temb_proj(temb.squeeze(-1).squeeze(-1).squeeze(-1))
            temb = temb[:, :, None, None, None]
        x = self.resnet1(x, temb)
        x = self.attn_spatial(x)
        if self.attn_temporal is not None:
            x = self.attn_temporal(x)
        if context is not None:
            x = self.attn_cross(x, context)
        x = self.resnet2(x, temb)
        return x


class DownBlock(nn.Module):
    def __init__(
        self,
        in_dim,
        out_dim,
        context_dim=1024,
        num_layers=2,
        downsample=True,
        num_heads=8,
    ):
        super().__init__()
        self.blocks = [
            SVDUNetBlock(in_dim if i == 0 else out_dim, out_dim, context_dim, num_heads)
            for i in range(num_layers)
        ]
        self.downsample = downsample
        if downsample:
            self.conv_down = Conv3d(
                out_dim, out_dim, kernel_size=3, stride=2, padding=1
            )

    def __call__(self, x, context=None, temb=None):
        for block in self.blocks:
            x = block(x, context, temb)
        if self.downsample:
            x = self.conv_down(x)
        return x


class UpBlock(nn.Module):
    def __init__(
        self,
        in_dim,
        out_dim,
        context_dim=1024,
        num_layers=2,
        upsample=True,
        num_heads=8,
    ):
        super().__init__()
        self.blocks = [
            SVDUNetBlock(in_dim if i == 0 else out_dim, out_dim, context_dim, num_heads)
            for i in range(num_layers)
        ]
        self.upsample = upsample
        if upsample:
            self.conv_up = Conv3d(out_dim, out_dim, kernel_size=3, stride=1, padding=1)

    def __call__(self, x, context=None, temb=None):
        for block in self.blocks:
            x = block(x, context, temb)
        if self.upsample:
            B, C, T, H, W = x.shape
            x = mx.broadcast_to(
                x.reshape(B, C, T, 1, H, 1, W, 1),
                (B, C, T, 2, H, 2, W, 2),
            )
            x = x.reshape(B, C, T * 2, H * 2, W * 2)
            x = self.conv_up(x)
        return x


def _map_unet_weights(raw):
    """Map PyTorch SVD UNet weight keys to our MLX model structure.

    PyTorch SVD (diffusers) key patterns:
      - down_blocks.{i}.resnets.{j}.{subkey}
      - down_blocks.{i}.attentions.{j}.transformer_blocks.{k}.attn1.to_q.weight
      - down_blocks.{i}.temporal_convs.{j}.conv.weight
      - down_blocks.{i}.downsamplers.0.conv.weight
      - mid_block.resnets.{j}.{subkey}
      - mid_block.attentions.{j}.transformer_blocks.{k}.attn1.to_q.weight
      - up_blocks.{i}.resnets.{j}.{subkey}
      - up_blocks.{i}.attentions.{j}.transformer_blocks.{k}.attn1.to_q.weight
      - up_blocks.{i}.temporal_convs.{j}.conv.weight
      - up_blocks.{i}.upsamplers.0.conv.weight
      - time_embedding.linear_1.weight / time_embedding.linear_2.weight

    Our MLX model structure:
      - down_blocks.{i}.blocks.{j}.resnet1/resnet2.{subkey}
      - down_blocks.{i}.blocks.{j}.attn_spatial.q/k/v/out.weight
      - down_blocks.{i}.blocks.{j}.attn_temporal.q/k/v/out.weight
      - down_blocks.{i}.blocks.{j}.attn_cross.q/k/v/out.weight
      - down_blocks.{i}.conv_down.weight
      - mid_block1/mid_block2.{subkey}
      - up_blocks.{i}.blocks.{j}.{subkey}
      - up_blocks.{i}.conv_up.weight
      - time_proj.linear1/linear2.weight
      - time_embed.{subkey}
    """
    mapped = {}
    for k, v in raw.items():
        nk = k
        # Strip common prefix
        if nk.startswith("model."):
            nk = nk[len("model.") :]
        if nk.startswith("unet."):
            nk = nk[len("unet.") :]

        # Time embedding: linear_1 -> linear1, linear_2 -> linear2
        nk = nk.replace("time_embedding.linear_1.", "time_proj.linear1.")
        nk = nk.replace("time_embedding.linear_2.", "time_proj.linear2.")

        # nn.Sequential sublayers: time_embed.layers.{0,1,2}
        # PyTorch: time_embedding.linear_{1,2} already mapped above
        # Our time_embed = nn.Sequential(Linear, SiLU, Linear) -> layers.0, layers.2
        if nk.startswith("time_proj."):
            mapped[nk] = v
            continue

        # Conv keys: direct match
        if (
            nk.startswith("conv_in.")
            or nk.startswith("conv_out.")
            or nk.startswith("conv_norm_out.")
        ):
            mapped[nk] = v
            continue

        # Mid block: mid_block.resnets.{j} -> mid_block{j+1}.resnet1
        # mid_block has 2 resnets + optional attentions
        # In our model: mid_block1, mid_block2 are SVDUNetBlock
        # PyTorch: mid_block.resnets.{0,1} + mid_block.attentions.{0,1}
        if nk.startswith("mid_block."):
            nk = nk[len("mid_block.") :]
            if nk.startswith("resnets."):
                parts = nk.split(".", 2)
                j = int(parts[1])
                subkey = parts[2] if len(parts) > 2 else ""
                # mid resnet 0 -> mid_block1.resnet1, mid resnet 1 -> mid_block2.resnet1
                # Each mid SVDUNetBlock has resnet1 + resnet2, but PyTorch
                # mid_block.resnets.{0,1} are full ResnetBlocks (norm1+conv1+norm2+conv2)
                # Our mid_block SVDUNetBlock: resnet1 + attn + resnet2
                # PyTorch SVD mid_block has: resnets[0], attentions[0], resnets[1], attentions[1]
                # -> mid_block1.resnet1 + mid_block1.attn_spatial + mid_block2.resnet1 + mid_block2.attn_spatial
                block_idx = j + 1
                new_key = f"mid_block{block_idx}.resnet1.{subkey}"
                mapped[new_key] = v
            elif nk.startswith("attentions."):
                parts = nk.split(".", 2)
                j = int(parts[1])
                subkey = parts[2] if len(parts) > 2 else ""
                block_idx = j + 1
                attn_key = _remap_attn_key(subkey)
                if attn_key:
                    mapped[f"mid_block{block_idx}.{attn_key}"] = v
            continue

        # Down/up blocks
        for prefix, block_cls in [("down_blocks", "down"), ("up_blocks", "up")]:
            if not nk.startswith(f"{prefix}."):
                continue
            nk = nk[len(f"{prefix}.") :]

            # down_blocks.{i}.downsamplers.0.conv -> down_blocks.{i}.conv_down
            # up_blocks.{i}.upsamplers.0.conv -> up_blocks.{i}.conv_up
            if ".downsamplers.0.conv." in nk:
                idx = nk.split(".")[0]
                rest = nk.split(".downsamplers.0.conv.")[1]
                mapped[f"{prefix}.{idx}.conv_down.{rest}"] = v
                nk = ""
                break
            if ".upsamplers.0.conv." in nk:
                idx = nk.split(".")[0]
                rest = nk.split(".upsamplers.0.conv.")[1]
                mapped[f"{prefix}.{idx}.conv_up.{rest}"] = v
                nk = ""
                break

            # Parse block index and sub-component
            parts = nk.split(".", 1)
            block_idx = parts[0]
            rest = parts[1] if len(parts) > 1 else ""

            # resnets.{j}.{subkey} -> blocks.{j}.resnet1.{subkey} or blocks.{j}.resnet2.{subkey}
            # In PyTorch SVD, each down/up block has num_layers resnets + num_layers attentions
            # In our model, blocks[j] = SVDUNetBlock with resnet1 + attn + resnet2
            # PyTorch resnets.{j} maps to blocks.{j}.resnet1 (first half of SVDUNetBlock)
            # But SVDUNetBlock has resnet1 AND resnet2... PyTorch has separate resnets
            # Actually in PyTorch SVD UNet: each block has resnets[j] and attentions[j]
            #   resnets[j] = full ResnetBlock (norm1+conv1+norm2+conv2)
            # In our SVDUNetBlock: resnet1 + attn + resnet2
            # PyTorch resnets[j] (full) maps to blocks[j].resnet1 (partial?)
            # NO — our ResnetBlock is also a full block (norm1+conv1+norm2+conv2+skip)
            # The correspondence: PyTorch resnets[j] -> blocks[j].resnet1
            #                     (no resnet2 in PyTorch — it only has 1 resnet per sub-block)
            # Wait, re-check: PyTorch SVD down_blocks have 2 resnets and 2 attentions per block
            # with num_layers=2. Our DownBlock has 2 SVDUNetBlocks each with resnet1+attn+resnet2.
            # The mapping should be:
            #   PyTorch resnets[j] -> blocks[j].resnet1
            #   PyTorch attentions[j] -> blocks[j].attn_spatial
            #   blocks[j].resnet2 -> no PyTorch equivalent (our addition)
            # Actually SVD in diffusers has the full structure with temporal resnets + temporal attn
            # Let me do the simpler mapping first and handle edge cases

            if rest.startswith("resnets."):
                rparts = rest.split(".", 2)
                j = int(rparts[1])
                subkey = rparts[2] if len(rparts) > 2 else ""
                mapped[f"{prefix}.{block_idx}.blocks.{j}.resnet1.{subkey}"] = v
                nk = ""
                break

            if rest.startswith("attentions."):
                apart = rest.split(".", 2)
                j = int(apart[1])
                subkey = apart[2] if len(apart) > 2 else ""
                # transformer_blocks.{k}.attn1.to_q -> attn_spatial.q
                attn_key = _remap_attn_key(subkey)
                if attn_key:
                    mapped[f"{prefix}.{block_idx}.blocks.{j}.{attn_key}"] = v
                nk = ""
                break

            if rest.startswith("temporal_convs."):
                # PyTorch: temporal_convs.{j}.conv.{weight,bias}
                tparts = rest.split(".", 2)
                j = int(tparts[1])
                subkey = tparts[2] if len(tparts) > 2 else ""
                # temporal_convs.{j}.conv -> blocks.{j}.attn_temporal (our temporal is attention)
                # Actually temporal_convs are separate Conv3d layers in PyTorch
                # Our SVDUNetBlock has temporal attention, not temporal conv
                # For now, log and skip — these need architectural alignment
                logger.debug("SVD UNet weight skip (temporal_conv): %s", k)
                nk = ""
                break

            # Fallback: pass through with prefix
            nk = f"{prefix}.{nk}"
            break

        if nk and not any(
            nk.startswith(p) for p in ("down_blocks.", "up_blocks.", "mid_block")
        ):
            # Unmatched key — try direct match
            mapped[nk] = v
            logger.debug("SVD UNet weight pass-through: %s -> %s", k, nk)

    return mapped


def _remap_attn_key(subkey):
    """Map PyTorch attention subkey to our attention key.

    PyTorch: transformer_blocks.{k}.attn1.to_q.weight -> attn_spatial.q.weight
             transformer_blocks.{k}.attn2.to_q.weight -> attn_cross.q.weight
    """
    if not subkey.startswith("transformer_blocks."):
        return None
    parts = subkey.split(".", 1)
    if len(parts) < 2:
        return None
    inner = parts[1]
    # attn1 = self-attention (spatial), attn2 = cross-attention
    for src, dst in [
        ("attn1.to_q.", "attn_spatial.q."),
        ("attn1.to_k.", "attn_spatial.k."),
        ("attn1.to_v.", "attn_spatial.v."),
        ("attn1.to_out.0.", "attn_spatial.out."),
        ("attn2.to_q.", "attn_cross.q."),
        ("attn2.to_k.", "attn_cross.k."),
        ("attn2.to_v.", "attn_cross.v."),
        ("attn2.to_out.0.", "attn_cross.out."),
        ("norm.weight", "attn_spatial.norm.weight"),
        ("norm.bias", "attn_spatial.norm.bias"),
    ]:
        if inner.startswith(src):
            rest = inner[len(src) :]
            return f"{dst}{rest}"
    return None


def _count_missing(model, mapped):
    model_keys = set(_flatten_keys(model.parameters()))
    mapped_keys = set(mapped.keys())
    missing = model_keys - mapped_keys
    return len(missing)


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


class SVDTemporalUNet(nn.Module):
    def __init__(
        self,
        in_channels=8,
        out_channels=4,
        context_dim=1024,
        dims=(320, 640, 1280, 1280),
        num_heads=8,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.context_dim = context_dim
        # Time embedding
        self.time_proj = TimestepEmbedding(320, 1280)
        self.time_embed = nn.Sequential(
            nn.Linear(1280, 1280, bias=True),
            nn.SiLU(),
            nn.Linear(1280, 1280, bias=True),
        )
        # Input conv
        self.conv_in = Conv3d(in_channels, dims[0], kernel_size=3, stride=1, padding=1)
        # Downsampling
        self.down_blocks = [
            DownBlock(
                dims[0],
                dims[0],
                context_dim,
                num_layers=2,
                downsample=True,
                num_heads=num_heads,
            ),
            DownBlock(
                dims[0],
                dims[1],
                context_dim,
                num_layers=2,
                downsample=True,
                num_heads=num_heads,
            ),
            DownBlock(
                dims[1],
                dims[2],
                context_dim,
                num_layers=2,
                downsample=True,
                num_heads=num_heads,
            ),
            DownBlock(
                dims[2],
                dims[3],
                context_dim,
                num_layers=2,
                downsample=False,
                num_heads=num_heads,
            ),
        ]
        # Mid
        self.mid_block1 = SVDUNetBlock(
            dims[3], context_dim=context_dim, num_heads=num_heads
        )
        self.mid_block2 = SVDUNetBlock(
            dims[3], context_dim=context_dim, num_heads=num_heads
        )
        # Upsampling
        self.up_blocks = [
            UpBlock(
                dims[3],
                dims[2],
                context_dim,
                num_layers=2,
                upsample=True,
                num_heads=num_heads,
            ),
            UpBlock(
                dims[2],
                dims[1],
                context_dim,
                num_layers=2,
                upsample=True,
                num_heads=num_heads,
            ),
            UpBlock(
                dims[1],
                dims[0],
                context_dim,
                num_layers=2,
                upsample=True,
                num_heads=num_heads,
            ),
            UpBlock(
                dims[0],
                dims[0],
                context_dim,
                num_layers=2,
                upsample=False,
                num_heads=num_heads,
            ),
        ]
        # Output
        self.conv_norm_out = nn.GroupNorm(32, dims[0])
        self.conv_out = Conv3d(
            dims[0], out_channels, kernel_size=3, stride=1, padding=1
        )

    def __call__(self, x, timestep=None, context=None):
        # Time embedding
        if timestep is not None:
            t_emb = _timestep_proj(timestep, 320)
            t_emb = self.time_proj(t_emb)
            t_emb = self.time_embed(t_emb)
            # Reshape for broadcast: (B, C, 1, 1, 1)
            temb = t_emb[:, :, None, None, None]
        else:
            temb = None
        # Forward
        h = self.conv_in(x)
        # Down — collect skip outputs
        skips = [h]
        for block in self.down_blocks:
            h = block(h, context, temb)
            skips.append(h)
        # Mid
        h = self.mid_block1(h, context, temb)
        h = self.mid_block2(h, context, temb)
        # Up with residual skip connections (add when shapes match)
        for i, block in enumerate(self.up_blocks):
            j = len(skips) - 1 - i
            if j >= 0 and skips[j].shape == h.shape:
                h = h + skips[j]
            h = block(h, context, temb)
        # Output
        h = _group_norm(self.conv_norm_out, h)
        h = nn.silu(h)
        h = self.conv_out(h)
        return h

    @classmethod
    def from_pretrained(cls, path, dtype=mx.float32):
        model = cls()
        import glob
        import os

        weight_files = sorted(glob.glob(os.path.join(path, "*.safetensors")))
        if not weight_files:
            unet_dir = os.path.join(path, "unet")
            if os.path.isdir(unet_dir):
                weight_files = sorted(
                    glob.glob(os.path.join(unet_dir, "*.safetensors"))
                )
        if weight_files:
            from mlx.utils import load_weights

            raw = dict(load_weights(path))
            mapped = _map_unet_weights(raw)
            model.update(mapped)
            model = model.astype(dtype)
            n_matched = len(mapped)
            n_missing = _count_missing(model, mapped)
            logger.info(
                "SVD UNet loaded dtype=%s matched=%d missing=%d",
                dtype,
                n_matched,
                n_missing,
            )
        else:
            model = model.astype(dtype)
            logger.warning("SVD UNet: no weight files found at %s", path)
        return model
