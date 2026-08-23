# SPDX-License-Identifier: Apache-2.0
# MiniMax-H3 AudioVAE：推理用解码器（DAC + BigVGAN）。
# 上游 dac_audio_vae.py DacAudioVAE + minimax_h3_audio_vae.py wrapper。
#
# 解码路径：z (B,32,T) → dec_in_proj Conv1d(32→2048,k1) → BigVGAN → (B,1,T*800)。
# 编码器/mean_proj/logs_proj/attn_proj/pre_block 推理不用，跳过（173 权重）。
#
# 权重映射（PyTorch → MLX，通道 last）：
#   dec_in_proj.weight        (out,in,k)  → (out,k,in)   transpose(1,2)
#   *.weight_g + *.weight_v   weight_norm → reconstruct → 见下
#     Conv1d        v (out,in,k) → (out,k,in)  transpose(1,2)
#     ConvTranspose1d v (in,out,k) → (out,k,in)  transpose(1,2,0)
#   act.alpha/beta            (C,)        → 直用
#   *.filter                  (1,1,k)     → 直用
#   *.bias                    (out,)      → 直用
import logging
import os

import mlx.core as mx
import mlx.nn as nn

from .bigvgan import BigVGAN
from .weight_norm import reconstruct_weight_norm

logger = logging.getLogger(__name__)


class MiniMaxH3AudioVAE(nn.Module):
    # vae_latent_channels: 32。latent_dim: 2048。decoder_dim: 1024。
    # decoder_rates: [5,5,2,2,2,2,2]。sample_rate: 32000。
    def __init__(
        self,
        vae_latent_channels=32,
        latent_dim=2048,
        decoder_dim=1024,
        decoder_rates=(5, 5, 2, 2, 2, 2, 2),
        upsample_kernel_sizes=(9, 9, 4, 4, 4, 4, 4),
        resblock_kernel_sizes=(3, 7, 11),
        resblock_dilation_sizes=((1, 3, 5), (1, 3, 5), (1, 3, 5)),
        sample_rate=32000,
        snake_logscale=True,
    ):
        super().__init__()
        self.vae_latent_channels = vae_latent_channels
        self.latent_dim = latent_dim
        self.sample_rate = sample_rate
        # dec_in_proj：vae_latent_channels → latent_dim，k1。
        self.dec_in_proj = nn.Conv1d(vae_latent_channels, latent_dim, 1)
        self.decoder = BigVGAN(
            num_mels=latent_dim,
            upsample_rates=list(decoder_rates),
            upsample_kernel_sizes=list(upsample_kernel_sizes),
            upsample_initial_channel=decoder_dim,
            resblock_kernel_sizes=resblock_kernel_sizes,
            resblock_dilation_sizes=resblock_dilation_sizes,
            snake_logscale=snake_logscale,
            use_tanh_at_final=False,
            use_bias_at_final=False,
        )

    def decode(self, z):
        # z (B, T, 32) 通道 last。返回 (B, T_out, 1)。
        z = self.dec_in_proj(z)
        return self.decoder(z)

    @classmethod
    def from_pretrained(cls, path):
        # path: audio_vae 目录或 model.safetensors 路径。
        # 加载 914 decode 权重，weight_norm 重建 + 布局转置。
        model_file = path
        if os.path.isdir(path):
            model_file = os.path.join(path, "model.safetensors")
        if not os.path.isfile(model_file):
            raise FileNotFoundError(
                f"AudioVAE weights not found: {model_file}. "
                f"Download via HF_ENDPOINT=https://hf-mirror.com "
                f"huggingface-cli download MiniMaxAI/MiniMax-H3 FL2VA/audio_vae/model.safetensors"
            )
        from safetensors import safe_open

        logger.info("Loading AudioVAE weights from %s", model_file)
        with safe_open(model_file, framework="numpy") as f:
            keys = list(f.keys())
            tensors = {k: f.get_tensor(k) for k in keys}
        logger.info("AudioVAE: %d total keys, mapping decode path", len(keys))
        model = cls()
        _apply_decode_weights(model, tensors)
        mx.eval(model.parameters())
        logger.info("AudioVAE loaded, decode path ready")
        return model


__all__ = ["MiniMaxH3AudioVAE"]


def _conv1d_to_mlx(weight):
    # PyTorch Conv1d weight (out, in, k) → MLX (out, k, in)。
    return mx.array(weight, dtype=mx.float32).transpose(0, 2, 1)


def _convt1d_to_mlx(weight):
    # PyTorch ConvTranspose1d weight (in, out, k) → MLX (out, k, in)。
    return mx.array(weight, dtype=mx.float32).transpose(1, 2, 0)


def _apply_decode_weights(model, tensors):
    # 将 safetensors 权重映射到 MLX 模块（strict，fail-visible）。
    # 直接参数：bias / act.alpha / act.beta / *.filter。
    # weight_norm：weight_g + weight_v → reconstruct → 布局转置。
    import mlx.utils

    def has(k):
        return k in tensors

    out = {}

    # dec_in_proj（plain Conv1d，无 weight_norm）。
    out["dec_in_proj.weight"] = _conv1d_to_mlx(tensors["dec_in_proj.weight"])
    out["dec_in_proj.bias"] = mx.array(tensors["dec_in_proj.bias"], dtype=mx.float32)

    # decoder.conv_pre（weight_norm Conv1d）。
    out["decoder.conv_pre.weight"] = _conv1d_to_mlx(
        reconstruct_weight_norm(
            tensors["decoder.conv_pre.weight_g"], tensors["decoder.conv_pre.weight_v"]
        )
    )
    out["decoder.conv_pre.bias"] = mx.array(
        tensors["decoder.conv_pre.bias"], dtype=mx.float32
    )

    # decoder.ups.N.0（weight_norm ConvTranspose1d）。
    # 上游 ups 为 ModuleList-of-ModuleList → key ups.N.0.*；MLX 用扁平 list → ups.N.*。
    # 仅剥离末尾的 ".0"（不replace 全部，避免 ups.0.0 误删前导 0）。
    def _strip_inner_ups(key):
        assert (
            key.endswith(".0.weight_g")
            or key.endswith(".0.weight_v")
            or key.endswith(".0.bias")
        ), key
        return (
            key[: -len(".0.weight_g")]
            if key.endswith(".0.weight_g")
            else (
                key[: -len(".0.weight_v")]
                if key.endswith(".0.weight_v")
                else key[: -len(".0.bias")]
            )
        )

    for k in list(tensors.keys()):
        if k.startswith("decoder.ups.") and k.endswith(".weight_g"):
            mlx_base = _strip_inner_ups(k)  # decoder.ups.N
            base_v = k[: -len(".weight_g")]  # decoder.ups.N.0
            out[mlx_base + ".weight"] = _convt1d_to_mlx(
                reconstruct_weight_norm(tensors[k], tensors[base_v + ".weight_v"])
            )
        elif k.startswith("decoder.ups.") and k.endswith(".0.bias"):
            mlx_key = _strip_inner_ups(k) + ".bias"  # decoder.ups.N.bias
            out[mlx_key] = mx.array(tensors[k], dtype=mx.float32)

    # decoder.resblocks.R.convs{1,2}.M（weight_norm Conv1d）。
    for k in list(tensors.keys()):
        if (".convs1." in k or ".convs2." in k) and k.endswith(".weight_g"):
            base = k[: -len(".weight_g")]
            out[base + ".weight"] = _conv1d_to_mlx(
                reconstruct_weight_norm(tensors[k], tensors[base + ".weight_v"])
            )
        elif (".convs1." in k or ".convs2." in k) and k.endswith(".bias"):
            out[k] = mx.array(tensors[k], dtype=mx.float32)

    # decoder.conv_post（weight_norm Conv1d，无 bias）。
    out["decoder.conv_post.weight"] = _conv1d_to_mlx(
        reconstruct_weight_norm(
            tensors["decoder.conv_post.weight_g"], tensors["decoder.conv_post.weight_v"]
        )
    )

    # act.alpha / act.beta / *.filter（直用）。
    for k in list(tensors.keys()):
        if k.endswith(".act.alpha") or k.endswith(".act.beta") or k.endswith(".filter"):
            out[k] = mx.array(tensors[k], dtype=mx.float32)

    # strict 加载：MLX 模块 779 参数必须全部命中。
    mlx_keys = set(dict(mlx.utils.tree_flatten(model.parameters())).keys())
    out_keys = set(out.keys())
    missing = mlx_keys - out_keys
    extra = out_keys - mlx_keys
    if missing or extra:
        raise RuntimeError(
            f"AudioVAE weight mapping mismatch: missing={sorted(missing)[:10]} "
            f"extra={sorted(extra)[:10]} (missing {len(missing)} extra {len(extra)})"
        )
    model.load_weights(list(out.items()), strict=True)
    logger.info("AudioVAE: %d weights mapped strict", len(out))
