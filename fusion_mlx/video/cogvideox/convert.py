# SPDX-License-Identifier: Apache-2.0
import argparse
import logging
import shutil
from pathlib import Path

import numpy as np
import torch

logger = logging.getLogger(__name__)

TRANSFORMER_REMAP = {
    "time_embedding.linear_1": "time_embed.0",
    "time_embedding.linear_2": "time_embed.2",
    "patch_embed.proj": "patch_embed.proj",
    "patch_embed.text_proj": "patch_embed.text_proj",
    "patch_embed.position_embedding": "patch_embed.position_embedding",
    "norm_final": "norm_final",
    "norm_out": "norm_out",
    "proj_out": "proj_out",
}

BLOCK_REMAP = {
    "norm1.linear": "norm1.linear",
    "norm1.context_linear": "norm1.context_linear",
    "attn1.to_q": "attn1.to_q",
    "attn1.to_k": "attn1.to_k",
    "attn1.to_v": "attn1.to_v",
    "attn1.to_out.0": "attn1.to_out",
    "norm2.linear": "norm2.linear",
    "norm2.context_linear": "norm2.context_linear",
    "ff.net.0.proj": "ff.net.0.proj",
    "ff.net.2": "ff.net.2",
}


def _remap_transformer_key(key: str) -> str:
    parts = key.split(".")
    if parts[0] == "transformer_blocks":
        block_idx = parts[1]
        rest = ".".join(parts[2:])
        remapped = BLOCK_REMAP.get(rest, rest)
        return f"blocks.{block_idx}.{remapped}"
    for src, dst in TRANSFORMER_REMAP.items():
        if key.startswith(src + ".") or key == src:
            suffix = key[len(src) :]
            return dst + suffix
    return key


VAE_REMAP = {
    "encoder.conv_in": "encoder.conv_in",
    "encoder.conv_out": "encoder.conv_out",
    "encoder.conv_norm_out": "encoder.conv_norm_out",
    "decoder.conv_in": "decoder.conv_in",
    "decoder.conv_out": "decoder.conv_out",
    "decoder.conv_norm_out": "decoder.conv_norm_out",
    "quant_conv": "quant_conv",
    "post_quant_conv": "post_quant_conv",
}


def _remap_vae_key(key: str) -> str:
    parts = key.split(".")
    if parts[0] in ("encoder", "decoder"):
        if parts[1] == "down_blocks":
            block_idx = parts[2]
            rest = ".".join(parts[3:])
            return f"{parts[0]}.blocks.{block_idx}.{rest}"
        if parts[1] == "up_blocks":
            block_idx = parts[2]
            rest = ".".join(parts[3:])
            return f"{parts[0]}.blocks.{block_idx}.{rest}"
        if parts[1] == "mid_block":
            rest = ".".join(parts[2:])
            return f"{parts[0]}.mid_block.{rest}"
    for src, dst in VAE_REMAP.items():
        if key.startswith(src + ".") or key == src:
            suffix = key[len(src) :]
            return dst + suffix
    return key


def convert_weights(state_dict: dict, model_type: str = "transformer") -> dict:
    if model_type == "transformer":
        remap_fn = _remap_transformer_key
    elif model_type == "vae":
        remap_fn = _remap_vae_key
    else:
        remap_fn = lambda k: k

    new_sd = {}
    for key, val in state_dict.items():
        new_key = remap_fn(key)
        arr = (
            val.numpy().astype(np.float32)
            if isinstance(val, torch.Tensor)
            else np.array(val, dtype=np.float32)
        )
        new_sd[new_key] = arr
        if new_key != key:
            logger.debug(f"  {key} -> {new_key}")
        else:
            logger.debug(f"  {key} (unchanged)")

    logger.info(f"Converted {len(new_sd)} weights ({model_type})")
    return new_sd


def _save_safetensors(weights: dict, path: Path):
    from safetensors.numpy import save_file

    path.parent.mkdir(parents=True, exist_ok=True)
    save_file(weights, str(path))
    logger.info(f"Saved {len(weights)} tensors to {path}")


def convert_model(
    src_dir: str,
    dst_dir: str,
    model_type: str = "transformer",
):
    src = Path(src_dir).expanduser().resolve()
    dst = Path(dst_dir).expanduser().resolve()
    dst.mkdir(parents=True, exist_ok=True)

    if model_type == "transformer":
        from diffusers import CogVideoXTransformer3DModel

        logger.info(f"Loading CogVideoX transformer from {src}...")
        model = CogVideoXTransformer3DModel.from_pretrained(
            str(src), torch_dtype=torch.float32
        )
        sd = model.state_dict()
        logger.info(f"Original keys: {len(sd)}")
        converted = convert_weights(sd, "transformer")
        _save_safetensors(converted, dst / "model.safetensors")

    elif model_type == "vae":
        from diffusers import AutoencoderKLCogVideoX

        logger.info(f"Loading CogVideoX VAE from {src}...")
        model = AutoencoderKLCogVideoX.from_pretrained(
            str(src), torch_dtype=torch.float32
        )
        sd = model.state_dict()
        logger.info(f"Original keys: {len(sd)}")
        converted = convert_weights(sd, "vae")
        _save_safetensors(converted, dst / "vae.safetensors")

    elif model_type == "text_encoder":
        logger.info("T5 encoder reuse: copy from Wan2 T5 or convert standalone")
        safetensors_files = list(src.glob("*.safetensors"))
        if safetensors_files:
            from safetensors.torch import load_file

            all_weights = {}
            for f in safetensors_files:
                w = load_file(str(f))
                for k, v in w.items():
                    all_weights[k] = v.numpy().astype(np.float32)
            _save_safetensors(all_weights, dst / "t5_encoder.safetensors")
        else:
            logger.warning(f"No safetensors found in {src}")

    elif model_type == "all":
        convert_model(src_dir, dst_dir, "transformer")
        convert_model(src_dir, dst_dir, "vae")
        convert_model(src_dir, dst_dir, "text_encoder")
        if (src / "config.json").exists():
            shutil.copy2(src / "config.json", dst / "config.json")
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    logger.info(f"Conversion complete: {dst}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert CogVideoX PyTorch weights to MLX format"
    )
    parser.add_argument(
        "--src-dir", type=str, required=True, help="Source diffusers model directory"
    )
    parser.add_argument(
        "--dst-dir", type=str, required=True, help="Destination MLX model directory"
    )
    parser.add_argument(
        "--model-type",
        type=str,
        default="all",
        choices=["transformer", "vae", "text_encoder", "all"],
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    convert_model(args.src_dir, args.dst_dir, args.model_type)


if __name__ == "__main__":
    main()
