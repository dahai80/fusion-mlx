import logging
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn

from ..ltx2.utils import get_model_path as get_model_path

logger = logging.getLogger(__name__)


def _load_safetensors(path: Path) -> dict[str, mx.array]:
    if path.is_dir():
        weights = {}
        for f in sorted(path.glob("*.safetensors")):
            weights.update(mx.load(str(f)))
        if weights:
            logger.info("loaded %d keys from sharded dir %s", len(weights), path)
            return weights
    if not path.exists():
        parent = path.parent
        index = parent / (path.stem + ".safetensors.index.json")
        if index.exists():
            import json

            with open(index) as f:
                meta = json.load(f)
            files = sorted(set(meta.get("weight_map", {}).values()))
            weights = {}
            for fn in files:
                fp = parent / fn
                if fp.exists():
                    weights.update(mx.load(str(fp)))
            if weights:
                logger.info("loaded %d keys via index %s", len(weights), index)
                return weights
    return mx.load(str(path))


def load_wan_model(
    model_path: Path,
    config,
    quantization: dict | None = None,
    loras: list | None = None,
):
    from .wan_2 import WanModel

    model = WanModel(config)

    if quantization:
        from .convert import _quantize_predicate

        nn.quantize(
            model,
            group_size=quantization["group_size"],
            bits=quantization["bits"],
            class_predicate=lambda path, m: _quantize_predicate(path, m),
        )

    weights = _load_safetensors(model_path)
    weights = model.sanitize(weights)

    # Apply LoRAs: dequantize+merge for quantized models, weight merge for bf16
    if loras:
        if quantization:
            # Dequantize LoRA-targeted layers, merge delta, replace with bf16 Linear.
            # Non-LoRA layers stay 4-bit. Zero per-step overhead.
            from .convert import _load_lora_configs
            from .lora import apply_loras_to_model

            model.load_weights(list(weights.items()), strict=False)
            mx.eval(model.parameters())
            module_to_loras = _load_lora_configs(loras)
            apply_loras_to_model(model, module_to_loras)
            mx.eval(model.parameters())
            return model
        else:
            # Weight merging: fold LoRA into bf16 weights before loading
            from .convert import load_and_apply_loras

            weights = load_and_apply_loras(dict(weights), loras)

    model.load_weights(list(weights.items()), strict=False)
    mx.eval(model.parameters())
    return model


def _remap_t5_weights(weights: dict) -> dict:
    import re

    remapped = {}
    for k, v in weights.items():
        nk = k
        if k == "shared.weight" or k == "encoder.embed_tokens.weight":
            nk = "token_embedding.weight"
        elif k == "encoder.final_layer_norm.weight":
            nk = "norm.weight"
        elif k.startswith("encoder.block."):
            m = re.match(r"encoder\.block\.(\d+)\.layer\.(\d+)\.(.+)", k)
            if m:
                block_n, layer_n, rest = m.group(1), m.group(2), m.group(3)
                prefix = f"blocks.{block_n}"
                if layer_n == "0":
                    if rest == "layer_norm.weight":
                        nk = f"{prefix}.norm1.weight"
                    elif rest.startswith("SelfAttention."):
                        sa_rest = rest[len("SelfAttention."):]
                        sa_map = {
                            "q.weight": "attn.q.weight",
                            "q.scale_weight": "attn.q.scale_weight",
                            "k.weight": "attn.k.weight",
                            "k.scale_weight": "attn.k.scale_weight",
                            "v.weight": "attn.v.weight",
                            "v.scale_weight": "attn.v.scale_weight",
                            "o.weight": "attn.o.weight",
                            "o.scale_weight": "attn.o.scale_weight",
                            "relative_attention_bias.weight": "pos_embedding.embedding.weight",
                        }
                        if sa_rest in sa_map:
                            nk = f"{prefix}.{sa_map[sa_rest]}"
                elif layer_n == "1":
                    ffn_map = {
                        "DenseReluDense.wi_0.weight": "ffn.gate_proj.weight",
                        "DenseReluDense.wi_0.scale_weight": "ffn.gate_proj.scale_weight",
                        "DenseReluDense.wi_1.weight": "ffn.fc1.weight",
                        "DenseReluDense.wi_1.scale_weight": "ffn.fc1.scale_weight",
                        "DenseReluDense.wo.weight": "ffn.fc2.weight",
                        "DenseReluDense.wo.scale_weight": "ffn.fc2.scale_weight",
                        "layer_norm.weight": "norm2.weight",
                    }
                    if rest in ffn_map:
                        nk = f"{prefix}.{ffn_map[rest]}"
        remapped[nk] = v
    return remapped


def load_t5_encoder(model_path: Path, config, dtype: str | None = None):
    import os

    from .text_encoder import T5Encoder

    encoder = T5Encoder(
        vocab_size=config.t5_vocab_size,
        dim=config.t5_dim,
        dim_attn=config.t5_dim_attn,
        dim_ffn=config.t5_dim_ffn,
        num_heads=config.t5_num_heads,
        num_layers=config.t5_num_layers,
        num_buckets=config.t5_num_buckets,
        shared_pos=False,
    )
    weights = mx.load(str(model_path))
    # Drop non-weight entries (tokenizer, metadata) that load_weights can't handle
    weights = {k: v for k, v in weights.items() if k not in ("spiece_model", "scaled_fp8")}
    # Remap HuggingFace T5 keys to wan2 T5Encoder attribute names
    weights = _remap_t5_weights(weights)
    # fp16 overflows in T5 attention for long sequences (q/k/v dot-product >65504)
    # Use bfloat16 by default — same memory as fp16 but with fp32 dynamic range
    has_fp8 = any(v.dtype in (mx.uint8, mx.int8) for v in weights.values())
    default_dtype = "bfloat16"
    target_dtype = dtype or os.environ.get("FUSION_T5_DTYPE", default_dtype)
    dtype_map = {
        "float32": mx.float32,
        "float16": mx.float16,
        "bf16": mx.bfloat16,
        "bfloat16": mx.bfloat16,
    }
    mx_dtype = dtype_map.get(target_dtype, mx.float16)
    logger.info(
        "Loading T5 encoder with dtype=%s (env FUSION_T5_DTYPE=%s)",
        target_dtype,
        os.environ.get("FUSION_T5_DTYPE"),
    )
    # FP8 (uint8 + scale_weight) weights: skip astype, load_weights handles dequantization
    weights = {
        k: v.astype(mx_dtype) if v.dtype not in (mx.uint8, mx.int8) else v
        for k, v in weights.items()
    }
    encoder.load_weights(list(weights.items()), strict=False)
    mx.eval(encoder.parameters())
    return encoder


def _transpose_conv2d_weights(weights: dict) -> dict:
    out = {}
    for k, v in weights.items():
        if v.ndim == 4 and k.endswith(".weight") and "gamma" not in k:
            out[k] = v.transpose(0, 2, 3, 1)
        else:
            out[k] = v
    return out


def load_vae_decoder(model_path: Path, config=None):
    is_wan22 = config is not None and config.vae_z_dim == 48

    if is_wan22:
        from .vae22 import Wan22VAEDecoder, sanitize_wan22_vae_weights

        vae = Wan22VAEDecoder(z_dim=48)
    else:
        from .vae import WanVAE

        vae = WanVAE(z_dim=16)

    weights = mx.load(str(model_path))
    if is_wan22:
        weights = sanitize_wan22_vae_weights(weights)
    else:
        weights = _transpose_conv2d_weights(weights)
    weights = {k: v.astype(mx.float32) for k, v in weights.items()}
    vae.load_weights(list(weights.items()), strict=False)
    mx.eval(vae.parameters())
    return vae


def load_vae_encoder(model_path: Path, config=None):
    is_wan22 = config is not None and config.vae_z_dim != 16
    if config is not None and config.vae_z_dim == 16:
        from .vae import WanVAE

        vae = WanVAE(z_dim=16, encoder=True)
    else:
        from .vae22 import Wan22VAEEncoder, sanitize_wan22_vae_weights

        vae = Wan22VAEEncoder(z_dim=config.vae_z_dim if config else 48)

    weights = mx.load(str(model_path))
    if is_wan22:
        weights = sanitize_wan22_vae_weights(weights)
    else:
        weights = _transpose_conv2d_weights(weights)
    weights = {k: v.astype(mx.float32) for k, v in weights.items()}
    vae.load_weights(list(weights.items()), strict=False)
    mx.eval(vae.parameters())
    return vae


def _clean_text(text: str) -> str:
    import html
    import re

    try:
        import ftfy

        text = ftfy.fix_text(text)
    except ImportError:
        pass
    text = html.unescape(html.unescape(text))
    text = re.sub(r"\s+", " ", text).strip()
    return text


def encode_text(
    encoder,
    tokenizer,
    prompt: str,
    text_len: int = 512,
) -> mx.array:
    prompt = _clean_text(prompt)
    tokens = tokenizer(
        prompt,
        max_length=text_len,
        padding="max_length",
        truncation=True,
        return_tensors="np",
    )
    ids = mx.array(tokens["input_ids"])
    mask = mx.array(tokens["attention_mask"])

    embeddings = encoder(ids, mask=mask)

    # Return only non-padding tokens
    seq_len = int(mask.sum().item())
    return embeddings[0, :seq_len]
