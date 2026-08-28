import logging
import os
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
        # Probe the index for the requested stem (e.g. model.safetensors),
        # then fall back to the diffusers sharded layout
        # (diffusion_pytorch_model.safetensors.index.json + -0000X-of-0000Y
        # shards), which is the standard HF distribution format for 14B Wan
        # checkpoints.
        for index_name in (
            path.stem + ".safetensors.index.json",
            "diffusion_pytorch_model.safetensors.index.json",
        ):
            index = parent / index_name
            if not index.exists():
                continue
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


def _probe_in_dim_from_weights(model_dir: Path) -> int | None:
    # Read patch_embedding weight from the DiT safetensors and derive in_dim.
    # patch_embedding (Conv3d) weight is (out, in, kt, kh, kw); after sanitize
    # it becomes patch_embedding_proj.weight (out, in*kt*kh*kw). We probe the
    # raw file so this works regardless of sanitize. Returns None if no weight
    # file is found. pt, ph, pw are fixed at (1,2,2) for all Wan models, so
    # in_dim = second_dim // 4 for the Linear form, or shape[1] for Conv3d.
    candidates = []
    dit = model_dir / "dit"
    if dit.is_dir():
        candidates.extend(sorted(dit.glob("*.safetensors")))
    for name in ("model.safetensors", "high_noise_model.safetensors"):
        p = model_dir / name
        if p.exists():
            candidates.append(p)
    if not candidates:
        return None
    probe = _load_safetensors(candidates[0])
    in_dim = None
    for k, v in probe.items():
        if k.endswith("patch_embedding.weight") and v.ndim == 5:
            in_dim = int(v.shape[1])
            break
        if k.endswith("patch_embedding_proj.weight") and v.ndim == 2:
            in_dim = int(v.shape[1]) // 4
            break
    del probe
    return in_dim


def correct_in_dim(config, model_dir: Path):
    # config.json in_dim is sometimes stale (e.g. Wan2.1-14B dir holds an
    # i2v checkpoint with in_dim=36 but config.json says 32). The patch
    # embedding weight is authoritative: patch_embedding.weight.shape[1] is
    # the true in_dim. A mismatch makes the channel-concat build a y tensor
    # with the wrong channel count -> addmm shape error at the first DiT
    # block (input 128 vs weight 144). Re-derive in_dim from weights and
    # override the config when they disagree. Issue #456.
    true_in_dim = _probe_in_dim_from_weights(Path(model_dir))
    if true_in_dim is None or true_in_dim == config.in_dim:
        return config
    logger.warning(
        "correct_in_dim: config.in_dim=%d disagrees with patch_embedding "
        "weight in_dim=%d (model_dir=%s); overriding config from weights",
        config.in_dim,
        true_in_dim,
        model_dir,
    )
    fields = {
        f.name: getattr(config, f.name) for f in config.__dataclass_fields__.values()
    }
    fields["in_dim"] = true_in_dim
    from .config import WanModelConfig

    return WanModelConfig(**fields)


def _is_fp8_weights(weights: dict[str, mx.array]) -> bool:
    for k, v in weights.items():
        if k.endswith(".weight") and v.dtype == mx.uint8:
            return True
    return False


def _dequantize_fp8_weights(weights: dict[str, mx.array]) -> dict[str, mx.array]:
    _FP8_SKIP_SUFFIXES = (".scale_weight", ".scale_input", ".scaled_fp8")
    out = {}
    for k, v in weights.items():
        if any(k.endswith(s) for s in _FP8_SKIP_SUFFIXES):
            logger.debug("FP8 dequant: dropping meta key %s", k)
            continue
        if k.endswith(".weight") and v.dtype == mx.uint8:
            scale_key = k.rsplit(".", 1)[0] + ".scale_weight"
            scale = weights.get(scale_key)
            if scale is not None:
                v = v.astype(mx.bfloat16) * scale.astype(mx.bfloat16)
                logger.debug(
                    "FP8 dequant: %s (uint8 -> bf16, scale=%s)", k, scale.shape
                )
            else:
                v = v.astype(mx.bfloat16)
                logger.warning(
                    "FP8 dequant: %s has uint8 weight but no scale_weight", k
                )
        out[k] = v
    return out


def load_wan_model(
    model_path: Path,
    config,
    quantization: dict | None = None,
    loras: list | None = None,
):
    from .wan_2 import WanModel

    model = WanModel(config)

    weights = _load_safetensors(model_path)
    weights = model.sanitize(weights)

    is_fp8 = _is_fp8_weights(weights)
    if is_fp8:
        logger.info("Detected FP8 weights (uint8 + scale_weight), dequantizing to bf16")
        weights = _dequantize_fp8_weights(weights)
        quantization = None

    if quantization:
        from .convert import _quantize_predicate

        nn.quantize(
            model,
            group_size=quantization["group_size"],
            bits=quantization["bits"],
            class_predicate=lambda path, m: _quantize_predicate(path, m),
        )

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
                        sa_rest = rest[len("SelfAttention.") :]
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
    weights = {
        k: v for k, v in weights.items() if k not in ("spiece_model", "scaled_fp8")
    }
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


# Conv2d kernel spatial dims never exceed this; used to tell PyTorch
# [out, in, kH, kW] layout from already-MLX [out, kH, kW, in] layout.
_MAX_CONV_KERNEL = 8


def _transpose_conv2d_weights(weights: dict) -> dict:
    # PyTorch Conv2d weight is [out, in, kH, kW]; MLX wants [out, kH, kW, in] ->
    # transpose(0, 2, 3, 1). But some shipped VAE checkpoints (e.g.
    # Wan2.1-1.3B/vae.safetensors) are ALREADY in MLX layout; transposing them
    # again corrupts 1x1 convs like to_qkv (1152,1,1,384) -> (1152,1,384,1) and
    # crashes mx.conv2d with an input/weight channel mismatch. So only transpose
    # weights whose axes (2, 3) are kernel-sized (the PyTorch signature); an
    # already-MLX weight carries in_channels at axis 3 and is left alone.
    out = {}
    transposed = 0
    skipped = 0
    for k, v in weights.items():
        is_pytorch_layout = (
            v.ndim == 4
            and k.endswith(".weight")
            and "gamma" not in k
            and v.shape[2] <= _MAX_CONV_KERNEL
            and v.shape[3] <= _MAX_CONV_KERNEL
        )
        if is_pytorch_layout:
            out[k] = v.transpose(0, 2, 3, 1)
            transposed += 1
        else:
            out[k] = v
            if v.ndim == 4 and k.endswith(".weight") and "gamma" not in k:
                skipped += 1
    logger.info(
        "conv2d weight layout: transposed %d PyTorch-layout, skipped %d already-MLX",
        transposed,
        skipped,
    )
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


def _has_encoder_keys(weights: dict) -> bool:
    # Encoder weights live under the encoder.* prefix (encoder.conv1 is the
    # 3->96 input conv). A bare top-level conv1.* also exists in full VAEs
    # but is a different module (32->32 1x1x1x1), not an encoder weight —
    # matching it would let a decoder-only subset that kept that conv1
    # false-positive as encoder-bearing and skip the #670 fallback.
    return any(k.startswith("encoder.") for k in weights)


def _resolve_full_wan_vae(model_path: Path) -> Path | None:
    # Some Wan2.1 T2V checkpoints ship a DECODER-ONLY vae.safetensors (111
    # keys: decoder.*, conv2, mean, inv_std, std — no encoder.*). Loading
    # that into WanVAE(encoder=True) with strict=False leaves ~59/197 encoder
    # conv params at init, producing a degenerate latent (std ~0.19) that
    # does not roundtrip through decode (corr ~-0.34). Fall back to a full
    # VAE checkpoint (encoder+decoder) when the model-local file lacks
    # encoder weights. See issue #670.
    candidates = []
    env_full = os.environ.get("FUSION_WAN_VAE_FULL")
    if env_full:
        candidates.append(Path(env_full))
    candidates.append(
        Path.home() / ".fusion-mlx/models/wan-vae/wan_2.1_vae.safetensors"
    )
    for cand in candidates:
        if cand.exists():
            try:
                probe = _load_safetensors(cand)
            except Exception:
                continue
            if _has_encoder_keys(probe):
                logger.warning(
                    "vae encoder: %s has no encoder.* keys; falling back to "
                    "full VAE %s (issue #670)",
                    model_path,
                    cand,
                )
                return cand
    return None


def load_vae_encoder(model_path: Path, config=None):
    is_wan22 = config is not None and config.vae_z_dim != 16
    if config is not None and config.vae_z_dim == 16:
        from .vae import WanVAE

        vae = WanVAE(z_dim=16, encoder=True)
    else:
        from .vae22 import Wan22VAEEncoder, sanitize_wan22_vae_weights

        vae = Wan22VAEEncoder(z_dim=config.vae_z_dim if config else 48)

    weights = _load_safetensors(model_path)
    if is_wan22:
        weights = sanitize_wan22_vae_weights(weights)
    else:
        if not _has_encoder_keys(weights):
            full = _resolve_full_wan_vae(model_path)
            if full is None:
                raise FileNotFoundError(
                    f"vae encoder weights missing: {model_path} has no "
                    f"encoder.* keys and no full VAE fallback found. Set "
                    f"FUSION_WAN_VAE_FULL to a full Wan2.1 VAE safetensors "
                    f"(encoder+decoder) or place one at "
                    f"~/.fusion-mlx/models/wan-vae/wan_2.1_vae.safetensors "
                    f"(issue #670)."
                )
            weights = _load_safetensors(full)
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
