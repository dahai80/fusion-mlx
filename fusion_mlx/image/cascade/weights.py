import logging

import mlx.core as mx

logger = logging.getLogger(__name__)


def _is_conv_weight(v: mx.array, k: str) -> bool:
    # 4D conv weights (Conv2d / DepthwiseConv2d / ConvTranspose2d) are the
    # only 4D tensors in the cascade checkpoints. All end in ".weight".
    return k.endswith(".weight") and v.ndim == 4


def _is_conv_transpose(k: str) -> bool:
    # ConvTranspose2d only appears as up_upscalers.{i}.1.weight in the
    # decoder (switch_level None). UpDownBlock2d mapping is a 1x1 Conv2d
    # keyed up_upscalers.{i}.1.blocks.1.weight, so ".blocks." excludes it.
    if k.startswith("up_upscalers.") and ".1.weight" in k and ".blocks." not in k:
        return True
    # vqgan: ConvTranspose2d is appended directly to up_blocks (not nested
    # in a list / not a MixingResidualBlock), so its key is
    # up_blocks.{N}.weight with no extra dotted component. The plain 1x1
    # Conv2d at up_blocks.0 is nested -> up_blocks.0.0.weight.
    if k.startswith("up_blocks.") and k.endswith(".weight"):
        rest = k[len("up_blocks.") :]
        if rest.count(".") == 1 and rest.split(".")[0].isdigit():
            return True
    return False


def _find_missing(model, loaded: set) -> list:
    flat = []
    _flatten(model.parameters(), "", flat)
    return [p for p in flat if p not in loaded]


def _flatten(tree, prefix, out):
    if isinstance(tree, dict):
        for k, v in tree.items():
            _flatten(v, f"{prefix}.{k}" if prefix else k, out)
    elif isinstance(tree, list):
        for i, v in enumerate(tree):
            _flatten(v, f"{prefix}.{i}", out)
    elif isinstance(tree, mx.array):
        out.append(prefix)


def remap_unet_weights(raw: dict) -> list:
    # diffusers StableCascadeUNet bf16 keys already mirror the MLX module
    # param paths (verified key-for-key against the decoder checkpoint), so
    # no key renaming is needed. Layout conversion:
    #   Conv2d/DepthwiseConv2d OIHW (out,in,k,k) -> OHWI (out,k,k,in): (0,2,3,1)
    #   ConvTranspose2d (in,out,k,k) -> OHWI (out,k,k,in): (1,2,3,0)
    # nn.Linear/nn.Embedding 2D weights match PyTorch and are left as-is.

    pairs = []
    n_conv = 0
    n_ct = 0
    for k, v in raw.items():
        if _is_conv_weight(v, k):
            if _is_conv_transpose(k):
                v = mx.transpose(v, (1, 2, 3, 0))
                n_ct += 1
            else:
                v = mx.transpose(v, (0, 2, 3, 1))
            n_conv += 1
        pairs.append((k, v))
    logger.info(
        "Cascade unet remapped %d / %d keys (conv=%d conv_transpose=%d)",
        len(pairs),
        len(raw),
        n_conv,
        n_ct,
    )
    return pairs


def load_unet(unet, raw: dict) -> None:
    pairs = remap_unet_weights(raw)
    unet.load_weights(pairs, strict=False)
    loaded = set(p for p, _ in pairs)
    missing = _find_missing(unet, loaded)
    if missing:
        logger.warning(
            "Cascade unet missing %d params (first 10): %s", len(missing), missing[:10]
        )


def remap_vqgan_weights(raw: dict) -> list:
    # PaellaVQModel decode-only. vquantizer.embedding.weight is unused
    # (force_not_quantize=True) and is dropped here. Conv2d/DepthwiseConv2d
    # OIHW -> OHWI (0,2,3,1); ConvTranspose2d (in,out,k,k) -> OHWI (1,2,3,0).
    # nn.Linear/nn.Embedding 2D weights match PyTorch and are left as-is.

    pairs = []
    dropped = 0
    n_ct = 0
    for k, v in raw.items():
        if k.startswith("vquantizer."):
            dropped += 1
            continue
        if _is_conv_weight(v, k):
            if _is_conv_transpose(k):
                v = mx.transpose(v, (1, 2, 3, 0))
                n_ct += 1
            else:
                v = mx.transpose(v, (0, 2, 3, 1))
        pairs.append((k, v))
    logger.info(
        "Cascade vqgan remapped %d / %d keys (dropped vquantizer=%d)",
        len(pairs),
        len(raw),
        dropped,
    )
    return pairs


def load_vqgan(vqgan, raw: dict) -> None:
    pairs = remap_vqgan_weights(raw)
    vqgan.load_weights(pairs, strict=False)
    loaded = set(p for p, _ in pairs)
    missing = _find_missing(vqgan, loaded)
    if missing:
        logger.warning(
            "Cascade vqgan missing %d params (first 10): %s", len(missing), missing[:10]
        )


def remap_clip_weights(raw: dict) -> list:
    # CLIP bigG text model: keys mirror HF naming, pass through. Only conv
    # weights (none in the text encoder) would need OHWI; nn.Linear/
    # nn.Embedding 2D weights match PyTorch layout and are left as-is.

    pairs = []
    for k, v in raw.items():
        if _is_conv_weight(v, k):
            v = mx.transpose(v, (0, 2, 3, 1))
        pairs.append((k, v))
    logger.info("Cascade clip remapped %d / %d keys", len(pairs), len(raw))
    return pairs


def load_clip(model, raw: dict) -> None:
    pairs = remap_clip_weights(raw)
    model.load_weights(pairs, strict=False)
    loaded = set(p for p, _ in pairs)
    flat = []
    _flatten(model.parameters(), "", flat)
    missing = [p for p in flat if p not in loaded]
    if missing:
        logger.warning(
            "Cascade clip missing %d params (first 10): %s", len(missing), missing[:10]
        )
