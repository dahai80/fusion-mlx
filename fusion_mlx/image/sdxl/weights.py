import logging

import mlx.core as mx

logger = logging.getLogger(__name__)


_CONV_SUFFIXES = (
    ".conv_in.weight",
    ".conv_out.weight",
    ".conv1.weight",
    ".conv2.weight",
    ".conv_shortcut.weight",
    ".downsamplers.0.conv.weight",
    ".upsamplers.0.conv.weight",
)


_CONV_SUFFIXES_FLAT = (
    "conv_in.weight",
    "conv_out.weight",
) + _CONV_SUFFIXES


def _is_conv_key(k: str) -> bool:
    return k.endswith(_CONV_SUFFIXES_FLAT)


def remap_unet_weights(raw: dict) -> list:
    # diffusers UNet2DConditionModel fp16 keys -> SDXLUNet module param paths.
    # Conv weights (OIHW) transposed to OHWI for mlx conv2d.

    pairs = []
    for k, v in raw.items():
        new = _map_key(k)
        if new is None:
            logger.debug("SDXL unet skip unmapped key: %s", k)
            continue
        if _is_conv_key(new) and v.ndim == 4:
            v = mx.transpose(v, (0, 2, 3, 1))
        pairs.append((new, v))
    logger.info("SDXL unet remapped %d / %d keys", len(pairs), len(raw))
    return pairs


def _map_key(k: str) -> str | None:
    if k.startswith("time_embedding.linear_1."):
        return "time_embedding.0." + k.split(".")[-1]
    if k.startswith("time_embedding.linear_2."):
        return "time_embedding.1." + k.split(".")[-1]
    if k.startswith("add_embedding.linear_1."):
        return "add_embedding.0." + k.split(".")[-1]
    if k.startswith("add_embedding.linear_2."):
        return "add_embedding.1." + k.split(".")[-1]
    if ".ff.net.0.proj." in k:
        return k.replace(".ff.net.0.proj.", ".ff.net_0_proj.")
    if ".ff.net.2." in k:
        return k.replace(".ff.net.2.", ".ff.net_2.")
    return k


def load_unet(unet, raw: dict) -> None:
    pairs = remap_unet_weights(raw)
    unet.load_weights(pairs, strict=False)
    loaded = set(p for p, _ in pairs)
    missing = _find_missing(unet, loaded)
    if missing:
        logger.warning(
            "SDXL unet missing %d params (first 10): %s", len(missing), missing[:10]
        )


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


_VAE_CONV_SUFFIXES = (
    ".conv_in.weight",
    ".conv_out.weight",
    ".conv1.weight",
    ".conv2.weight",
    ".conv_shortcut.weight",
    ".downsamplers.0.conv.weight",
    ".upsamplers.0.conv.weight",
    ".quant_conv.weight",
    ".post_quant_conv.weight",
    "quant_conv.weight",
    "post_quant_conv.weight",
)


def remap_vae_weights(raw: dict) -> list:
    pairs = []
    attn_suffixes = (
        ".to_q.weight",
        ".to_k.weight",
        ".to_v.weight",
        ".to_out.0.weight",
    )
    for k, v in raw.items():
        if k.endswith(_VAE_CONV_SUFFIXES):
            v = mx.transpose(v, (0, 2, 3, 1))
        elif k.endswith(attn_suffixes) and v.ndim == 4:
            v = v.reshape(v.shape[0], v.shape[1])
        pairs.append((k, v))
    logger.info("SDXL vae remapped %d / %d keys", len(pairs), len(raw))
    return pairs


def load_vae(vae, raw: dict) -> None:
    pairs = remap_vae_weights(raw)
    vae.load_weights(pairs, strict=False)
    loaded = set(p for p, _ in pairs)
    flat = []
    _flatten(vae.parameters(), "", flat)
    missing = [p for p in flat if p not in loaded]
    if missing:
        logger.warning(
            "SDXL vae missing %d params (first 10): %s", len(missing), missing[:10]
        )


def remap_clip_weights(raw: dict) -> list:
    # diffusers CLIP text model keys -> SDXLCLIPTextModel param paths.
    # Module structure mirrors HF naming (text_model.*), so keys pass
    # through unchanged.

    pairs = [(k, v) for k, v in raw.items()]
    logger.info("SDXL clip remapped %d / %d keys", len(pairs), len(raw))
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
            "SDXL clip missing %d params (first 10): %s", len(missing), missing[:10]
        )
