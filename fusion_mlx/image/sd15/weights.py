import logging

import mlx.core as mx

# SD1.5 weight remap differs from SDXL in one place: use_linear_projection=False
# means Transformer2D proj_in/proj_out are Conv2d 1x1 (4D OIHW) and must be
# transposed to OHWI for mlx conv2d. SDXL's _CONV_SUFFIXES does NOT list these,
# so we add an SD1.5-specific remap. SD1.5 also has NO add_embedding keys.
from fusion_mlx.image.sdxl.weights import (
    _CONV_SUFFIXES,
    _CONV_SUFFIXES_FLAT,
    _VAE_CONV_SUFFIXES,
    _find_missing,
    _flatten,
    _map_key,
    load_clip,
)

logger = logging.getLogger(__name__)

# Conv keys present in SD1.5 UNet beyond the SDXL _CONV_SUFFIXES set:
# Transformer2D 1x1 projections (use_linear_projection=False).
_PROJ_CONV_SUFFIXES = (".proj_in.weight", ".proj_out.weight")
_PROJ_CONV_SUFFIXES_FLAT = ("proj_in.weight", "proj_out.weight")


def _is_sd15_conv_key(k: str) -> bool:
    return (
        k.endswith(_CONV_SUFFIXES)
        or k.endswith(_CONV_SUFFIXES_FLAT)
        or k.endswith(_PROJ_CONV_SUFFIXES)
        or k.endswith(_PROJ_CONV_SUFFIXES_FLAT)
    )


def remap_unet_weights(raw: dict) -> list:
    # diffusers UNet2DConditionModel (SD1.5) keys -> SD15UNet module param paths.
    # Conv weights (OIHW) transposed to OHWI for mlx conv2d, including the
    # Transformer2D proj_in/proj_out 1x1 convs that SDXL lacks.
    # add_embedding.* keys are dropped (SD1.5 has none).

    pairs = []
    skipped = 0
    for k, v in raw.items():
        if k.startswith("add_embedding."):
            skipped += 1
            continue
        new = _map_key(k)
        if new is None:
            logger.debug("SD15 unet skip unmapped key: %s", k)
            continue
        if _is_sd15_conv_key(new) and v.ndim == 4:
            v = mx.transpose(v, (0, 2, 3, 1))
        pairs.append((new, v))
    if skipped:
        logger.debug("SD15 unet skipped %d add_embedding keys", skipped)
    logger.info("SD15 unet remapped %d / %d keys", len(pairs), len(raw))
    return pairs


def load_unet(unet, raw: dict) -> None:
    pairs = remap_unet_weights(raw)
    unet.load_weights(pairs, strict=False)
    loaded = set(p for p, _ in pairs)
    missing = _find_missing(unet, loaded)
    if missing:
        logger.warning(
            "SD15 unet missing %d params (first 10): %s", len(missing), missing[:10]
        )
    else:
        logger.info("SD15 unet loaded %d / %d keys (strict)", len(pairs), len(raw))


# SD1.5 VAE mid_block attentions use legacy diffusers naming:
#   query/key/value/proj_attn  -> module to_q/to_k/to_v/to_out.0
# (SDXL VAE uses the newer to_q/to_k/to_v/to_out.0 naming directly, so the
# SDXL remap leaves these keys unmapped and they go missing on SD1.5.)
_VAE_ATTN_LEGACY = (
    (".query.", ".to_q."),
    (".key.", ".to_k."),
    (".value.", ".to_v."),
    (".proj_attn.", ".to_out.0."),
)


def _remap_vae_attn_key(k: str) -> str:
    for old, new in _VAE_ATTN_LEGACY:
        if old in k:
            return k.replace(old, new)
    return k


def remap_vae_weights(raw: dict) -> list:
    pairs = []
    for k, v in raw.items():
        new = _remap_vae_attn_key(k)
        if new.endswith(_VAE_CONV_SUFFIXES) and v.ndim == 4:
            v = mx.transpose(v, (0, 2, 3, 1))
        # legacy mid attn weights are nn.Linear 2D [512,512] -> no reshape.
        pairs.append((new, v))
    logger.info("SD15 vae remapped %d / %d keys", len(pairs), len(raw))
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
            "SD15 vae missing %d params (first 10): %s", len(missing), missing[:10]
        )
    else:
        logger.info("SD15 vae loaded %d / %d keys (strict)", len(pairs), len(raw))
