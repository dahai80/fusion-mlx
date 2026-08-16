import logging

# SD2.1 UNet weight remap == SDXL UNet remap (use_linear_projection=True ->
# proj_in/proj_out are nn.Linear 2D, no conv transpose; time_embedding.
# linear_1/2 -> .0/.1; ff.net.0.proj/.ff.net.2 -> net_0_proj/net_2). SD2.1
# has NO add_embedding keys (like SD1.5), so the add_embedding.* remap in
# the shared _map_key is simply never hit. The VAE is identical to SD1.5
# (legacy mid-attn naming query/key/value/proj_attn), so reuse the SD15 VAE
# remap. The CLIP text model keys pass through unchanged (CLIPTextModel
# layout, no text_projection).
from fusion_mlx.image.sd15.weights import load_vae, remap_vae_weights  # noqa: F401
from fusion_mlx.image.sdxl.weights import (
    _find_missing,
    load_clip,  # noqa: F401  — re-exported for sd2.generate
    remap_unet_weights,
)

logger = logging.getLogger(__name__)


def load_unet(unet, raw: dict) -> None:
    pairs = remap_unet_weights(raw)
    unet.load_weights(pairs, strict=False)
    loaded = set(p for p, _ in pairs)
    missing = _find_missing(unet, loaded)
    if missing:
        logger.warning(
            "SD2 unet missing %d params (first 10): %s", len(missing), missing[:10]
        )
    else:
        logger.info("SD2 unet loaded %d / %d keys (strict)", len(pairs), len(raw))
