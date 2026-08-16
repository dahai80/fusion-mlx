import logging

from fusion_mlx.image.sdxl.text_encoder import SDXLCLIPTextModel

logger = logging.getLogger(__name__)


class SD15CLIPTextModel(SDXLCLIPTextModel):
    # SD1.5 uses CLIP-L only (single text encoder): dims=768, layers=12,
    # heads=12, intermediate=3072, quick_gelu. SDXLCLIPTextModel with
    # projection_dim=None (no text_projection) matches SD1.5's weight keys
    # (197 keys, all under text_model.*, no pooled/text_projection).
    # Reuse the SDXL CLIP implementation + weight loader verbatim.
    pass
