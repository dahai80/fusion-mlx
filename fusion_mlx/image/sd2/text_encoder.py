import logging

from fusion_mlx.image.sdxl.text_encoder import SDXLCLIPTextModel

logger = logging.getLogger(__name__)


class SD2TextEncoder(SDXLCLIPTextModel):
    # SD2.1 text encoder = ViT-H/14 in CLIPTextModel layout: hidden=1024,
    # 23 layers, 16 heads, intermediate=4096, hidden_act=gelu, NO
    # text_projection (projection_dim=None). SDXLCLIPTextModel with these
    # params + projection_dim=None matches the CLIPTextModel weight keys
    # (text_model.*, no text_projection). Reuse the SDXL CLIP implementation
    # + weight loader verbatim.
    pass
