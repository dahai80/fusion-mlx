"""pulid-mlx — Apple-MLX port of PuLID (Pure and Lightning ID Customization).

Identity-preserving image generation via contrastive alignment, integrated
with Flux. Zero PyTorch dependency.

Core components:
- IDFormer: Perceiver-resampler that fuses ArcFace + EVA-CLIP embeddings
- PerceiverAttentionCA: Cross-attention injection into Flux DiT layers
- PuLIDPipeline: Full pipeline (face detect -> EVA-CLIP -> IDFormer -> Flux)
"""

from .encoders import IDFormer
from .attention import PerceiverAttentionCA, IDAttnProcessor
from .eva_clip import EVACLIPEncoder, EVAVisionTransformer
from .pipeline import PuLIDPipeline

__version__ = "0.1.0"
