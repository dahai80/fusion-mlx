import logging

import mlx.core as mx

from fusion_mlx.image.sdxl.text_encoder import SDXLCLIPTextModel, _causal_mask

logger = logging.getLogger(__name__)


class CascadeCLIPTextModel(SDXLCLIPTextModel):
    # CLIP-ViT-bigG text encoder for Stable Cascade prior. Reuses the
    # SDXL single-encoder implementation (CLIPTextModelWithProjection
    # semantics). Unlike SDXL, diffusers StableCascadePriorPipeline feeds
    # clip_text = hidden_states[-1] - the last encoder layer output BEFORE
    # final_layer_norm - while pooled stays post-norm + projection (#473).
    # Config defaults: dims=1280, num_layers=32, num_heads=20,
    # intermediate=5120, act=gelu, projection_dim=1280, vocab=49408,
    # max_pos=77.

    def __init__(
        self,
        dims: int = 1280,
        num_layers: int = 32,
        num_heads: int = 20,
        intermediate: int = 5120,
        act: str = "gelu",
        vocab: int = 49408,
        max_pos: int = 77,
        projection_dim: int = 1280,
    ):
        super().__init__(
            dims=dims,
            num_layers=num_layers,
            num_heads=num_heads,
            intermediate=intermediate,
            act=act,
            vocab=vocab,
            max_pos=max_pos,
            projection_dim=projection_dim,
        )
        logger.info(
            "CascadeCLIPTextModel dims=%d layers=%d heads=%d proj=%d",
            dims,
            num_layers,
            num_heads,
            projection_dim,
        )

    def __call__(self, tokens: mx.array):
        hidden = self.text_model.embeddings(tokens)
        mask = _causal_mask(hidden.shape).astype(hidden.dtype)
        hidden = self.text_model.encoder(hidden, mask)
        prenorm = hidden
        normed = self.text_model.final_layer_norm(prenorm)
        eos = mx.argmax(tokens, axis=-1)
        pooled = normed[mx.arange(normed.shape[0]), eos]
        if self.has_projection:
            pooled = self.text_projection(pooled)
        return prenorm, pooled
