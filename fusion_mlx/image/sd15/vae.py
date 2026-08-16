import logging

from fusion_mlx.image.sdxl.vae import SDXLVAE

logger = logging.getLogger(__name__)


class SD15VAE(SDXLVAE):
    # SD1.5 VAE is architecturally identical to SDXL VAE (same
    # AutoencoderKL: block_out_channels=(128,256,512,512), latent_channels=4),
    # differing only in scaling_factor: 0.18215 (SD1.5) vs 0.13025 (SDXL).
    # Reuse the SDXL encoder/decoder/weight layout verbatim.
    scaling_factor = 0.18215
    spatial_scale = 8
    latent_channels = 4
