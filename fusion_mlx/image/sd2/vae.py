import logging

from fusion_mlx.image.sd15.vae import SD15VAE

logger = logging.getLogger(__name__)


class SD2VAE(SD15VAE):
    # SD2.1 VAE is architecturally identical to SD1.5 VAE (AutoencoderKL,
    # block_out_channels=(128,256,512,512), latent_channels=4) with the same
    # scaling_factor 0.18215. Reuse SD15VAE (which reuses SDXLVAE) verbatim.
    scaling_factor = 0.18215
    spatial_scale = 8
    latent_channels = 4
