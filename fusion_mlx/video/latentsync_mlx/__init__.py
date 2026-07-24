"""latentsync-mlx — Apple-MLX port of ByteDance LatentSync.

Pure MLX audio-driven lip-sync: whisper audio encoding + UNet denoising + VAE decode.
Zero PyTorch dependency.
"""

from .pipeline import LipsyncPipelineMLX
from .unet import UNet3DConditionModel
from .vae import Autoencoder
from .sampler import DDIMSampler

__version__ = "0.2.0"
