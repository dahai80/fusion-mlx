"""latentsync-mlx — Apple-MLX port of ByteDance LatentSync.

Pure MLX audio-driven lip-sync: whisper audio encoding + UNet denoising + VAE decode.
Zero PyTorch dependency.
"""

from .pipeline import LipsyncPipelineMLX
from .sampler import DDIMSampler
from .unet import UNet3DConditionModel
from .vae import Autoencoder

__version__ = "0.2.0"
