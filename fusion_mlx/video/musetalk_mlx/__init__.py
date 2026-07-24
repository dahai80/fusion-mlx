"""musetalk-mlx — Apple-MLX port of MuseTalk 1.5 (realtime lip-sync via single-step latent inpainting)."""

from . import config  # noqa: F401
from .pipeline_mlx import MuseTalkPipeline

__version__ = "0.2.0"
