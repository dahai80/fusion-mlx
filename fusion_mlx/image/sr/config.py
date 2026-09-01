import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class RealESRGANConfig:
    # Mirrors official RealESRGAN_x4plus (basicsr.arch.rrdb_net.RRDBNet):
    # num_in_ch=3, num_out_ch=3, scale=4, num_feat=64, num_block=23,
    # num_grow_ch=32, res_scale=1.0. The ~64MB .pth holds these defaults.
    num_in_ch: int = 3
    num_out_ch: int = 3
    scale: int = 4
    num_feat: int = 64
    num_block: int = 23
    num_grow_ch: int = 32
    res_scale: float = 1.0
