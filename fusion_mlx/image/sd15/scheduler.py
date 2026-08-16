import logging

from fusion_mlx.image.sdxl.scheduler import SDXLEulerDiscreteScheduler

logger = logging.getLogger(__name__)


class SD15EulerDiscreteScheduler(SDXLEulerDiscreteScheduler):
    # SD1.5 uses the same scaled_linear beta schedule as SDXL
    # (beta_start=0.00085, beta_end=0.012), epsilon prediction, and
    # steps_offset=1. The diffusers default scheduler for SD1.5 is PNDM,
    # but EulerDiscrete is a valid drop-in and matches the SDXL MLX
    # implementation we reuse here. No behavioral change.
    pass
