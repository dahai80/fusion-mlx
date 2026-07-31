# SPDX-License-Identifier: Apache-2.0
import logging
import os
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

MAGE_FLOW_MODEL_DIR = os.environ.get(
    "FUSION_MAGE_FLOW_MODEL_DIR",
    os.path.expanduser("~/.fusion-mlx/models/microsoft/Mage-Flow-4B"),
)


@dataclass
class MageFlowConfig:
    model_path: str = MAGE_FLOW_MODEL_DIR
    static_shift: float = 6.0
    num_steps: int = 30
    cfg_scale: float = 5.0
    max_size: int = 1024
    guidance_dtype: str = "bf16"
    vae_path: str | None = None
    txt_enc_path: str | None = None
    lora_paths: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, d: dict) -> "MageFlowConfig":
        return cls(
            **{k: v for k, v in d.items() if k in cls.__dataclass_fields__},
        )

    def to_dict(self) -> dict:
        import dataclasses

        return dataclasses.asdict(self)
