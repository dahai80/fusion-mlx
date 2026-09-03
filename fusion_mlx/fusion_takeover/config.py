from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class FusionConfig:
    enabled: bool = False
    quant: str | None = None
    paged_kv_enabled: bool = False
    paged_kv_block_size: int = 16
    paged_kv_num_blocks: int = 256
    target_model_types: tuple[str, ...] = field(default_factory=tuple)
    fused_decode_enabled: bool = False
    pool_enabled: bool = False
    pool_num_blocks: int = 256

    def __post_init__(self) -> None:
        if self.quant not in (None, "q4_k_m", "nvfp4", "mxfp8", "bf16"):
            logger.warning(
                "FusionConfig.quant=%r unknown, treating as passthrough", self.quant
            )
        if self.paged_kv_block_size < 1:
            raise ValueError("paged_kv_block_size must be >= 1")
        if self.paged_kv_num_blocks < 1:
            raise ValueError("paged_kv_num_blocks must be >= 1")
        if self.pool_num_blocks < 1:
            raise ValueError("pool_num_blocks must be >= 1")

    @classmethod
    def from_model_settings(cls, model_settings: Any) -> FusionConfig:
        enabled = bool(getattr(model_settings, "fusion_takeover_enabled", False))
        quant = getattr(model_settings, "fusion_quant", None)
        paged = bool(getattr(model_settings, "fusion_paged_kv_enabled", False))
        block_size = int(getattr(model_settings, "fusion_paged_kv_block_size", 16))
        num_blocks = int(getattr(model_settings, "fusion_paged_kv_num_blocks", 256))
        target_types = tuple(
            getattr(model_settings, "fusion_target_model_types", ()) or ()
        )
        fused = getattr(model_settings, "fusion_paged_fused_kernel", None) == "on"
        pool_enabled = getattr(model_settings, "fusion_paged_pool", None) == "on"
        pool_num_blocks = int(
            getattr(model_settings, "fusion_paged_pool_num_blocks", 256)
        )
        logger.info(
            "FusionConfig.from_model_settings: fused_decode_enabled=%s "
            "pool_enabled=%s pool_num_blocks=%d",
            fused,
            pool_enabled,
            pool_num_blocks,
        )
        return cls(
            enabled=enabled,
            quant=quant,
            paged_kv_enabled=paged,
            paged_kv_block_size=block_size,
            paged_kv_num_blocks=num_blocks,
            target_model_types=target_types,
            fused_decode_enabled=fused,
            pool_enabled=pool_enabled,
            pool_num_blocks=pool_num_blocks,
        )

    def is_supported_model_type(self, model_type: str | None) -> bool:
        if not model_type:
            return False
        if not self.target_model_types:
            return True
        return model_type in self.target_model_types
