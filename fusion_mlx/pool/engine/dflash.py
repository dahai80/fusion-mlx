# SPDX-License-Identifier: Apache-2.0
"""DFlash engine wrapper — speculative decoding with draft model."""

from typing import Any

from ..engines.batched import BatchedEngine


class DFlashEngine(BatchedEngine):
    """Wrapper around BatchedEngine that adds DFlash speculative decoding."""

    def __init__(
        self,
        model_name: str,
        draft_model_path: str,
        model_settings: Any = None,
        fallback_engine_type: Any = BatchedEngine,
        scheduler_config: Any = None,
        omlx_ssd_cache_dir: str | None = None,
        draft_quant_enabled: bool = False,
        draft_quant_weight_bits: int = 4,
        draft_quant_activation_bits: int = 16,
        draft_quant_group_size: int = 64,
        **kwargs: Any,
    ):
        super().__init__(model_name, model_settings=model_settings, scheduler_config=scheduler_config, **kwargs)
        self.draft_model_path = draft_model_path
        self.draft_quant_enabled = draft_quant_enabled
