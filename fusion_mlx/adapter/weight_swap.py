# SPDX-License-Identifier: Apache-2.0
"""In-place LoRA swap (issue #389).

The default per-adapter path (pool/engine_pool._make_adapter_entry) reloads
the full base model for each adapter via ``mlx_lm.load(adapter_path=...)`` —
every adapter pays for a second copy of the base weights, tokenizer,
scheduler and KV cache state. On a 128GB machine several LoRAs across one
base model can exhaust memory.

This module keeps a single base engine resident and swaps the LoRA adapter
in place using mlx_lm's own ``LoRALinear`` machinery (``load_adapters`` /
``remove_lora_layers``), which correctly handles quantized base weights —
the low-rank ``lora_a``/``lora_b`` arrays are added beside the base
quantized linear, never fused into the packed weights, so no second base
copy is allocated.

The trade-off (matching the PRD "single base + N LoRA co-resident,
millisecond switch") is that only one adapter is active on a given base at
a time: concurrent multi-LoRA would race on the shared model graph. Callers
MUST hold a per-base lock for the whole apply->infer->restore window. The
EnginePool gates this when FUSION_LORA_INPLACE_SWAP=1.

This mirrors the video DiT in-place inject/remove pattern
(video/adapters/animatediff.py).
"""

import logging
import time

import mlx.core as mx
import mlx.nn as nn

logger = logging.getLogger(__name__)


def _lora_module_count(model: nn.Module) -> int:
    try:
        from mlx_lm.tuner.utils import LoRALinear
    except Exception:
        return 0
    return sum(1 for _, m in model.named_modules() if isinstance(m, LoRALinear))


class InPlaceLoRASwap:
    """Apply then remove a LoRA adapter on a resident base model.

    Uses mlx_lm ``load_adapters`` (wraps base linears as ``LoRALinear``) and
    ``remove_lora_layers`` (restores the original base linears). Symmetric,
    safe for quantized base weights — no delta fused into packed weights.

    Usage::

        swap = InPlaceLoRASwap(model, adapter_path)
        swap.apply()        # wrap linears with LoRALinear + load adapter
        try:
            ... infer with model ...
        finally:
            swap.restore()  # unwrap back to base linears
    """

    def __init__(self, model: nn.Module, adapter_path: str):
        self._model = model
        self._adapter_path = adapter_path
        self._applied = False
        self._lora_count = 0

    def apply(self) -> float:
        """Wrap base linears with LoRALinear and load adapter weights.

        Returns elapsed seconds. No-op if already applied.
        """
        if self._applied:
            logger.debug("lora_swap: already applied, no-op")
            return 0.0
        from mlx_lm.utils import load_adapters

        before = _lora_module_count(self._model)
        if before > 0:
            raise RuntimeError(
                f"lora_swap: model already has {before} LoRALinear modules; "
                "concurrent in-place swap on one base is forbidden (race)."
            )
        start = time.monotonic()
        load_adapters(self._model, self._adapter_path)
        mx.eval(self._model.parameters())
        elapsed = time.monotonic() - start
        self._lora_count = _lora_module_count(self._model)
        self._applied = True
        logger.info(
            "lora_swap: applied %s -> %d LoRALinear modules in %.3fms",
            self._adapter_path,
            self._lora_count,
            elapsed * 1000,
        )
        return elapsed

    def restore(self) -> float:
        """Remove LoRA layers, restoring the original base linears.

        Returns elapsed seconds. No-op if not applied.
        """
        if not self._applied:
            logger.debug("lora_swap: not applied, restore no-op")
            return 0.0
        from mlx_lm.tuner.utils import remove_lora_layers

        start = time.monotonic()
        remove_lora_layers(self._model)
        mx.eval(self._model.parameters())
        elapsed = time.monotonic() - start
        remaining = _lora_module_count(self._model)
        self._applied = False
        count = self._lora_count
        self._lora_count = 0
        logger.info(
            "lora_swap: restored %d LoRALinear modules in %.3fms (remaining=%d)",
            count,
            elapsed * 1000,
            remaining,
        )
        if remaining != 0:
            logger.warning(
                "lora_swap: %d LoRALinear modules remain after restore", remaining
            )
        return elapsed

    @property
    def applied(self) -> bool:
        return self._applied

    @property
    def lora_count(self) -> int:
        return self._lora_count


__all__ = ["InPlaceLoRASwap"]
