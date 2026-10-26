# SPDX-License-Identifier: Apache-2.0
"""mx.repeat array-repeats compat for mlx-vlm qwen2_5_vl vision tower.

mlx-vlm 0.5.0 builds cu_seqlens with:
    mx.repeat(seq_len, grid_thw[i, 0])
where ``grid_thw[i, 0]`` is a length-1 mx.array. MLX 0.32 tightened the
nanobind signature to require a Python int, so every qwen2.5-VL image
request fails with:
    TypeError: repeat(): incompatible function arguments. ...
    Invoked with types: mlx.core.array, mlx.core.array
and the OpenAI chat route returns 500.

Upstream fixed this on main (mx.repeat(seq_len, int(grid_thw[i, 0])));
this patch self-guards against that fix and can be removed once the
pinned mlx-vlm carries it.

Scoped fix: swap the vision module's ``mx`` attribute for a delegating
proxy whose repeat() converts array repeats via .item(). The global
mlx.core module is left untouched.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

_APPLIED = False


class _MxRepeatProxy:
    # Delegates every attribute to the real mlx.core except repeat(),
    # which converts length-1 array repeats to int before dispatching.
    __slots__ = ("_real",)

    def __init__(self, real):
        object.__setattr__(self, "_real", real)

    def __getattr__(self, name):
        return getattr(self._real, name)

    @staticmethod
    def repeat(array, repeats, axis=None, *, stream=None):
        import mlx.core as mx

        if isinstance(repeats, mx.array):
            repeats = int(repeats.item())
        if stream is not None:
            return mx.repeat(array, repeats, axis=axis, stream=stream)
        return mx.repeat(array, repeats, axis=axis)


def apply_mlx_vlm_qwen_vl_repeat_patch() -> bool:
    global _APPLIED
    if _APPLIED:
        return True
    try:
        import inspect

        import mlx.core as mx
        from mlx_vlm.models.qwen2_5_vl import vision
    except Exception as e:
        logger.debug("qwen2_5_vl vision import unavailable: %s", e)
        return False
    if getattr(vision, "_fusion_mlx_repeat_proxy", False):
        _APPLIED = True
        return True
    # Upstream already fixed on main — leave the real module alone.
    try:
        source = inspect.getsource(vision)
        if "int(grid_thw" in source:
            logger.debug("qwen2_5_vl repeat already fixed upstream; skipping patch")
            _APPLIED = True
            return True
    except (OSError, TypeError):
        pass
    vision.mx = _MxRepeatProxy(mx)
    vision._fusion_mlx_repeat_proxy = True
    _APPLIED = True
    logger.info("qwen2_5_vl mx.repeat array-repeats compat applied")
    return True


def is_applied() -> bool:
    return _APPLIED


from .registry import register as _reg

_reg(
    "mlx_vlm_qwen_vl_repeat_compat",
    "qwen2_5_vl",
    "mlx-vlm 0.5.0 passes mx.array repeats to mx.repeat; MLX 0.32 requires int (self-guards if upstream fixed)",
    lambda model, config: apply_mlx_vlm_qwen_vl_repeat_patch(),
)
