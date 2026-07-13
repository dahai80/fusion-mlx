# SPDX-License-Identifier: Apache-2.0
# xfuser attention strategy MLX port - window residual + CFG share + calibration.
# Ported from xfuser/core/fast_attention/attn_layer.py and utils.py.

from __future__ import annotations

import json
import logging
from enum import Flag, auto
from pathlib import Path
from typing import Any

import mlx.core as mx

from .mfa_bridge import flash_attention as _mfa_flash_attention

logger = logging.getLogger(__name__)


class FastAttnMethod(Flag):
    FULL_ATTN = auto()
    RESIDUAL_WINDOW_ATTN = auto()
    SPARSE_ATTN = auto()
    OUTPUT_SHARE = auto()
    CFG_SHARE = auto()
    RESIDUAL_WINDOW_ATTN_CFG_SHARE = RESIDUAL_WINDOW_ATTN | CFG_SHARE
    FULL_ATTN_CFG_SHARE = FULL_ATTN | CFG_SHARE
    SPARSE_ATTN_CFG_SHARE = SPARSE_ATTN | CFG_SHARE

    def has(self, method: FastAttnMethod) -> bool:
        return bool(self & method)


class MLXFastAttention:

    def __init__(
        self,
        window_size: int = -1,
        cond_first: bool = False,
    ):
        self.window_size = [window_size, window_size]
        self.cond_first = cond_first
        self.steps_method: list[FastAttnMethod] = []
        self.need_compute_residual: list[bool] = []
        self.need_cache_output = True
        self.cached_output: mx.array | None = None
        self.cached_residual: mx.array | None = None

    def set_methods(
        self,
        steps_method: list[FastAttnMethod],
        selecting: bool = False,
    ):
        self.steps_method = steps_method
        if selecting:
            self.need_compute_residual = [False] * len(steps_method)
        else:
            self.need_compute_residual = self._compute_need_residual()

    def _compute_need_residual(self) -> list[bool]:
        need = []
        for i, method in enumerate(self.steps_method):
            n = False
            if method.has(FastAttnMethod.FULL_ATTN):
                for j in range(i + 1, len(self.steps_method)):
                    if self.steps_method[j].has(FastAttnMethod.RESIDUAL_WINDOW_ATTN):
                        n = True
                    if self.steps_method[j].has(FastAttnMethod.FULL_ATTN):
                        break
            need.append(n)
        return need

    def _make_window_mask(
        self, q_len: int, k_len: int, dtype: mx.Dtype
    ) -> mx.array | None:
        ws = self.window_size[0]
        if ws < 0:
            return None
        i = mx.arange(q_len, dtype=mx.float32)[:, None]
        j = mx.arange(k_len, dtype=mx.float32)[None, :]
        mask = mx.where(mx.abs(i - j) < ws, 0.0, -float("inf")).astype(dtype)
        return mask.reshape(1, 1, q_len, k_len)

    def __call__(
        self,
        q: mx.array,
        k: mx.array,
        v: mx.array,
        step_idx: int,
        *,
        scale: float | None = None,
        mask: mx.array | None = None,
        is_causal: bool = False,
        batch_size: int | None = None,
    ) -> mx.array:
        if step_idx >= len(self.steps_method):
            method = FastAttnMethod.FULL_ATTN
        else:
            method = self.steps_method[step_idx]

        need_residual = (
            step_idx < len(self.need_compute_residual)
            and self.need_compute_residual[step_idx]
        )

        if scale is None:
            scale = q.shape[-1] ** -0.5

        # Output Share: reuse cached output
        if method.has(FastAttnMethod.OUTPUT_SHARE) and self.cached_output is not None:
            return self.cached_output

        # CFG Share: use only half the batch
        if method.has(FastAttnMethod.CFG_SHARE) and batch_size is not None:
            if self.cond_first:
                q = q[: batch_size // 2]
                k = k[: batch_size // 2]
                v = v[: batch_size // 2]
            else:
                q = q[batch_size // 2 :]
                k = k[batch_size // 2 :]
                v = v[batch_size // 2 :]

        # Full Attention
        if method.has(FastAttnMethod.FULL_ATTN):
            out = _mfa_flash_attention(
                q, k, v, scale=scale, mask=mask, causal=is_causal
            )

            if need_residual:
                w_mask = self._make_window_mask(q.shape[-2], k.shape[-2], q.dtype)
                w_out = _mfa_flash_attention(
                    q, k, v, scale=scale, mask=w_mask, causal=is_causal
                )
                residual = out - w_out

                if method.has(FastAttnMethod.CFG_SHARE):
                    residual = mx.concatenate([residual, residual], axis=0)
                self.cached_residual = residual

        # Window Residual Attention
        elif method.has(FastAttnMethod.RESIDUAL_WINDOW_ATTN):
            w_mask = self._make_window_mask(q.shape[-2], k.shape[-2], q.dtype)
            w_out = _mfa_flash_attention(
                q, k, v, scale=scale, mask=w_mask, causal=is_causal
            )
            if self.cached_residual is not None:
                out = w_out + self.cached_residual[: q.shape[0]]
            else:
                out = w_out

        # Sparse Attention
        elif method.has(FastAttnMethod.SPARSE_ATTN):
            from .mfa.sparse_video_attention import sparse_attention as _sparse_attn

            seq_q = q.shape[-2]
            seq_k = k.shape[-2]
            block_size = 64
            n_q = (seq_q + block_size - 1) // block_size
            n_k = (seq_k + block_size - 1) // block_size
            block_mask = mx.zeros((n_q, n_k), dtype=mx.uint8)
            ws = (
                max(1, self.window_size[0] // block_size)
                if self.window_size[0] > 0
                else n_k
            )
            for i in range(n_q):
                for j in range(max(0, i - ws), min(n_k, i + ws + 1)):
                    if not is_causal or j <= i:
                        block_mask[i, j] = 1
            out = _sparse_attn(q, k, v, block_mask, scale=scale, block_size=block_size)

        # CFG Share: duplicate result
        if method.has(FastAttnMethod.CFG_SHARE) and batch_size is not None:
            out = mx.concatenate([out, out], axis=0)

        # Cache output
        if self.need_cache_output:
            self.cached_output = out

        return out


def compression_loss(
    output_a: mx.array | list[mx.array],
    output_b: mx.array | list[mx.array],
) -> float:
    if isinstance(output_a, list):
        losses = []
        for a, b in zip(output_a, output_b):
            if isinstance(a, mx.array):
                diff = mx.abs(a - b) / (mx.maximum(mx.abs(a), mx.abs(b)) + 1e-6)
                losses.append(mx.clip(diff, 0.0, 10.0).mean())
        return float(sum(losses) / len(losses))
    else:
        diff = mx.abs(output_a - output_b) / (
            mx.maximum(mx.abs(output_a), mx.abs(output_b)) + 1e-6
        )
        return float(mx.clip(diff, 0.0, 10.0).mean())


def _run_calib_model(
    model: Any,
    calib_prompts: list[str],
    n_steps: int,
) -> mx.array | list[mx.array]:
    # Contract: model is callable(prompts, n_steps) -> outputs, or exposes
    # calibration_forward(prompts, n_steps). Outputs are compared via
    # compression_loss, so a single array or a list of arrays both work.
    if hasattr(model, "calibration_forward"):
        return model.calibration_forward(calib_prompts, n_steps)
    if callable(model):
        return model(calib_prompts, n_steps)
    raise TypeError(
        "calibrate_attention_strategy: model must be callable(prompts, n_steps) "
        "or expose calibration_forward(prompts, n_steps)"
    )


def _reset_module_caches(attention_modules: list[MLXFastAttention]) -> None:
    for m in attention_modules:
        m.cached_output = None
        m.cached_residual = None


# Candidate methods ordered most->least aggressive. Only standalone-safe
# methods are eligible: OUTPUT_SHARE is excluded because it requires a cached
# output from a prior FULL_ATTN step and cannot run at step 0 on its own.
_CALIB_CANDIDATES: list[FastAttnMethod] = [
    FastAttnMethod.SPARSE_ATTN,
    FastAttnMethod.RESIDUAL_WINDOW_ATTN,
    FastAttnMethod.FULL_ATTN_CFG_SHARE,
]


def calibrate_attention_strategy(
    model: Any,
    attention_modules: list[MLXFastAttention],
    n_steps: int,
    calib_prompts: list[str],
    *,
    threshold: float = 0.1,
    device: mx.Device | None = None,
    verbose: bool = False,
) -> list[list[FastAttnMethod]]:
    if n_steps <= 0:
        raise ValueError("n_steps must be >= 1")
    if not attention_modules:
        return []
    if device is not None:
        mx.set_default_device(device)

    baseline = [FastAttnMethod.FULL_ATTN] * n_steps

    def run(strategies: list[list[FastAttnMethod]]):
        _reset_module_caches(attention_modules)
        for m, methods in zip(attention_modules, strategies):
            m.set_methods(methods)
        return _run_calib_model(model, calib_prompts, n_steps)

    ref = run([list(baseline) for _ in attention_modules])

    # Greedy per-module: upgrade each module to the most aggressive candidate
    # whose output stays within `threshold` of the FULL_ATTN baseline. Earlier
    # modules' chosen strategies are kept when probing later ones, so the
    # search reflects a realistic incremental compression cascade.
    strategies = [list(baseline) for _ in attention_modules]
    for mi in range(len(attention_modules)):
        chosen = FastAttnMethod.FULL_ATTN
        for cand in _CALIB_CANDIDATES:
            trial = [list(s) for s in strategies]
            trial[mi] = [cand] * n_steps
            out = run(trial)
            loss = compression_loss(out, ref)
            if verbose:
                logger.info(
                    "calibrate module=%d candidate=%s loss=%.5f threshold=%.5f",
                    mi,
                    cand.name,
                    loss,
                    threshold,
                )
            if loss <= threshold:
                chosen = cand
                break
        strategies[mi] = [chosen] * n_steps

    _reset_module_caches(attention_modules)
    for m, methods in zip(attention_modules, strategies):
        m.set_methods(methods)
    logger.info(
        "calibrate_attention_strategy: %d modules, %d steps, threshold=%.3f -> %s",
        len(attention_modules),
        n_steps,
        threshold,
        {s[0].name for s in strategies if s},
    )
    return strategies


def save_strategy_config(
    strategies: list[list[FastAttnMethod]],
    file_path: str | Path,
) -> None:
    data = {
        f"block{bi}": {f"step{si}": m.name for si, m in enumerate(methods)}
        for bi, methods in enumerate(strategies)
    }
    with open(file_path, "w") as f:
        json.dump(data, f, indent=2)
    logger.info("Strategy config saved to %s", file_path)


def load_strategy_config(file_path: str | Path) -> list[list[FastAttnMethod]]:
    with open(file_path) as f:
        data = json.load(f)
    strategies = [
        [FastAttnMethod[m] for m in methods.values()] for methods in data.values()
    ]
    return strategies


class _FastAttnRuntime:
    __slots__ = ("active", "step", "batch_size")

    def __init__(self):
        self.active = False
        self.step = 0
        self.batch_size = None


_runtime = _FastAttnRuntime()


class fast_attn_step:
    def __init__(self, step_idx: int, *, batch_size: int | None = None):
        self._step = step_idx
        self._bs = batch_size
        self._prev_active = False
        self._prev_step = 0
        self._prev_bs: int | None = None

    def __enter__(self):
        self._prev_active = _runtime.active
        self._prev_step = _runtime.step
        self._prev_bs = _runtime.batch_size
        _runtime.active = True
        _runtime.step = self._step
        _runtime.batch_size = self._bs
        return self

    def __exit__(self, *exc):
        _runtime.active = self._prev_active
        _runtime.step = self._prev_step
        _runtime.batch_size = self._prev_bs
        return False


def current_step() -> int:
    return _runtime.step


def is_active() -> bool:
    return _runtime.active


def set_fast_attn_step(step_idx: int, *, batch_size: int | None = None) -> None:
    _runtime.active = True
    _runtime.step = step_idx
    _runtime.batch_size = batch_size


def deactivate_fast_attn() -> None:
    _runtime.active = False


def _walk_modules(module: Any, seen: set | None = None) -> list:
    if seen is None:
        seen = set()
    out: list = []
    mid = id(module)
    if mid in seen:
        return out
    seen.add(mid)
    out.append(module)
    children = getattr(module, "_modules", None)
    if isinstance(children, dict):
        for child in children.values():
            if isinstance(child, (list, tuple)):
                for c in child:
                    out.extend(_walk_modules(c, seen))
            else:
                out.extend(_walk_modules(child, seen))
    return out


def apply_fast_attention(
    model: Any,
    n_steps: int,
    *,
    window_size: int = -1,
    cond_first: bool = False,
) -> list[MLXFastAttention]:
    from fusion_mlx.video.ltx2.attention import Attention as LtxAttention
    from fusion_mlx.video.wan2.attention import (
        WanCrossAttention,
        WanSelfAttention,
    )

    target_types = (WanSelfAttention, WanCrossAttention, LtxAttention)
    mods = [m for m in _walk_modules(model) if isinstance(m, target_types)]
    fas: list[MLXFastAttention] = []
    for m in mods:
        fa = MLXFastAttention(window_size=window_size, cond_first=cond_first)
        fa.set_methods([FastAttnMethod.FULL_ATTN] * n_steps)
        m._fast_attn = fa
        fas.append(fa)
    logger.info(
        "apply_fast_attention: attached MLXFastAttention to %d attention modules "
        "(n_steps=%d, window=%d)",
        len(fas),
        n_steps,
        window_size,
    )
    return fas


def calibrate_and_apply(
    model: Any,
    n_steps: int,
    calib_prompts: list[str],
    *,
    window_size: int = -1,
    threshold: float = 0.1,
    verbose: bool = False,
) -> list[list[FastAttnMethod]]:
    fas = apply_fast_attention(model, n_steps, window_size=window_size)
    if not fas:
        logger.warning("calibrate_and_apply: no attention modules found on model")
        return []
    strategies = calibrate_attention_strategy(
        model,
        fas,
        n_steps,
        calib_prompts,
        threshold=threshold,
        verbose=verbose,
    )
    return strategies


__all__ = [
    "FastAttnMethod",
    "MLXFastAttention",
    "compression_loss",
    "calibrate_attention_strategy",
    "save_strategy_config",
    "load_strategy_config",
    "fast_attn_step",
    "current_step",
    "is_active",
    "set_fast_attn_step",
    "deactivate_fast_attn",
    "apply_fast_attention",
    "calibrate_and_apply",
]
