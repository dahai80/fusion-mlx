# SPDX-License-Identifier: Apache-2.0
"""PR-L: imatrix metadata + IQ/GGUF mixed-quant consumption (v2 doc §3.3).

Three pieces:

  - imatrix loader: parse a llama.cpp ``imatrix`` GGUF file (each entry
    key = tensor name, value = F32 array of accumulated activation
    statistics, optionally chunked). Entries feed per-layer sensitivity
    scoring.
  - Mixed-quant plan: score each tensor by imatrix-weighted relative MSE
    of a hypothetical Q4_0 requant, then assign sensitive tensors to
    bf16 and the rest to the preferred production format (Q4_0 /
    MXFP4). Deterministic — no model decisions, pure arithmetic.
  - Per-layer dispatch: raw GGUF block data → numpy dequant handlers for
    the unambiguous layouts (q4_0, q8_0, iq4_nl, mxfp4). iq4_nl is
    gated behind FUSION_SHIM_IQ (default OFF → falls back like the
    other IQ formats). Lattice-based IQ2/IQ3/IQ1 and Q2_K..Q6_K raise
    UnsupportedQuantError — loud fallback to native mlx_lm, never a
    silently-wrong decode.

Degrade switches (default OFF for the new paths):
  FUSION_SHIM_IQ=1  — allow iq4_nl consumption through the shim handler
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..migrate.asfw import ASFWConverter, UnsupportedQuantError
from ..migrate.gguf_reader import GGUFReader, TensorInfo

logger = logging.getLogger(__name__)

_IMATRIX_CHUNK_KEYS = ("imatrix.chunks", "imatrix.n_chunks", "imatrix.chunk_count")
_TENSOR_KEY_SUFFIXES = (".weight", ".bias", ".exp_probs_b")


def _env_on(name: str) -> bool:
    return os.environ.get(name, "0") == "1"


def is_iq_enabled() -> bool:
    return _env_on("FUSION_SHIM_IQ")


# --- imatrix loading --------------------------------------------------------


@dataclass
class ImatrixData:
    entries: dict[str, np.ndarray] = field(default_factory=dict)
    n_chunks: int = 1
    source: str = ""

    def entry_for(self, tensor_name: str) -> np.ndarray | None:
        # Exact key, then basename fallback (imatrix files often store
        # "blk.0.attn_q.weight" while a model may prefix differently).
        if tensor_name in self.entries:
            return self.entries[tensor_name]
        short = tensor_name.rsplit(".", 1)[-1]
        for key, arr in self.entries.items():
            if key.rsplit(".", 1)[-1] == short:
                return arr
        return None

    @property
    def n_tensors(self) -> int:
        return len(self.entries)

    def total_elements(self) -> int:
        return sum(int(a.size) for a in self.entries.values())


def load_imatrix(path: str | Path) -> ImatrixData:
    # llama.cpp imatrix GGUF: one F32-array KV per scored tensor, value =
    # accumulated sums (optionally n_chunks chunks concatenated), plus an
    # optional chunk-count key. Unknown keys are ignored loudly-quietly:
    # only F32 arrays whose key looks like a weight/bias are treated as
    # entries (Rule: parse what we know, log the rest).
    path = Path(path)
    reader = GGUFReader(path)
    meta = reader.read_header()
    kv = meta.raw_kv

    n_chunks = 1
    for key in _IMATRIX_CHUNK_KEYS:
        if key in kv:
            try:
                n_chunks = max(1, int(kv[key]))
                break
            except (TypeError, ValueError):
                logger.warning("imatrix %s: bad chunk count %r", path.name, kv[key])

    entries: dict[str, np.ndarray] = {}
    for key, val in kv.items():
        if key.startswith("imatrix."):
            continue
        if not isinstance(val, list) or not val:
            continue
        if not key.endswith(_TENSOR_KEY_SUFFIXES):
            continue
        arr = np.asarray(val, dtype=np.float32)
        if arr.size % n_chunks != 0:
            logger.warning(
                "imatrix %s: entry %s size %d not divisible by chunks %d; "
                "treating as single chunk",
                path.name,
                key,
                arr.size,
                n_chunks,
            )
        entries[key] = arr

    data = ImatrixData(entries=entries, n_chunks=n_chunks, source=str(path))
    logger.info(
        "imatrix %s: %d entries, chunks=%d, %.1fM values",
        path.name,
        data.n_tensors,
        data.n_chunks,
        data.total_elements() / 1e6,
    )
    return data


def chunk_mean(entry: np.ndarray, n_chunks: int) -> np.ndarray:
    # Average the accumulated per-chunk sums into one importance vector.
    if entry.size % n_chunks != 0 or n_chunks <= 1:
        return entry.astype(np.float32)
    return entry.reshape(n_chunks, -1).mean(axis=0).astype(np.float32)


# --- sensitivity scoring + mixed plan ---------------------------------------


def q40_sensitivity(w: np.ndarray, imp: np.ndarray | None = None) -> float:
    # Relative imatrix-weighted MSE of a Q4_0 requant of w.
    # Q4_0 symmetric per-row (last dim) scale d = max|w_row|/8; round-to-
    # nearest error is uniform on ±d/2 → variance d²/12. Weighted by the
    # activation importance (imatrix ≈ diag of E[x xᵀ]), the estimate is
    # Σ_j imp_j · d_j² /12 normalized by the weighted signal power.
    w32 = np.asarray(w, dtype=np.float32)
    if w32.ndim == 1:
        w32 = w32[None, :]
    n_in = w32.shape[-1]
    if imp is not None:
        imp = np.asarray(imp, dtype=np.float32)
        if imp.size < n_in:
            imp = np.concatenate(
                [imp, np.full(n_in - imp.size, imp.mean() if imp.size else 1.0)]
            )
        elif imp.size > n_in:
            imp = imp[:n_in]
        if not np.isfinite(imp).all() or imp.sum() <= 0:
            imp = None
    if imp is None:
        imp = np.full(n_in, 1.0 / n_in, dtype=np.float32)
    else:
        s = imp.sum()
        imp = imp / s if s > 0 else np.full(n_in, 1.0 / n_in, dtype=np.float32)

    d = np.abs(w32).max(axis=-1) / 8.0
    err_var = (d.astype(np.float32) ** 2) / 12.0
    weighted_err = float(np.mean(err_var[:, None] * imp[None, :]))
    signal = float(np.mean((w32.astype(np.float32) ** 2) * imp[None, :]))
    if signal <= 0:
        return 0.0
    return weighted_err / signal


@dataclass
class MixedQuantPlan:
    assignments: dict[str, str] = field(default_factory=dict)
    scores: dict[str, float] = field(default_factory=dict)
    threshold: float = 0.0

    @property
    def n_high_precision(self) -> int:
        return sum(1 for v in self.assignments.values() if v != "q4_0")


def build_mixed_plan(
    tensors: list[TensorInfo],
    imatrix: ImatrixData | None,
    weights_by_name: dict[str, np.ndarray] | None = None,
    threshold: float = 0.05,
    preferred: str = "q4_0",
) -> MixedQuantPlan:
    # Deterministic policy: sensitivity >= threshold → bf16, else the
    # preferred production format (Q4_0 default; MXFP4 opt-in). Tensors
    # without weights or imatrix coverage score 0 → preferred (log it —
    # missing coverage must be visible, not assumed good).
    plan = MixedQuantPlan(threshold=threshold)
    if preferred not in ("q4_0", "mxfp4"):
        raise ValueError(f"mixed_quant: unsupported preferred format {preferred!r}")
    for info in tensors:
        score = 0.0
        w = (weights_by_name or {}).get(info.name)
        imp = imatrix.entry_for(info.name) if imatrix is not None else None
        if imp is not None and imatrix is not None:
            imp = chunk_mean(imp, imatrix.n_chunks)
        if w is not None:
            score = q40_sensitivity(w, imp)
            if imp is None:
                logger.debug(
                    "mixed_quant: %s no imatrix entry, unweighted score", info.name
                )
        else:
            logger.debug(
                "mixed_quant: %s no weights available, score defaults to 0",
                info.name,
            )
        assignment = "bf16" if score >= threshold else preferred
        plan.assignments[info.name] = assignment
        plan.scores[info.name] = score
    n_hp = plan.n_high_precision
    logger.info(
        "mixed_quant plan: %d tensors, %d bf16 (threshold %.4g), %d %s",
        len(plan.assignments),
        n_hp,
        threshold,
        len(plan.assignments) - n_hp,
        preferred,
    )
    return plan


# --- raw block dequant dispatch ---------------------------------------------

_IQ4NL_KVALUES = np.array(
    [-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 27, 42, 58, 74, 96, 120],
    dtype=np.float32,
)
_MXFP4_LUT = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=np.float32)


def dequant_iq4_nl(raw: np.ndarray, n_elements: int) -> np.ndarray:
    # block_iq4_nl: 32 elems, 18 bytes = f16 scale + 16 bytes of 4-bit
    # indices into the 16-entry kvalues lookup (ggml-quants.h).
    if n_elements % 32 != 0:
        raise ValueError(f"iq4_nl: element count {n_elements} not a multiple of 32")
    n_blocks = n_elements // 32
    dtype = np.dtype([("scale", np.float16), ("qs", np.uint8, 16)], align=False)
    blocks = np.frombuffer(raw, dtype=dtype, count=n_blocks)
    scales = blocks["scale"].astype(np.float32)
    qs = blocks["qs"]
    lo = _IQ4NL_KVALUES[qs & 0x0F]
    hi = _IQ4NL_KVALUES[qs >> 4]
    out = np.empty((n_blocks, 32), dtype=np.float32)
    out[:, 0::2] = lo
    out[:, 1::2] = hi
    return (out * scales[:, None]).reshape(-1)


def dequant_mxfp4(raw: np.ndarray, n_elements: int) -> np.ndarray:
    # MXFP4 block: 32 elems, 17 bytes = e8m0 scale byte + 16 bytes of
    # 4-bit E2M1 values (sign bit + 3-bit LUT index).
    if n_elements % 32 != 0:
        raise ValueError(f"mxfp4: element count {n_elements} not a multiple of 32")
    n_blocks = n_elements // 32
    scales_bytes = np.frombuffer(raw, dtype=np.uint8, count=n_blocks * 17)
    scales_bytes = scales_bytes.reshape(n_blocks, 17)
    scale_bits = scales_bytes[:, 0].astype(np.int16) - 127
    scales = np.ldexp(np.ones(n_blocks, dtype=np.float32), scale_bits).astype(
        np.float32
    )
    packed = scales_bytes[:, 1:].reshape(n_blocks, 16)
    lo = np.where(
        (packed & 0x08) != 0, -_MXFP4_LUT[packed & 0x07], _MXFP4_LUT[packed & 0x07]
    )
    hi_nib = (packed >> 4) & 0x0F
    hi = np.where(
        (hi_nib & 0x08) != 0, -_MXFP4_LUT[hi_nib & 0x07], _MXFP4_LUT[hi_nib & 0x07]
    )
    out = np.empty((n_blocks, 32), dtype=np.float32)
    out[:, 0::2] = lo
    out[:, 1::2] = hi
    return (out * scales[:, None]).reshape(-1)


_DISPATCHERS = {
    "q4_0": "_q40_handler",
    "q8_0": "_q80_handler",
    "iq4_nl": "_iq4nl_handler",
    "mxfp4": "_mxfp4_handler",
}


def _q40_handler(raw: bytes, n_elements: int) -> np.ndarray:
    n_blocks = n_elements // 32
    dtype = np.dtype([("scale", np.float16), ("packed", np.uint8, 16)], align=False)
    blocks = np.frombuffer(raw, dtype=dtype, count=n_blocks)
    return ASFWConverter._dequant_q4_0(
        blocks["scale"].astype(np.float32), blocks["packed"]
    )


def _q80_handler(raw: bytes, n_elements: int) -> np.ndarray:
    n_blocks = n_elements // 32
    dtype = np.dtype([("scale", np.float16), ("qs", np.int8, 32)], align=False)
    blocks = np.frombuffer(raw, dtype=dtype, count=n_blocks)
    return (
        blocks["qs"].astype(np.float32) * blocks["scale"].astype(np.float32)[:, None]
    ).reshape(-1)


def _iq4nl_handler(raw: bytes, n_elements: int) -> np.ndarray:
    return dequant_iq4_nl(raw, n_elements)


def _mxfp4_handler(raw: bytes, n_elements: int) -> np.ndarray:
    return dequant_mxfp4(raw, n_elements)


def dispatch_tensor(
    info: TensorInfo, raw: bytes, *, iq_allowed: bool | None = None
) -> np.ndarray:
    # Raw GGUF block data → dequantized FP32 numpy array (flat). Raises
    # UnsupportedQuantError for formats without an unambiguous layout
    # handler — the caller (gguf_loader) falls back to native mlx_lm.
    if iq_allowed is None:
        iq_allowed = is_iq_enabled()
    handler_name = _DISPATCHERS.get(info.dtype_name)
    if handler_name is None:
        raise UnsupportedQuantError(
            f"mixed_quant: no dequant handler for dtype {info.dtype_name} "
            f"(tensor {info.name}); falling back to native mlx_lm"
        )
    if info.dtype_name == "iq4_nl" and not iq_allowed:
        logger.info(
            "mixed_quant: %s is iq4_nl but FUSION_SHIM_IQ is OFF; "
            "falling back to native mlx_lm",
            info.name,
        )
        raise UnsupportedQuantError(
            f"iq4_nl consumption disabled (FUSION_SHIM_IQ=0) for {info.name}"
        )
    n_elements = 1
    for d in info.dims:
        n_elements *= d
    handler = globals()[handler_name]
    out = handler(raw, n_elements)
    logger.debug(
        "mixed_quant dispatch %s: dtype=%s elems=%d",
        info.name,
        info.dtype_name,
        n_elements,
    )
    return out


def supported_dtype(dtype_name: str) -> bool:
    # iq4_nl counts as supported only when FUSION_SHIM_IQ is on.
    if dtype_name == "iq4_nl":
        return is_iq_enabled()
    return dtype_name in _DISPATCHERS
