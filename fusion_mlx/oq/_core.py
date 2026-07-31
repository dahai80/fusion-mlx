# SPDX-License-Identifier: Apache-2.0
"""oQ: FusionMLX Universal Dynamic Quantization.

Mixed-precision quantization combining GGUF K-quant layer position strategy,
unsloth Dynamic 2.0 selective non-quantization, and BnB MSE-optimal clipping.

Supported levels: oQ2, oQ2.5, oQ2.7, oQ3, oQ3.5, oQ4, oQ5, oQ6, oQ8 (base
bits differ, same predicate). Fractional levels keep the lower level's base
bits and add a mandatory boost for routed expert down_proj (Super Weights
protection; see _LEVEL_EXPERT_DOWN_BOOST) plus a higher bpw budget.
"""

import logging

try:
    import mlx.core as mx
    import mlx.nn as nn  # noqa: F401 - availability check (HAS_MLX is the signal)
    from mlx.utils import tree_flatten  # noqa: F401 - availability check
    from mlx_lm.models.base import (
        create_attention_mask,  # noqa: F401 - availability check
    )

    HAS_MLX = True
except ImportError:
    HAS_MLX = False


logger = logging.getLogger(__name__)

OQ_LEVELS = {2, 2.5, 2.7, 3, 3.5, 4, 5, 6, 8}

OQ_DTYPES: tuple[str, ...] = ("bfloat16", "float16")

_OQ_DEFAULT_GROUP_SIZE = 64

_MAX_MODEL_RAM_FRACTION = 0.8

# Auto-built proxy for sensitivity measurement when the source model
# exceeds available RAM. Uniform 4-bit affine quant — same shape as a
# user-supplied --sensitivity-model, but built on demand.
_PROXY_QUANT_BITS = 4
_PROXY_QUANT_GROUP_SIZE = 64

_LEVEL_BITS: dict[float, int] = {
    2: 2,
    2.5: 2,
    2.7: 2,
    3: 3,
    3.5: 3,
    4: 4,
    5: 5,
    6: 6,
    8: 8,
}

_LEVEL_PROTECTION: dict[float, str] = {
    2: "full",
    2.5: "full",
    2.7: "full",
    3: "full",
    3.5: "full",
    4: "full",
    5: "full",
    6: "full",
    8: "full",
}

# Fractional levels: mandatory protection for routed expert down_proj
# (Super Weights), expressed as bits above the level's base bits.
# 2.5 -> 3-bit, 2.7 -> 4-bit, 3.5 -> 4-bit.
_LEVEL_EXPERT_DOWN_BOOST: dict[float, int] = {2.5: 1, 2.7: 2, 3.5: 1}

_OQ_BPW_TARGETS: dict[float, tuple[float, float]] = {
    2: (2.8, 3.0),
    2.5: (3.1, 3.3),
    2.7: (3.35, 3.45),
    3: (3.5, 3.7),
    3.5: (3.8, 4.0),
    4: (4.6, 4.7),
    5: (5.5, 5.7),
    6: (6.5, 6.7),
}


class _TrackedTensor:
    """Fake tensor proxy that records shape, dtype, lineage, and transforms
    applied during a sanitize() dry run. Holds no GPU data."""

    __slots__ = (
        "shape",
        "ndim",
        "dtype",
        "sources",
        "transform",
        "axis",
        "recipe",
        "expr",
    )

    def __init__(
        self,
        shape,
        dtype,
        sources=None,
        transform="passthrough",
        axis=None,
        recipe=None,
        expr=None,
    ):
        self.shape = tuple(shape)
        self.ndim = len(self.shape)
        self.dtype = dtype
        self.sources = sources or []
        self.transform = transform
        self.axis = axis
        self.recipe = list(recipe or [])
        if expr is None and transform == "passthrough" and len(self.sources) == 1:
            expr = ("source", self.sources[0])
        self.expr = expr

    def _clone(self, shape=None, dtype=None, transform=None):
        new_transform = transform if transform is not None else self.transform
        return _TrackedTensor(
            shape if shape is not None else self.shape,
            dtype if dtype is not None else self.dtype,
            list(self.sources),
            new_transform,
            recipe=list(self.recipe),
            expr=self.expr if new_transform == self.transform else None,
        )

    # Arithmetic — recipe is "fp8_dequant" for the whole sanitize block if weight came from FP8
    def __add__(self, other):
        return self._clone(transform="add")

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        return self._clone(transform="sub")

    def __mul__(self, other):
        return self._clone(transform="mul")

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        return self._clone(transform="div")

    @staticmethod
    def _slice_length(dim, sl):
        start, stop, step = sl.indices(dim)
        return len(range(start, stop, step))

    @staticmethod
    def _detect_half_split(dim, sl):
        start, stop, step = sl.indices(dim)
        if step != 1 or dim <= 0 or dim % 2 != 0:
            return None
        length = len(range(start, stop, step))
        if length != dim // 2:
            return None
        if start == 0:
            return 0
        if start == dim // 2:
            return 1
        return None

    @staticmethod
    def _expand_index(idx, rank):
        if not isinstance(idx, tuple):
            return idx
        if Ellipsis not in idx:
            return idx
        explicit = sum(1 for p in idx if p is not Ellipsis and p is not None)
        pad = max(0, rank - explicit)
        expanded: list = []
        seen = False
        for part in idx:
            if part is Ellipsis:
                if seen:
                    raise ValueError("only one Ellipsis allowed in index")
                seen = True
                expanded.extend([slice(None)] * pad)
            else:
                expanded.append(part)
        return tuple(expanded)

    def _with_recipe(self, shape, transform, op, axis=None):
        expr = self.as_expr()
        if expr is not None:
            expr = self._wrap_expr_op(expr, op)
        return _TrackedTensor(
            shape,
            self.dtype,
            list(self.sources),
            transform,
            axis=axis,
            recipe=list(self.recipe) + [op],
            expr=expr,
        )

    @staticmethod
    def _wrap_expr_op(expr, op):
        kind = op[0]
        if kind == "reshape":
            return ("reshape", op[1], expr)
        if kind == "slice":
            return ("slice", op[1], expr)
        if kind == "transpose":
            return ("transpose", op[1], expr)
        if kind == "moveaxis":
            return ("moveaxis", op[1], op[2], expr)
        if kind == "astype":
            return ("astype", op[1], expr)
        if kind == "expand_dims":
            return ("expand_dims", op[1], expr)
        return None

    def as_expr(self):
        if self.expr is not None:
            return self.expr
        if self.recipe and len(self.sources) == 1:
            expr = ("source", self.sources[0])
            for op in self.recipe:
                expr = self._wrap_expr_op(expr, op)
                if expr is None:
                    return None
            return expr
        if self.transform == "passthrough" and len(self.sources) == 1:
            return ("source", self.sources[0])
        if self.transform == "stack":
            axis = self.axis if self.axis is not None else 0
            return ("stack", axis, [("source", src) for src in self.sources])
        if self.transform == "concatenate":
            axis = self.axis if self.axis is not None else 0
            return ("concatenate", axis, [("source", src) for src in self.sources])
        return None

    @staticmethod
    def _normalize_expand_axes(axis, ndim):
        axes = (axis,) if isinstance(axis, int) else tuple(axis)
        out_ndim = ndim + len(axes)
        normalized = []
        for ax in axes:
            ax = ax + out_ndim if ax < 0 else ax
            if ax < 0 or ax >= out_ndim:
                raise ValueError(f"axis {ax} is out of bounds for expand_dims")
            normalized.append(ax)
        if len(set(normalized)) != len(normalized):
            raise ValueError("repeated axis in expand_dims")
        return tuple(sorted(normalized))

    def expand_dims(self, axis):
        axes = self._normalize_expand_axes(axis, self.ndim)
        axis_set = set(axes)
        src_i = 0
        new_shape = []
        for i in range(self.ndim + len(axes)):
            if i in axis_set:
                new_shape.append(1)
            else:
                new_shape.append(self.shape[src_i])
                src_i += 1
        stored_axis = axes[0] if len(axes) == 1 else axes
        return self._with_recipe(
            tuple(new_shape),
            "expand_dims",
            ("expand_dims", axes),
            axis=stored_axis,
        )

    def __getitem__(self, idx):
        new_shape = list(self.shape)
        idx = self._expand_index(idx, len(new_shape))
        if isinstance(idx, tuple):
            result_shape = []
            axis = 0
            split_info = None
            for part in idx:
                if part is None:
                    result_shape.append(1)
                elif isinstance(part, slice):
                    if axis < len(new_shape):
                        dim = new_shape[axis]
                        length = self._slice_length(dim, part)
                        result_shape.append(length)
                        half = self._detect_half_split(dim, part)
                        if half is not None:
                            split_info = (axis, half, 2)
                        axis += 1
                    else:
                        result_shape.append(1)
                else:
                    if axis < len(new_shape):
                        axis += 1
            while axis < len(new_shape):
                result_shape.append(new_shape[axis])
                axis += 1
            if split_info is not None:
                ax, idx_n, total = split_info
                return _TrackedTensor(
                    result_shape,
                    self.dtype,
                    list(self.sources),
                    f"split_{idx_n}_{total}",
                    axis=ax,
                    recipe=list(self.recipe) + [("slice", idx)],
                )
            return self._with_recipe(result_shape, "slice", ("slice", idx))
        if isinstance(idx, slice):
            dim = new_shape[0] if new_shape else 0
            length = self._slice_length(dim, idx) if dim > 0 else 0
            half = self._detect_half_split(dim, idx) if dim > 0 else None
            if half is not None:
                return _TrackedTensor(
                    [length] + new_shape[1:],
                    self.dtype,
                    list(self.sources),
                    f"split_{half}_2",
                    axis=0,
                    recipe=list(self.recipe) + [("slice", idx)],
                )
            result = list(new_shape)
            if result:
                result[0] = length
            return self._with_recipe(result, "slice", ("slice", idx))
        # int or other
        if new_shape:
            return self._with_recipe(new_shape[1:], "slice", ("slice", idx))
        return self._with_recipe(self.shape, "slice", ("slice", idx))

    def reshape(self, *new_shape):
        if len(new_shape) == 1 and isinstance(new_shape[0], (tuple, list)):
            new_shape = tuple(new_shape[0])
        # Resolve any -1 using total element count
        total = 1
        for d in self.shape:
            total *= d
        resolved = []
        unknown_idx = -1
        known_prod = 1
        for i, d in enumerate(new_shape):
            if d == -1:
                unknown_idx = i
                resolved.append(-1)
            else:
                resolved.append(d)
                known_prod *= d
        if unknown_idx >= 0 and known_prod > 0:
            resolved[unknown_idx] = total // known_prod
        shape = tuple(resolved)
        return _TrackedTensor(
            shape,
            self.dtype,
            list(self.sources),
            "reshape",
            recipe=list(self.recipe) + [("reshape", shape)],
        )

    def astype(self, dtype):
        return _TrackedTensor(
            self.shape,
            dtype,
            list(self.sources),
            "astype",
            recipe=list(self.recipe) + [("astype", dtype)],
        )

    def moveaxis(self, src_ax, dst_ax):
        src_ax = src_ax % self.ndim if src_ax < 0 else src_ax
        dst_ax = dst_ax % self.ndim if dst_ax < 0 else dst_ax
        dims = list(range(self.ndim))
        dims.insert(dst_ax, dims.pop(src_ax))
        new_shape = tuple(self.shape[d] for d in dims)
        return _TrackedTensor(
            new_shape,
            self.dtype,
            list(self.sources),
            f"moveaxis_{src_ax}_{dst_ax}",
            recipe=list(self.recipe) + [("moveaxis", src_ax, dst_ax)],
        )

    def transpose(self, *axes):
        if not axes:
            axes_list = list(reversed(range(self.ndim)))
        elif len(axes) == 1 and isinstance(axes[0], (list, tuple)):
            axes_list = list(axes[0])
        else:
            axes_list = list(axes)
        axes_list = [a % self.ndim if a < 0 else a for a in axes_list]
        new_shape = tuple(self.shape[a] for a in axes_list)
        return _TrackedTensor(
            new_shape,
            self.dtype,
            list(self.sources),
            "transpose_" + "_".join(str(a) for a in axes_list),
            recipe=list(self.recipe) + [("transpose", tuple(axes_list))],
        )

    def swapaxes(self, axis1, axis2):
        axes_list = list(range(self.ndim))
        axis1 = axis1 % self.ndim if axis1 < 0 else axis1
        axis2 = axis2 % self.ndim if axis2 < 0 else axis2
        axes_list[axis1], axes_list[axis2] = axes_list[axis2], axes_list[axis1]
        return self.transpose(axes_list)

    @property
    def T(self):
        axes = tuple(reversed(range(self.ndim)))
        return _TrackedTensor(
            tuple(reversed(self.shape)),
            self.dtype,
            list(self.sources),
            "transpose",
            recipe=list(self.recipe) + [("transpose", axes)],
        )

    @property
    def size(self):
        r = 1
        for d in self.shape:
            r *= d
        return r


_FP8_WEIGHT_DTYPES = frozenset(("F8_E4M3", "F8_E5M2", "I8"))


def _block_dequant_fp8(weight_raw, scale_raw, w_dtype, s_dtype):
    """Block-scaled dequant of a single FP8/I8 weight+scale pair to BF16."""
    if s_dtype == "F8_E8M0":
        scale = mx.power(mx.array(2.0), scale_raw.astype(mx.float32) - 127.0)
    else:
        scale = scale_raw

    if w_dtype in ("F8_E4M3", "F8_E5M2"):
        weight = mx.from_fp8(weight_raw, dtype=mx.bfloat16)
    else:
        weight = weight_raw.astype(mx.bfloat16)

    m, n = weight.shape
    sm, sn = scale.shape
    if sm == 0 or sn == 0:
        raise ValueError(f"degenerate scale shape {scale.shape}")

    def _infer_block(dim: int, blocks: int) -> int | None:
        if dim % blocks == 0:
            return dim // blocks
        for block in (128, 64, 32, 256, 16, 8):
            if (dim + block - 1) // block == blocks:
                return block
        return None

    bs_row = _infer_block(m, sm)
    bs_col = _infer_block(n, sn)
    if bs_row is None or bs_col is None:
        raise ValueError(
            f"weight shape ({m},{n}) not divisible by scale shape ({sm},{sn})"
        )

    if bs_row > 1:
        target_m = sm * bs_row
        target_n = sn * bs_col
        pad_bottom = max(0, target_m - m)
        pad_side = max(0, target_n - n)
        if pad_bottom or pad_side:
            weight = mx.pad(weight, ((0, pad_bottom), (0, pad_side)))
        weight = weight.reshape(sm, bs_row, sn, bs_col)
        weight = (weight * scale[:, None, :, None]).reshape(
            m + pad_bottom, n + pad_side
        )
        if pad_bottom or pad_side:
            weight = weight[:m, :n]
    else:
        weight = weight.reshape(m, sn, bs_col)
        weight = (weight * scale[:, :, None]).reshape(m, n)

    weight = weight.astype(mx.bfloat16)
    mx.eval(weight)
    return weight


def _discover_sanitize_plan(sanitize_fn, lazy_index):
    """Run sanitize on _TrackedTensors to discover the key mapping and
    transforms without materializing any real data.

    Returns a dict: output_key -> {sources, transform, shape, axis}
    or None if discovery fails.
    """
    import mlx.core as mx

    # Build tracked dict mirroring the lazy index (logical view hides scale
    # keys and reports FP8 weights as BF16 so sanitize won't call from_fp8)
    tracked = {}
    initial_meta = {}
    if hasattr(lazy_index, "logical_metadata"):
        logical = lazy_index.logical_metadata()
        for k, (shape, dtype) in logical.items():
            tracked[k] = _TrackedTensor(shape, dtype, sources=[k])
            initial_meta[k] = (tuple(shape), dtype)
    else:
        for k in lazy_index._index:
            meta = lazy_index._index[k]
            shape, dtype = meta[4], meta[5]
            tracked[k] = _TrackedTensor(shape, dtype, sources=[k])
            initial_meta[k] = (tuple(shape), dtype)

    # Monkey-patch mx ops to work on tracked tensors
    _orig = {
        "stack": mx.stack,
        "concatenate": mx.concatenate,
        "split": mx.split,
        "eval": mx.eval,
        "clear_cache": mx.clear_cache,
        "synchronize": mx.synchronize,
        "moveaxis": mx.moveaxis,
        "transpose": mx.transpose,
        "swapaxes": getattr(mx, "swapaxes", None),
        "expand_dims": mx.expand_dims,
        "contiguous": getattr(mx, "contiguous", None),
        "from_fp8": getattr(mx, "from_fp8", None),
        "pad": getattr(mx, "pad", None),
    }

    def _is_plain_source(tensor):
        return (
            isinstance(tensor, _TrackedTensor)
            and tensor.transform == "passthrough"
            and not tensor.recipe
            and len(tensor.sources) == 1
        )

    def _fake_stack(tensors, axis=0):
        if tensors and isinstance(tensors[0], _TrackedTensor):
            n = len(tensors)
            base = list(tensors[0].shape)
            axis = axis + len(base) + 1 if axis < 0 else axis
            new_shape = base[:axis] + [n] + base[axis:]
            all_src = []
            for t in tensors:
                all_src.extend(t.sources)
            if not all(_is_plain_source(t) for t in tensors):
                exprs = [t.as_expr() for t in tensors]
                if any(expr is None for expr in exprs):
                    return _TrackedTensor(
                        new_shape,
                        tensors[0].dtype,
                        all_src,
                        "nested_unreplayable",
                    )
                return _TrackedTensor(
                    new_shape,
                    tensors[0].dtype,
                    all_src,
                    "expr",
                    axis=axis,
                    expr=("stack", axis, exprs),
                )
            return _TrackedTensor(
                new_shape, tensors[0].dtype, all_src, "stack", axis=axis
            )
        return _orig["stack"](tensors, axis=axis)

    def _fake_concatenate(tensors, axis=0):
        if tensors and isinstance(tensors[0], _TrackedTensor):
            all_src = []
            for t in tensors:
                all_src.extend(t.sources)
            base = list(tensors[0].shape)
            axis = axis + len(base) if axis < 0 else axis
            base[axis] = sum(t.shape[axis] for t in tensors)
            if not all(_is_plain_source(t) for t in tensors):
                exprs = [t.as_expr() for t in tensors]
                if any(expr is None for expr in exprs):
                    return _TrackedTensor(
                        base,
                        tensors[0].dtype,
                        all_src,
                        "nested_unreplayable",
                    )
                return _TrackedTensor(
                    base,
                    tensors[0].dtype,
                    all_src,
                    "expr",
                    axis=axis,
                    expr=("concatenate", axis, exprs),
                )
            return _TrackedTensor(
                base, tensors[0].dtype, all_src, "concatenate", axis=axis
            )
        return _orig["concatenate"](tensors, axis=axis)

    def _fake_split(tensor, indices_or_sections, axis=0):
        if isinstance(tensor, _TrackedTensor):
            if isinstance(indices_or_sections, int):
                n = indices_or_sections
                sz = tensor.shape[axis] // n
                parts = []
                for i in range(n):
                    sh = list(tensor.shape)
                    sh[axis] = sz
                    parts.append(
                        _TrackedTensor(
                            sh,
                            tensor.dtype,
                            list(tensor.sources),
                            f"split_{i}_{n}",
                            axis=axis,
                        )
                    )
                return parts
            # list of indices
            parts = []
            prev = 0
            idxs = list(indices_or_sections) + [tensor.shape[axis]]
            for i, idx in enumerate(idxs):
                sh = list(tensor.shape)
                sh[axis] = idx - prev
                parts.append(
                    _TrackedTensor(
                        sh, tensor.dtype, list(tensor.sources), f"split_{i}", axis=axis
                    )
                )
                prev = idx
            return parts
        return _orig["split"](tensor, indices_or_sections, axis=axis)

    def _fake_moveaxis(tensor, src_ax, dst_ax):
        if isinstance(tensor, _TrackedTensor):
            return tensor.moveaxis(src_ax, dst_ax)
        return _orig["moveaxis"](tensor, src_ax, dst_ax)

    def _fake_transpose(tensor, axes=None):
        if isinstance(tensor, _TrackedTensor):
            if axes is None:
                axes = list(reversed(range(tensor.ndim)))
            return tensor.transpose(tuple(axes))
        return _orig["transpose"](tensor, axes=axes)

    def _fake_swapaxes(tensor, axis1, axis2):
        if isinstance(tensor, _TrackedTensor):
            return tensor.swapaxes(axis1, axis2)
        return _orig["swapaxes"](tensor, axis1, axis2)

    def _fake_expand_dims(tensor, axis, **kwargs):
        if isinstance(tensor, _TrackedTensor):
            return tensor.expand_dims(axis)
        return _orig["expand_dims"](tensor, axis=axis, **kwargs)

    def _fake_contiguous(tensor, *args, **kwargs):
        if isinstance(tensor, _TrackedTensor):
            return tensor
        return _orig["contiguous"](tensor, *args, **kwargs)

    def _noop(*a, **kw):
        pass

    mx.stack = _fake_stack
    mx.concatenate = _fake_concatenate
    mx.split = _fake_split
    mx.eval = _noop
    mx.clear_cache = _noop
    mx.synchronize = _noop
    mx.moveaxis = _fake_moveaxis
    mx.transpose = _fake_transpose
    mx.expand_dims = _fake_expand_dims
    if _orig["swapaxes"] is not None:
        mx.swapaxes = _fake_swapaxes
    if _orig["contiguous"] is not None:
        mx.contiguous = _fake_contiguous

    def _fake_from_fp8(x, dtype=None, **kw):
        if isinstance(x, _TrackedTensor):
            return _TrackedTensor(
                x.shape, dtype or x.dtype, list(x.sources), "from_fp8"
            )
        return _orig["from_fp8"](x, dtype=dtype, **kw) if _orig["from_fp8"] else x

    def _fake_pad(x, pad_width, **kw):
        if isinstance(x, _TrackedTensor):
            new_shape = []
            for i, d in enumerate(x.shape):
                if i < len(pad_width):
                    lo, hi = (
                        pad_width[i]
                        if isinstance(pad_width[i], (tuple, list))
                        else (pad_width[i], pad_width[i])
                    )
                    new_shape.append(d + lo + hi)
                else:
                    new_shape.append(d)
            return _TrackedTensor(new_shape, x.dtype, list(x.sources), "pad")
        return _orig["pad"](x, pad_width, **kw) if _orig["pad"] else x

    if _orig["from_fp8"] is not None:
        mx.from_fp8 = _fake_from_fp8
    if _orig["pad"] is not None:
        mx.pad = _fake_pad

    try:
        result = sanitize_fn(tracked)
    finally:
        for name, fn in _orig.items():
            if fn is not None:
                setattr(mx, name, fn)

    # Extract plan
    _REPLAYABLE_PREFIXES = (
        "passthrough",
        "literal",
        "stack",
        "concatenate",
        "add",
        "add_if_mean_lt_0_5",
        "transpose_",
        "moveaxis_",
        "split_",
        "slice",
        "reshape",
        "astype",
        "expand_dims",
        "expr",
    )
    plan = {}
    for k, v in result.items():
        if isinstance(v, _TrackedTensor):
            t = v.transform
            if not any(t == p or t.startswith(p) for p in _REPLAYABLE_PREFIXES):
                raise ValueError(
                    f"non-replayable transform {t!r} for {k!r} — "
                    "falling back to eager sanitize"
                )
            plan[k] = {
                "sources": v.sources,
                "transform": t,
                "shape": v.shape,
                "axis": v.axis,
                "recipe": list(v.recipe),
            }
            if v.transform == "expr":
                if v.expr is None:
                    raise ValueError(
                        f"missing replay expression for {k!r} — "
                        "falling back to eager sanitize"
                    )
                plan[k]["expr"] = v.expr
            if v.recipe:
                if len(v.sources) != 1:
                    raise ValueError(
                        f"recipe with non-trivial sources for {k!r} — "
                        "falling back to eager sanitize"
                    )
            elif t in ("reshape", "astype"):
                # Only the LAST transform is tracked, so replay is sound
                # only when nothing else touched the tensor: an astype must
                # keep the source shape and a reshape must keep the source
                # dtype. Chains (e.g. reshape-then-astype) fall back to
                # eager sanitize, matching the pre-replay behavior.
                src_meta = (
                    initial_meta.get(v.sources[0]) if len(v.sources) == 1 else None
                )
                if src_meta is None:
                    raise ValueError(
                        f"{t} with non-trivial sources for {k!r} — "
                        "falling back to eager sanitize"
                    )
                if t == "astype" and tuple(v.shape) != src_meta[0]:
                    raise ValueError(
                        f"astype after a shape-changing op for {k!r} — "
                        "falling back to eager sanitize"
                    )
                if t == "reshape" and v.dtype != src_meta[1]:
                    raise ValueError(
                        f"reshape after a dtype-changing op for {k!r} — "
                        "falling back to eager sanitize"
                    )
            if t == "astype":
                # _TrackedTensor.astype records the target mx dtype.
                plan[k]["dtype"] = v.dtype
        else:
            plan[k] = {
                "sources": [],
                "transform": "literal",
                "shape": getattr(v, "shape", ()),
                "axis": None,
                "value": v,
            }

    return plan


class _DiscoveredPlan:
    """Dict-like wrapper that materializes tensors one at a time using
    a plan discovered by _discover_sanitize_plan. Supports chunked
    stacking for huge MoE expert tensors."""

    _STACK_CHUNK = 16  # experts per chunk during materialization

    def __init__(self, plan, lazy_index):
        self._plan = plan  # output_key -> {sources, transform, ...}
        self._lazy = lazy_index
        self._cache = {}  # output_key -> mx.array (for multi-consumer sources)

    def keys(self):
        return self._plan.keys()

    def __len__(self):
        return len(self._plan)

    def __contains__(self, k):
        return k in self._plan

    def __iter__(self):
        return iter(self._plan)

    def items(self):
        # Yield (key, shape_proxy) for the quantize loop shape inspection
        class _SP:
            __slots__ = ("shape", "ndim")

            def __init__(self, sh):
                self.shape = tuple(sh)
                self.ndim = len(self.shape)

        return ((k, _SP(info["shape"])) for k, info in self._plan.items())

    def nbytes(self):
        return self._lazy.nbytes()

    def plan_shape(self, key):
        """Logical output shape for a planned key without materializing."""
        return tuple(self._plan[key]["shape"])

    def source_quant_info(self, key):
        """Common pre-quantized source metadata for an output key, or None.

        Only meaningful for transforms that preserve the packed layout
        (passthrough, stack, single-source reshape) where every source is
        the same passthrough-capable format.
        """
        info = self._plan.get(key)
        if info is None or not hasattr(self._lazy, "source_quant_info"):
            return None
        transform = info["transform"]
        sources = info["sources"]
        recipe = info.get("recipe") or []
        if not sources:
            return None
        if recipe and not (
            transform == "reshape" and len(recipe) == 1 and recipe[0][0] == "reshape"
        ):
            return None
        if transform not in ("passthrough", "stack") and not (
            transform == "reshape" and len(sources) == 1
        ):
            return None
        first = self._lazy.source_quant_info(sources[0])
        if first is None:
            return None
        for src in sources[1:]:
            if self._lazy.source_quant_info(src) != first:
                return None
        return first

    def pop_packed(self, key):
        """Materialize a pre-quantized output tensor in mlx packed form.

        Returns (weight, scales). Only valid when source_quant_info(key)
        returned a dict; consumes the plan entry like pop().
        """
        info = self._plan.pop(key)
        transform = info["transform"]
        sources = info["sources"]

        if transform == "passthrough":
            return self._lazy._load_packed(sources[0])

        if transform == "reshape":
            w, s = self._lazy._load_packed(sources[0])
            lead = tuple(info["shape"][:-1])
            return mx.reshape(w, lead + (-1,)), mx.reshape(s, lead + (-1,))

        if transform == "stack":
            axis = info.get("axis", 0)
            chunk = self._STACK_CHUNK
            w_parts, s_parts = [], []
            for base in range(0, len(sources), chunk):
                w_piece, s_piece = [], []
                for src in sources[base : base + chunk]:
                    w, s = self._lazy._load_packed(src)
                    w_piece.append(w)
                    s_piece.append(s)
                w_stk = mx.stack(w_piece, axis=axis)
                s_stk = mx.stack(s_piece, axis=axis)
                mx.eval(w_stk, s_stk)
                del w_piece, s_piece
                mx.clear_cache()
                w_parts.append(w_stk)
                s_parts.append(s_stk)
            if len(w_parts) == 1:
                return w_parts[0], s_parts[0]
            w_res = mx.concatenate(w_parts, axis=axis)
            s_res = mx.concatenate(s_parts, axis=axis)
            mx.eval(w_res, s_res)
            del w_parts, s_parts
            mx.clear_cache()
            return w_res, s_res

        raise ValueError(f"cannot materialize packed {key!r}: transform={transform}")

    def _materialize_source(self, src_key):
        """Load a single source tensor from the lazy index."""
        from .io import _LazyTensor  # lazy: io imports from _core at module load

        if hasattr(self._lazy, "_fp8_pairs") and src_key in self._lazy._fp8_pairs:
            return self._lazy._dequant_one(src_key)
        meta = self._lazy._index.get(src_key)
        if meta is None:
            raise KeyError(f"source tensor {src_key!r} not in lazy index")
        sf_path, data_offset, start, end, shape, dtype = meta
        if len(shape) == 0:
            import numpy as _np

            with open(sf_path, "rb") as f:
                f.seek(data_offset + start)
                raw = f.read(end - start)
            lt_tmp = _LazyTensor(sf_path, data_offset, start, end, (1,), dtype)
            np_view = _np.frombuffer(raw, dtype=lt_tmp._np_view_dtype())
            arr = mx.array(np_view).view(lt_tmp._mlx_dtype()).reshape(())
            mx.eval(arr)
            return arr
        lt = _LazyTensor(sf_path, data_offset, start, end, shape, dtype)
        arr = lt[:]
        mx.eval(arr)
        return arr

    @staticmethod
    def _apply_recipe(arr, recipe):
        for op in recipe:
            kind = op[0]
            if kind == "reshape":
                arr = mx.reshape(arr, op[1])
            elif kind == "slice":
                arr = arr[op[1]]
            elif kind == "transpose":
                arr = mx.transpose(arr, axes=op[1])
            elif kind == "moveaxis":
                arr = mx.moveaxis(arr, op[1], op[2])
            elif kind == "astype":
                arr = arr.astype(op[1])
            elif kind == "expand_dims":
                arr = mx.expand_dims(arr, axis=op[1])
            else:
                raise ValueError(f"unsupported replay recipe op: {kind}")
            mx.eval(arr)
        return arr

    def _materialize_expr(self, expr):
        kind = expr[0]

        if kind == "source":
            return self._materialize_source(expr[1])

        if kind == "stack":
            axis = expr[1]
            children = expr[2]
            chunk = self._STACK_CHUNK
            partials = []
            for base in range(0, len(children), chunk):
                piece = [
                    self._materialize_expr(c) for c in children[base : base + chunk]
                ]
                stk = mx.stack(piece, axis=axis)
                mx.eval(stk)
                del piece
                mx.clear_cache()
                partials.append(stk)
            if len(partials) == 1:
                return partials[0]
            result = mx.concatenate(partials, axis=axis)
            mx.eval(result)
            del partials
            mx.clear_cache()
            return result

        child_kinds = {"reshape", "slice", "transpose", "expand_dims", "astype"}
        if kind in child_kinds:
            arr = self._materialize_expr(expr[2])
            if kind == "reshape":
                result = mx.reshape(arr, expr[1])
            elif kind == "slice":
                result = arr[expr[1]]
            elif kind == "transpose":
                result = mx.transpose(arr, axes=expr[1])
            elif kind == "expand_dims":
                result = mx.expand_dims(arr, axis=expr[1])
            else:
                result = arr.astype(expr[1])
            mx.eval(result)
            return result

        if kind == "moveaxis":
            arr = self._materialize_expr(expr[3])
            result = mx.moveaxis(arr, expr[1], expr[2])
            mx.eval(result)
            return result

        if kind == "concatenate":
            axis = expr[1]
            parts = [self._materialize_expr(c) for c in expr[2]]
            result = mx.concatenate(parts, axis=axis)
            mx.eval(result)
            del parts
            mx.clear_cache()
            return result

        raise ValueError(f"unsupported replay expression op: {kind}")

    def pop(self, key, *default):
        if key not in self._plan:
            if default:
                return default[0]
            raise KeyError(key)

        info = self._plan.pop(key)
        transform = info["transform"]
        sources = info["sources"]
        recipe = info.get("recipe") or []

        if transform == "literal":
            return info["value"]

        if transform == "expr":
            return self._materialize_expr(info["expr"])

        if recipe and len(sources) == 1:
            arr = self._materialize_source(sources[0])
            return self._apply_recipe(arr, recipe)

        if transform == "passthrough" and len(sources) == 1:
            arr = self._materialize_source(sources[0])
            return arr

        if transform == "stack":
            # Chunked stacking to bound peak memory
            axis = info.get("axis", 0)
            chunk = self._STACK_CHUNK
            partials = []
            for base in range(0, len(sources), chunk):
                piece = []
                for src in sources[base : base + chunk]:
                    piece.append(self._materialize_source(src))
                stk = mx.stack(piece, axis=axis)
                mx.eval(stk)
                del piece
                mx.clear_cache()
                partials.append(stk)
            if len(partials) == 1:
                return partials[0]
            result = mx.concatenate(partials, axis=axis)
            mx.eval(result)
            del partials
            mx.clear_cache()
            return result

        if transform == "concatenate":
            axis = info.get("axis", 0)
            parts = [self._materialize_source(src) for src in sources]
            result = mx.concatenate(parts, axis=axis)
            mx.eval(result)
            del parts
            mx.clear_cache()
            return result

        if transform == "add":
            arr = self._materialize_source(sources[0])
            return arr + 1.0  # norm weight += 1.0 pattern

        if transform == "add_if_mean_lt_0_5":
            arr = self._materialize_source(sources[0])
            mean = float(mx.mean(arr.astype(mx.float32)).item())
            if mean < 0.5:
                return arr + 1.0
            return arr

        if transform == "reshape":
            arr = self._materialize_source(sources[0])
            return mx.reshape(arr, info["shape"])

        if transform == "astype":
            arr = self._materialize_source(sources[0])
            return arr.astype(info["dtype"])

        if transform.startswith("transpose_"):
            axes = [int(a) for a in transform.split("_")[1:]]
            arr = self._materialize_source(sources[0])
            return mx.transpose(arr, axes=axes)

        if transform.startswith("moveaxis_"):
            parts = transform.split("_")
            src_ax, dst_ax = int(parts[1]), int(parts[2])
            arr = self._materialize_source(sources[0])
            return mx.moveaxis(arr, src_ax, dst_ax)

        if "split_" in transform:
            # split_N_M means take part N of M
            parts = transform.split("_")
            arr = self._materialize_source(sources[0])
            axis = info.get("axis", 0)
            if len(parts) == 3:  # split_idx_total
                idx, total = int(parts[1]), int(parts[2])
                chunks = mx.split(arr, total, axis=axis)
                result = chunks[idx]
                mx.eval(result)
                del arr, chunks
                mx.clear_cache()
                return result
            # split_idx (index-based split) — less common
            return arr

        if transform == "slice":
            raise ValueError(
                f"cannot replay arbitrary slice for {key!r} — "
                "discovery should fall back to eager sanitize"
            )

        # Fallback: passthrough (identity) — load first source unchanged
        if transform == "passthrough" and sources:
            return self._materialize_source(sources[0])

        raise ValueError(
            f"cannot materialize {key!r}: transform={transform}, no sources"
        )
