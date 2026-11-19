# SPDX-License-Identifier: Apache-2.0
"""Gate A' radial/polar 4-bit KV codec (Python reference).

Spec source: arXiv 2610.33076 section 3.3 "Python Polar reference":
seeded Walsh-Hadamard preconditioner, adjacent rotated coordinates
grouped into polar pairs, radius and angle quantized, optional
QJL-style residual sign sketch. The paper ships no code; the gap fills
used here are documented inline and in the experiment log.

Reconstruction path (quality arm): encode cache arrays -> packed bytes
-> decode back to the cache dtype; attention reads the DECODED values.

Bit budget (documented gap fill): 4 bits per stored element = 8 bits per
polar pair, default split 4 bits radius + 4 bits angle, plus one fp16
per-vector radius scale (2 bytes per head_dim=128 vector).

QJL residual sign sketch: sign sketches estimate inner products in the
compressed domain and are not invertible, so they contribute bytes but
cannot improve reconstruction. Kept as an encode-only path for byte
accounting; the quality arm runs with it disabled (paper marks it
optional: "a QJL-style residual sign sketch can be retained").
"""

import logging

import mlx.core as mx
import numpy as np

logger = logging.getLogger(__name__)

# Bit split frozen by the synthetic calibration run (2026-11-13, mixed-scale
# gaussian through the full codec): uniform 3r+5a rel-MSE 1.12% < uniform
# 4r+4a 1.46% < log 3r+5a 19.9%. Same 8 bits per polar pair either way.
R_BITS = 3
A_BITS = 5
R_LEVELS = 1 << R_BITS
A_LEVELS = 1 << A_BITS
RADIUS_MODE = "uniform"
LOG_EPS = 1e-4


def _hadamard_matrix(n):
    h = mx.ones((1, 1), mx.float32)
    while h.shape[0] < n:
        top = mx.concatenate([h, h], axis=1)
        bot = mx.concatenate([h, -h], axis=1)
        h = mx.concatenate([top, bot], axis=0)
    return h / mx.sqrt(mx.array(float(n), mx.float32))


class RadialPolarCodec:
    def __init__(self, dim, seed=0, r_bits=R_BITS, a_bits=A_BITS,
                 radius_mode=RADIUS_MODE):
        if dim % 2 != 0:
            raise ValueError("dim must be even for polar pairing")
        self.dim = dim
        self.r_bits = r_bits
        self.a_bits = a_bits
        self.r_levels = 1 << r_bits
        self.a_levels = 1 << a_bits
        self.radius_mode = radius_mode
        # Seeded sign flips before the Hadamard transform (paper: seeded
        # Walsh-Hadamard preconditioner). numpy RandomState keeps the
        # preconditioner identical across processes and machines.
        rng = np.random.RandomState(seed)
        sign = (rng.randint(0, 2, size=dim) * 2 - 1).astype(np.float32)
        self.sign = mx.array(sign)
        h = _hadamard_matrix(dim)
        # forward: rotated = (x * sign) @ H / sqrt(dim) = x @ (diag(sign) @ H)
        # (H already normalized inside _hadamard_matrix; diag(sign) @ H is
        # a row scaling of H).
        self.fwd = h * self.sign[:, None]
        # inverse: x = (rotated @ H / sqrt(dim)) * sign
        self.inv = h
        self.rmax_rel_eps = LOG_EPS

    def encode(self, x):
        x = x.astype(mx.float32)
        rot = x @ self.fwd
        pairs = rot.reshape(*rot.shape[:-1], self.dim // 2, 2)
        a = pairs[..., 0]
        b = pairs[..., 1]
        r = mx.sqrt(a * a + b * b)
        theta = mx.arctan2(b, a)
        rmax = mx.max(r, axis=-1, keepdims=True)
        rmax = mx.maximum(rmax, mx.array(1e-12, mx.float32))
        if self.radius_mode == "log":
            lo = rmax * self.rmax_rel_eps
            rc = mx.clip(r, lo, rmax)
            t = (mx.log(rc / lo) / mx.log(rmax / lo)) * (self.r_levels - 1)
        else:
            t = (r / rmax) * (self.r_levels - 1)
        rq = mx.clip(mx.round(t), 0, self.r_levels - 1).astype(mx.uint8)
        tq = mx.clip(
            mx.floor((theta + mx.pi) / (2.0 * mx.pi) * self.a_levels),
            0,
            self.a_levels - 1,
        ).astype(mx.uint8)
        packed = rq * mx.array(self.a_levels, mx.uint8) + tq
        scale = rmax.astype(mx.float16)
        return packed, scale

    def decode(self, packed, scale):
        rq = (packed >> self.a_bits).astype(mx.float32)
        tq = (packed & (self.a_levels - 1)).astype(mx.float32)
        rmax = scale.astype(mx.float32)
        if self.radius_mode == "log":
            lo = rmax * self.rmax_rel_eps
            r = lo * mx.power(rmax / lo, rq / (self.r_levels - 1))
        else:
            r = rq / (self.r_levels - 1) * rmax
        theta = (tq + 0.5) / self.a_levels * (2.0 * mx.pi) - mx.pi
        a = r * mx.cos(theta)
        b = r * mx.sin(theta)
        rot = mx.concatenate([a[..., None], b[..., None]], axis=-1)
        rot = rot.reshape(*packed.shape[:-1], self.dim)
        xf = rot @ self.inv
        return (xf * self.sign).astype(mx.bfloat16)

    def roundtrip(self, x, out_dtype=None):
        packed, scale = self.encode(x)
        y = self.decode(packed, scale)
        if out_dtype is not None:
            y = y.astype(out_dtype)
        return y

    def bytes_per_vector(self, sketch=False):
        # packed: dim/2 pairs at (r_bits + a_bits) bits; scale: one fp16
        main = self.dim // 2
        s = self.dim // 8 if sketch else 0
        return main + 2 + s


def sign_sketch_bytes(dim):
    # QJL-style residual sign sketch: 1 bit per rotated coordinate.
    return dim // 8


def radial_roundtrip_array(codec, x, out_dtype):
    y = codec.roundtrip(x, out_dtype=out_dtype)
    mx.eval(y)
    return y
