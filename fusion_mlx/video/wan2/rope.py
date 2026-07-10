import mlx.core as mx
import numpy as np


def rope_params(max_seq_len: int, dim: int, theta: float = 10000.0) -> mx.array:
    assert dim % 2 == 0
    freqs = (
        np.arange(max_seq_len, dtype=np.float64)[:, None]
        * (
            1.0
            / np.power(
                theta,
                np.arange(0, dim, 2, dtype=np.float64) / dim,
            )
        )[None, :]
    )
    # Store as (cos, sin) pairs: shape [max_seq_len, dim // 2, 2]
    cos_freqs = np.cos(freqs).astype(np.float32)
    sin_freqs = np.sin(freqs).astype(np.float32)
    return mx.array(np.stack([cos_freqs, sin_freqs], axis=-1))


def rope_apply(
    x: mx.array,
    grid_sizes: list,
    freqs: mx.array,
    precomputed_cos_sin: tuple | None = None,
) -> mx.array:
    b, s, n, d = x.shape
    half_d = d // 2

    if precomputed_cos_sin is not None:
        cos_f, sin_f = precomputed_cos_sin
        # Check if all batch elements have the same grid (common for CFG B=2)
        f0, h0, w0 = grid_sizes[0]
        seq_len = f0 * h0 * w0
        all_same_grid = (
            all(grid_sizes[i] == grid_sizes[0] for i in range(1, b)) if b > 1 else True
        )

        if all_same_grid:
            # Vectorized path: apply RoPE to all batch elements at once
            x_seq = x[:, :seq_len].reshape(b, seq_len, n, half_d, 2)
            x_real = x_seq[..., 0]
            x_imag = x_seq[..., 1]
            out_real = x_real * cos_f - x_imag * sin_f
            out_imag = x_real * sin_f + x_imag * cos_f
            x_rotated = mx.stack([out_real, out_imag], axis=-1).reshape(
                b, seq_len, n, d
            )
            if seq_len < s:
                x_rotated = mx.concatenate([x_rotated, x[:, seq_len:]], axis=1)
            return x_rotated
        else:
            # Per-element path for mixed grid sizes
            outputs = []
            for i in range(b):
                f, h, w = grid_sizes[i]
                sl = f * h * w
                x_i = x[i, :sl].reshape(sl, n, half_d, 2)
                x_real = x_i[..., 0]
                x_imag = x_i[..., 1]
                out_real = x_real * cos_f - x_imag * sin_f
                out_imag = x_real * sin_f + x_imag * cos_f
                x_rotated = mx.stack([out_real, out_imag], axis=-1).reshape(sl, n, d)
                if sl < s:
                    x_rotated = mx.concatenate([x_rotated, x[i, sl:]], axis=0)
                outputs.append(x_rotated)
            return mx.stack(outputs)

    # Cast freqs to input dtype to prevent float32 promotion cascade
    if freqs.dtype != x.dtype:
        freqs = freqs.astype(x.dtype)

    # Split frequency dimensions: temporal gets more capacity
    d_t = half_d - 2 * (half_d // 3)
    d_h = half_d // 3
    d_w = half_d // 3

    # Split freqs along dim axis
    freqs_t = freqs[:, :d_t]  # [1024, d_t, 2]
    freqs_h = freqs[:, d_t : d_t + d_h]  # [1024, d_h, 2]
    freqs_w = freqs[:, d_t + d_h : d_t + d_h + d_w]  # [1024, d_w, 2]

    outputs = []
    for i in range(b):
        f, h, w = grid_sizes[i]
        seq_len = f * h * w

        # Reshape x to pairs for rotation: [seq_len, n, half_d, 2]
        x_i = x[i, :seq_len].reshape(seq_len, n, half_d, 2)

        # Build per-position frequencies by expanding along grid dims
        # temporal: [f,1,1,d_t,2] -> [f,h,w,d_t,2]
        ft = mx.broadcast_to(freqs_t[:f].reshape(f, 1, 1, d_t, 2), (f, h, w, d_t, 2))
        # height: [1,h,1,d_h,2] -> [f,h,w,d_h,2]
        fh = mx.broadcast_to(freqs_h[:h].reshape(1, h, 1, d_h, 2), (f, h, w, d_h, 2))
        # width: [1,1,w,d_w,2] -> [f,h,w,d_w,2]
        fw = mx.broadcast_to(freqs_w[:w].reshape(1, 1, w, d_w, 2), (f, h, w, d_w, 2))

        # Concatenate: [f*h*w, half_d, 2]
        freqs_i = mx.concatenate([ft, fh, fw], axis=3).reshape(seq_len, 1, half_d, 2)

        # Apply rotation: (a + bi) * (cos + sin*i) = (a*cos - b*sin) + (a*sin + b*cos)i
        cos_f = freqs_i[..., 0]  # [seq_len, 1, half_d]
        sin_f = freqs_i[..., 1]  # [seq_len, 1, half_d]

        x_real = x_i[..., 0]  # [seq_len, n, half_d]
        x_imag = x_i[..., 1]  # [seq_len, n, half_d]

        out_real = x_real * cos_f - x_imag * sin_f
        out_imag = x_real * sin_f + x_imag * cos_f

        # Interleave back: [seq_len, n, half_d, 2] -> [seq_len, n, d]
        x_rotated = mx.stack([out_real, out_imag], axis=-1).reshape(seq_len, n, d)

        # Handle padding: keep non-rotated tokens after seq_len
        if seq_len < s:
            x_rotated = mx.concatenate([x_rotated, x[i, seq_len:]], axis=0)

        outputs.append(x_rotated)

    return mx.stack(outputs)


def rope_precompute_cos_sin(
    grid_sizes: list, freqs: mx.array, dtype: type = mx.float32
) -> tuple:
    if freqs.dtype != dtype:
        freqs = freqs.astype(dtype)

    f, h, w = grid_sizes[0]
    seq_len = f * h * w
    half_d = freqs.shape[1]

    d_t = half_d - 2 * (half_d // 3)
    d_h = half_d // 3
    d_w = half_d // 3

    freqs_t = freqs[:, :d_t]
    freqs_h = freqs[:, d_t : d_t + d_h]
    freqs_w = freqs[:, d_t + d_h : d_t + d_h + d_w]

    ft = mx.broadcast_to(freqs_t[:f].reshape(f, 1, 1, d_t, 2), (f, h, w, d_t, 2))
    fh = mx.broadcast_to(freqs_h[:h].reshape(1, h, 1, d_h, 2), (f, h, w, d_h, 2))
    fw = mx.broadcast_to(freqs_w[:w].reshape(1, 1, w, d_w, 2), (f, h, w, d_w, 2))

    freqs_i = mx.concatenate([ft, fh, fw], axis=3).reshape(seq_len, 1, half_d, 2)
    return freqs_i[..., 0], freqs_i[..., 1]
