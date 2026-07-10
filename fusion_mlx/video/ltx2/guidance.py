import mlx.core as mx


def cfg_delta(cond: mx.array, uncond: mx.array, scale: float) -> mx.array:
    return (scale - 1.0) * (cond - uncond)


def apg_delta(
    cond: mx.array,
    uncond: mx.array,
    scale: float,
    eta: float = 1.0,
    norm_threshold: float = 0.0,
) -> mx.array:
    guidance = cond - uncond

    if norm_threshold > 0:
        guidance_norm = mx.sqrt(
            mx.sum(guidance**2, axis=(-1, -2, -3), keepdims=True) + 1e-8
        )
        scale_factor = mx.minimum(
            mx.ones_like(guidance_norm), norm_threshold / guidance_norm
        )
        guidance = guidance * scale_factor

    batch_size = cond.shape[0]
    cond_flat = mx.reshape(cond, (batch_size, -1))
    guidance_flat = mx.reshape(guidance, (batch_size, -1))

    dot_product = mx.sum(guidance_flat * cond_flat, axis=1, keepdims=True)
    squared_norm = mx.sum(cond_flat**2, axis=1, keepdims=True) + 1e-8
    proj_coeff = dot_product / squared_norm

    proj_coeff = mx.reshape(proj_coeff, (batch_size,) + (1,) * (cond.ndim - 1))
    g_parallel = proj_coeff * cond
    g_orth = guidance - g_parallel

    g_apg = g_parallel * eta + g_orth

    return g_apg * (scale - 1.0)
