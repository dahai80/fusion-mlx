import math

import mlx.core as mx


def phi(j: int, neg_h: float) -> float:
    if abs(neg_h) < 1e-10:
        return 1.0 / math.factorial(j)

    remainder = sum(neg_h**k / math.factorial(k) for k in range(j))
    return (math.exp(neg_h) - remainder) / (neg_h**j)


def get_res2s_coefficients(
    h: float,
    phi_cache: dict,
    c2: float = 0.5,
) -> tuple[float, float, float]:
    def get_phi(j: int, neg_h: float) -> float:
        cache_key = (j, neg_h)
        if cache_key in phi_cache:
            return phi_cache[cache_key]
        result = phi(j, neg_h)
        phi_cache[cache_key] = result
        return result

    neg_h_c2 = -h * c2
    phi_1_c2 = get_phi(1, neg_h_c2)
    a21 = c2 * phi_1_c2

    neg_h_full = -h
    phi_2_full = get_phi(2, neg_h_full)
    b2 = phi_2_full / c2

    phi_1_full = get_phi(1, neg_h_full)
    b1 = phi_1_full - b2

    return a21, b1, b2


def get_sde_coeff(
    sigma_next: float,
) -> tuple[float, float, float]:
    sigma_up = sigma_next * 0.5
    sigma_up = min(sigma_up, sigma_next * 0.9999)

    sigma_signal = 1.0 - sigma_next
    sigma_residual = math.sqrt(max(sigma_next**2 - sigma_up**2, 0.0))
    alpha_ratio = sigma_signal + sigma_residual

    if alpha_ratio == 0:
        sigma_down = sigma_next
    else:
        sigma_down = sigma_residual / alpha_ratio

    if math.isnan(sigma_up):
        sigma_up = 0.0
    if math.isnan(sigma_down):
        sigma_down = sigma_next
    if math.isnan(alpha_ratio):
        alpha_ratio = 1.0

    return alpha_ratio, sigma_down, sigma_up


def sde_noise_step(
    sample: mx.array,
    denoised_sample: mx.array,
    sigma: float,
    sigma_next: float,
    noise: mx.array,
) -> mx.array:
    alpha_ratio, sigma_down, sigma_up = get_sde_coeff(sigma_next)

    if sigma_up == 0 or sigma_next == 0:
        return denoised_sample

    sample_f32 = sample.astype(mx.float32)
    denoised_f32 = denoised_sample.astype(mx.float32)
    noise_f32 = noise.astype(mx.float32)

    eps_next = (sample_f32 - denoised_f32) / (sigma - sigma_next)
    denoised_next = sample_f32 - sigma * eps_next

    x_noised = (
        alpha_ratio * (denoised_next + sigma_down * eps_next) + sigma_up * noise_f32
    )

    return x_noised


def channelwise_normalize(x: mx.array) -> mx.array:
    mean = mx.mean(x, axis=(-2, -1), keepdims=True)
    x = x - mean
    std = mx.sqrt(mx.mean(x * x, axis=(-2, -1), keepdims=True) + 1e-8)
    x = x / std
    return x


def get_new_noise(shape: tuple, key: mx.array) -> mx.array:
    noise = mx.random.normal(shape, dtype=mx.float32, key=key)
    noise = (noise - mx.mean(noise)) / (mx.sqrt(mx.mean(noise * noise)) + 1e-8)
    noise = channelwise_normalize(noise)
    return noise
