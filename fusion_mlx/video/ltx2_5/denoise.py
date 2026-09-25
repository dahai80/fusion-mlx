# SPDX-License-Identifier: Apache-2.0
# LTX-2.5 distilled 去噪 (T2V-only path)。
# 与 ltx2 denoise_distilled 的差异: 使用 ltx2_5.Modality (跨模块 Modality 类不兼容,
# 见 GOTCHA: ltx2_5.Modality is not ltx2.Modality) 并调 LTX2_5Model。
# audio / I2V conditioning (state) 不在本轮 T2V 路径, 留空 fail visible。
from __future__ import annotations

import logging
import os
from functools import partial

import mlx.core as mx

from fusion_mlx.engines.video_backends._inpaint import apply_inpaint_mask

from ..ltx2.conditioning import LatentState, apply_denoise_mask
from .transformer import Modality

logger = logging.getLogger(__name__)

# #gap1: LTX-2.5 latency fast path. Two levers, both env-gated:
#   FUSION_LTX_FAST_PATH (default 0): drop the 2 redundant per-step mx.eval
#     (velocity, denoised) — keep only the final per-step latents eval. Lets
#     MLX fuse the velocity->x0->renoise math across the step. OPT-IN: on
#     720P/8s (921K tokens) measured 2× SLOWER than 3-eval baseline — the
#     uncompiled transformer graph stays lazy across the fused step and MLX
#     re-traverses it; the per-step evals bound graph size. Win only on small
#     token counts (short low-res clips). Leave OFF for production.
#   FUSION_LTX_COMPILE_TRANSFORMER (default 0): mx.compile the DiT __call__
#     (Metal kernel fusion: RMSNorm+attn+FFN elementwise fused, fewer launch
#     round-trips). Opt-in — skyreels_v3 found whole-__call__ compile can
#     degrade for xfuser-injected DiTs; LTX-2.5 has no xfuser so expected to
#     help, but needs real-model A/B (42GB load) to confirm. First call pays
#     a one-time compile (~seconds), cached per shape for the remaining steps.
_FAST_PATH = os.environ.get("FUSION_LTX_FAST_PATH", "0") == "1"
_COMPILE_TRANSFORMER = os.environ.get("FUSION_LTX_COMPILE_TRANSFORMER", "0") == "1"


@partial(mx.compile, inputs=(), outputs=())
def _step_update(
    latents_flat_f32, timesteps_f32, velocity_f32, sigma_f32, sigma_next_f32
):
    # Compiled per-step update (state=None path): x0 = latent - t*velocity,
    # then DDIM renoise toward sigma_next, fused into 1 Metal kernel.
    # mx.where (not Python if) — can't branch on array values inside compile.
    x0 = latents_flat_f32 - timesteps_f32 * velocity_f32
    renoised = x0 + sigma_next_f32 * (latents_flat_f32 - x0) / sigma_f32
    return mx.where(sigma_next_f32 > 0, renoised, x0)


@partial(mx.compile, inputs=(), outputs=())
def _x0_only(latents_flat_f32, timesteps_f32, velocity_f32):
    # Compiled x0 prediction (state path): apply_denoise_mask must insert
    # between x0 and renoise, so we can't fuse renoise here. Still fuses the
    # subtract+mul into 1 kernel and avoids a separate velocity eval.
    return latents_flat_f32 - timesteps_f32 * velocity_f32


def denoise_distilled_t2v(
    latents: mx.array,
    positions: mx.array,
    text_embeddings: mx.array,
    transformer,
    sigmas: list,
    verbose: bool = True,
    controlnet_image=None,
    inpaint_mask=None,
    init_latent=None,
    state: LatentState | None = None,
    negative_context: mx.array | None = None,
    cfg_scale: float = 1.0,
) -> mx.array:
    # 两阶段 distilled T2V/I2V 去噪。latents (b,c,f,h,w), sigmas 降序 -> 0。
    # 每步: 展平 latent -> Modality(context=text_embeddings) -> transformer ->
    # velocity -> x0 = latent - sigma*velocity -> 重新加噪到 sigma_next。
    # #782 I2V: state (LatentState) 开启 latent-level 条件注入 — 条件帧
    # denoise_mask=0 -> timesteps=0 -> transformer 视为干净帧; 每步 x0 预测后
    # apply_denoise_mask 把条件帧夹回 clean_latent, 跨 re-noise 步冻结。
    # state=None 走原 T2V 路径 (uniform timesteps) bit-exact 不变。
    # #735 Surface B: ControlNet not fabricatable for ltx2_5 (shared adapter is
    # Wan2-arch, no per-backend model). Fail visible — refuse silent T2V degrade.
    # #735 Surface C: DiT-agnostic latent-space inpaint re-composite after each
    # step's x0 prediction, so frozen regions stay frozen across re-noise steps.
    if controlnet_image is not None:
        raise RuntimeError(
            "ltx2_5: ControlNet (Surface B) not available for this backend — "
            "no per-backend ControlNet model (see issue #735 follow-up). "
            "Refusing to silently degrade to T2V (#735)."
        )
    dtype = latents.dtype
    if state is not None:
        latents = state.latent
    latents = latents.astype(mx.float32)
    num_steps = len(sigmas) - 1
    if verbose:
        logger.info("Denoising T2V: %d steps", num_steps)
    logger.info(
        "ltx2_5 denoise: inpaint=%s controlnet=%s fast_path=%s compile_xf=%s",
        inpaint_mask is not None,
        controlnet_image is not None,
        _FAST_PATH,
        _COMPILE_TRANSFORMER,
    )

    # #gap1: optionally compile the DiT forward (Metal kernel fusion). mx.compile
    # requires array-tree args, but transformer.__call__ takes a Modality object
    # → compile a wrapper that takes the raw array fields and builds the Modality
    # inside. Caches per (shape, dtype) so all N steps reuse one compiled graph.
    _xf = transformer
    if _COMPILE_TRANSFORMER:
        _raw_xf = transformer

        def _xf_forward(latent, timesteps_a, positions_a, context, sigma_b):
            vm = Modality(
                latent=latent,
                timesteps=timesteps_a,
                positions=positions_a,
                context=context,
                context_mask=None,
                enabled=True,
                sigma=sigma_b,
            )
            return _raw_xf(video=vm, audio=None)

        try:
            _xf = mx.compile(_xf_forward)
            logger.info("ltx2_5 denoise: DiT __call__ compiled (Metal fusion)")
        except Exception as exc:
            logger.warning(
                "ltx2_5 denoise: mx.compile(transformer) failed (%s) — "
                "falling back to uncompiled forward",
                exc,
            )
            _xf = transformer

    for i in range(num_steps):
        sigma, sigma_next = sigmas[i], sigmas[i + 1]

        b, c, f, h, w = latents.shape
        num_tokens = f * h * w
        latents_flat = mx.transpose(mx.reshape(latents, (b, c, -1)), (0, 2, 1)).astype(
            dtype
        )

        if state is not None:
            denoise_mask_flat = mx.reshape(state.denoise_mask, (b, 1, f, 1, 1))
            denoise_mask_flat = mx.broadcast_to(denoise_mask_flat, (b, 1, f, h, w))
            denoise_mask_flat = mx.reshape(denoise_mask_flat, (b, num_tokens))
            timesteps = mx.array(sigma, dtype=dtype) * denoise_mask_flat
        else:
            timesteps = mx.full((b, num_tokens), sigma, dtype=dtype)

        sigma_b = mx.full((b,), sigma, dtype=dtype)
        use_cfg = negative_context is not None and cfg_scale != 1.0
        if _COMPILE_TRANSFORMER and _xf is not transformer:
            velocity, _audio_velocity = _xf(
                latents_flat, timesteps, positions, text_embeddings, sigma_b
            )
            if use_cfg:
                v_neg, _ = _xf(
                    latents_flat, timesteps, positions, negative_context, sigma_b
                )
                velocity = v_neg + cfg_scale * (velocity - v_neg)
        else:
            video_modality = Modality(
                latent=latents_flat,
                timesteps=timesteps,
                positions=positions,
                context=text_embeddings,
                context_mask=None,
                enabled=True,
                sigma=sigma_b,
            )
            velocity, _audio_velocity = _xf(video=video_modality, audio=None)
            if use_cfg:
                neg_modality = Modality(
                    latent=latents_flat,
                    timesteps=timesteps,
                    positions=positions,
                    context=negative_context,
                    context_mask=None,
                    enabled=True,
                    sigma=sigma_b,
                )
                v_neg, _ = _xf(video=neg_modality, audio=None)
                velocity = v_neg + cfg_scale * (velocity - v_neg)

        sigma_f32 = mx.array(sigma, dtype=mx.float32)
        sigma_next_f32 = mx.array(sigma_next, dtype=mx.float32)
        latents_flat_f32 = mx.transpose(mx.reshape(latents, (b, c, -1)), (0, 2, 1))
        timesteps_f32 = mx.expand_dims(timesteps.astype(mx.float32), axis=-1)

        if _FAST_PATH:
            # #gap1: single per-step eval (was 3). state=None fuses x0+renoise
            # into 1 kernel via _step_update; state path fuses x0 via _x0_only
            # then masks + renoises outside (mask must insert pre-renoise).
            if state is None:
                renoised_flat = _step_update(
                    latents_flat_f32,
                    timesteps_f32,
                    velocity.astype(mx.float32),
                    sigma_f32,
                    sigma_next_f32,
                )
                latents = mx.reshape(
                    mx.transpose(renoised_flat, (0, 2, 1)), (b, c, f, h, w)
                )
            else:
                x0_f32 = _x0_only(
                    latents_flat_f32, timesteps_f32, velocity.astype(mx.float32)
                )
                denoised = mx.reshape(mx.transpose(x0_f32, (0, 2, 1)), (b, c, f, h, w))
                denoised = apply_denoise_mask(
                    denoised,
                    state.clean_latent.astype(mx.float32),
                    state.denoise_mask,
                )
                if sigma_next > 0:
                    latents = (
                        denoised + sigma_next_f32 * (latents - denoised) / sigma_f32
                    )
                else:
                    latents = denoised
            mx.eval(latents)
        else:
            # original path: 3 evals/step, uncompiled math (bit-exact baseline)
            mx.eval(velocity)
            x0_f32 = latents_flat_f32 - timesteps_f32 * velocity.astype(mx.float32)
            denoised = mx.reshape(mx.transpose(x0_f32, (0, 2, 1)), (b, c, f, h, w))
            if state is not None:
                denoised = apply_denoise_mask(
                    denoised,
                    state.clean_latent.astype(mx.float32),
                    state.denoise_mask,
                )
            mx.eval(denoised)
            if sigma_next > 0:
                latents = denoised + sigma_next_f32 * (latents - denoised) / sigma_f32
            else:
                latents = denoised
            mx.eval(latents)

        if inpaint_mask is not None and init_latent is not None:
            latents = apply_inpaint_mask(latents, init_latent, inpaint_mask)
            mx.eval(latents)
        if verbose:
            logger.info("step %d/%d", i + 1, num_steps)

    return latents.astype(mx.float32)


__all__ = ["denoise_distilled_t2v"]
