# SPDX-License-Identifier: Apache-2.0
# Shared gen_kwargs builder for image generation (S3 subprocess isolation).
# Extracted from ImageGenEngine._generate() so both the in-process path and
# the subprocess worker use the same variant-specific kwarg logic.
from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


def build_gen_kwargs(
    variant: str,
    seed: int,
    prompt: str,
    steps: int,
    height: int,
    width: int,
    guidance: float,
    scheduler: str | None = None,
    negative_prompt: str | None = None,
    denoising_end: float | None = None,
    control_image: str | None = None,
    controlnet_strength: float | None = None,
    depth_image: str | None = None,
    image_strength: float | None = None,
    edit_image: str | None = None,
    mask_image: str | None = None,
    reference_images: list[str] | None = None,
    reference_strengths: list[float] | None = None,
    extra_kwargs: dict[str, Any] | None = None,
) -> dict[str, Any]:
    extra = extra_kwargs or {}
    gen_kwargs: dict[str, Any] = dict(
        seed=seed,
        prompt=prompt,
        num_inference_steps=steps,
        height=height,
        width=width,
        guidance=guidance,
    )
    if scheduler is not None:
        gen_kwargs["scheduler"] = scheduler
    if denoising_end is not None:
        logger.warning(
            "denoising_end=%.2f ignored: variant '%s' uses single-call "
            "generate_image (no staged denoise); use a Qwen-Image "
            "multi-stage variant for partial denoise",
            denoising_end,
            variant,
        )
    if variant in ("controlnet_canny", "controlnet_upscaler"):
        if control_image is None:
            raise ValueError(f"variant '{variant}' requires control_image")
        gen_kwargs["controlnet_image_path"] = control_image
        if controlnet_strength is not None:
            gen_kwargs["controlnet_strength"] = controlnet_strength
    elif variant == "depth":
        if depth_image is not None:
            gen_kwargs["depth_image_path"] = depth_image
        elif control_image is not None:
            gen_kwargs["image_path"] = control_image
        if image_strength is not None:
            gen_kwargs["image_strength"] = image_strength
    elif variant == "fill":
        if edit_image is None or mask_image is None:
            raise ValueError("variant 'fill' requires edit_image and mask_image")
        gen_kwargs["image_path"] = edit_image
        gen_kwargs["masked_image_path"] = mask_image
        if image_strength is not None:
            gen_kwargs["image_strength"] = image_strength
    elif variant == "kontext":
        if edit_image is not None:
            gen_kwargs["image_path"] = edit_image
        elif control_image is not None:
            gen_kwargs["image_path"] = control_image
        if image_strength is not None:
            gen_kwargs["image_strength"] = image_strength
    elif variant == "redux":
        if not reference_images:
            raise ValueError("variant 'redux' requires reference_images")
        gen_kwargs["redux_image_paths"] = reference_images
        if reference_strengths is not None:
            gen_kwargs["redux_image_strengths"] = reference_strengths
        if image_strength is not None:
            gen_kwargs["image_strength"] = image_strength
    elif variant in ("txt2img", "flux1_dev", "flux1_schnell", "flux2_dev"):
        if edit_image is not None or control_image is not None:
            gen_kwargs["image_path"] = edit_image or control_image
            if image_strength is not None:
                gen_kwargs["image_strength"] = image_strength
    elif variant == "sd3":
        if negative_prompt is not None:
            gen_kwargs["negative_prompt"] = negative_prompt
        shift = extra.get("shift")
        if shift is not None:
            gen_kwargs["shift"] = shift
    elif variant in ("sdxl", "cosxl", "sdxs", "sd15", "sd2"):
        if negative_prompt is not None:
            gen_kwargs["negative_prompt"] = negative_prompt
    elif variant == "qwen_image":
        if negative_prompt is not None:
            gen_kwargs["negative_prompt"] = negative_prompt
        if edit_image is not None or control_image is not None:
            gen_kwargs["image_path"] = edit_image or control_image
            if image_strength is not None:
                gen_kwargs["image_strength"] = image_strength
    elif variant == "qwen_image_edit":
        if edit_image is not None:
            gen_kwargs["image_path"] = edit_image
        if image_strength is not None:
            gen_kwargs["image_strength"] = image_strength
    if variant in ("sd3", "sdxl", "cosxl", "sdxs", "sd15", "sd2") and (
        edit_image is not None or control_image is not None
    ):
        gen_kwargs["image_path"] = edit_image or control_image
        if image_strength is not None:
            gen_kwargs["image_strength"] = image_strength
    elif variant == "stable_cascade":
        if negative_prompt is not None:
            gen_kwargs["negative_prompt"] = negative_prompt
        d_steps = extra.get("decoder_steps")
        if d_steps is not None:
            gen_kwargs["decoder_steps"] = d_steps
        d_guidance = extra.get("decoder_guidance")
        if d_guidance is not None:
            gen_kwargs["decoder_guidance"] = d_guidance
    if negative_prompt is not None and variant not in (
        "sd3",
        "sdxl",
        "cosxl",
        "sdxs",
        "sd15",
        "sd2",
        "stable_cascade",
        "qwen_image",
        "qwen_image_edit",
    ):
        logger.warning(
            "Flux does not support negative_prompt; ignoring (got %d chars)",
            len(negative_prompt),
        )
    return gen_kwargs
