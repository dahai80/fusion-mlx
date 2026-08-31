import logging

logger = logging.getLogger(__name__)


def apply_inpaint_mask(latents, init_latent, mask):
    # #653 Surface C: neutral per-step re-composite. mask=1 -> reactive
    # (keep denoised); mask=0 -> frozen (restore init). Orthogonal to
    # ControlState; never touches conditioning. None -> T2V passthrough.
    if mask is None or init_latent is None:
        return latents
    if latents.shape != init_latent.shape:
        raise ValueError(
            f"apply_inpaint_mask: init_latent shape {tuple(init_latent.shape)}"
            f" != latents {tuple(latents.shape)}"
        )
    return mask * latents + (1.0 - mask) * init_latent
