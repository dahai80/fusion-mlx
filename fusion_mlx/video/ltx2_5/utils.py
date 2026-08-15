# SPDX-License-Identifier: Apache-2.0
# Pure-MLX port of LTX-2.5 utility functions.
# Path resolution for the Lightricks/LTX-2.5 Comfy single-file repo. The repo
# layout is: diffusion_models/, vae/, text_encoders/, model_patches/,
# latent_upscale_models/. We resolve either a local dir or an HF snapshot.
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# LTX-2.5 Comfy 单文件仓的子目录与权重文件名（附录 A）。
_LTX2_5_REPO = "Lightricks/LTX-2.5"
_LTX2_5_FILES = {
    "transformer_distilled": "diffusion_models/ltx-2.5-22b-distilled-transformer-bf16.safetensors",
    "transformer_dev": "diffusion_models/ltx-2.5-22b-dev-transformer-bf16.safetensors",
    "video_vae": "vae/ltx-2.5-video-vae-bf16.safetensors",
    "audio_vae": "vae/ltx-2.5-audio-vae-bf16.safetensors",
    "text_encoder": "text_encoders/gemma4-12b-with-proj-ltx-2.5-bf16.safetensors",
    "duration_head": "model_patches/ltx-2.5-duration-head-bf16.safetensors",
    "spatial_upscaler": "latent_upscale_models/ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors",
    "temporal_upscaler": "latent_upscale_models/ltx-2.5-latent-temporal-upscaler-x2-bf16-1.0.safetensors",
}


def get_model_path(model_repo: str = _LTX2_5_REPO) -> Path:
    # 解析 LTX-2.5 仓根目录：本地路径优先，否则 HF snapshot 下载（限 safetensors/json）。
    try:
        if Path(model_repo).exists():
            logger.info("ltx2_5 get_model_path: local dir %s", model_repo)
            return Path(model_repo)
        from huggingface_hub import snapshot_download

        return Path(snapshot_download(repo_id=model_repo, local_files_only=True))
    except Exception:
        logger.info("ltx2_5 get_model_path: downloading %s", model_repo)
        from huggingface_hub import snapshot_download

        return Path(
            snapshot_download(
                repo_id=model_repo,
                local_files_only=False,
                resume_download=True,
                allow_patterns=["*.safetensors", "*.json"],
            )
        )


def resolve_component(
    root: Path,
    key: str,
    *,
    variant: str = "distilled",
) -> Path:
    # 在已解析的仓根目录下定位单个组件文件。
    if key == "transformer":
        key = "transformer_distilled" if variant == "distilled" else "transformer_dev"
    if key not in _LTX2_5_FILES:
        raise ValueError(f"unknown LTX-2.5 component key: {key!r}")
    rel = _LTX2_5_FILES[key]
    candidate = root / rel
    if not candidate.exists():
        logger.warning("ltx2_5 component %s not found at %s", key, candidate)
    return candidate


def component_keys() -> list[str]:
    return list(_LTX2_5_FILES.keys())
