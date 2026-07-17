# SPDX-License-Identifier: Apache-2.0
"""SkyReels-V3 统一推理入口 (对齐原版 generate_video.py).

替换原版 argparse + task_type 调度,
复用底座 fusion_mlx.cli 注册命令.

用法 (Python):
    from fusion_mlx.video.skyreels_v3 import generate_video
    video = generate_video(
        task_type="reference_to_video",
        ref_imgs=["img1.png", "img2.png"],
        prompt="...",
        duration=5,
        output_path="output.mp4",
    )

用法 (CLI):
    fusion-mlx generate-skyreels \
        --task-type reference_to_video \
        --prompt "..." \
        --ref-imgs img1.png,img2.png \
        --duration 5 \
        --output output.mp4
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import mlx.core as mx

from . import _device
from .config import get_branch_config, BRANCH_CONFIGS
from .pipelines import (
    SkyReelsR2VPipeline,
    SkyReelsV2VPipeline,
    SkyReelsA2VPipeline,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# task_type -> model_key 映射 (对齐原版 generate_video.py)
# ---------------------------------------------------------------------------
TASK_TO_MODEL: dict[str, str] = {
    "reference_to_video": "skyreels-v3-r2v-14b",
    "single_shot_extension": "skyreels-v3-v2v-14b",
    "shot_switching_extension": "skyreels-v3-v2v-14b",
    "talking_avatar": "skyreels-v3-a2v-19b",
}


def generate_video(
    task_type: str,
    *,
    prompt: str = "",
    ref_imgs: list[str] | None = None,
    input_video: str | None = None,
    audio: str | None = None,
    ref_image: Any | None = None,
    duration: int = 5,
    output_path: str = "output.mp4",
    model_path: str | None = None,
    seed: int | None = None,
    width: int = 1280,
    height: int = 720,
    fps: int = 24,
    num_inference_steps: int = 50,
    guidance_scale: float = 5.0,
    tiling: bool = False,
) -> str:
    """统一视频生成入口.

    Args:
        task_type: 任务类型
          - "reference_to_video": 参考图 -> 视频 (R2V)
          - "single_shot_extension": 单镜头续写 (V2V)
          - "shot_switching_extension": 镜头切换续写 (V2V)
          - "talking_avatar": 音频数字人 (A2V)
        prompt: 文本 prompt
        ref_imgs: 参考图路径列表 (R2V, 1~4 张)
        input_video: 输入视频路径 (V2V)
        audio: 音频路径 (A2V)
        ref_image: 参考人脸图 (A2V)
        duration: 视频时长 (秒)
        output_path: 输出视频路径
        model_path: 模型权重路径 (None 则用默认)
        seed: 随机种子
        width/height/fps: 输出规格
        num_inference_steps: 采样步数
        guidance_scale: CFG 引导强度
        tiling: 是否启用 tiling 解码

    Returns:
        输出视频文件路径
    """
    if task_type not in TASK_TO_MODEL:
        raise ValueError(
            f"Unknown task_type: {task_type}. "
            f"Valid: {list(TASK_TO_MODEL)}"
        )

    model_key = TASK_TO_MODEL[task_type]
    branch_cfg = get_branch_config(model_key)

    # 自动推断 model_path (如果未指定)
    if model_path is None:
        model_path = _default_model_path(model_key)

    logger.info(
        "generate_video: task=%s model=%s branch=%s",
        task_type, model_key, branch_cfg.branch,
    )

    # 分派到对应 Pipeline
    if branch_cfg.branch == "r2v":
        pipeline = SkyReelsR2VPipeline(model_path)
        # 加载参考图
        ref_images = _load_ref_imgs(ref_imgs) if ref_imgs else None
        video = pipeline.generate(
            prompt=prompt,
            ref_images=ref_images,
            duration=duration,
            seed=seed,
        )
    elif branch_cfg.branch == "v2v":
        pipeline = SkyReelsV2VPipeline(model_path)
        if input_video is None:
            raise ValueError("V2V task requires input_video")
        video = pipeline.generate(
            input_video=input_video,
            prompt=prompt,
            duration=duration,
            seed=seed,
        )
    elif branch_cfg.branch == "a2v":
        pipeline = SkyReelsA2VPipeline(model_path)
        if audio is None or ref_image is None:
            raise ValueError("A2V task requires audio + ref_image")
        video = pipeline.generate(
            audio=audio,
            ref_image=ref_image,
            prompt=prompt,
            duration=duration,
            seed=seed,
        )
    else:
        raise ValueError(f"Unknown branch: {branch_cfg.branch}")

    # 保存视频
    pipeline.save(video, output_path)
    logger.info("Video saved: %s", output_path)
    return output_path


def _default_model_path(model_key: str) -> str:
    """返回默认模型路径 (HuggingFace cache)."""
    branch_cfg = get_branch_config(model_key)
    hf_id = branch_cfg.hf_model_id
    # 默认从 HuggingFace cache 加载
    return f"~/.cache/huggingface/hub/{hf_id.replace('/', '--')}"


def _load_ref_imgs(paths: list[str]) -> list[Any]:
    """加载参考图列表."""
    from PIL import Image

    images = []
    for p in paths:
        if p.startswith("http"):
            # URL: 下载到临时文件
            import tempfile
            import urllib.request

            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tf:
                urllib.request.urlretrieve(p, tf.name)
                images.append(Image.open(tf.name).convert("RGB"))
        else:
            images.append(Image.open(p).convert("RGB"))
    return images


# ---------------------------------------------------------------------------
# CLI 入口
# ---------------------------------------------------------------------------
def cli_main() -> None:
    """CLI 入口: fusion-mlx generate-skyreels."""
    import argparse

    parser = argparse.ArgumentParser(
        description="SkyReels-V3 video generation (fusion-mlx MLX port)."
    )
    parser.add_argument(
        "--task-type", required=True,
        choices=list(TASK_TO_MODEL.keys()),
        help="Task type",
    )
    parser.add_argument("--prompt", default="", help="Text prompt")
    parser.add_argument(
        "--ref-imgs", default=None,
        help="Reference images (comma-separated paths/URLs, R2V only)",
    )
    parser.add_argument(
        "--input-video", default=None,
        help="Input video path (V2V only)",
    )
    parser.add_argument(
        "--audio", default=None,
        help="Audio path (A2V only)",
    )
    parser.add_argument(
        "--ref-image", default=None,
        help="Reference face image (A2V only)",
    )
    parser.add_argument("--duration", type=int, default=5, help="Duration in seconds")
    parser.add_argument("--output", default="output.mp4", help="Output video path")
    parser.add_argument("--model-path", default=None, help="Model weights path")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--width", type=int, default=1280, help="Output width")
    parser.add_argument("--height", type=int, default=720, help="Output height")
    parser.add_argument("--fps", type=int, default=24, help="Output fps")
    parser.add_argument(
        "--num-inference-steps", type=int, default=50,
        help="Number of inference steps",
    )
    parser.add_argument(
        "--guidance-scale", type=float, default=5.0,
        help="CFG guidance scale",
    )
    parser.add_argument(
        "--tiling", action="store_true",
        help="Enable tiling decode (save memory for large resolution)",
    )

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    # 拆分 ref_imgs
    ref_imgs = None
    if args.ref_imgs:
        ref_imgs = [p.strip() for p in args.ref_imgs.split(",")]

    generate_video(
        task_type=args.task_type,
        prompt=args.prompt,
        ref_imgs=ref_imgs,
        input_video=args.input_video,
        audio=args.audio,
        ref_image=args.ref_image,
        duration=args.duration,
        output_path=args.output,
        model_path=args.model_path,
        seed=args.seed,
        width=args.width,
        height=args.height,
        fps=args.fps,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        tiling=args.tiling,
    )


if __name__ == "__main__":
    cli_main()
