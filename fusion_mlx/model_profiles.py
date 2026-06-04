"""Model profile definitions for fusion-mlx."""

from typing import Dict, List


# Models that should not be included in auto-profile detection
EXCLUDED_FROM_PROFILES: List[str] = [
    "google/t5-v1_1-xxl",
    "OpenSuper/CLIP-ViT-bigG-14",
    "laion/CLIP-ViT-H-14-laion2B-s32B-b79K",
]

# Default model profiles: model_id -> {params, type, context}
DEFAULT_PROFILES: Dict[str, Dict[str, str]] = {}
