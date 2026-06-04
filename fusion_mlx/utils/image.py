# SPDX-License-Identifier: Apache-2.0
"""Image processing utilities for VLM support."""

import base64
import hashlib
import io
import logging
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image, ImageOps

logger = logging.getLogger(__name__)


def load_image(url_or_base64: str) -> Image.Image:
    """Load an image from URL, base64 data URI, or local file path."""
    if url_or_base64.startswith("data:"):
        _, data_part = url_or_base64.split(",", 1)
        img_bytes = base64.b64decode(data_part)
        img = Image.open(io.BytesIO(img_bytes))
    elif url_or_base64.startswith(("http://", "https://")):
        import urllib.request

        with urllib.request.urlopen(url_or_base64, timeout=30) as response:
            img_bytes = response.read()
        img = Image.open(io.BytesIO(img_bytes))
    else:
        img = Image.open(url_or_base64)

    img = ImageOps.exif_transpose(img)
    return img.convert("RGB")


def extract_images_from_messages(
    messages: List[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], List[Image.Image]]:
    """Extract images from OpenAI-format messages.

    Returns (text_messages, images) where text_messages have image parts
    removed and images is a list of loaded PIL Image objects.
    """
    text_messages = []
    images = []

    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")

        if not isinstance(content, list):
            text_messages.append({"role": role, "content": content or ""})
            for key in msg:
                if key not in ("role", "content"):
                    text_messages[-1][key] = msg[key]
            continue

        text_parts = []
        for part in content:
            part_type = part.get("type", "") if isinstance(part, dict) else getattr(part, "type", "")

            if part_type == "text":
                text = part.get("text", "") if isinstance(part, dict) else getattr(part, "text", "")
                if text:
                    text_parts.append(text)
            elif part_type in ("image_url", "input_image"):
                image_url_obj = part.get("image_url") if isinstance(part, dict) else getattr(part, "image_url", None)
                if image_url_obj is None and isinstance(part, dict):
                    image_url_obj = part.get("input_image")

                url = None
                if isinstance(image_url_obj, str):
                    url = image_url_obj
                elif isinstance(image_url_obj, dict):
                    url = image_url_obj.get("url")
                elif image_url_obj is not None:
                    url = getattr(image_url_obj, "url", None)

                if url:
                    try:
                        images.append(load_image(url))
                    except Exception as e:
                        logger.warning(f"Failed to load image: {e}")

        new_msg = {"role": role, "content": "\n".join(text_parts) if text_parts else ""}
        for key in msg:
            if key not in ("role", "content"):
                new_msg[key] = msg[key]
        text_messages.append(new_msg)

    return text_messages, images


def compute_image_hash(images: List[Image.Image]) -> Optional[str]:
    """Compute SHA256 hash from images for prefix cache deduplication."""
    if not images:
        return None

    hasher = hashlib.sha256()
    for img in images:
        hasher.update(f"{img.size[0]}x{img.size[1]}".encode())
        hasher.update(img.convert("RGB").tobytes())
    return hasher.hexdigest()


def compute_per_image_hashes(images: List[Image.Image]) -> List[str]:
    """Compute individual SHA256 hashes for each image."""
    hashes = []
    for img in images:
        hasher = hashlib.sha256()
        hasher.update(f"{img.size[0]}x{img.size[1]}".encode())
        hasher.update(img.convert("RGB").tobytes())
        hashes.append(hasher.hexdigest())
    return hashes
