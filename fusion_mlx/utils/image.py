# SPDX-License-Identifier: Apache-2.0
"""Image processing utilities for VLM support."""

import base64
import binascii
import hashlib
import io
import logging
from typing import Any

from PIL import Image, ImageOps

logger = logging.getLogger(__name__)


def load_image(url_or_base64: str) -> Image.Image:
    """Load an image from a base64 data URI or a local file path."""
    if url_or_base64.startswith("data:"):
        _, data_part = url_or_base64.split(",", 1)
        img_bytes = base64.b64decode(data_part)
        img = Image.open(io.BytesIO(img_bytes))
    elif url_or_base64.startswith(("http://", "https://")):
        # SSRF protection: never make an outbound request to a remote URL.
        # The server must not fetch attacker-controlled or RFC1918/loopback
        # addresses (e.g. cloud metadata endpoints). Clients must inline
        # images as data: URIs or pass a local file path. The caller
        # (extract_images_from_messages) catches this and skips the part.
        logger.warning(
            "load_image: refused remote URL (SSRF protection); require a "
            "data: URI or a local file path"
        )
        raise ValueError(
            "Remote image URLs are not supported; use a data: URI or a "
            "local file path"
        )
    else:
        img = Image.open(url_or_base64)

    img = ImageOps.exif_transpose(img)
    return img.convert("RGB")


def extract_images_from_messages(
    messages: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[Image.Image], list, list[str]]:
    """Extract images, videos, and audio from OpenAI-format messages.

    Returns (text_messages, images, videos, audio) where text_messages have
    media parts removed, images is a list of loaded PIL Image objects, videos
    is a list of video URL/path/base64 strings, and audio is a list of
    BytesIO/str audio references.
    """
    text_messages = []
    images = []
    videos = []
    audio = []

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
            part_type = (
                part.get("type", "")
                if isinstance(part, dict)
                else getattr(part, "type", "")
            )

            if part_type == "text":
                text = (
                    part.get("text", "")
                    if isinstance(part, dict)
                    else getattr(part, "text", "")
                )
                if text:
                    text_parts.append(text)

            elif part_type in ("image_url", "input_image"):
                image_url_obj = (
                    part.get("image_url")
                    if isinstance(part, dict)
                    else getattr(part, "image_url", None)
                )
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

            elif part_type in ("video_url", "video"):
                video_url_obj = (
                    part.get("video_url")
                    if isinstance(part, dict)
                    else getattr(part, "video_url", None)
                )
                if video_url_obj is None and isinstance(part, dict):
                    video_url_obj = part.get("video")

                url = None
                if isinstance(video_url_obj, str):
                    url = video_url_obj
                elif isinstance(video_url_obj, dict):
                    url = video_url_obj.get("url")
                elif video_url_obj is not None:
                    url = getattr(video_url_obj, "url", None)

                if url:
                    videos.append(url)
                    text_parts.append("[video]")

            elif part_type == "input_audio":
                input_audio = (
                    part.get("input_audio")
                    if isinstance(part, dict)
                    else getattr(part, "input_audio", None)
                )
                if input_audio and isinstance(input_audio, dict):
                    data = input_audio.get("data", "")
                    if isinstance(data, str):
                        stripped = data.strip()
                        if stripped.startswith("data:"):
                            prefix, separator, encoded = stripped.partition(",")
                            if separator == "," and ";base64" in prefix:
                                try:
                                    audio.append(
                                        io.BytesIO(
                                            base64.b64decode(encoded, validate=True)
                                        )
                                    )
                                except (binascii.Error, ValueError) as exc:
                                    logger.warning(
                                        f"Failed to decode input_audio base64: {exc}"
                                    )
                                continue
                        try:
                            audio.append(
                                io.BytesIO(base64.b64decode(stripped, validate=True))
                            )
                        except (binascii.Error, ValueError):
                            audio.append(stripped)
                    elif isinstance(data, bytes):
                        audio.append(io.BytesIO(data))
                    else:
                        audio.append(data)
                logger.debug(f"Extracted audio part, total audio count: {len(audio)}")

        new_msg = {"role": role, "content": "\n".join(text_parts) if text_parts else ""}
        for key in msg:
            if key not in ("role", "content"):
                new_msg[key] = msg[key]
        text_messages.append(new_msg)

    return text_messages, images, videos, audio


def compute_image_hash(images: list[Image.Image]) -> str | None:
    """Compute SHA256 hash from images for prefix cache deduplication."""
    if not images:
        return None

    hasher = hashlib.sha256()
    for img in images:
        hasher.update(f"{img.size[0]}x{img.size[1]}".encode())
        hasher.update(img.convert("RGB").tobytes())
    return hasher.hexdigest()


def compute_per_image_hashes(images: list[Image.Image]) -> list[str]:
    """Compute individual SHA256 hashes for each image."""
    hashes = []
    for img in images:
        hasher = hashlib.sha256()
        hasher.update(f"{img.size[0]}x{img.size[1]}".encode())
        hasher.update(img.convert("RGB").tobytes())
        hashes.append(hasher.hexdigest())
    return hashes
