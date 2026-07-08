# SPDX-License-Identifier: Apache-2.0
"""Video processing utilities for VLM support."""

import atexit
import base64
import logging
import math
import os
import tempfile
import threading
from pathlib import Path
from urllib.parse import urlparse

import numpy as np

logger = logging.getLogger(__name__)

FRAME_FACTOR = 2
DEFAULT_FPS = 2.0
MIN_FRAMES = 4
MAX_FRAMES = 128
IMAGE_FACTOR = 28

MAX_IMAGE_SIZE = 20 * 1024 * 1024
MAX_VIDEO_SIZE = 500 * 1024 * 1024
MAX_BASE64_IMAGE_LENGTH = 30 * 1024 * 1024
MAX_BASE64_VIDEO_LENGTH = 700 * 1024 * 1024


class FileSizeExceededError(Exception):
    pass


class TempFileManager:
    def __init__(self):
        self._files: set[str] = set()
        self._lock = threading.Lock()
        atexit.register(self.cleanup_all)

    def register(self, path: str) -> str:
        with self._lock:
            self._files.add(path)
        return path

    def cleanup(self, path: str) -> bool:
        with self._lock:
            if path in self._files:
                self._files.discard(path)
        try:
            if os.path.exists(path):
                os.unlink(path)
                logger.debug("Cleaned up temp file: %s", path)
                return True
        except OSError as e:
            logger.warning("Failed to clean up temp file %s: %s", path, e)
        return False

    def cleanup_all(self) -> int:
        with self._lock:
            files_to_clean = list(self._files)
            self._files.clear()
        cleaned = 0
        for path in files_to_clean:
            try:
                if os.path.exists(path):
                    os.unlink(path)
                    cleaned += 1
            except OSError:
                pass
        if cleaned:
            logger.info("Cleaned up %d temp files", cleaned)
        return cleaned


_temp_manager = TempFileManager()


def cleanup_temp_file(path: str) -> bool:
    return _temp_manager.cleanup(path)


def cleanup_all_temp_files() -> int:
    return _temp_manager.cleanup_all()


def is_url(s: str) -> bool:
    return s.startswith(("http://", "https://"))


def is_base64_image(s: str) -> bool:
    return s.startswith("data:image/") or (
        len(s) > 100 and not s.startswith(("http://", "https://", "/"))
    )


def is_base64_video(s: str) -> bool:
    return s.startswith("data:video/")


def decode_base64_image(
    base64_string: str, max_length: int = MAX_BASE64_IMAGE_LENGTH
) -> bytes:
    if len(base64_string) > max_length:
        raise FileSizeExceededError(
            f"Base64 image data exceeds maximum size: "
            f"{len(base64_string) / 1024 / 1024:.1f} MB > "
            f"{max_length / 1024 / 1024:.1f} MB limit"
        )
    if base64_string.startswith("data:"):
        _, data = base64_string.split(",", 1)
        return base64.b64decode(data)
    return base64.b64decode(base64_string)


def download_image(url: str, timeout: int = 30, max_size: int = MAX_IMAGE_SIZE) -> str:
    import requests

    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"
    }
    try:
        head_response = requests.head(
            url, timeout=timeout, headers=headers, allow_redirects=True, verify=True
        )
        content_length = head_response.headers.get("content-length")
        if content_length and int(content_length) > max_size:
            raise FileSizeExceededError(
                f"Image at {url} exceeds maximum size: "
                f"{int(content_length) / 1024 / 1024:.1f} MB > "
                f"{max_size / 1024 / 1024:.1f} MB limit"
            )
    except requests.RequestException:
        pass

    response = requests.get(
        url, timeout=timeout, headers=headers, stream=True, verify=True
    )
    response.raise_for_status()

    content_length = response.headers.get("content-length")
    if content_length and int(content_length) > max_size:
        raise FileSizeExceededError(
            f"Image at {url} exceeds maximum size: "
            f"{int(content_length) / 1024 / 1024:.1f} MB > "
            f"{max_size / 1024 / 1024:.1f} MB limit"
        )

    content_type = response.headers.get("content-type", "")
    if "jpeg" in content_type or "jpg" in content_type:
        ext = ".jpg"
    elif "png" in content_type:
        ext = ".png"
    elif "gif" in content_type:
        ext = ".gif"
    elif "webp" in content_type:
        ext = ".webp"
    else:
        path = urlparse(url).path
        ext = Path(path).suffix or ".jpg"

    temp_file = tempfile.NamedTemporaryFile(suffix=ext, delete=False)
    downloaded_size = 0
    try:
        for chunk in response.iter_content(chunk_size=8192):
            downloaded_size += len(chunk)
            if downloaded_size > max_size:
                temp_file.close()
                os.unlink(temp_file.name)
                raise FileSizeExceededError(
                    f"Image at {url} exceeds maximum size during download: "
                    f"{downloaded_size / 1024 / 1024:.1f} MB > "
                    f"{max_size / 1024 / 1024:.1f} MB limit"
                )
            temp_file.write(chunk)
        temp_file.close()
    except FileSizeExceededError:
        raise
    except Exception:
        temp_file.close()
        if os.path.exists(temp_file.name):
            os.unlink(temp_file.name)
        raise

    return _temp_manager.register(temp_file.name)


def download_video(url: str, timeout: int = 120, max_size: int = MAX_VIDEO_SIZE) -> str:
    import requests

    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"
    }
    logger.info("Downloading video from: %s", url)

    try:
        head_response = requests.head(
            url, timeout=timeout, headers=headers, allow_redirects=True, verify=True
        )
        content_length = head_response.headers.get("content-length")
        if content_length and int(content_length) > max_size:
            raise FileSizeExceededError(
                f"Video at {url} exceeds maximum size: "
                f"{int(content_length) / 1024 / 1024:.1f} MB > "
                f"{max_size / 1024 / 1024:.1f} MB limit"
            )
    except requests.RequestException:
        pass

    response = requests.get(
        url, timeout=timeout, headers=headers, stream=True, verify=True
    )
    response.raise_for_status()

    content_length = response.headers.get("content-length")
    if content_length and int(content_length) > max_size:
        raise FileSizeExceededError(
            f"Video at {url} exceeds maximum size: "
            f"{int(content_length) / 1024 / 1024:.1f} MB > "
            f"{max_size / 1024 / 1024:.1f} MB limit"
        )

    content_type = response.headers.get("content-type", "")
    if "mp4" in content_type:
        ext = ".mp4"
    elif "webm" in content_type:
        ext = ".webm"
    elif "avi" in content_type:
        ext = ".avi"
    elif "mov" in content_type or "quicktime" in content_type:
        ext = ".mov"
    elif "mkv" in content_type:
        ext = ".mkv"
    else:
        path = urlparse(url).path
        ext = Path(path).suffix or ".mp4"

    temp_file = tempfile.NamedTemporaryFile(suffix=ext, delete=False)
    downloaded_size = 0
    try:
        for chunk in response.iter_content(chunk_size=8192):
            downloaded_size += len(chunk)
            if downloaded_size > max_size:
                temp_file.close()
                os.unlink(temp_file.name)
                raise FileSizeExceededError(
                    f"Video at {url} exceeds maximum size during download: "
                    f"{downloaded_size / 1024 / 1024:.1f} MB > "
                    f"{max_size / 1024 / 1024:.1f} MB limit"
                )
            temp_file.write(chunk)
        temp_file.close()
    except FileSizeExceededError:
        raise
    except Exception:
        temp_file.close()
        if os.path.exists(temp_file.name):
            os.unlink(temp_file.name)
        raise

    file_size = Path(temp_file.name).stat().st_size
    logger.info(
        "Video downloaded: %s (%.1f MB)", temp_file.name, file_size / 1024 / 1024
    )

    return _temp_manager.register(temp_file.name)


def decode_base64_video(
    base64_string: str, max_length: int = MAX_BASE64_VIDEO_LENGTH
) -> str:
    if len(base64_string) > max_length:
        raise FileSizeExceededError(
            f"Base64 video data exceeds maximum size: "
            f"{len(base64_string) / 1024 / 1024:.1f} MB > "
            f"{max_length / 1024 / 1024:.1f} MB limit"
        )

    if base64_string.startswith("data:video/"):
        header, data = base64_string.split(",", 1)
        format_part = header.split(";")[0]
        ext = "." + format_part.split("/")[-1]
    else:
        data = base64_string
        ext = ".mp4"

    video_bytes = base64.b64decode(data)
    temp_file = tempfile.NamedTemporaryFile(suffix=ext, delete=False)
    temp_file.write(video_bytes)
    temp_file.close()

    logger.info(
        "Base64 video decoded: %s (%.1f MB)",
        temp_file.name,
        len(video_bytes) / 1024 / 1024,
    )

    return _temp_manager.register(temp_file.name)


def process_video_input(video: str | dict) -> str:
    if isinstance(video, dict):
        url = video.get("url", video.get("video_url", ""))
        if isinstance(url, dict):
            url = url.get("url", "")
        video = url

    if not video:
        raise ValueError("Empty video input")

    if Path(video).exists():
        return video

    if is_url(video):
        return download_video(video)

    if is_base64_video(video):
        return decode_base64_video(video)

    raise ValueError(f"Cannot process video: {video[:50]}...")


_base64_image_cache: dict[str, str] = {}


def save_base64_image(base64_string: str) -> str:
    import hashlib

    image_hash = hashlib.md5(base64_string.encode()).hexdigest()

    if image_hash in _base64_image_cache:
        cached_path = _base64_image_cache[image_hash]
        if Path(cached_path).exists():
            return cached_path

    image_bytes = decode_base64_image(base64_string)

    if image_bytes[:8] == b"\x89PNG\r\n\x1a\n":
        ext = ".png"
    elif image_bytes[:2] == b"\xff\xd8":
        ext = ".jpg"
    elif image_bytes[:6] in (b"GIF87a", b"GIF89a"):
        ext = ".gif"
    elif image_bytes[:4] == b"RIFF" and image_bytes[8:12] == b"WEBP":
        ext = ".webp"
    else:
        ext = ".jpg"

    temp_file = tempfile.NamedTemporaryFile(suffix=ext, delete=False)
    temp_file.write(image_bytes)
    temp_file.close()

    path = _temp_manager.register(temp_file.name)
    _base64_image_cache[image_hash] = path
    return path


def process_image_input(image: str | dict) -> str:
    if isinstance(image, dict):
        url = image.get("url", image.get("image_url", ""))
        if isinstance(url, dict):
            url = url.get("url", "")
        image = url

    if not image:
        raise ValueError("Empty image input")

    if not isinstance(image, str):
        raise ValueError(f"image_url.url must be a string (got {type(image).__name__})")

    if is_base64_image(image):
        return save_base64_image(image)

    if is_url(image):
        return download_image(image)

    if len(image) < 4096 and Path(image).exists():
        return image

    raise ValueError(f"Cannot process image: {image[:50]}...")


def round_by_factor(x: int, factor: int) -> int:
    return round(x / factor) * factor


def ceil_by_factor(x: float, factor: int) -> int:
    return math.ceil(x / factor) * factor


def floor_by_factor(x: float, factor: int) -> int:
    return math.floor(x / factor) * factor


def smart_nframes(
    total_frames: int,
    video_fps: float,
    target_fps: float = DEFAULT_FPS,
    min_frames: int = MIN_FRAMES,
    max_frames: int = MAX_FRAMES,
) -> int:
    # Non-positive total_frames (cap.get returns 0/-1 on a broken or empty
    # stream) would otherwise force nframes >= FRAME_FACTOR and make
    # np.linspace(0, total_frames-1, n) emit negative indices downstream.
    if total_frames <= 0:
        logger.warning(
            "smart_nframes: non-positive total_frames=%d -> 0 frames", total_frames
        )
        return 0
    duration = total_frames / video_fps if video_fps > 0 else 0
    nframes = duration * target_fps
    nframes = max(min_frames, min(nframes, max_frames, total_frames))
    nframes = max(FRAME_FACTOR, floor_by_factor(nframes, FRAME_FACTOR))
    return int(nframes)


def extract_video_frames_smart(
    video_path: str,
    fps: float = DEFAULT_FPS,
    max_frames: int = MAX_FRAMES,
    resize: tuple[int, int] | None = None,
) -> list[np.ndarray]:
    try:
        import cv2
    except ImportError:
        raise ImportError("opencv-python is required for video processing")

    cap = cv2.VideoCapture(video_path)
    try:
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        video_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

        nframes = smart_nframes(
            total_frames=total_frames,
            video_fps=video_fps,
            target_fps=fps,
            max_frames=max_frames,
        )

        indices = np.linspace(0, total_frames - 1, nframes).round().astype(int)

        logger.info(
            "Video: %d total frames @ %.1f fps, extracting %d frames",
            total_frames,
            video_fps,
            nframes,
        )

        frames = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret:
                continue
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if resize:
                frame = cv2.resize(frame, resize)
            frames.append(frame)
    finally:
        cap.release()
    return frames


def save_frames_to_temp(frames: list[np.ndarray]) -> list[str]:
    from PIL import Image

    paths = []
    for i, frame in enumerate(frames):
        img = Image.fromarray(frame)
        temp_file = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
        try:
            img.save(temp_file.name, "JPEG", quality=85)
        except Exception:
            logger.warning("save_frames_to_temp: img.save failed for frame %d", i)
            temp_file.close()
            try:
                os.unlink(temp_file.name)
            except OSError:
                pass
            raise
        temp_file.close()
        paths.append(_temp_manager.register(temp_file.name))

    return paths


def describe_video(video_path: str) -> dict:
    try:
        import cv2
    except ImportError:
        raise ImportError("opencv-python is required for video processing")

    cap = cv2.VideoCapture(video_path)
    try:
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration = total_frames / fps if fps > 0 else 0.0
    finally:
        cap.release()

    return {
        "fps": fps,
        "total_frames": total_frames,
        "width": width,
        "height": height,
        "duration": duration,
        "path": video_path,
    }
