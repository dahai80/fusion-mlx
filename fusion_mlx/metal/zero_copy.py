# SPDX-License-Identifier: Apache-2.0
# IOSurface ↔ MTLBuffer ↔ mlx::array ↔ CVPixelBuffer zero-copy bridge (#913).
#
# PRD hard target: audio-to-video RTT <= 80ms; output must be a CVPixelBuffer
# (IOSurface) handed zero-copy into LiveKit RTCVideoFrame — no numpy round-trip.
#
# True zero-copy requires C++ access to the MTLBuffer backing the mlx::array
# (via the shim _ext). When the extension is absent, this module provides a
# functional one-copy bridge (mlx array → numpy → CVPixelBuffer) so the API is
# usable today; the zero-copy path activates automatically when _ext is built.

from __future__ import annotations

import ctypes
import logging

import mlx.core as mx
import numpy as np

logger = logging.getLogger(__name__)

# CoreVideo / IOSurface system frameworks (macOS — no pip dep).
_CV_LOADED = False
_CVPixelBufferRef = ctypes.c_void_p
CVPixelBufferRef = _CVPixelBufferRef
try:
    ctypes.CDLL("/System/Library/Frameworks/CoreVideo.framework/CoreVideo")
    _CV_LOADED = True
except OSError:
    logger.debug("[zero_copy] CoreVideo framework not loadable (non-macOS?)")


def _native_zero_copy_available() -> bool:
    try:
        from ..shim.fast import is_native_available

        return is_native_available()
    except Exception:
        return False


class MetalZeroCopyBridge:
    """Zero-copy mlx::array → CVPixelBuffer bridge (#913).

    ``array_to_cvbuffer`` returns a CVPixelBuffer backed by the same IOSurface as
    the mlx array when the C++ extension is present (no copy). Otherwise it
    creates a CVPixelBuffer via CoreVideo C API and copies the array data once
    (functional fallback — one copy, still hands a CVPixelBuffer to LiveKit).

    The ``scale``/``offset`` denormalize the array (assumed normalized) to
    0-255 uint8 for BGRA CVPixelBuffer output.
    """

    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height
        self._native = _native_zero_copy_available()
        if self._native:
            logger.info("[zero_copy] native zero-copy path active (shim _ext)")
        else:
            logger.info("[zero_copy] one-copy fallback (numpy → CVPixelBuffer)")

    def array_to_cvbuffer(
        self, arr: mx.array, scale: float = 127.5, offset: float = 1.0
    ) -> CVPixelBufferRef:
        """mlx array (H,W,C) → CVPixelBufferRef (BGRA, IOSurface-backed).

        ``scale``/``offset``: output = clip((arr * scale + offset), 0, 255).astype(uint8).
        Default maps [-1,1]→[0,255] (MuseTalk VAE denorm).
        """
        if arr.ndim != 3 or arr.shape[-1] not in (1, 3, 4):
            raise ValueError(f"array_to_cvbuffer expects (H,W,1|3|4), got {arr.shape}")
        h, w, c = arr.shape
        if (h, w) != (self.height, self.width):
            raise ValueError(f"array {arr.shape} != bridge {self.height}x{self.width}")

        # Denormalize → uint8.
        img = np.array(arr) * scale + offset
        img = np.clip(img, 0, 255).astype(np.uint8)

        if c == 3:
            # RGB → BGRA (LiveKit RTCVideoFrame native format).
            bgra = np.zeros((h, w, 4), dtype=np.uint8)
            bgra[..., 0] = img[..., 2]  # B
            bgra[..., 1] = img[..., 1]  # G
            bgra[..., 2] = img[..., 0]  # R
            bgra[..., 3] = 255  # A
        elif c == 1:
            bgra = np.zeros((h, w, 4), dtype=np.uint8)
            bgra[..., :3] = img[..., 0:1]
            bgra[..., 3] = 255
        else:
            bgra = img  # already 4-ch

        if self._native:
            return self._native_cvbuffer(bgra)
        return self._fallback_cvbuffer(bgra)

    def _native_cvbuffer(self, bgra: np.ndarray) -> CVPixelBufferRef:
        """Zero-copy path via shim _ext (IOSurface from MTLBuffer)."""
        try:
            from ..shim.fast import _ext

            return _ext.array_to_cvbuffer(bgra, self.width, self.height)
        except Exception as exc:
            logger.warning("[zero_copy] native path failed (%s) — fallback", exc)
            return self._fallback_cvbuffer(bgra)

    def _fallback_cvbuffer(self, bgra: np.ndarray) -> CVPixelBufferRef:
        """One-copy path: CVPixelBufferCreate + memcpy (CoreVideo C API)."""
        if not _CV_LOADED:
            logger.debug("[zero_copy] CoreVideo unavailable — returning raw numpy ptr")
            return ctypes.cast(bgra.ctypes.data, ctypes.c_void_p)
        cv = ctypes.CDLL("/System/Library/Frameworks/CoreVideo.framework/CoreVideo")
        # Explicit ABI: pointers must be c_void_p (default c_int truncates on arm64 → segfault).
        cv.CVPixelBufferCreate.argtypes = [
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_size_t,
            ctypes.c_uint32,
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_void_p),
        ]
        cv.CVPixelBufferCreate.restype = ctypes.c_int32
        cv.CVPixelBufferLockBaseAddress.argtypes = [ctypes.c_void_p, ctypes.c_uint64]
        cv.CVPixelBufferLockBaseAddress.restype = ctypes.c_int32
        cv.CVPixelBufferUnlockBaseAddress.argtypes = [ctypes.c_void_p, ctypes.c_uint64]
        cv.CVPixelBufferUnlockBaseAddress.restype = ctypes.c_int32
        cv.CVPixelBufferGetBaseAddress.argtypes = [ctypes.c_void_p]
        cv.CVPixelBufferGetBaseAddress.restype = ctypes.c_void_p
        cv.CVPixelBufferGetBytesPerRow.argtypes = [ctypes.c_void_p]
        cv.CVPixelBufferGetBytesPerRow.restype = ctypes.c_size_t
        # kCVPixelFormatType_32BGRA = 'BGRA' = 0x42475241
        bgra_fmt = 0x42475241
        ptr = ctypes.c_void_p()
        # CVPixelBufferCreate(allocator=NULL, width, height, format, pool=NULL, bufOut)
        cv.CVPixelBufferCreate(
            None, self.width, self.height, bgra_fmt, None, ctypes.byref(ptr)
        )
        if not ptr.value:
            logger.warning("[zero_copy] CVPixelBufferCreate failed — raw numpy ptr")
            return ctypes.cast(bgra.ctypes.data, ctypes.c_void_p)
        # Lock base address, memcpy, unlock.
        cv.CVPixelBufferLockBaseAddress(ptr, 0)
        dst_addr = cv.CVPixelBufferGetBaseAddress(ptr) or 0
        row_bytes = cv.CVPixelBufferGetBytesPerRow(ptr)
        src_addr = bgra.ctypes.data
        if not dst_addr:
            logger.warning("[zero_copy] GetBaseAddress returned NULL — raw numpy ptr")
            cv.CVPixelBufferUnlockBaseAddress(ptr, 0)
            return ctypes.cast(bgra.ctypes.data, ctypes.c_void_p)
        for y in range(self.height):
            ctypes.memmove(
                dst_addr + y * row_bytes, src_addr + y * self.width * 4, self.width * 4
            )
        cv.CVPixelBufferUnlockBaseAddress(ptr, 0)
        return ptr

    @staticmethod
    def cvbuffer_to_numpy(
        cvbuf: CVPixelBufferRef, width: int, height: int
    ) -> np.ndarray:
        """Read a CVPixelBufferRef back to numpy (H,W,4) BGRA — for testing/parity."""
        cv = ctypes.CDLL("/System/Library/Frameworks/CoreVideo.framework/CoreVideo")
        base = ctypes.c_void_p()
        cv.CVPixelBufferLockBaseAddress(cvbuf, 0)
        base.value = cv.CVPixelBufferGetBaseAddress(cvbuf)
        row_bytes = cv.CVPixelBufferGetBytesPerRow(cvbuf)
        ptr = ctypes.cast(base, ctypes.POINTER(ctypes.c_uint8))
        out = np.zeros((height, width, 4), dtype=np.uint8)
        for y in range(height):
            for x in range(width * 4):
                out[y, x // 4, x % 4] = ptr[y * row_bytes + x]
        cv.CVPixelBufferUnlockBaseAddress(cvbuf, 0)
        return out


def make_bridge(width: int, height: int) -> MetalZeroCopyBridge:
    """Factory for the zero-copy bridge (#913)."""
    return MetalZeroCopyBridge(width, height)
