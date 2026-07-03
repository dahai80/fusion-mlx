# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from mlx_lm.models.cache import RotatingKVCache


class PrefillReadyRotatingKVCache(RotatingKVCache):
    """RotatingKVCache that reports actual buffer length from ``size()``.

    The default ``size()`` returns ``min(offset, max_size)`` which is the
    logical token count. For caches restored from SSD whose buffer was
    sliced shorter than ``max_size`` (e.g. extract() stripped left
    padding), the logical count can exceed ``keys.shape[2]``. mlx-lm's
    merge then either over-reads the RHS or, when pre-pads with
    zeros, lets those zeros leak into attention.

    Clamping to ``keys.shape[2]`` keeps merge consistent: the row gets
    exactly ``keys.shape[2]`` real entries, padded on the left by the
    enclosing batch (via ``left_padding``) instead of by phantom zeros.
    """

    def size(self):
        """Return logical token count clamped to physical buffer length."""
        if self.keys is None:
            return 0
        buffer_len = self.keys.shape[2]
        if buffer_len == 0:
            return 0
        return min(super().size(), buffer_len)

    @property
    def physical_resident_size(self) -> int:
        """Return min of logical length and physical buffer capacity.

        Used by attention kernels for masking, not by the scheduler.
        """
        if self.keys is None:
            return 0
        buffer_len = self.keys.shape[2]
        if buffer_len == 0:
            return 0
        return min(super().size(), buffer_len)
