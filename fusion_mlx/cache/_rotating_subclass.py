# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from mlx_lm.models.cache import RotatingKVCache


class PrefillReadyRotatingKVCache(RotatingKVCache):
    """RotatingKVCache: logical length vs physical window.

    size() returns total tokens processed (chunked prefill offset).
    physical_resident_size returns buffer capacity for attention masking.
    """

    def size(self):
        """Return logical token count for chunked prefill offset tracking."""
        if self.keys is None:
            return 0
        return super().size()

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
