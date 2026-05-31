# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from mlx_lm.models.cache import RotatingKVCache


class PrefillReadyRotatingKVCache(RotatingKVCache):
    """RotatingKVCache that reports actual buffer length from size()."""

    def size(self):
        if self.keys is None:
            return 0
        buffer_len = self.keys.shape[2]
        if buffer_len == 0:
            return 0
        return min(super().size(), buffer_len)
