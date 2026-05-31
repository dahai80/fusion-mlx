# SPDX-License-Identifier: Apache-2.0
"""TurboQuant KV cache - stub module. Full implementation migrated when needed."""

from __future__ import annotations
from collections import namedtuple
from typing import List

TurboQuantMSEState = namedtuple("TurboQuantMSEState", [])
TurboQuantProdState = namedtuple("TurboQuantProdState", [])
TurboQuantPolarState = namedtuple("TurboQuantPolarState", [])
TurboQuantPolarProdState = namedtuple("TurboQuantPolarProdState", [])
TurboQuantSplitState = namedtuple("TurboQuantSplitState", [])


def _state_length(keys):
    if hasattr(keys, "norms") and hasattr(keys.norms, "shape"):
        return keys.norms.shape[2]
    if hasattr(keys, "shape") and len(keys.shape) >= 3:
        return keys.shape[2]
    return 0


def _slice_state_range(state, start, end):
    return state


def _concat_state(a, b):
    return a


def _rebuild_codecs(tq, keys, values):
    pass
