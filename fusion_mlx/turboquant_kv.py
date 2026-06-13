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
    try:
        return keys.norms.shape[2]
    except AttributeError:
        try:
            shape = keys.shape
            if len(shape) >= 3:
                return shape[2]
        except AttributeError:
            pass
    return 0


def _slice_state_range(state, start, end):
    return state


def _concat_state(a, b):
    return a


def _rebuild_codecs(tq, keys, values):
    pass
