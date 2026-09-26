# SPDX-License-Identifier: Apache-2.0
# Marching cubes surface extraction (classic Lorensen-Cline tables), pure numpy.
# Port of ddalcu/mlx-serve src/marching_cubes.zig. Turns a scalar field sampled
# on a regular grid into an indexed triangle mesh with per-vertex normals.
# Sign convention: INSIDE = positive field values, surface at `level`. Triangles
# wound CCW-outward (matches glTF front-face) via a global winding vote.
from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)

# Cube corner offsets (x,y,z) for corner index 0..7.
_CORNER = np.array(
    [
        [0, 0, 0],
        [1, 0, 0],
        [1, 1, 0],
        [0, 1, 0],
        [0, 0, 1],
        [1, 0, 1],
        [1, 1, 1],
        [0, 1, 1],
    ],
    dtype=np.int32,
)

# Edge -> (corner_a, corner_b) pair.
_EDGE_CORNERS = np.array(
    [
        [0, 1],
        [1, 2],
        [2, 3],
        [3, 0],
        [4, 5],
        [5, 6],
        [6, 7],
        [7, 4],
        [0, 4],
        [1, 5],
        [2, 6],
        [3, 7],
    ],
    dtype=np.int32,
)

# Edge -> canonical origin corner (the lower-index endpoint of the axis).
_EDGE_ORIGIN = np.array(
    [
        [0, 0, 0],
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 0],
        [0, 0, 1],
        [1, 0, 1],
        [0, 1, 1],
        [0, 0, 1],
        [0, 0, 0],
        [1, 0, 0],
        [1, 1, 0],
        [0, 1, 0],
    ],
    dtype=np.int32,
)

# Edge -> axis (0=x, 1=y, 2=z).
_EDGE_AXIS = np.array([0, 1, 0, 1, 0, 1, 0, 1, 2, 2, 2, 2], dtype=np.int32)

# 256-entry edge bitmask: which of the 12 edges are crossed for each cube index.
_EDGE_TABLE = np.array(
    [
        0x0,
        0x109,
        0x203,
        0x30A,
        0x406,
        0x50F,
        0x605,
        0x70C,
        0x80C,
        0x905,
        0xA0F,
        0xB06,
        0xC0A,
        0xD03,
        0xE09,
        0xF00,
        0x190,
        0x99,
        0x393,
        0x29A,
        0x596,
        0x49F,
        0x795,
        0x69C,
        0x99C,
        0x895,
        0xB9F,
        0xA96,
        0xD9A,
        0xC93,
        0xF99,
        0xE90,
        0x230,
        0x339,
        0x33,
        0x13A,
        0x636,
        0x73F,
        0x435,
        0x53C,
        0xA3C,
        0xB35,
        0x83F,
        0x936,
        0xE3A,
        0xF33,
        0xC39,
        0xD30,
        0x3A0,
        0x2A9,
        0x1A3,
        0xAA,
        0x7A6,
        0x6AF,
        0x5A5,
        0x4AC,
        0xBAC,
        0xAA5,
        0x9AF,
        0x8A6,
        0xFAA,
        0xEA3,
        0xDA9,
        0xCA0,
        0x460,
        0x569,
        0x663,
        0x76A,
        0x66,
        0x16F,
        0x265,
        0x36C,
        0xC6C,
        0xD65,
        0xE6F,
        0xF66,
        0x86A,
        0x963,
        0xA69,
        0xB60,
        0x5F0,
        0x4F9,
        0x7F3,
        0x6FA,
        0x1F6,
        0xFF,
        0x3F5,
        0x2FC,
        0xDFC,
        0xCF5,
        0xFFF,
        0xEF6,
        0x9FA,
        0x8F3,
        0xBF9,
        0xAF0,
        0x650,
        0x759,
        0x453,
        0x55A,
        0x256,
        0x35F,
        0x55,
        0x15C,
        0xE5C,
        0xF55,
        0xC5F,
        0xD56,
        0xA5A,
        0xB53,
        0x859,
        0x950,
        0x7C0,
        0x6C9,
        0x5C3,
        0x4CA,
        0x3C6,
        0x2CF,
        0x1C5,
        0xCC,
        0xFCC,
        0xEC5,
        0xDCF,
        0xCC6,
        0xBCA,
        0xAC3,
        0x9C9,
        0x8C0,
        0x8C0,
        0x9C9,
        0xAC3,
        0xBCA,
        0xCC6,
        0xDCF,
        0xEC5,
        0xFCC,
        0xCC,
        0x1C5,
        0x2CF,
        0x3C6,
        0x4CA,
        0x5C3,
        0x6C9,
        0x7C0,
        0x950,
        0x859,
        0xB53,
        0xA5A,
        0xD56,
        0xC5F,
        0xF55,
        0xE5C,
        0x15C,
        0x55,
        0x35F,
        0x256,
        0x55A,
        0x453,
        0x759,
        0x650,
        0xAF0,
        0xBF9,
        0x8F3,
        0x9FA,
        0xEF6,
        0xFFF,
        0xCF5,
        0xDFC,
        0x2FC,
        0x3F5,
        0xFF,
        0x1F6,
        0x6FA,
        0x7F3,
        0x4F9,
        0x5F0,
        0xB60,
        0xA69,
        0x963,
        0x86A,
        0xF66,
        0xE6F,
        0xD65,
        0xC6C,
        0x36C,
        0x265,
        0x16F,
        0x66,
        0x76A,
        0x663,
        0x569,
        0x460,
        0xCA0,
        0xDA9,
        0xEA3,
        0xFAA,
        0x8A6,
        0x9AF,
        0xAA5,
        0xBAC,
        0x4AC,
        0x5A5,
        0x6AF,
        0x7A6,
        0xAA,
        0x1A3,
        0x2A9,
        0x3A0,
        0xD30,
        0xC39,
        0xF33,
        0xE3A,
        0x936,
        0x83F,
        0xB35,
        0xA3C,
        0x53C,
        0x435,
        0x73F,
        0x636,
        0x13A,
        0x33,
        0x339,
        0x230,
        0xE90,
        0xF99,
        0xC93,
        0xD9A,
        0xA96,
        0xB9F,
        0x895,
        0x99C,
        0x69C,
        0x795,
        0x49F,
        0x596,
        0x29A,
        0x393,
        0x99,
        0x190,
        0xF00,
        0xE09,
        0xD03,
        0xC0A,
        0xB06,
        0xA0F,
        0x905,
        0x80C,
        0x70C,
        0x605,
        0x50F,
        0x406,
        0x30A,
        0x203,
        0x109,
        0x0,
    ],
    dtype=np.int64,
)

# 256-entry triangle table: up to 5 triangles (15 edge indices), -1 sentinel.
# Flattened 256*16; rows are the zig TRI_TABLE verbatim.
_TRI_TABLE_FLAT = """
-1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
0 8 3 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
0 1 9 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
1 8 3 9 8 1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
1 2 10 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
0 8 3 1 2 10 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
9 2 10 0 2 9 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
2 8 3 2 10 8 10 9 8 -1 -1 -1 -1 -1 -1 -1
3 11 2 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
0 11 2 8 11 0 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
1 9 0 2 3 11 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
1 11 2 1 9 11 9 8 11 -1 -1 -1 -1 -1 -1 -1
3 10 1 11 10 3 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
0 10 1 0 8 10 8 11 10 -1 -1 -1 -1 -1 -1 -1
3 9 0 3 11 9 11 10 9 -1 -1 -1 -1 -1 -1 -1
9 8 10 10 8 11 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
4 7 8 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
4 3 0 7 3 4 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
0 1 9 8 4 7 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
4 1 9 4 7 1 7 3 1 -1 -1 -1 -1 -1 -1 -1
1 2 10 8 4 7 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
3 4 7 3 0 4 1 2 10 -1 -1 -1 -1 -1 -1 -1
9 2 10 9 0 2 8 4 7 -1 -1 -1 -1 -1 -1 -1
2 10 9 2 9 7 2 7 3 7 9 4 -1 -1 -1 -1
8 4 7 3 11 2 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
11 4 7 11 2 4 2 0 4 -1 -1 -1 -1 -1 -1 -1
9 0 1 8 4 7 2 3 11 -1 -1 -1 -1 -1 -1 -1
4 7 11 9 4 11 9 11 2 9 2 1 -1 -1 -1 -1
3 10 1 3 11 10 7 8 4 -1 -1 -1 -1 -1 -1 -1
1 11 10 1 4 11 1 0 4 7 11 4 -1 -1 -1 -1
4 7 8 9 0 11 9 11 10 11 0 3 -1 -1 -1 -1
4 7 11 4 11 9 9 11 10 -1 -1 -1 -1 -1 -1 -1
9 5 4 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
9 5 4 0 8 3 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
0 5 4 1 5 0 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
8 5 4 8 3 5 3 1 5 -1 -1 -1 -1 -1 -1 -1
1 2 10 9 5 4 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
3 0 8 1 2 10 4 9 5 -1 -1 -1 -1 -1 -1 -1
5 2 10 5 4 2 4 0 2 -1 -1 -1 -1 -1 -1 -1
2 10 5 3 2 5 3 5 4 3 4 8 -1 -1 -1 -1
9 5 4 2 3 11 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
0 11 2 0 8 11 4 9 5 -1 -1 -1 -1 -1 -1 -1
0 5 4 0 1 5 2 3 11 -1 -1 -1 -1 -1 -1 -1
2 1 5 2 5 8 2 8 11 4 8 5 -1 -1 -1 -1
10 3 11 10 1 3 9 5 4 -1 -1 -1 -1 -1 -1 -1
4 9 5 0 8 1 8 10 1 8 11 10 -1 -1 -1 -1
5 4 0 5 0 11 5 11 10 11 0 3 -1 -1 -1 -1
5 4 8 5 8 10 10 8 11 -1 -1 -1 -1 -1 -1 -1
9 7 8 5 7 9 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
9 3 0 9 5 3 5 7 3 -1 -1 -1 -1 -1 -1 -1
0 7 8 0 1 7 1 5 7 -1 -1 -1 -1 -1 -1 -1
1 5 3 3 5 7 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
9 7 8 9 5 7 10 1 2 -1 -1 -1 -1 -1 -1 -1
10 1 2 9 5 0 5 3 0 5 7 3 -1 -1 -1 -1
8 0 2 8 2 5 8 5 7 10 5 2 -1 -1 -1 -1
2 10 5 2 5 3 3 5 7 -1 -1 -1 -1 -1 -1 -1
7 9 5 7 8 9 3 11 2 -1 -1 -1 -1 -1 -1 -1
9 5 7 9 7 2 9 2 0 2 7 11 -1 -1 -1 -1
2 3 11 0 1 8 1 7 8 1 5 7 -1 -1 -1 -1
11 2 1 11 1 7 7 1 5 -1 -1 -1 -1 -1 -1 -1
9 5 8 8 5 7 10 1 3 10 3 11 -1 -1 -1 -1
5 7 0 5 0 9 7 11 0 1 0 10 11 10 0 -1
11 10 0 11 0 3 10 5 0 8 0 7 5 7 0 -1
11 10 5 7 11 5 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
10 6 5 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
0 8 3 5 10 6 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
9 0 1 5 10 6 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
1 8 3 1 9 8 5 10 6 -1 -1 -1 -1 -1 -1 -1
1 6 5 2 6 1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
1 6 5 1 2 6 3 0 8 -1 -1 -1 -1 -1 -1 -1
9 6 5 9 0 6 0 2 6 -1 -1 -1 -1 -1 -1 -1
5 9 8 5 8 2 5 2 6 3 2 8 -1 -1 -1 -1
2 3 11 10 6 5 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
11 0 8 11 2 0 10 6 5 -1 -1 -1 -1 -1 -1 -1
0 1 9 2 3 11 5 10 6 -1 -1 -1 -1 -1 -1 -1
5 10 6 1 9 2 9 11 2 9 8 11 -1 -1 -1 -1
6 3 11 6 5 3 5 1 3 -1 -1 -1 -1 -1 -1 -1
0 8 11 0 11 5 0 5 1 5 11 6 -1 -1 -1 -1
3 11 6 0 3 6 0 6 5 0 5 9 -1 -1 -1 -1
6 5 9 6 9 11 11 9 8 -1 -1 -1 -1 -1 -1 -1
5 10 6 4 7 8 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
4 3 0 4 7 3 6 5 10 -1 -1 -1 -1 -1 -1 -1
1 9 0 5 10 6 8 4 7 -1 -1 -1 -1 -1 -1 -1
10 6 5 1 9 7 1 7 3 7 9 4 -1 -1 -1 -1
6 1 2 6 5 1 4 7 8 -1 -1 -1 -1 -1 -1 -1
1 2 5 5 2 6 3 0 4 3 4 7 -1 -1 -1 -1
8 4 7 9 0 5 0 6 5 0 2 6 -1 -1 -1 -1
7 3 9 7 9 4 3 2 9 5 9 6 2 6 9 -1
3 11 2 7 8 4 10 6 5 -1 -1 -1 -1 -1 -1 -1
5 10 6 4 7 2 4 2 0 2 7 11 -1 -1 -1 -1
0 1 9 4 7 8 2 3 11 5 10 6 -1 -1 -1 -1
9 2 1 9 11 2 9 4 11 7 11 4 5 10 6 -1
8 4 7 3 11 5 3 5 1 5 11 6 -1 -1 -1 -1
5 1 11 5 11 6 1 0 11 7 11 4 0 4 11 -1
0 5 9 0 6 5 0 3 6 11 6 3 8 4 7 -1
6 5 9 6 9 11 4 7 9 7 11 9 -1 -1 -1 -1
10 4 9 6 4 10 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
4 10 6 4 9 10 0 8 3 -1 -1 -1 -1 -1 -1 -1
10 0 1 10 6 0 6 4 0 -1 -1 -1 -1 -1 -1 -1
8 3 1 8 1 6 8 6 4 6 1 10 -1 -1 -1 -1
1 4 9 1 2 4 2 6 4 -1 -1 -1 -1 -1 -1 -1
3 0 8 1 2 9 2 4 9 2 6 4 -1 -1 -1 -1
0 2 4 4 2 6 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
8 3 2 8 2 4 4 2 6 -1 -1 -1 -1 -1 -1 -1
10 4 9 10 6 4 11 2 3 -1 -1 -1 -1 -1 -1 -1
0 8 2 2 8 11 4 9 10 4 10 6 -1 -1 -1 -1
3 11 2 0 1 6 0 6 4 6 1 10 -1 -1 -1 -1
6 4 1 6 1 10 4 8 1 2 1 11 8 11 1 -1
9 6 4 9 3 6 9 1 3 11 6 3 -1 -1 -1 -1
8 11 1 8 1 0 11 6 1 9 1 4 6 4 1 -1
3 11 6 3 6 0 0 6 4 -1 -1 -1 -1 -1 -1 -1
6 4 8 11 6 8 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
7 10 6 7 8 10 8 9 10 -1 -1 -1 -1 -1 -1 -1
0 7 3 0 10 7 0 9 10 6 7 10 -1 -1 -1 -1
10 6 7 1 10 7 1 7 8 1 8 0 -1 -1 -1 -1
10 6 7 10 7 1 1 7 3 -1 -1 -1 -1 -1 -1 -1
1 2 6 1 6 8 1 8 9 8 6 7 -1 -1 -1 -1
2 6 9 2 9 1 6 7 9 0 9 3 7 3 9 -1
7 8 0 7 0 6 6 0 2 -1 -1 -1 -1 -1 -1 -1
7 3 2 6 7 2 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
2 3 11 10 6 8 10 8 9 8 6 7 -1 -1 -1 -1
2 0 7 2 7 11 0 9 7 6 7 10 9 10 7 -1
1 8 0 1 7 8 1 10 7 6 7 10 2 3 11 -1
11 2 1 11 1 7 10 6 1 6 7 1 -1 -1 -1 -1
8 9 6 8 6 7 9 1 6 11 6 3 1 3 6 -1
0 9 1 11 6 7 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
7 8 0 7 0 6 3 11 0 11 6 0 -1 -1 -1 -1
7 11 6 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
7 6 11 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
3 0 8 11 7 6 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
0 1 9 11 7 6 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
8 1 9 8 3 1 11 7 6 -1 -1 -1 -1 -1 -1 -1
10 1 2 6 11 7 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
1 2 10 3 0 8 6 11 7 -1 -1 -1 -1 -1 -1 -1
2 9 0 2 10 9 6 11 7 -1 -1 -1 -1 -1 -1 -1
6 11 7 2 10 3 10 8 3 10 9 8 -1 -1 -1 -1
7 2 3 6 2 7 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
7 0 8 7 6 0 6 2 0 -1 -1 -1 -1 -1 -1 -1
2 7 6 2 3 7 0 1 9 -1 -1 -1 -1 -1 -1 -1
1 6 2 1 8 6 1 9 8 8 7 6 -1 -1 -1 -1
10 7 6 10 1 7 1 3 7 -1 -1 -1 -1 -1 -1 -1
10 7 6 1 7 10 1 8 7 1 0 8 -1 -1 -1 -1
0 3 7 0 7 10 0 10 9 6 10 7 -1 -1 -1 -1
7 6 10 7 10 8 8 10 9 -1 -1 -1 -1 -1 -1 -1
6 8 4 11 8 6 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
3 6 11 3 0 6 0 4 6 -1 -1 -1 -1 -1 -1 -1
8 6 11 8 4 6 9 0 1 -1 -1 -1 -1 -1 -1 -1
9 4 6 9 6 3 9 3 1 11 3 6 -1 -1 -1 -1
6 8 4 6 11 8 2 10 1 -1 -1 -1 -1 -1 -1 -1
1 2 10 3 0 11 0 6 11 0 4 6 -1 -1 -1 -1
4 11 8 4 6 11 0 2 9 2 10 9 -1 -1 -1 -1
10 9 3 10 3 2 9 4 3 11 3 6 4 6 3 -1
8 2 3 8 4 2 4 6 2 -1 -1 -1 -1 -1 -1 -1
0 4 2 4 6 2 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
1 9 0 2 3 4 2 4 6 4 3 8 -1 -1 -1 -1
1 9 4 1 4 2 2 4 6 -1 -1 -1 -1 -1 -1 -1
8 1 3 8 6 1 8 4 6 6 10 1 -1 -1 -1 -1
10 1 0 10 0 6 6 0 4 -1 -1 -1 -1 -1 -1 -1
4 6 3 4 3 8 6 10 3 0 3 9 10 9 3 -1
10 9 4 6 10 4 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
4 9 5 7 6 11 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
0 8 3 4 9 5 11 7 6 -1 -1 -1 -1 -1 -1 -1
5 0 1 5 4 0 7 6 11 -1 -1 -1 -1 -1 -1 -1
11 7 6 8 3 4 3 5 4 3 1 5 -1 -1 -1 -1
9 5 4 10 1 2 7 6 11 -1 -1 -1 -1 -1 -1 -1
6 11 7 1 2 10 0 8 3 4 9 5 -1 -1 -1 -1
7 6 11 5 4 10 4 2 10 4 0 2 -1 -1 -1 -1
3 4 8 3 5 4 3 2 5 10 5 2 11 7 6 -1
7 2 3 7 6 2 5 4 9 -1 -1 -1 -1 -1 -1 -1
9 5 4 0 8 6 0 6 2 6 8 7 -1 -1 -1 -1
3 6 2 3 7 6 1 5 0 5 4 0 -1 -1 -1 -1
6 2 8 6 8 7 2 1 8 4 8 5 1 5 8 -1
9 5 4 10 1 6 1 7 6 1 3 7 -1 -1 -1 -1
1 6 10 1 7 6 1 0 7 8 7 0 9 5 4 -1
4 0 10 4 10 5 0 3 10 6 10 7 3 7 10 -1
7 6 10 7 10 8 5 4 10 4 8 10 -1 -1 -1 -1
6 9 5 6 11 9 11 8 9 -1 -1 -1 -1 -1 -1 -1
3 6 11 0 6 3 0 5 6 0 9 5 -1 -1 -1 -1
0 11 8 0 5 11 0 1 5 5 6 11 -1 -1 -1 -1
6 11 3 6 3 5 5 3 1 -1 -1 -1 -1 -1 -1 -1
1 2 10 9 5 11 9 11 8 11 5 6 -1 -1 -1 -1
0 11 3 0 6 11 0 9 6 5 6 9 1 2 10 -1
11 8 5 11 5 6 8 0 5 10 5 2 0 2 5 -1
6 11 3 6 3 5 2 10 3 10 5 3 -1 -1 -1 -1
5 8 9 5 2 8 5 6 2 3 8 2 -1 -1 -1 -1
9 5 6 9 6 0 0 6 2 -1 -1 -1 -1 -1 -1 -1
1 5 8 1 8 0 5 6 8 3 8 2 6 2 8 -1
1 5 6 2 1 6 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
1 3 6 1 6 10 3 8 6 5 6 9 8 9 6 -1
10 1 0 10 0 6 9 5 0 5 6 0 -1 -1 -1 -1
0 3 8 5 6 10 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
10 5 6 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
11 5 10 7 5 11 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
11 5 10 11 7 5 8 3 0 -1 -1 -1 -1 -1 -1 -1
5 11 7 5 10 11 1 9 0 -1 -1 -1 -1 -1 -1 -1
10 7 5 10 11 7 9 8 1 8 3 1 -1 -1 -1 -1
11 1 2 11 7 1 7 5 1 -1 -1 -1 -1 -1 -1 -1
0 8 3 1 2 7 1 7 5 7 2 11 -1 -1 -1 -1
9 7 5 9 2 7 9 0 2 2 11 7 -1 -1 -1 -1
7 5 2 7 2 11 5 9 2 3 2 8 9 8 2 -1
2 5 10 2 3 5 3 7 5 -1 -1 -1 -1 -1 -1 -1
8 2 0 8 5 2 8 7 5 10 2 5 -1 -1 -1 -1
9 0 1 5 10 3 5 3 7 3 10 2 -1 -1 -1 -1
9 8 2 9 2 1 8 7 2 10 2 5 7 5 2 -1
1 3 5 3 7 5 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
0 8 7 0 7 1 1 7 5 -1 -1 -1 -1 -1 -1 -1
9 0 3 9 3 5 5 3 7 -1 -1 -1 -1 -1 -1 -1
9 8 7 5 9 7 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
5 8 4 5 10 8 10 11 8 -1 -1 -1 -1 -1 -1 -1
5 0 4 5 11 0 5 10 11 11 3 0 -1 -1 -1 -1
0 1 9 8 4 10 8 10 11 10 4 5 -1 -1 -1 -1
10 11 4 10 4 5 11 3 4 9 4 1 3 1 4 -1
2 5 1 2 8 5 2 11 8 4 5 8 -1 -1 -1 -1
0 4 11 0 11 3 4 5 11 2 11 1 5 1 11 -1
0 2 5 0 5 9 2 11 5 4 5 8 11 8 5 -1
9 4 5 2 11 3 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
2 5 10 3 5 2 3 4 5 3 8 4 -1 -1 -1 -1
5 10 2 5 2 4 4 2 0 -1 -1 -1 -1 -1 -1 -1
3 10 2 3 5 10 3 8 5 4 5 8 0 1 9 -1
5 10 2 5 2 4 1 9 2 9 4 2 -1 -1 -1 -1
8 4 5 8 5 3 3 5 1 -1 -1 -1 -1 -1 -1 -1
0 4 5 1 0 5 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
8 4 5 8 5 3 9 0 5 0 3 5 -1 -1 -1 -1
9 4 5 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
4 11 7 4 9 11 9 10 11 -1 -1 -1 -1 -1 -1 -1
0 8 3 4 9 7 9 11 7 9 10 11 -1 -1 -1 -1
1 10 11 1 11 4 1 4 0 7 4 11 -1 -1 -1 -1
3 1 4 3 4 8 1 10 4 7 4 11 10 11 4 -1
4 11 7 9 11 4 9 2 11 9 1 2 -1 -1 -1 -1
9 7 4 9 11 7 9 1 11 2 11 1 0 8 3 -1
11 7 4 11 4 2 2 4 0 -1 -1 -1 -1 -1 -1 -1
11 7 4 11 4 2 8 3 4 3 2 4 -1 -1 -1 -1
2 9 10 2 7 9 2 3 7 7 4 9 -1 -1 -1 -1
9 10 7 9 7 4 10 2 7 8 7 0 2 0 7 -1
3 7 10 3 10 2 7 4 10 1 10 0 4 0 10 -1
1 10 2 8 7 4 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
4 9 1 4 1 7 7 1 3 -1 -1 -1 -1 -1 -1 -1
4 9 1 4 1 7 0 8 1 8 7 1 -1 -1 -1 -1
4 0 3 7 4 3 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
4 8 7 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
9 10 8 10 11 8 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
3 0 9 3 9 11 11 9 10 -1 -1 -1 -1 -1 -1 -1
0 1 10 0 10 8 8 10 11 -1 -1 -1 -1 -1 -1 -1
3 1 10 11 3 10 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
1 2 11 1 11 9 9 11 8 -1 -1 -1 -1 -1 -1 -1
3 0 9 3 9 11 1 2 9 2 11 9 -1 -1 -1 -1
0 2 11 8 0 11 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
3 2 11 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
2 3 8 2 8 10 10 8 9 -1 -1 -1 -1 -1 -1 -1
9 10 2 0 9 2 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
2 3 8 2 8 10 0 1 8 1 10 8 -1 -1 -1 -1
1 10 2 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
1 3 8 9 1 8 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
0 9 1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
0 3 8 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
-1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
"""
_TRI_TABLE = np.array(
    [int(x) for x in _TRI_TABLE_FLAT.split()], dtype=np.int64
).reshape(256, 16)


def extract(
    grid: np.ndarray,
    level: float = 0.0,
    scale: tuple[float, float, float] = (1.0, 1.0, 1.0),
    offset: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # grid: (n0, n1, n2) scalar field, x-major ij order: idx=(i*n1+j)*n2+k.
    # Returns (vertices [V,3], normals [V,3], indices [F,3] uint32).
    # Inside-positive sign; surface at `level`; CCW faces outward.
    n0, n1, n2 = grid.shape
    if n0 < 2 or n1 < 2 or n2 < 2:
        return (
            np.zeros((0, 3), dtype=np.float32),
            np.zeros((0, 3), dtype=np.float32),
            np.zeros((0, 3), dtype=np.uint32),
        )
    scale_arr = np.asarray(scale, dtype=np.float32)
    offset_arr = np.asarray(offset, dtype=np.float32)

    # Corner samples for every cell: shape (ncells, 8).
    # Cell (i,j,k) occupies corners [i..i+1, j..j+1, k..k+1].
    gi = grid[: n0 - 1, : n1 - 1, : n2 - 1]
    corners = np.empty((8, n0 - 1, n1 - 1, n2 - 1), dtype=grid.dtype)
    for c in range(8):
        di, dj, dk = _CORNER[c]
        corners[c] = grid[di : n0 - 1 + di, dj : n1 - 1 + dj, dk : n2 - 1 + dk]
    # cube index: bit c set if corners[c] < level (outside).
    cube_idx = np.zeros((n0 - 1, n1 - 1, n2 - 1), dtype=np.int64)
    for c in range(8):
        cube_idx[corners[c] < level] |= 1 << c

    ncells = cube_idx.size
    edge_bits = _EDGE_TABLE[cube_idx.reshape(-1)]  # (ncells,)
    has_geom = edge_bits != 0
    if not has_geom.any():
        return (
            np.zeros((0, 3), dtype=np.float32),
            np.zeros((0, 3), dtype=np.float32),
            np.zeros((0, 3), dtype=np.uint32),
        )

    # Flatten cell coords for active cells only.
    flat_idx = np.arange(ncells)[has_geom]
    ix = flat_idx // ((n1 - 1) * (n2 - 1))
    rem = flat_idx % ((n1 - 1) * (n2 - 1))
    iy = rem // (n2 - 1)
    iz = rem % (n2 - 1)
    cube_idx_act = cube_idx.reshape(-1)[has_geom]
    edge_bits_act = edge_bits[has_geom]

    # Build vertex per crossed edge. Edge key = corner_lin * 3 + axis (watertight
    # dedup across adjacent cells sharing the edge).
    verts: list[np.ndarray] = []
    norms: list[np.ndarray] = []
    tri_buf: list[tuple[int, int, int]] = []
    edge_vert_idx: dict[int, int] = {}
    v_count = 0

    # Precompute world coords of all grid corners once.
    ii = np.arange(n0)
    jj = np.arange(n1)
    kk = np.arange(n2)
    # world[i,j,k] = [i*sx+ox, j*sy+oy, k*sz+oz]
    wx = ii.astype(np.float32) * scale_arr[0] + offset_arr[0]
    wy = jj.astype(np.float32) * scale_arr[1] + offset_arr[1]
    wz = kk.astype(np.float32) * scale_arr[2] + offset_arr[2]

    # Gradient via central difference (one-sided at faces), per axis.
    grad = np.empty((n0, n1, n2, 3), dtype=np.float32)
    for axis in range(3):
        g = grid.astype(np.float32)
        dim = g.shape[axis]
        dlo = np.take(g, np.clip(np.arange(dim) - 1, 0, dim - 1), axis=axis)
        dhi = np.take(g, np.clip(np.arange(dim) + 1, 0, dim - 1), axis=axis)
        # span: 2 interior, 1 at faces.
        span = np.ones(dim, dtype=np.float32) * 2.0
        span[0] = 1.0
        span[-1] = 1.0
        grad[..., axis] = (dhi - dlo) / (
            span.reshape([dim if a == axis else 1 for a in range(3)]) * scale_arr[axis]
        )

    for cell_i in range(flat_idx.size):
        ebits = int(edge_bits_act[cell_i])
        if ebits == 0:
            continue
        ci = int(ix[cell_i])
        cj = int(iy[cell_i])
        ck = int(iz[cell_i])
        cidx = int(cube_idx_act[cell_i])
        # corner values
        vals = np.empty(8, dtype=np.float32)
        for c in range(8):
            vals[c] = float(
                grid[ci + _CORNER[c, 0], cj + _CORNER[c, 1], ck + _CORNER[c, 2]]
            )
        vidx = [-1] * 12
        for e in range(12):
            if ebits & (1 << e):
                eo = _EDGE_ORIGIN[e]
                oi = ci + eo[0]
                oj = cj + eo[1]
                ok = ck + eo[2]
                axis = int(_EDGE_AXIS[e])
                corner_lin = (oi * n1 + oj) * n2 + ok
                key = corner_lin * 3 + axis
                v = edge_vert_idx.get(key)
                if v is None:
                    a, b = int(_EDGE_CORNERS[e, 0]), int(_EDGE_CORNERS[e, 1])
                    ai = ci + _CORNER[a, 0]
                    aj = cj + _CORNER[a, 1]
                    ak = ck + _CORNER[a, 2]
                    bi = ci + _CORNER[b, 0]
                    bj = cj + _CORNER[b, 1]
                    bk = ck + _CORNER[b, 2]
                    va = vals[a]
                    vb = vals[b]
                    d = vb - va
                    t = 0.5 if abs(d) < 1e-12 else (level - va) / d
                    pa = np.array([wx[ai], wy[aj], wz[ak]], dtype=np.float32)
                    pb = np.array([wx[bi], wy[bj], wz[bk]], dtype=np.float32)
                    pos = pa + t * (pb - pa)
                    ga = grad[ai, aj, ak]
                    gb = grad[bi, bj, bk]
                    nrm = -(ga + t * (gb - ga))
                    nlen = float(np.linalg.norm(nrm))
                    if nlen < 1e-9:
                        nrm = np.zeros(3, dtype=np.float32)
                    else:
                        nrm = (nrm / nlen).astype(np.float32)
                    verts.append(pos)
                    norms.append(nrm)
                    v = v_count
                    edge_vert_idx[key] = v
                    v_count += 1
                vidx[e] = v
        tris = _TRI_TABLE[cidx]
        for t in range(0, 16, 3):
            if tris[t] < 0:
                break
            tri_buf.append((vidx[tris[t]], vidx[tris[t + 1]], vidx[tris[t + 2]]))

    if v_count == 0:
        return (
            np.zeros((0, 3), dtype=np.float32),
            np.zeros((0, 3), dtype=np.float32),
            np.zeros((0, 3), dtype=np.uint32),
        )

    vert_arr = np.stack(verts).astype(np.float32)
    norm_arr = np.stack(norms).astype(np.float32)
    idx_arr = np.array(tri_buf, dtype=np.uint32).reshape(-1, 3)

    # Global winding vote: flip all if CCW face normals disagree with -grad normals.
    if idx_arr.shape[0] > 0:
        v0 = vert_arr[idx_arr[:, 0]]
        v1 = vert_arr[idx_arr[:, 1]]
        v2 = vert_arr[idx_arr[:, 2]]
        face_n = np.cross(v1 - v0, v2 - v0)
        vert_n = (
            norm_arr[idx_arr[:, 0]] + norm_arr[idx_arr[:, 1]] + norm_arr[idx_arr[:, 2]]
        ) / 3.0
        agree = float(np.sum(face_n * vert_n))
        if agree < 0:
            idx_arr[:, [1, 2]] = idx_arr[:, [2, 1]]

    logger.info(
        "marching_cubes: %d verts, %d tris (grid %dx%dx%d, level=%.3f)",
        vert_arr.shape[0],
        idx_arr.shape[0],
        n0,
        n1,
        n2,
        level,
    )
    return vert_arr, norm_arr, idx_arr


def write_obj(
    path: str,
    vertices: np.ndarray,
    normals: np.ndarray,
    indices: np.ndarray,
) -> None:
    # Minimal Wavefront OBJ: v/vn/f (triangulated, 1-indexed).
    with open(path, "w") as f:
        f.write("# fusion-mlx Hunyuan3D-2.1 mesh\n")
        for v in vertices:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        for n in normals:
            f.write(f"vn {n[0]:.6f} {n[1]:.6f} {n[2]:.6f}\n")
        for tri in indices:
            a, b, c = int(tri[0]) + 1, int(tri[1]) + 1, int(tri[2]) + 1
            f.write(f"f {a}//{a} {b}//{b} {c}//{c}\n")
    logger.info("write_obj: %s (%d verts, %d tris)", path, len(vertices), len(indices))
