# SPDX-License-Identifier: Apache-2.0
"""Unit tests for fusion_mlx.admin.subkey — _hash_key helper.

The route handlers (create_sub_key/delete_sub_key) require a live global
settings + admin auth fixture; covered by integration suites. This file
covers the pure helper _hash_key (deterministic sha256) which is the only
pure-logic surface in subkey.py. Aims at lifting subkey.py off 0% without
spinning a server.
"""

from __future__ import annotations

import hashlib

from fusion_mlx.admin.subkey import _hash_key


class TestHashKey:
    def test_deterministic(self):
        assert _hash_key("abc") == _hash_key("abc")

    def test_matches_sha256_hexdigest(self):
        assert _hash_key("abc") == hashlib.sha256(b"abc").hexdigest()

    def test_empty_string(self):
        assert _hash_key("") == hashlib.sha256(b"").hexdigest()

    def test_different_inputs_different_hashes(self):
        assert _hash_key("a") != _hash_key("b")

    def test_unicode_input(self):
        assert _hash_key("中文") == hashlib.sha256("中文".encode()).hexdigest()
