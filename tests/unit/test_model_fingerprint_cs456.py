# SPDX-License-Identifier: Apache-2.0
"""CS-4 / CS-5 / CS-6 (#811 audit 0906): content-aware cache fingerprinting.

A model/adapter path is not a content revision. Overwriting weights at the
same path must change the cache key so stale KV/completions are never reused.
"""

import os
import time

from fusion_mlx.cache.model_fingerprint import (
    adapter_path_signature,
    model_path_signature,
)


def _touch(path, size):
    with open(path, "wb") as f:
        f.truncate(size)
    # bump mtime so the second write is strictly newer (same-second writes
    # on some FSes collapse to identical mtime -> no signal change).
    os.utime(path, (time.time() + 10, time.time() + 10))


def test_model_path_signature_changes_on_weight_rewrite(tmp_path):
    d = tmp_path / "model"
    d.mkdir()
    _touch(d / "weights.safetensors", 1024)
    sig_a = model_path_signature(str(d))
    assert sig_a and sig_a != str(d)
    # rewrite at same path, different size+mtime -> different signature
    _touch(d / "weights.safetensors", 2048)
    sig_b = model_path_signature(str(d))
    assert sig_b != sig_a, "in-place weight rewrite must change cache key"


def test_model_path_signature_stable_when_unchanged(tmp_path):
    d = tmp_path / "model"
    d.mkdir()
    _touch(d / "w.gguf", 512)
    assert model_path_signature(str(d)) == model_path_signature(str(d))


def test_model_path_signature_falls_back_for_non_weight_dir(tmp_path):
    d = tmp_path / "empty"
    d.mkdir()
    (d / "config.json").write_text("{}")
    # no weight files -> falls back to raw path (no regression)
    assert model_path_signature(str(d)) == str(d)


def test_model_path_signature_empty():
    assert model_path_signature("") == ""


def test_model_path_signature_missing_path():
    # non-existent path -> os.walk raises OSError -> (0,0) -> raw path back
    assert model_path_signature("/no/such/model/xyz") == "/no/such/model/xyz"


def test_adapter_path_signature_changes_on_rewrite(tmp_path):
    d = tmp_path / "lora"
    d.mkdir()
    _touch(d / "adapter.safetensors", 256)
    a = adapter_path_signature(str(d))
    _touch(d / "adapter.safetensors", 512)
    b = adapter_path_signature(str(d))
    assert a != b
    assert a and a != str(d)


def test_adapter_path_signature_empty():
    assert adapter_path_signature("") == ""


def test_adapter_path_signature_non_weight_falls_back(tmp_path):
    d = tmp_path / "lora"
    d.mkdir()
    (d / "adapter_config.json").write_text("{}")
    assert adapter_path_signature(str(d)) == str(d)
