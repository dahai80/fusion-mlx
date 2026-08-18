# SPDX-License-Identifier: Apache-2.0
"""Regression tests for #500: Wan2.2-TI2V-5B all-NaN at large self-attn seq.

The diffusion denoising overflows Metal on a single (B,H,seq,seq) attention
matrix when seq > ~16384, producing all-NaN latents; the VAE then zeros NaN
and emits a static video. The fix auto-enables Q-chunking
(FUSION_WAN2_ATTN_CHUNK) when seq exceeds the safe threshold, and makes the
VAE-decode NaN path fail visibly instead of silently zeroing.
"""

import pytest

pytest.importorskip("mlx")

from fusion_mlx.video.wan2.generate import WAN2_SAFE_SEQ, _auto_enable_attn_chunk


def os_env_chunk():
    import os

    return os.getenv("FUSION_WAN2_ATTN_CHUNK")


def test_auto_chunk_below_threshold_noop(monkeypatch):
    monkeypatch.delenv("FUSION_WAN2_ATTN_CHUNK", raising=False)
    _auto_enable_attn_chunk(WAN2_SAFE_SEQ)
    assert os_env_chunk() is None


def test_auto_chunk_above_threshold_enables(monkeypatch):
    monkeypatch.delenv("FUSION_WAN2_ATTN_CHUNK", raising=False)
    _auto_enable_attn_chunk(WAN2_SAFE_SEQ + 1)
    assert os_env_chunk() == "8192"


def test_auto_chunk_respects_user_override(monkeypatch):
    monkeypatch.setenv("FUSION_WAN2_ATTN_CHUNK", "4096")
    _auto_enable_attn_chunk(WAN2_SAFE_SEQ + 99999)
    assert os_env_chunk() == "4096"


def test_auto_chunk_user_force_off_honored(monkeypatch):
    monkeypatch.setenv("FUSION_WAN2_ATTN_CHUNK", "0")
    _auto_enable_attn_chunk(WAN2_SAFE_SEQ + 99999)
    assert os_env_chunk() == "0"


def test_auto_chunk_exact_issue_seq(monkeypatch):
    monkeypatch.delenv("FUSION_WAN2_ATTN_CHUNK", raising=False)
    _auto_enable_attn_chunk(27280)
    assert os_env_chunk() == "8192"


def test_vae_nan_raises_not_silenced():
    import inspect

    from fusion_mlx.video.wan2 import generate as gen_mod

    src = inspect.getsource(gen_mod)
    assert (
        "video = np.nan_to_num(video" not in src
    ), "#500 regression: VAE decode silently zeros NaN instead of failing"
    assert "Wan2 VAE decode produced" in src
