# SPDX-License-Identifier: Apache-2.0
"""Tests for H3 denoise cache clear (#1163).

Verifies mx.clear_cache() is called between denoise steps and before VAE
decode in all three generate functions, preventing MLX metal cache
accumulation that causes OOM on 128GB machines.
"""

from __future__ import annotations

import inspect


def _import_generate():
    from fusion_mlx.video.minimax_h3 import generate

    return generate


class TestClearMetalCacheHelper:
    def test_clear_metal_cache_exists(self):
        gen = _import_generate()
        assert hasattr(gen, "_clear_metal_cache")
        assert callable(gen._clear_metal_cache)


class TestT2vaCacheClear:
    def test_clear_cache_called_between_steps(self):
        gen = _import_generate()
        source = inspect.getsource(gen.generate_t2va_video)
        assert "_clear_metal_cache()" in source
        for_block = source[source.find("for i, t") : source.find("if i % 10")]
        assert "_clear_metal_cache()" in for_block

    def test_clear_cache_called_before_vae_decode(self):
        gen = _import_generate()
        source = inspect.getsource(gen.generate_t2va_video)
        clear_pos = source.find("_clear_metal_cache()")
        decode_pos = source.find("vae.decode(latents)")
        assert clear_pos < decode_pos


class TestT2vaAvCacheClear:
    def test_clear_cache_in_av_loop(self):
        gen = _import_generate()
        source = inspect.getsource(gen.generate_t2va_av)
        assert "_clear_metal_cache()" in source


class TestFl2vaCacheClear:
    def test_clear_cache_in_fl2va_loop(self):
        gen = _import_generate()
        source = inspect.getsource(gen.generate_fl2va_video)
        assert "_clear_metal_cache()" in source
