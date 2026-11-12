# SPDX-License-Identifier: Apache-2.0
"""P1-4: SSD cold-tier TurboQuant compression (cold_tq module)."""

import mlx.core as mx

from fusion_mlx.cache.cold_tq import (
    TQ_COLD_MARKER,
    TQ_COLD_TYPE,
    _codec,
    maybe_quantize_block_kv,
    quantize_pair,
)


def _kv_layer(tokens=8, heads=2, head_dim=64):
    keys = mx.random.normal((1, heads, tokens, head_dim))
    values = mx.random.normal((1, heads, tokens, head_dim))
    return keys, values


class TestMaybeQuantizeBlockKv:
    def test_converts_plain_kv_layer(self):
        keys, values = _kv_layer()
        block = [(keys, values)]
        types = ["KVCache"]
        meta = [()]
        out = maybe_quantize_block_kv(block, types, meta, token_count=8)
        assert out is not None
        new_block, new_types, new_meta = out
        marker, (ks, vs) = new_block[0]
        assert marker == TQ_COLD_MARKER
        assert new_types[0] == TQ_COLD_TYPE
        # meta triple: (token_count, bits, seed) — restore reads [1]/[2]
        assert new_meta[0][0] == "8"
        assert int(new_meta[0][1]) in (3, 4)
        assert new_meta[0][2] == "0"

    def test_non_kv_layers_untouched(self):
        keys, values = _kv_layer()
        gdn_state = mx.zeros((4, 8))
        block = [(keys, values), ("__nstate__", "MiniMaxM3KVCache", [gdn_state]), 42]
        types = ["KVCache", "MiniMaxM3KVCache", "ArraysCache"]
        meta = [(), (), ()]
        out = maybe_quantize_block_kv(block, types, meta, token_count=8)
        assert out is not None
        new_block, new_types, _ = out
        assert isinstance(new_block[1], tuple) and new_block[1][0] == "__nstate__"
        assert new_types[1] == "MiniMaxM3KVCache"
        assert new_block[2] == 42

    def test_respects_layer_type_gate(self):
        keys, values = _kv_layer()
        block = [(keys, values)]
        types = ["RotatingKVCache"]
        out = maybe_quantize_block_kv(block, types, None, token_count=8)
        assert out is None

    def test_non_array_pair_untouched(self):
        block = [(("raw", "tuple"),)]
        types = ["KVCache"]
        assert maybe_quantize_block_kv(block, types, None, token_count=4) is None

    def test_three_dim_pair_untouched(self):
        keys = mx.random.normal((2, 8, 64))
        values = mx.random.normal((2, 8, 64))
        assert (
            maybe_quantize_block_kv([(keys, values)], ["KVCache"], None, 8) is None
        )

    def test_quantize_failure_keeps_fp16(self, monkeypatch):
        keys, values = _kv_layer()

        def _boom(*a, **k):
            raise RuntimeError("codec exploded")

        monkeypatch.setattr(
            "fusion_mlx.cache.cold_tq.quantize_pair", _boom
        )
        block = [(keys, values)]
        out = maybe_quantize_block_kv(block, ["KVCache"], [()], token_count=8)
        assert out is None
        # block list unchanged (no marker injected)
        assert isinstance(block[0][0], mx.array)


class TestRoundtrip:
    def test_rebuild_dequantized_kv_restores_fp16(self):
        keys, values = _kv_layer(tokens=8)
        mx.eval(keys, values)
        ks, vs, bits = quantize_pair(keys, values, seed=0)
        from fusion_mlx.turboquant_kv import rebuild_dequantized_kv

        keys2, values2 = rebuild_dequantized_kv(
            TQ_COLD_TYPE, ks, vs, float(bits), 0
        )
        assert keys2.shape == keys.shape
        assert values2.shape == values.shape
        # 4-bit Lloyd-Max per-dim: error is bounded but not tiny; sanity
        # check reconstruction quality beats a naive zero baseline.
        err = mx.abs(keys2.astype(mx.float32) - keys).mean().item()
        zero_err = mx.abs(keys).mean().item()
        assert err < zero_err * 0.5

    def test_block_roundtrip_via_marker_format(self):
        keys, values = _kv_layer(tokens=8)
        mx.eval(keys, values)
        block = [(keys, values)]
        types = ["KVCache"]
        meta = [()]
        new_block, new_types, new_meta = maybe_quantize_block_kv(
            block, types, meta, token_count=8
        )
        marker, (ks, vs) = new_block[0]
        assert marker == TQ_COLD_MARKER
        bits = int(new_meta[0][1])
        seed = int(new_meta[0][2])
        from fusion_mlx.turboquant_kv import rebuild_dequantized_kv

        keys2, values2 = rebuild_dequantized_kv(
            new_types[0], ks, vs, float(bits), seed
        )
        assert keys2.shape == keys.shape


class TestCodecReuse:
    def test_same_codec_for_same_config(self):
        c1 = _codec(64, 4, 0)
        c2 = _codec(64, 4, 0)
        assert c1 is c2

    def test_packed_width_matches_bits(self):
        keys, _ = _kv_layer(tokens=8)
        ks, _, bits = quantize_pair(keys, mx.zeros((1, 2, 8, 64)), seed=0)
        # norms carry tokens at axis 2 (what _state_length reads); 64-dim at
        # 4-bit packs to 64/8=8 uint32 words per vector on the last axis.
        assert ks.norms.shape == (1, 2, 8)
        assert ks.indices.shape == (1, 2, 8, 8)
        assert bits == 4


class TestSharedTypesList:
    def test_shared_layer_cache_types_not_mutated(self):
        # Regression: layer_cache_types is shared across blocks of one
        # store_cache call; mutating it made blocks 2..N skip conversion.
        keys, values = _kv_layer()
        types = ["KVCache"]
        maybe_quantize_block_kv([(keys, values)], types, None, token_count=8)
        assert types == ["KVCache"]

    def test_consecutive_blocks_all_convert(self):
        types = ["KVCache"]
        blocks = []
        for _ in range(3):
            keys, values = _kv_layer()
            out = maybe_quantize_block_kv(
                [(keys, values)], types, None, token_count=8
            )
            assert out is not None
            blocks.append(out[0])
        for b in blocks:
            assert b[0][0] == TQ_COLD_MARKER
