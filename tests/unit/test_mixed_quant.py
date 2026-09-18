# SPDX-License-Identifier: Apache-2.0
"""Tests for PR-L: imatrix metadata + IQ/GGUF mixed-quant consumption."""

from __future__ import annotations

import numpy as np
import pytest

from fusion_mlx.migrate.asfw import UnsupportedQuantError
from fusion_mlx.migrate.gguf_reader import TensorInfo
from fusion_mlx.shim.mixed_quant import (
    ImatrixData,
    build_mixed_plan,
    chunk_mean,
    dequant_iq4_nl,
    dequant_mxfp4,
    dispatch_tensor,
    load_imatrix,
    q40_sensitivity,
    supported_dtype,
)

rng = np.random.default_rng(7)

_IQ4NL_KVALUES = np.array(
    [-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 27, 42, 58, 74, 96, 120],
    np.float32,
)
_MXFP4_LUT = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], np.float32)


def _encode_iq4_nl(w: np.ndarray) -> bytes:
    blocks = w.reshape(-1, 32)
    n = blocks.shape[0]
    raw = np.empty(n, dtype=[("scale", np.float16), ("qs", np.uint8, 16)])
    for i in range(n):
        s = np.abs(blocks[i]).max() / 120.0
        raw["scale"][i] = np.float16(s)
        v = blocks[i] / s
        idx = np.array(
            [int(np.argmin(np.abs(_IQ4NL_KVALUES - x))) for x in v], np.uint8
        )
        raw["qs"][i] = (idx[0::2] | (idx[1::2] << 4)).astype(np.uint8)
    return raw.tobytes()


def _encode_mxfp4(w: np.ndarray) -> tuple[bytes, np.ndarray]:
    blocks = w.reshape(-1, 32)
    n = blocks.shape[0]
    raw = np.zeros((n, 17), np.uint8)
    ref = np.empty((n, 32), np.float32)
    for i in range(n):
        e8 = int(np.ceil(np.log2(max(np.abs(blocks[i]).max(), 1e-9) / 6.0))) + 127
        raw[i, 0] = np.uint8(e8)
        s = 2.0 ** (e8 - 127)
        enc = np.zeros(32, np.uint8)
        for j, v in enumerate(blocks[i] / s):
            best = int(np.argmin(np.abs(_MXFP4_LUT - abs(v))))
            enc[j] = np.uint8(best | (8 if v < 0 else 0))
            ref[i, j] = _MXFP4_LUT[best] * (-1 if v < 0 else 1) * s
        raw[i, 1:] = (enc[0::2] | (enc[1::2] << 4)).astype(np.uint8)
    return raw.tobytes(), ref.reshape(-1)


def _info(name="blk.0.attn_q.weight", dtype="q4_0", n=128):
    return TensorInfo(
        name=name,
        n_dims=2,
        dims=[n, n],
        dtype_id=2,
        dtype_name=dtype,
        data_offset=0,
        raw_offset=0,
    )


def _dinfo(n_elements, dtype="q4_0", name="t.weight"):
    return TensorInfo(
        name=name,
        n_dims=1,
        dims=[n_elements],
        dtype_id=2,
        dtype_name=dtype,
        data_offset=0,
        raw_offset=0,
    )


class TestIq4Nl:
    def test_lut_exactness(self):
        # One block, scale 1, single known index -> exact LUT value out.
        raw = np.empty(1, dtype=[("scale", np.float16), ("qs", np.uint8, 16)])
        raw["scale"][0] = np.float16(1.0)
        idx = np.array([0, 3, 7, 15] * 8, np.uint8)
        raw["qs"][0] = (idx[0::2] | (idx[1::2] << 4)).astype(np.uint8)
        out = dequant_iq4_nl(raw.tobytes(), 32)
        expect = np.empty(32, np.float32)
        expect[0::2] = _IQ4NL_KVALUES[idx[0::2]]
        expect[1::2] = _IQ4NL_KVALUES[idx[1::2]]
        np.testing.assert_allclose(out, expect, rtol=0, atol=0)

    def test_error_within_lut_granularity(self):
        w = rng.standard_normal(256).astype(np.float32)
        out = dequant_iq4_nl(_encode_iq4_nl(w), 256)
        assert np.abs(out - w).max() < 0.35

    def test_rejects_non_multiple_of_32(self):
        with pytest.raises(ValueError):
            dequant_iq4_nl(b"\x00" * 18, 33)


class TestMxfp4:
    def test_matches_numpy_reference(self):
        w = rng.standard_normal(128).astype(np.float32)
        raw, ref = _encode_mxfp4(w)
        out = dequant_mxfp4(raw, 128)
        np.testing.assert_allclose(out, ref, rtol=1e-6, atol=1e-7)

    def test_negative_scale_bit(self):
        # Sign bit = nibble bit 3 -> negative values.
        raw = np.zeros(17, np.uint8)
        raw[0] = np.uint8(127)  # scale = 2^0 = 1
        raw[1] = np.uint8(0x08 | 0x03)  # lo nibble: -1.5
        raw[1] |= np.uint8((0x08 | 0x07) << 4)  # hi nibble: -6.0
        out = dequant_mxfp4(raw.tobytes(), 32)
        assert out[0] == -1.5 and out[1] == -6.0

    def test_e8m0_scale(self):
        raw = np.zeros(17, np.uint8)
        raw[0] = np.uint8(130)  # scale = 2^3 = 8
        raw[1] = np.uint8(0x07)  # lo nibble: +6 -> 48
        out = dequant_mxfp4(raw.tobytes(), 32)
        assert out[0] == 48.0


class TestSensitivity:
    def test_positive_finite(self):
        w = rng.standard_normal((64, 128)).astype(np.float32)
        s = q40_sensitivity(w)
        assert np.isfinite(s) and s > 0

    def test_unweighted_close_to_d2_over_12_normalized(self):
        # Uniform importance -> score ~= mean(d^2/12) / mean(w^2).
        w = rng.standard_normal((64, 128)).astype(np.float32)
        s = q40_sensitivity(w)
        d2 = (np.abs(w).max(axis=-1) / 8.0) ** 2 / 12.0
        ref = d2.mean() / (w**2).mean()
        assert abs(s - ref) / ref < 1e-3

    def test_scale_invariance(self):
        w = rng.standard_normal((64, 128)).astype(np.float32)
        assert abs(q40_sensitivity(w) - q40_sensitivity(w * 7.0)) < 1e-6

    def test_skewed_importance_reduces_score(self):
        # Importance concentrated on the largest-magnitude columns lowers
        # the relative error estimate.
        w = rng.standard_normal((64, 128)).astype(np.float32)
        w[:, :8] *= 10.0
        imp = np.zeros(128, np.float32)
        imp[:8] = 1.0
        assert q40_sensitivity(w, imp) < q40_sensitivity(w)

    def test_imp_length_mismatch_padded(self):
        w = rng.standard_normal((4, 64)).astype(np.float32)
        s = q40_sensitivity(w, np.ones(16, np.float32))
        assert np.isfinite(s)

    def test_bad_importance_falls_back_uniform(self):
        w = rng.standard_normal((4, 64)).astype(np.float32)
        assert q40_sensitivity(w, np.zeros(64, np.float32)) > 0


class TestChunkMean:
    def test_averages_chunks(self):
        e = np.arange(8, dtype=np.float32)
        np.testing.assert_allclose(
            chunk_mean(e, 2), np.array([2.0, 3.0, 4.0, 5.0], np.float32)
        )

    def test_single_chunk_passthrough(self):
        e = np.arange(4, dtype=np.float32)
        np.testing.assert_allclose(chunk_mean(e, 1), e)

    def test_non_divisible_passthrough(self):
        e = np.arange(5, dtype=np.float32)
        np.testing.assert_allclose(chunk_mean(e, 2), e)


class TestBuildMixedPlan:
    def test_threshold_routing(self):
        infos = [_info("a.weight"), _info("b.weight")]
        w = rng.standard_normal((128, 128)).astype(np.float32) * 0.01
        plan = build_mixed_plan(
            infos,
            None,
            weights_by_name={"a.weight": w, "b.weight": w},
            threshold=1e9,
        )
        assert plan.assignments == {"a.weight": "q4_0", "b.weight": "q4_0"}
        plan2 = build_mixed_plan(
            infos,
            None,
            weights_by_name={"a.weight": w, "b.weight": w},
            threshold=-1.0,
        )
        assert all(v == "bf16" for v in plan2.assignments.values())
        assert plan2.n_high_precision == 2

    def test_deterministic(self):
        infos = [_info(f"t{i}.weight") for i in range(4)]
        ws = {i.name: rng.standard_normal((64, 64)).astype(np.float32) for i in infos}
        p1 = build_mixed_plan(infos, None, weights_by_name=ws, threshold=0.0)
        p2 = build_mixed_plan(infos, None, weights_by_name=ws, threshold=0.0)
        assert p1.assignments == p2.assignments
        assert p1.scores == p2.scores

    def test_preferred_mxfp4(self):
        infos = [_info()]
        plan = build_mixed_plan(infos, None, threshold=1e9, preferred="mxfp4")
        assert plan.assignments[_info().name] == "mxfp4"

    def test_rejects_unknown_preferred(self):
        with pytest.raises(ValueError):
            build_mixed_plan([_info()], None, preferred="q6_K")

    def test_missing_weights_score_zero(self):
        infos = [_info("a.weight")]
        plan = build_mixed_plan(infos, None, threshold=1e9)
        assert plan.assignments["a.weight"] == "q4_0"
        assert plan.scores["a.weight"] == 0.0


class TestDispatch:
    def test_q40_matches_asfw(self):
        from fusion_mlx.migrate.asfw import ASFWConverter

        w = rng.standard_normal(256).astype(np.float32)
        blocks = w.reshape(8, 32)
        raw = np.empty(8, dtype=[("scale", np.float16), ("packed", np.uint8, 16)])
        for i in range(8):
            s = np.abs(blocks[i]).max() / 8.0
            raw["scale"][i] = np.float16(s)
            q = np.clip(np.round(blocks[i] / s) + 8, 0, 15).astype(np.uint8)
            raw["packed"][i] = (q[0::2] | (q[1::2] << 4)).astype(np.uint8)
        got = dispatch_tensor(_dinfo(256, "q4_0"), raw.tobytes())
        ref = ASFWConverter._dequant_q4_0(
            raw["scale"].astype(np.float32), raw["packed"]
        )
        np.testing.assert_allclose(got, ref, rtol=1e-5)

    def test_q80_handler(self):
        raw = np.empty(1, dtype=[("scale", np.float16), ("qs", np.int8, 32)])
        raw["scale"][0] = np.float16(0.5)
        raw["qs"][0] = np.arange(32, dtype=np.int8) - 16
        out = dispatch_tensor(_dinfo(32, "q8_0"), raw.tobytes())
        np.testing.assert_allclose(out, (np.arange(32) - 16) * 0.5, rtol=1e-6)

    def test_mxfp4_handler(self):
        w = rng.standard_normal(64).astype(np.float32)
        raw, ref = _encode_mxfp4(w)
        out = dispatch_tensor(_dinfo(64, "mxfp4"), raw)
        np.testing.assert_allclose(out, ref, rtol=1e-6)

    def test_iq4_nl_gated_off(self, monkeypatch):
        monkeypatch.setenv("FUSION_SHIM_IQ", "0")
        w = rng.standard_normal(32).astype(np.float32)
        with pytest.raises(UnsupportedQuantError):
            dispatch_tensor(_dinfo(32, "iq4_nl"), _encode_iq4_nl(w))

    def test_iq4_nl_gated_on(self, monkeypatch):
        monkeypatch.setenv("FUSION_SHIM_IQ", "1")
        assert supported_dtype("iq4_nl") is True
        w = rng.standard_normal(32).astype(np.float32)
        out = dispatch_tensor(_dinfo(32, "iq4_nl"), _encode_iq4_nl(w))
        assert out.shape == (32,)

    def test_iq_allowed_override(self, monkeypatch):
        monkeypatch.setenv("FUSION_SHIM_IQ", "0")
        w = rng.standard_normal(32).astype(np.float32)
        out = dispatch_tensor(_dinfo(32, "iq4_nl"), _encode_iq4_nl(w), iq_allowed=True)
        assert out.shape == (32,)

    def test_unsupported_raises(self):
        for dt in ("q6_K", "q4_K", "iq2_xs", "iq1_s", "f16"):
            with pytest.raises(UnsupportedQuantError):
                dispatch_tensor(_info(dtype=dt), b"\x00" * 64)

    def test_supported_dtype_table(self, monkeypatch):
        monkeypatch.setenv("FUSION_SHIM_IQ", "0")
        assert supported_dtype("q4_0") and supported_dtype("q8_0")
        assert supported_dtype("mxfp4")
        assert not supported_dtype("iq4_nl")
        assert not supported_dtype("q2_K")


class TestLoadImatrix:
    def _write_gguf(self, path, kvs, tensors=()):
        import struct

        def s(x):
            b = x.encode()
            return struct.pack("<Q", len(b)) + b

        buf = b"GGUF" + struct.pack("<IQQ", 3, 0, len(kvs))
        for k, (vt, v) in kvs.items():
            buf += s(k) + struct.pack("<I", vt)
            if vt == 8:
                buf += s(v)
            elif vt == 6:
                buf += struct.pack("<f", v)
            elif vt == 4:
                buf += struct.pack("<I", v)
            elif vt == 9:
                it, vals = v
                buf += struct.pack("<IQ", it, len(vals))
                for x in vals:
                    if it == 6:
                        buf += struct.pack("<f", x)
                    elif it == 4:
                        buf += struct.pack("<I", x)
        buf += struct.pack("<Q", len(tensors))
        for name, dims, dt, raw in tensors:
            buf += s(name) + struct.pack("<I", len(dims))
            buf += b"".join(struct.pack("<Q", d) for d in dims)
            buf += struct.pack("<IQ", dt, 0)
        pad = (32 - (len(buf) % 32)) % 32
        buf += b"\x00" * pad
        for _, _, _, raw in tensors:
            buf += raw
        path.write_bytes(buf)

    def test_roundtrip(self, tmp_path):
        p = tmp_path / "imatrix.gguf"
        kvs = {
            "imatrix.chunks": (4, 3),
            "a.weight": (9, (6, [6.0, 4.0])),
            "ignored.key": (8, "nope"),
            "b.bias": (9, (6, [1.0, 2.0])),
        }
        self._write_gguf(p, kvs)
        data = load_imatrix(p)
        assert data.n_chunks == 3
        assert "a.weight" in data.entries
        assert "ignored.key" not in data.entries
        np.testing.assert_allclose(data.entries["a.weight"], [6.0, 4.0])
        np.testing.assert_allclose(data.entries["b.bias"], [1.0, 2.0])
        assert data.source == str(p)

    def test_entry_for_exact_only(self):
        im = ImatrixData(entries={"a.weight": np.ones(4, np.float32)})
        assert im.entry_for("a.weight") is not None
        assert im.entry_for("blk.0.a.weight") is None

    def test_missing_file(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_imatrix(tmp_path / "nope.gguf")
