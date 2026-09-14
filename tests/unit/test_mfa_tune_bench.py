# SPDX-License-Identifier: Apache-2.0
# O4.2 bench->tuning pipeline unit test. Validates shape-key encoding +
# table schema (backend/block_size/candidates). Real timing requires a
# Metal device — gated to the tune_bench __main__ run, not unit-tested here.

import json
import os
import tempfile


def test_tuning_key_encoding():
    from fusion_mlx.custom_kernels.mfa.tune_bench import _key

    assert _key(128, True, 4) == "d128_decode_b4"
    assert _key(64, False, 2) == "d64_prefill_b2"
    assert _key(256, True, 1) == "d256_decode_b1"


def test_tuning_table_schema():
    from fusion_mlx.custom_kernels.mfa.tune_bench import run_tuning_bench

    # Run the real microbench (fast — 18 shapes, 10 iters each on small tensors).
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tf:
        path = tf.name
    try:
        table = run_tuning_bench(output_path=path)
        assert len(table) >= 10, f"expected >=10 entries, got {len(table)}"
        with open(path) as f:
            raw = json.load(f)
        for key, entry in raw.items():
            assert "backend" in entry, f"{key} missing backend"
            assert "block_size" in entry, f"{key} missing block_size"
            assert isinstance(entry["block_size"], list)
            assert "candidates" in entry, f"{key} missing candidates"
            assert "ms" in entry, f"{key} missing ms"
            # backend must be a known AttentionBackend name
            assert entry["backend"] in (
                "MLX_SDPA",
                "NAX",
                "STEEL",
                "STEEL_DSPLIT",
                "PAGED_FUSED",
                "TURBOQUANT",
            ), f"{key} unknown backend {entry['backend']}"
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass


def test_tuning_lookup_reads_table():
    # dispatch_policy._tuning_lookup must read the emitted table and return
    # a DispatchDecision matching the measured winner.
    from fusion_mlx.custom_kernels.mfa import dispatch_policy as dp
    from fusion_mlx.custom_kernels.mfa.tune_bench import run_tuning_bench

    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tf:
        path = tf.name
    os.environ["FUSION_MFA_TUNING_TABLE"] = path
    dp._TUNING_TABLE_PATH = path  # path captured at import; override for test
    dp._TUNING_TABLE = None  # force reload
    try:
        table = run_tuning_bench(output_path=path)
        # Pick any shape that benched and confirm lookup returns it.
        first_key = next(iter(table))
        # parse key d{hd}_{phase}_b{batch}
        parts = first_key.split("_")
        hd = int(parts[0][1:])
        is_decode = parts[1] == "decode"
        b = int(parts[2][1:])
        dec = dp._tuning_lookup(hd, is_decode, b)
        assert dec is not None, f"{first_key} not found by lookup"
        assert dec.reason.startswith("tuning-table override")
    finally:
        os.environ.pop("FUSION_MFA_TUNING_TABLE", None)
        dp._TUNING_TABLE_PATH = ""
        dp._TUNING_TABLE = None
        try:
            os.unlink(path)
        except OSError:
            pass
