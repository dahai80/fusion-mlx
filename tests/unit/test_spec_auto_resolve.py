# SPDX-License-Identifier: Apache-2.0
import logging
from types import SimpleNamespace
from unittest.mock import patch

from fusion_mlx.speculative.auto_resolve import (
    AutoResolution,
    apply_resolution,
    resolve_spec_auto,
)
from fusion_mlx.speculative.auto_router import METHOD_MTP, METHOD_NGRAM

# MTP-eligible: supported model_type + at least one MTP layer.
MTP_CONFIG = {"model_type": "qwen3_5", "mtp_num_hidden_layers": 1}
# Non-MTP: unsupported model_type.
NON_MTP_CONFIG = {"model_type": "llama", "mtp_num_hidden_layers": 0}


class TestResolveSpecAuto:
    def test_mtp_eligible_picks_mtp(self):
        r = resolve_spec_auto(MTP_CONFIG)
        assert r.method == METHOD_MTP
        assert r.cli_target == "mtp"
        assert "MTP-eligible" in r.reason

    def test_non_mtp_picks_suffix(self):
        r = resolve_spec_auto(NON_MTP_CONFIG)
        assert r.method == METHOD_NGRAM
        assert r.cli_target == "suffix"

    def test_none_config_picks_suffix(self):
        r = resolve_spec_auto(None)
        assert r.method == METHOD_NGRAM

    def test_mtp_model_without_layers_picks_suffix(self):
        r = resolve_spec_auto({"model_type": "qwen3_5"})
        assert r.method == METHOD_NGRAM

    def test_probe_failure_falls_back_to_suffix(self, caplog):
        caplog.set_level(logging.WARNING, logger="fusion_mlx.speculative.auto_resolve")
        with patch(
            "fusion_mlx.speculative.auto_resolve.detect_mtp_eligibility",
            side_effect=RuntimeError("boom"),
        ):
            r = resolve_spec_auto(MTP_CONFIG)
        assert r.method == METHOD_NGRAM
        assert any("probe failed" in rec.message for rec in caplog.records)


class TestApplyResolution:
    def _args(self):
        return SimpleNamespace(
            spec_decode="auto",
            suffix_decoding=False,
            enable_mtp=False,
            enable_dflash=False,
            enable_dspark=False,
        )

    def test_apply_mtp_rides_spec_decode_slot(self):
        args = self._args()
        apply_resolution(args, AutoResolution(METHOD_MTP, "mtp", "r"))
        assert args.spec_decode == "mtp"
        assert args.suffix_decoding is False

    def test_apply_suffix_uses_suffix_decoding_flag(self):
        args = self._args()
        apply_resolution(args, AutoResolution(METHOD_NGRAM, "suffix", "r"))
        assert args.spec_decode == "none"
        assert args.suffix_decoding is True
