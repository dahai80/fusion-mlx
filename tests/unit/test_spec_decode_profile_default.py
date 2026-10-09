# SPDX-License-Identifier: Apache-2.0
"""Spec-decode profile-default bridge — efficiency fix.

The profile presets advertise spec decode as default-on for
standard/full/turbo (_PRESET_SPEC_DEFAULT) and the startup banner prints
"spec_decode=ON(default)", but the argparse default was "none" and nothing
bridged the preset to args.spec_decode. Every default-profile serve ran
stock decode with MTP/suffix speedups silently off. These tests verify the
resolve_spec_decode_from_profile bridge closes that gap.
"""

from fusion_mlx.profile import resolve_spec_decode_from_profile


class TestResolveSpecDecodeFromProfile:
    def test_standard_profile_resolves_to_auto(self):
        result = resolve_spec_decode_from_profile("none", "standard")
        assert result == "auto"

    def test_full_profile_resolves_to_auto(self):
        result = resolve_spec_decode_from_profile("none", "full")
        assert result == "auto"

    def test_turbo_profile_resolves_to_auto(self):
        result = resolve_spec_decode_from_profile("none", "turbo")
        assert result == "auto"

    def test_lite_profile_stays_none(self):
        result = resolve_spec_decode_from_profile("none", "lite")
        assert result is None

    def test_no_profile_stays_none(self):
        result = resolve_spec_decode_from_profile("none", None)
        assert result is None

    def test_operator_chose_mtp_stays_none(self):
        result = resolve_spec_decode_from_profile("mtp", "standard")
        assert result is None

    def test_operator_chose_auto_stays_none(self):
        result = resolve_spec_decode_from_profile("auto", "standard")
        assert result is None

    def test_operator_chose_dflash_stays_none(self):
        result = resolve_spec_decode_from_profile("dflash", "standard")
        assert result is None

    def test_unknown_profile_defaults_to_auto(self):
        result = resolve_spec_decode_from_profile("none", "custom_profile")
        assert result == "auto"

    def test_empty_string_spec_decode_treated_as_none(self):
        result = resolve_spec_decode_from_profile("", "standard")
        assert result == "auto"
