# SPDX-License-Identifier: Apache-2.0
"""Unit tests for fusion_mlx.audio.registry — audio alias registry.

Covers AudioAliasEntry, _load_registry (skip _keys, validate type/hf_id/family),
resolve_audio_alias (lc match, hf_id index, None), is_audio_name,
list_audio_aliases, stt_aliases, tts_aliases.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from fusion_mlx.audio import registry as mod


class TestAudioAliasEntry:
    def test_frozen(self):
        e = mod.AudioAliasEntry(
            alias="kokoro", type="tts", hf_id="hex/kokoro", family="kokoro"
        )
        assert e.alias == "kokoro"
        assert e.default_voice is None
        with pytest.raises(Exception):
            e.alias = "x"


class TestLoadRegistry:
    def test_loads_real_registry(self):
        mod._reset_registry_cache()
        reg = mod._load_registry()
        assert isinstance(reg, dict)
        assert len(reg) > 0  # aliases.json has entries

    def test_caches_result(self):
        mod._reset_registry_cache()
        reg1 = mod._load_registry()
        reg2 = mod._load_registry()
        assert reg1 is reg2  # same cached object

    def test_skips_underscore_keys(self, tmp_path, monkeypatch):
        fake = {
            "_meta": {"version": 1},
            "kokoro": {"type": "tts", "hf_id": "hex/kokoro", "family": "kokoro"},
        }
        f = tmp_path / "aliases.json"
        f.write_text(json.dumps(fake))
        monkeypatch.setattr(mod, "_registry_path", lambda: str(f))
        mod._reset_registry_cache()
        reg = mod._load_registry()
        assert "_meta" not in reg
        assert "kokoro" in reg

    def test_invalid_type_raises(self, tmp_path, monkeypatch):
        fake = {"x": {"type": "invalid", "hf_id": "a/b", "family": "f"}}
        f = tmp_path / "aliases.json"
        f.write_text(json.dumps(fake))
        monkeypatch.setattr(mod, "_registry_path", lambda: str(f))
        mod._reset_registry_cache()
        with pytest.raises(ValueError, match="invalid type"):
            mod._load_registry()

    def test_missing_field_raises(self, tmp_path, monkeypatch):
        fake = {"x": {"type": "tts"}}  # missing hf_id, family
        f = tmp_path / "aliases.json"
        f.write_text(json.dumps(fake))
        monkeypatch.setattr(mod, "_registry_path", lambda: str(f))
        mod._reset_registry_cache()
        with pytest.raises(ValueError, match="missing required"):
            mod._load_registry()

    def test_bad_hf_id_raises(self, tmp_path, monkeypatch):
        fake = {"x": {"type": "tts", "hf_id": "no-slash", "family": "f"}}
        f = tmp_path / "aliases.json"
        f.write_text(json.dumps(fake))
        monkeypatch.setattr(mod, "_registry_path", lambda: str(f))
        mod._reset_registry_cache()
        with pytest.raises(ValueError, match="must be a HuggingFace"):
            mod._load_registry()

    def test_non_dict_value_raises(self, tmp_path, monkeypatch):
        f = tmp_path / "aliases.json"
        f.write_text(json.dumps({"x": "not a dict"}))
        monkeypatch.setattr(mod, "_registry_path", lambda: str(f))
        mod._reset_registry_cache()
        with pytest.raises(ValueError, match="must be an object"):
            mod._load_registry()


class TestResolveAudioAlias:
    def test_none_returns_none(self):
        assert mod.resolve_audio_alias(None) is None
        assert mod.resolve_audio_alias("") is None
        assert mod.resolve_audio_alias(123) is None

    def test_alias_match(self):
        mod._reset_registry_cache()
        reg = mod._load_registry()
        first_key = next(iter(reg))
        result = mod.resolve_audio_alias(first_key)
        assert result is not None
        assert result.alias == first_key

    def test_case_insensitive(self):
        mod._reset_registry_cache()
        reg = mod._load_registry()
        first_key = next(iter(reg))
        result = mod.resolve_audio_alias(first_key.upper())
        assert result is not None

    def test_no_match_returns_none(self):
        mod._reset_registry_cache()
        assert mod.resolve_audio_alias("nonexistent-alias-xyz") is None


class TestIsAudioName:
    def test_known_returns_true(self):
        mod._reset_registry_cache()
        reg = mod._load_registry()
        first_key = next(iter(reg))
        assert mod.is_audio_name(first_key) is True

    def test_unknown_returns_false(self):
        mod._reset_registry_cache()
        assert mod.is_audio_name("nonexistent-xyz") is False


class TestListAudioAliases:
    def test_returns_sorted_list(self):
        mod._reset_registry_cache()
        result = mod.list_audio_aliases()
        assert isinstance(result, list)
        assert len(result) > 0
        # sorted by alias
        aliases = [e.alias for e in result]
        assert aliases == sorted(aliases)


class TestSttAliases:
    def test_returns_dict(self):
        mod._reset_registry_cache()
        result = mod.stt_aliases()
        assert isinstance(result, dict)
        # all values are hf_ids
        for alias, hf_id in result.items():
            assert "/" in hf_id


class TestTtsAliases:
    def test_returns_dict(self):
        mod._reset_registry_cache()
        result = mod.tts_aliases()
        assert isinstance(result, dict)
        for alias, hf_id in result.items():
            assert "/" in hf_id


# ── speculative config ───────────────────────────────────────────────


from fusion_mlx.speculative import config as spec_mod


class TestSpeculativeConfig:
    def test_to_json_minimal(self):
        c = spec_mod.SpeculativeConfig(method="ddtree")
        j = c.to_json()
        assert "ddtree" in j
        assert "model" not in j  # None omitted

    def test_to_json_with_model(self):
        c = spec_mod.SpeculativeConfig(method="ddtree", model="qwen")
        assert "qwen" in c.to_json()

    def test_to_json_with_tree_budget(self):
        c = spec_mod.SpeculativeConfig(method="ddtree", tree_budget=10)
        assert "tree_budget" in c.to_json()

    def test_to_json_with_raw(self):
        c = spec_mod.SpeculativeConfig(method="ddtree", raw={"custom": "val"})
        assert "custom" in c.to_json()


class TestParseSpeculativeConfig:
    def test_empty_raises(self):
        with pytest.raises(spec_mod.SpeculativeConfigError, match="empty"):
            spec_mod.parse_speculative_config("")

    def test_invalid_json_raises(self):
        with pytest.raises(spec_mod.SpeculativeConfigError, match="invalid JSON"):
            spec_mod.parse_speculative_config("not json")

    def test_non_dict_raises(self):
        with pytest.raises(
            spec_mod.SpeculativeConfigError, match="must be a JSON object"
        ):
            spec_mod.parse_speculative_config("[1,2]")

    def test_missing_method_raises(self):
        with pytest.raises(
            spec_mod.SpeculativeConfigError, match="'method' key is required"
        ):
            spec_mod.parse_speculative_config('{"num_speculative_tokens": 5}')

    def test_unknown_method_raises(self):
        with patch.object(spec_mod, "get_spec_decoder", return_value=None):
            with patch.object(spec_mod, "iter_spec_decoders", return_value=[]):
                with pytest.raises(
                    spec_mod.SpeculativeConfigError, match="unknown method"
                ):
                    spec_mod.parse_speculative_config('{"method": "unknown"}')

    def test_invalid_num_spec_raises(self):
        with patch.object(
            spec_mod, "get_spec_decoder", return_value=MagicMock(method="ddtree")
        ):
            with pytest.raises(
                spec_mod.SpeculativeConfigError, match="num_speculative_tokens"
            ):
                spec_mod.parse_speculative_config(
                    '{"method": "ddtree", "num_speculative_tokens": 0}'
                )

    def test_valid_config(self):
        with patch.object(
            spec_mod, "get_spec_decoder", return_value=MagicMock(method="ddtree")
        ):
            cfg = spec_mod.parse_speculative_config(
                '{"method": "ddtree", "num_speculative_tokens": 3, "model": "qwen"}'
            )
            assert cfg.method == "ddtree"
            assert cfg.num_speculative_tokens == 3
            assert cfg.model == "qwen"

    def test_raw_keys_passed_through(self):
        with patch.object(
            spec_mod, "get_spec_decoder", return_value=MagicMock(method="dflash")
        ):
            cfg = spec_mod.parse_speculative_config(
                '{"method": "dflash", "block_size": 64}'
            )
            assert cfg.raw.get("block_size") == 64


class TestLegacyConfigs:
    def test_legacy_ddtree(self):
        c = spec_mod.legacy_ddtree_config(
            num_speculative_tokens=3, tree_budget=10, model="q"
        )
        assert c.method == "ddtree"
        assert c.num_speculative_tokens == 3
        assert c.tree_budget == 10
        assert c.model == "q"

    def test_legacy_dflash(self):
        c = spec_mod.legacy_dflash_config(num_speculative_tokens=2, model="q")
        assert c.method == "dflash"
        assert c.num_speculative_tokens == 2

    def test_legacy_mtp(self):
        c = spec_mod.legacy_mtp_config(num_speculative_tokens=4, model="q")
        assert c.method == "mtp"


class TestRequireMigrated:
    def test_ddtree_migrates(self):
        with patch.object(
            spec_mod, "get_spec_decoder", return_value=MagicMock(method="ddtree")
        ):
            cfg = spec_mod.require_migrated_speculative_config('{"method": "ddtree"}')
            assert cfg.method == "ddtree"

    def test_dflash_migrates(self):
        with patch.object(
            spec_mod, "get_spec_decoder", return_value=MagicMock(method="dflash")
        ):
            cfg = spec_mod.require_migrated_speculative_config('{"method": "dflash"}')
            assert cfg.method == "dflash"


# ── utils/decode ─────────────────────────────────────────────────────


from fusion_mlx.utils.decode import IncrementalDecoder


class TestIncrementalDecoder:
    def test_add_token_returns_delta(self):
        tok = MagicMock()
        tok.decode.side_effect = ["Hello", "Hello world", "Hello world!"]
        dec = IncrementalDecoder(tok)
        assert dec.add_token(1) == "Hello"
        assert dec.add_token(2) == " world"
        assert dec.add_token(3) == "!"

    def test_add_token_with_replacement_char_returns_empty(self):
        tok = MagicMock()
        tok.decode.return_value = "text�"
        dec = IncrementalDecoder(tok)
        assert dec.add_token(1) == ""

    def test_get_full_text_empty(self):
        tok = MagicMock()
        dec = IncrementalDecoder(tok)
        assert dec.get_full_text() == ""

    def test_get_full_text_with_tokens(self):
        tok = MagicMock()
        tok.decode.return_value = "full text"
        dec = IncrementalDecoder(tok)
        dec.add_token(1)
        assert dec.get_full_text() == "full text"

    def test_token_ids_property(self):
        tok = MagicMock()
        tok.decode.return_value = "x"
        dec = IncrementalDecoder(tok)
        dec.add_token(1)
        dec.add_token(2)
        assert dec.token_ids == [1, 2]

    def test_prev_text_property(self):
        tok = MagicMock()
        tok.decode.return_value = "hello"
        dec = IncrementalDecoder(tok)
        dec.add_token(1)
        assert dec.prev_text == "hello"

    def test_reset(self):
        tok = MagicMock()
        tok.decode.return_value = "x"
        dec = IncrementalDecoder(tok)
        dec.add_token(1)
        dec.reset()
        assert dec.token_ids == []
        assert dec.prev_text == ""
