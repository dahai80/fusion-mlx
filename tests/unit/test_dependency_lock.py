# SPDX-License-Identifier: Apache-2.0
"""G5 dependency lock + public_api validation tests."""

from pathlib import Path

import pytest

from fusion_mlx.dependency_lock import (
    _sha256,
    generate_manifest,
    verify_manifest,
    verify_or_raise,
)
from fusion_mlx.public_api import __all__ as public_all
from fusion_mlx.public_api import validate_public_api


class TestVendorChecksums:
    def test_generate_manifest_nonempty(self):
        m = generate_manifest()
        assert len(m) > 0
        for rel, digest in m.items():
            assert "patches/" in rel
            assert len(digest) == 64

    def test_verify_manifest_clean(self):
        errors = verify_manifest()
        assert errors == []

    def test_verify_detects_tamper(self, tmp_path):
        src = Path(__file__).parent.parent.parent / "fusion_mlx" / "patches"
        m = generate_manifest()
        # Simulate tamper: change a hash
        first_key = next(iter(m))
        m_tampered = dict(m)
        m_tampered[first_key] = "0" * 64
        errors = verify_manifest(m_tampered)
        assert any("MISMATCH" in e for e in errors)

    def test_verify_detects_missing(self):
        m = generate_manifest()
        first_key = next(iter(m))
        m_missing = dict(m)
        del m_missing[first_key]
        errors = verify_manifest(m_missing)
        assert any("UNTRACKED" in e for e in errors)

    def test_verify_or_raise_passes(self):
        verify_or_raise()

    def test_verify_or_raise_raises_on_tamper(self):
        m = generate_manifest()
        first_key = next(iter(m))
        m[first_key] = "0" * 64
        with pytest.raises(RuntimeError, match="checksum verification failed"):
            verify_or_raise(m)

    def test_sha256_consistent(self, tmp_path):
        f = tmp_path / "test.txt"
        f.write_text("hello")
        h1 = _sha256(f)
        h2 = _sha256(f)
        assert h1 == h2
        assert len(h1) == 64


class TestPublicApiValidation:
    def test_validate_all_importable(self):
        missing = validate_public_api()
        assert missing == []

    def test_all_contains_version(self):
        assert "__version__" in public_all

    def test_all_contains_server_and_config(self):
        assert "Server" in public_all
        assert "create_app" in public_all
        assert "get_config" in public_all
        assert "ServerConfig" in public_all

    def test_all_contains_engines(self):
        for eng in ("TTSEngine", "ImageGenEngine", "VideoGenEngine", "EmbeddingEngine"):
            assert eng in public_all

    def test_all_contains_profile(self):
        assert "ServerProfile" in public_all
        assert "profile_from_config" in public_all
        assert "resolve_profile" in public_all
