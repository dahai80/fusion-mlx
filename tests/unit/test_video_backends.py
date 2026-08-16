# SPDX-License-Identifier: Apache-2.0
# Tests for the video backend registry: resolve_backend auto-detection,
# constraints_for, validate_params, and Wan2Backend generate (stubbed pure-MLX
# wan2 port - no real model loading or generation).
from pathlib import Path

import pytest

from fusion_mlx.engines.video_backends import (
    BACKENDS,
    CogVideoBackend,
    LegacyLTXBackend,
    LTX2Backend,
    Wan2Backend,
    resolve_backend,
)


def _install_wan2_stub(monkeypatch):
    # Phase 5: Wan2 runs on the vendored pure-MLX port (fusion_mlx.video.wan2).
    # Stub the port's get_model_path + generate_video - no real weights/compute.
    calls = {"resolve": [], "generate": []}

    from fusion_mlx.video.wan2 import generate as port_gen
    from fusion_mlx.video.wan2 import utils as port_utils

    monkeypatch.setattr(
        port_utils,
        "get_model_path",
        lambda repo: calls["resolve"].append(repo) or Path("/tmp/fake-wan2"),
    )

    def generate_video(model_dir, prompt, **kwargs):
        calls["generate"].append({"model_dir": model_dir, "prompt": prompt, **kwargs})
        with open(kwargs["output_path"], "wb") as f:
            f.write(b"WANMP4" + str(kwargs.get("seed", 0)).encode())
        return None

    monkeypatch.setattr(port_gen, "generate_video", generate_video)
    return calls


class TestResolveBackend:
    def test_ltx2_autodetect(self):
        b = resolve_backend("ltx-2")
        assert isinstance(b, LTX2Backend)

    def test_wan2_autodetect_by_name(self):
        assert isinstance(resolve_backend("wan2.1"), Wan2Backend)

    def test_wan2_autodetect_by_repo_id(self):
        assert isinstance(resolve_backend("Wan-AI/Wan2.2-TI2V-5B"), Wan2Backend)

    def test_unknown_falls_back_to_ltx2(self):
        # Preserves Phase 0 single-backend fallback behavior.
        assert isinstance(resolve_backend("some-custom-video-model"), LTX2Backend)

    def test_legacy_ltx_autodetect(self):
        # Legacy LTX-Video (0.9.x) has a pure-MLX port (Phase 3). Detected by
        # name substring and by HF repo id.
        assert isinstance(resolve_backend("ltx-video"), LegacyLTXBackend)
        assert isinstance(resolve_backend("Lightricks/LTX-Video"), LegacyLTXBackend)

    def test_cogvideo_autodetect(self):
        # CogVideoX has no MLX port -> stub.
        assert isinstance(resolve_backend("cogvideo"), CogVideoBackend)
        assert isinstance(resolve_backend("THUDM/CogVideoX-2b"), CogVideoBackend)

    def test_legacy_does_not_shadow_modern_ltx(self):
        # Critical: ltx-2 / ltx-2.3 (shipped by mlx-video) must still resolve to
        # the real LTX2Backend, NOT the legacy stub.
        assert isinstance(resolve_backend("ltx-2"), LTX2Backend)
        assert isinstance(resolve_backend("ltx-2.3"), LTX2Backend)

    def test_explicit_legacy_and_cogvideo_aliases(self):
        assert isinstance(resolve_backend("x", explicit="ltx-video"), LegacyLTXBackend)
        assert isinstance(resolve_backend("x", explicit="ltx_video"), LegacyLTXBackend)
        assert isinstance(resolve_backend("x", explicit="cogvideo"), CogVideoBackend)
        assert isinstance(resolve_backend("x", explicit="cogvideox"), CogVideoBackend)

    def test_explicit_wan2(self):
        assert isinstance(resolve_backend("anything", explicit="wan2"), Wan2Backend)

    def test_explicit_alias(self):
        assert isinstance(resolve_backend("anything", explicit="wan2.2"), Wan2Backend)

    def test_explicit_invalid_raises(self):
        with pytest.raises(ValueError, match="unknown video backend"):
            resolve_backend("anything", explicit="bogus")

    def test_backends_registry_has_all(self):
        # ltx2 + wan2 ship real mlx-video impls; ltx_video_legacy is a pure-MLX
        # port (Phase 3); cogvideo graduated to a real port
        # (no MLX port exists upstream); skyreels is a pure-MLX port
        # (SkyReels-V3 R2V/V2V/A2V, Phase 4); ltx2_5 (22B) + minimax_h3 (33B)
        # are independent pure-MLX ports.
        assert set(BACKENDS) == {
            "ltx2",
            "cosmos",
            "hunyuanvideo",
            "wan2",
            "skyreels",
            "ltx_video_legacy",
            "svd",
            "cogvideo",
            "opensora",
            "uniworld",
            "ltx2_5",
            "minimax_h3",
        }

    def test_vace_alias_resolves_to_wan2(self):
        assert isinstance(resolve_backend("x", explicit="vace"), Wan2Backend)
        assert isinstance(resolve_backend("x", explicit="wan-vace"), Wan2Backend)
        assert isinstance(resolve_backend("x", explicit="wan2.1-vace"), Wan2Backend)

    def test_vace_autodetect_by_name(self):
        assert isinstance(resolve_backend("Wan2.1-VACE-14B"), Wan2Backend)


class TestInferConfigFromPath:
    def test_14b_i2v(self):
        from fusion_mlx.engines.video_backends.wan2 import _infer_config_from_path

        cfg = _infer_config_from_path("/models/Wan2.2-14B")
        assert cfg.dim == 5120
        assert cfg.num_layers == 40
        assert cfg.model_type == "t2v"

    def test_14b_i2v_path(self):
        from fusion_mlx.engines.video_backends.wan2 import _infer_config_from_path

        cfg = _infer_config_from_path("/models/Wan2.2-i2v-14B")
        assert cfg.model_type == "i2v"
        assert cfg.dim == 5120

    def test_5b_ti2v(self):
        from fusion_mlx.engines.video_backends.wan2 import _infer_config_from_path

        cfg = _infer_config_from_path("/models/Wan2.2-TI2V-5B")
        assert cfg.dim == 3072
        assert cfg.num_heads == 24
        assert cfg.model_type == "ti2v"

    def test_vace(self):
        from fusion_mlx.engines.video_backends.wan2 import _infer_config_from_path

        cfg = _infer_config_from_path("/models/Wan2.1-VACE-14B")
        assert cfg.model_type == "vace"
        assert cfg.dim == 5120

    def test_unknown_defaults_to_1_3b(self):
        from fusion_mlx.engines.video_backends.wan2 import _infer_config_from_path

        cfg = _infer_config_from_path("/models/some-wan-model")
        assert cfg.dim == 1536
        assert cfg.num_layers == 30
