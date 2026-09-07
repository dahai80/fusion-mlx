# SPDX-License-Identifier: Apache-2.0
# Tests for issue #805: slim default install. The core `dependencies` list must
# exclude heavy modality deps (mlx-vlm, dflash-mlx, transformers, markitdown,
# mlx-embeddings) so a bare `pip install fusion-mlx` boots text-LM serve
# without them. They live in extras; `[full]` aggregates every extra to
# reproduce the pre-#805 all-deps install. No network: pyproject is parsed with
# tomllib and the heavy deps are blocked via a sys.meta_path shim to prove the
# server import chain survives their absence.

import sys
import tomllib

import pytest

_HEAVY = {
    "mlx-vlm",
    "mlx-embeddings",
    "transformers",
    "dflash-mlx",
    "markitdown",
}


def _core_dep_names(pyproject_path):
    with open(pyproject_path, "rb") as f:
        data = tomllib.load(f)
    return data["project"]["dependencies"]


def _extras(pyproject_path):
    with open(pyproject_path, "rb") as f:
        data = tomllib.load(f)
    return data["project"]["optional-dependencies"]


def _dep_top_level(dep_spec):
    # "markitdown[pdf,docx,pptx]>=0.1.0" -> "markitdown"
    # "dflash==0.1.0; sys_platform == ..." -> "dflash"
    name = dep_spec.split(";")[0].split("[")[0]
    for sep in (">=", "<=", "==", "!=", "~=", ">", "<"):
        name = name.split(sep)[0]
    return name.strip().lower()


def test_heavy_modality_deps_absent_from_core():
    # Issue #805 acceptance: bare install must NOT pull mlx-vlm / dflash-mlx /
    # transformers / markitdown / mlx-embeddings.
    core = _core_dep_names("pyproject.toml")
    found = {d for d in core if _dep_top_level(d) in _HEAVY}
    assert not found, f"heavy deps still in core: {found}"


def test_heavy_deps_live_in_extras():
    # The five heavy deps moved out of core must still be installable via extras.
    extras = _extras("pyproject.toml")
    all_extra_specs = []
    for specs in extras.values():
        all_extra_specs.extend(specs)
    present = {_dep_top_level(s) for s in all_extra_specs}
    missing = _HEAVY - present
    assert not missing, f"heavy deps missing from extras: {missing}"


def test_full_meta_extra_aggregates_all_modality_extras():
    # `[full]` must self-reference every modality extra so the old all-deps
    # install is reproduced by `pip install "fusion-mlx[full]"`.
    extras = _extras("pyproject.toml")
    assert "full" in extras, "no [full] meta-extra"
    full_spec = extras["full"][0]
    # "fusion-mlx[vlm,dflash,embeddings,document,...]"
    assert full_spec.startswith("fusion-mlx[") and full_spec.endswith("]")
    inner = full_spec[len("fusion-mlx[") : -1]
    referenced = {g.strip() for g in inner.split(",")}
    # Every modality extra (skip full itself + dev) should be aggregated.
    expected = set(extras.keys()) - {"full", "dev"}
    missing = expected - referenced
    assert not missing, f"[full] does not aggregate: {missing}"


def test_slim_server_imports_without_heavy_deps(monkeypatch):
    # Simulate a slim install: block the heavy deps at import time and prove the
    # server boot import chain (fusion_mlx, .server, .cli) survives.
    blocked = {"mlx_vlm", "mlx_embeddings", "transformers", "dflash", "markitdown"}

    class _Blocker:
        def find_module(self, name, path=None):
            if name.split(".")[0] in blocked:
                return self

        def load_module(self, name):
            raise ImportError(f"blocked-sim: {name} absent (slim install)")

    monkeypatch.setattr(sys, "meta_path", [_Blocker()] + sys.meta_path)
    import fusion_mlx
    import fusion_mlx.cli
    import fusion_mlx.server

    assert fusion_mlx is not None


def test_modality_extras_still_installable_standalone():
    # Each modality extra (vision/audio/image/video) must remain independently
    # installable: present and non-empty.
    extras = _extras("pyproject.toml")
    for extra in ("vision", "audio", "image", "video"):
        assert extra in extras and extras[extra], f"extra [{extra}] empty/missing"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
